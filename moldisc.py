#!/usr/bin/env python3
"""MolDisc closed-loop molecular discovery orchestration."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "results" / ".matplotlib"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "results" / ".cache" / "huggingface"))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from models.utilsReport import (
    calculate_pairwise_tanimoto,
    plot_and_save_combined_tanimoto_histogram,
    smiles_to_fingerprints,
)
from sascore import SAscore
from SMILESX import inference, loadmodel, main


_OPERATIONAL_ARGUMENTS = {
    "project_folder",
    "data_err",
    "reuse_smilesx_models",
    "log_verbose",
    "train_verbose",
    "sub_GPU_ids",
    "gpt_device",
    "gpt_environment",
    "gpt_python_executable",
    "resume",
    "pause_after_cycles",
    "option_save",
    "option_show",
    "num_mol_top",
    "num_mol_bottom",
    "tanimoto_max_pairs",
    "cuda",
    "remove_tmp",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _jsonable(value):
    """Return a stable JSON-compatible representation for provenance records."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_allowed_elements(elements):
    if elements is None:
        return None
    if isinstance(elements, str):
        raise ValueError("gpt_allowed_elements must be a sequence, not a string.")
    periodic_table = Chem.GetPeriodicTable()
    normalized = []
    for symbol in elements:
        symbol = str(symbol).strip()
        if not symbol:
            continue
        try:
            atomic_number = int(periodic_table.GetAtomicNumber(symbol))
        except Exception as exc:
            raise ValueError(f"Unknown or non-canonical element symbol: {symbol!r}") from exc
        if atomic_number <= 0 or periodic_table.GetElementSymbol(atomic_number) != symbol:
            raise ValueError(f"Unknown or non-canonical element symbol: {symbol!r}")
        if symbol not in normalized:
            normalized.append(symbol)
    if not normalized:
        raise ValueError("gpt_allowed_elements must contain at least one element symbol.")
    return normalized


def _safe_remove_run_artifact(project_dir: Path, path: Path) -> None:
    """Remove one explicitly scoped run artifact after containment checks."""
    project_dir = project_dir.resolve()
    candidate = path.resolve()
    if candidate == project_dir or project_dir not in candidate.parents:
        raise RuntimeError(f"Refusing to remove path outside the run directory: {candidate}")
    if candidate.is_dir():
        shutil.rmtree(candidate)
    elif candidate.exists():
        candidate.unlink()


def _gpt_environment_info(python_executable: str, pretrained_model: str) -> dict:
    """Query the isolated generator environment and resolved Hub revision."""
    query = r'''
import json, platform, sys
payload = {"python": platform.python_version(), "executable": sys.executable}
try:
    import torch
    payload.update({
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    })
except Exception as exc:
    payload["torch_error"] = repr(exc)
try:
    import transformers, accelerate
    payload["transformers"] = transformers.__version__
    payload["accelerate"] = accelerate.__version__
    from transformers.utils.hub import cached_file
    resolved = cached_file(sys.argv[1], "config.json")
    payload["resolved_model_config"] = str(resolved)
    parts = str(resolved).replace("\\", "/").split("/")
    payload["resolved_model_revision"] = (
        parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
    )
except Exception as exc:
    payload["transformers_error"] = repr(exc)
print(json.dumps(payload))
'''
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", query, str(pretrained_model)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"query_error": repr(exc), "executable": str(python_executable)}


def _canonicalize(smiles: str):
    try:
        molecule = Chem.MolFromSmiles(str(smiles).strip())
        if molecule is None:
            return None
        Chem.SanitizeMol(molecule)
        return Chem.MolToSmiles(molecule, canonical=True)
    except Exception:
        return None


def _clean_input_data(
    labeled: pd.DataFrame,
    unlabeled: pd.DataFrame,
    model_type: str,
    duplicate_policy: str | None = None,
):
    """Validate, canonicalize, and de-duplicate the input tables.

    Regression datasets commonly contain replicate measurements for the same
    canonical structure.  Their default treatment is a mean aggregation, which
    retains the molecule without allowing it to occur in more than one split.
    Contradictory classification labels remain an error by default because an
    arithmetic mean would silently change a binary endpoint.
    """
    required_labeled = {"smiles", "property"}
    required_unlabeled = {"smiles"}
    if not required_labeled.issubset(labeled.columns):
        raise ValueError("labeled.csv must contain 'smiles' and 'property' columns.")
    if not required_unlabeled.issubset(unlabeled.columns):
        raise ValueError("unlabeled.csv must contain a 'smiles' column.")

    labeled = labeled.loc[:, ["smiles", "property"]].copy()
    unlabeled = unlabeled.loc[:, ["smiles"]].copy()
    labeled["property"] = pd.to_numeric(labeled["property"], errors="raise")
    if not np.isfinite(labeled["property"].to_numpy(dtype=float)).all():
        raise ValueError("All labeled property values must be finite.")
    if model_type == "classification" and not set(labeled["property"].unique()).issubset({0, 1}):
        raise ValueError("Classification labels must contain only 0 and 1.")

    labeled["canonical_smiles"] = labeled["smiles"].map(_canonicalize)
    unlabeled["canonical_smiles"] = unlabeled["smiles"].map(_canonicalize)
    invalid_labeled = labeled.loc[labeled["canonical_smiles"].isna(), "smiles"].astype(str).tolist()
    invalid_unlabeled = unlabeled.loc[unlabeled["canonical_smiles"].isna(), "smiles"].astype(str).tolist()
    labeled = labeled.dropna(subset=["canonical_smiles"]).copy()
    unlabeled = unlabeled.dropna(subset=["canonical_smiles"]).copy()

    if duplicate_policy is None:
        duplicate_policy = "mean" if model_type == "regression" else "error"
    allowed_policies = {"error", "mean", "median", "first", "drop"}
    if duplicate_policy not in allowed_policies:
        raise ValueError(
            "duplicate_policy must be one of: " + ", ".join(sorted(allowed_policies))
        )
    if model_type == "classification" and duplicate_policy in {"mean", "median"}:
        raise ValueError(
            "Classification duplicate_policy cannot be 'mean' or 'median'; "
            "use 'error', 'first', or 'drop'."
        )

    property_counts = labeled.groupby("canonical_smiles")["property"].nunique()
    conflicting = property_counts[property_counts > 1].index.tolist()
    if conflicting and duplicate_policy == "error":
        raise ValueError(
            "Conflicting property values were found for canonical duplicates: "
            + ", ".join(conflicting[:5])
        )

    conflicting_details = []
    for canonical_smiles in conflicting:
        values = labeled.loc[
            labeled["canonical_smiles"] == canonical_smiles, "property"
        ].astype(float)
        conflicting_details.append(
            {
                "smiles": canonical_smiles,
                "replicate_count": int(len(values)),
                "values": values.tolist(),
                "range": float(values.max() - values.min()),
            }
        )

    labeled_duplicates = int(labeled.duplicated("canonical_smiles").sum())
    unlabeled_duplicates = int(unlabeled.duplicated("canonical_smiles").sum())
    if duplicate_policy == "drop" and conflicting:
        labeled = labeled.loc[~labeled["canonical_smiles"].isin(conflicting)].copy()
    elif duplicate_policy in {"mean", "median"}:
        aggregation = duplicate_policy
        labeled = (
            labeled.groupby("canonical_smiles", as_index=False, sort=False)["property"]
            .agg(aggregation)
            .loc[:, ["canonical_smiles", "property"]]
        )
    else:
        labeled = labeled.drop_duplicates("canonical_smiles", keep="first")
    unlabeled = unlabeled.drop_duplicates("canonical_smiles", keep="first")

    overlap = sorted(set(labeled["canonical_smiles"]) & set(unlabeled["canonical_smiles"]))
    unlabeled = unlabeled.loc[~unlabeled["canonical_smiles"].isin(overlap)].copy()
    labeled["smiles"] = labeled.pop("canonical_smiles")
    unlabeled["smiles"] = unlabeled.pop("canonical_smiles")
    labeled = labeled.reset_index(drop=True)
    unlabeled = unlabeled.reset_index(drop=True)

    report = {
        "labeled_rows_after_cleaning": len(labeled),
        "unlabeled_rows_after_cleaning": len(unlabeled),
        "invalid_labeled_count": len(invalid_labeled),
        "invalid_labeled_smiles": invalid_labeled,
        "invalid_unlabeled_count": len(invalid_unlabeled),
        "invalid_unlabeled_smiles": invalid_unlabeled,
        "canonical_labeled_duplicates_removed": labeled_duplicates,
        "conflicting_labeled_duplicate_count": len(conflicting),
        "conflicting_labeled_duplicate_policy": duplicate_policy,
        "conflicting_labeled_duplicates": conflicting_details,
        "canonical_unlabeled_duplicates_removed": unlabeled_duplicates,
        "labeled_unlabeled_overlap_removed_from_unlabeled_count": len(overlap),
        "labeled_unlabeled_overlap_removed_from_unlabeled": overlap,
    }
    return labeled, unlabeled, report


def _package_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in (
        "numpy",
        "pandas",
        "tensorflow",
        "rdkit",
        "scikit-learn",
        "torch",
        "transformers",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return versions


def _smilesx_artifacts_complete(
    outdir: Path,
    data_name: str,
    augmentation: bool,
    k_fold_number: int,
    n_runs: int,
    scale_output: bool,
) -> bool:
    """Return whether the complete requested SMILES-X ensemble can be reused."""
    train_dir = outdir / data_name / ("Augm" if augmentation else "Can") / "Train"
    required = [train_dir / "Other" / f"{data_name}_Vocabulary.txt"]
    required.extend(
        train_dir
        / "Models"
        / f"{data_name}_Model_Fold_{fold}_Run_{run}.hdf5"
        for fold in range(int(k_fold_number))
        for run in range(int(n_runs))
    )
    if scale_output:
        required.extend(
            train_dir
            / "Other"
            / "Scalers"
            / f"{data_name}_Scaler_Outputs_Fold_{fold}.pkl"
            for fold in range(int(k_fold_number))
        )
    return all(path.is_file() for path in required)


def _smilesx_required_artifacts(
    outdir: Path,
    data_name: str,
    augmentation: bool,
    k_fold_number: int,
    n_runs: int,
    scale_output: bool,
) -> list[Path]:
    train_dir = outdir / data_name / ("Augm" if augmentation else "Can") / "Train"
    required = [train_dir / "Other" / f"{data_name}_Vocabulary.txt"]
    required.extend(
        train_dir / "Models" / f"{data_name}_Model_Fold_{fold}_Run_{run}.hdf5"
        for fold in range(int(k_fold_number))
        for run in range(int(n_runs))
    )
    if scale_output:
        required.extend(
            train_dir
            / "Other"
            / "Scalers"
            / f"{data_name}_Scaler_Outputs_Fold_{fold}.pkl"
            for fold in range(int(k_fold_number))
        )
    return required


def _smilesx_signature_path(outdir: Path, data_name: str, augmentation: bool) -> Path:
    train_dir = outdir / data_name / ("Augm" if augmentation else "Can") / "Train"
    return train_dir / "Other" / "moldisc_smilesx_signature.json"


def _smilesx_artifacts_reusable(
    outdir: Path,
    data_name: str,
    augmentation: bool,
    k_fold_number: int,
    n_runs: int,
    scale_output: bool,
    expected_specification: dict,
) -> tuple[bool, str]:
    signature_path = _smilesx_signature_path(outdir, data_name, augmentation)
    if not signature_path.is_file():
        return False, "predictor signature is missing"
    try:
        signature = json.loads(signature_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"predictor signature is unreadable: {exc}"
    if signature.get("specification") != expected_specification:
        return False, "predictor specification changed"
    required = _smilesx_required_artifacts(
        outdir, data_name, augmentation, k_fold_number, n_runs, scale_output
    )
    if not all(path.is_file() for path in required):
        return False, "one or more predictor artifacts are missing"
    recorded = signature.get("artifact_sha256", {})
    for path in required:
        relative = str(path.relative_to(outdir)).replace("\\", "/")
        if recorded.get(relative) != _sha256(path):
            return False, f"predictor artifact digest changed: {relative}"
    return True, "signed predictor artifacts match"


def _archive_fresh_run_artifacts(project_dir: Path, preserve_smilesx: bool) -> Path | None:
    """Move only known mutable outputs into a recoverable, scoped archive."""
    known = [
        project_dir / "GPT",
        project_dir / "tmp",
        project_dir / "final",
        project_dir / "analysis",
        project_dir / "run_state.pkl",
        project_dir / "run_manifest.json",
        project_dir / "cycle_metrics.csv",
        project_dir / "data_validation.json",
    ]
    if not preserve_smilesx:
        known.append(project_dir / "SMILESX")
    existing = [path for path in known if path.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = project_dir / ".archive" / stamp
    suffix = 1
    while archive_root.exists():
        archive_root = project_dir / ".archive" / f"{stamp}-{suffix}"
        suffix += 1
    archive_root.mkdir(parents=True)
    resolved_project = project_dir.resolve()
    for path in existing:
        source = path.resolve()
        if resolved_project not in source.parents:
            raise RuntimeError(f"Fresh-run artifact escaped the project directory: {source}")
        destination = (archive_root / path.name).resolve()
        if archive_root.resolve() not in destination.parents:
            raise RuntimeError(f"Fresh-run archive escaped its root: {destination}")
        shutil.move(str(source), str(destination))
    return archive_root


def _target_reached(values: pd.Series, target, mode: str) -> bool:
    if target is None or values.empty:
        return False
    if mode == "max":
        return bool(values.max() >= target)
    return bool(values.min() <= target)


def moldisc(
    project_name="data_1",
    input_file_labeled="labeled.csv",
    input_file_unlabeled="unlabeled.csv",
    project_folder="projects",
    data_err=None,
    data_name="test",
    data_units="units",
    data_label="Test label",
    smiles_concat=True,
    geomopt_mode="off",
    bayopt_mode="off",
    train_mode="on",
    model_type="regression",
    duplicate_policy=None,
    scale_output=True,
    bs_bounds=[64],
    lr_bounds=[2.0],
    embed_bounds=[64],
    lstm_bounds=[64],
    tdense_bounds=[64],
    bs_ref=128,
    lr_ref=3.0,
    embed_ref=8,
    lstm_ref=8,
    tdense_ref=8,
    dense_depth=3,
    k_fold_number=2,
    n_runs=5,
    check_smiles=True,
    augmentation=True,
    augmentation_max_per_molecule=None,
    reuse_smilesx_models=True,
    bayopt_n_rounds=5,
    bayopt_n_epochs=25,
    bayopt_n_runs=25,
    n_gpus=1,
    n_epochs=3,
    log_verbose=True,
    train_verbose=True,
    sub_GPU_ids="0",
    gpt_pretrained_model="gpt2",
    gpt_augmentation=2,
    gpt_data_split=0.8,
    gpt_tr_epochs=10,
    gpt_initial_epochs=None,
    gpt_cycle_epochs=None,
    gpt_tr_batch_size=16,
    gpt_eval_batch_size=16,
    gpt_warmup_steps=10,
    gpt_decay=0.01,
    gpt_patience=3,
    gpt_device=0,
    gpt_num_generation=10000,
    gpt_initial_generation=None,
    gpt_remove_ionic="all",
    gpt_num_attempts=5,
    gpt_generation_batch_size=128,
    gpt_max_new_tokens=100,
    gpt_min_carbon_atoms=1,
    gpt_min_heavy_atoms=2,
    gpt_max_heavy_atoms=None,
    gpt_max_molecular_weight=None,
    gpt_max_logp=None,
    gpt_allowed_elements=None,
    gpt_environment="subGPT",
    gpt_python_executable=None,
    smilesx_inference_augmentation=False,
    smilesx_inference_batch_size=512,
    smilesx_inference_max_augmentations=None,
    max_generation=100,
    max_generated_molecules=-1,
    cycles=1,
    target_property_value=None,
    target_property_mode="max",
    sa_score=5,
    cutoff=0.5,
    classification_threshold=0.5,
    feedback_max_per_cycle=None,
    collapse_min_validity=None,
    collapse_min_uniqueness=None,
    collapse_min_novelty=None,
    collapse_min_yield=None,
    collapse_patience=2,
    random_seed=42,
    resume=False,
    pause_after_cycles=None,
    option_save=True,
    option_show=False,
    num_mol_top=5,
    num_mol_bottom=5,
    tanimoto_max_pairs=200000,
    cuda=0,
    patience=100,
    remove_tmp=False,
):
    """Run a reproducible, bounded MolDisc discovery campaign.

    ``max_generation`` limits selected molecules for backward compatibility.
    ``max_generated_molecules`` limits globally novel, valid generated candidates.
    At least one deterministic stopping rule (cycle, selected count, generated
    count, or target property) must be enabled.
    """
    invocation_arguments = dict(locals())
    del data_err  # Retained in the public API for compatibility.

    signature_arguments = set(inspect.signature(moldisc).parameters)
    if signature_arguments != set(invocation_arguments):
        missing = sorted(signature_arguments - set(invocation_arguments))
        raise RuntimeError(f"Provenance capture missed public arguments: {missing}")

    if model_type not in {"regression", "classification"}:
        raise ValueError("model_type must be 'regression' or 'classification'.")
    if target_property_mode not in {"max", "min"}:
        raise ValueError("target_property_mode must be 'max' or 'min'.")
    if not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be between 0 and 1.")
    if not 0 <= classification_threshold <= 1:
        raise ValueError("classification_threshold must be between 0 and 1.")
    if collapse_patience < 1:
        raise ValueError("collapse_patience must be at least 1.")
    if feedback_max_per_cycle is not None and int(feedback_max_per_cycle) < 1:
        raise ValueError("feedback_max_per_cycle must be None or a positive integer.")
    for cap_name, cap_value in (
        ("augmentation_max_per_molecule", augmentation_max_per_molecule),
        ("smilesx_inference_max_augmentations", smilesx_inference_max_augmentations),
        ("gpt_max_heavy_atoms", gpt_max_heavy_atoms),
        ("gpt_max_molecular_weight", gpt_max_molecular_weight),
    ):
        if cap_value is not None and int(cap_value) < 1:
            raise ValueError(f"{cap_name} must be None or a positive integer.")
    for epoch_value in (gpt_tr_epochs, gpt_initial_epochs, gpt_cycle_epochs):
        if epoch_value is not None and int(epoch_value) < 1:
            raise ValueError("GPT epoch counts must be positive integers.")
    if gpt_initial_generation is not None and int(gpt_initial_generation) < 1:
        raise ValueError("gpt_initial_generation must be a positive integer.")
    if pause_after_cycles is not None and int(pause_after_cycles) < 1:
        raise ValueError("pause_after_cycles must be None or a positive integer.")
    if cycles == 0 or cycles < -1:
        raise ValueError("cycles must be -1 or a positive integer.")
    if max_generation < -1 or max_generated_molecules < -1:
        raise ValueError("Molecule limits must be -1 or non-negative integers.")
    if (
        cycles == -1
        and max_generation == -1
        and max_generated_molecules == -1
        and target_property_value is None
    ):
        raise ValueError("Enable at least one deterministic stopping rule.")
    gpt_allowed_elements = _normalize_allowed_elements(gpt_allowed_elements)

    random_seed = int(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(random_seed)
    except Exception:
        pass

    if (
        not isinstance(project_name, str)
        or not project_name.strip()
        or Path(project_name).name != project_name
        or project_name in {".", ".."}
    ):
        raise ValueError("project_name must be a single, non-empty directory name.")

    project_folder_path = Path(project_folder).expanduser()
    if not project_folder_path.is_absolute():
        project_folder_path = REPO_ROOT / project_folder_path
    project_folder_path = project_folder_path.resolve()
    project_dir = (project_folder_path / project_name).resolve()
    if project_folder_path != project_dir.parent:
        raise ValueError("The resolved project directory must remain under project_folder.")
    smilesx_output_dir = project_dir / "SMILESX"
    gpt_output_dir = project_dir / "GPT"
    gpt_log_dir = gpt_output_dir / "logs"
    tmp_output_dir = project_dir / "tmp"
    final_output_dir = project_dir / "final"
    project_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "data" / project_name
    labeled_path = data_dir / input_file_labeled
    unlabeled_path = data_dir / input_file_unlabeled
    if not labeled_path.exists() or not unlabeled_path.exists():
        raise FileNotFoundError(f"Input files were not found under {data_dir}.")

    labeled_raw = pd.read_csv(labeled_path)
    unlabeled_raw = pd.read_csv(unlabeled_path)
    input_labeled, input_unlabeled, validation_report = _clean_input_data(
        labeled_raw, unlabeled_raw, model_type, duplicate_policy=duplicate_policy
    )
    validation_report.update(
        {
            "labeled_source": str(labeled_path),
            "unlabeled_source": str(unlabeled_path),
            "labeled_sha256": _sha256(labeled_path),
            "unlabeled_sha256": _sha256(unlabeled_path),
        }
    )
    print(
        f"Loaded {len(input_labeled)} cleaned labeled and "
        f"{len(input_unlabeled)} cleaned unlabeled molecules from {data_dir}."
    )

    if gpt_python_executable is None:
        sibling_env_python = (
            Path(sys.executable).resolve().parent.parent
            / gpt_environment
            / ("python.exe" if os.name == "nt" else "bin/python")
        )
        if sibling_env_python.exists():
            gpt_python_executable = str(sibling_env_python)
        else:
            conda_executable = shutil.which("conda")
            if conda_executable is None:
                raise RuntimeError(
                    "Could not locate the GPT environment. Set gpt_python_executable."
                )
    else:
        gpt_python_executable = str(Path(gpt_python_executable).expanduser().resolve())

    invocation_arguments.update(
        {
            "project_folder": str(project_folder_path),
            "duplicate_policy": validation_report["conflicting_labeled_duplicate_policy"],
            "gpt_python_executable": gpt_python_executable,
            "gpt_allowed_elements": gpt_allowed_elements,
            "random_seed": random_seed,
        }
    )
    scientific_configuration = {
        name: _jsonable(value)
        for name, value in invocation_arguments.items()
        if name not in _OPERATIONAL_ARGUMENTS
    }
    scientific_configuration.update(
        {
            "labeled_sha256": validation_report["labeled_sha256"],
            "unlabeled_sha256": validation_report["unlabeled_sha256"],
        }
    )
    operational_configuration = {
        name: _jsonable(invocation_arguments[name])
        for name in sorted(_OPERATIONAL_ARGUMENTS)
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(scientific_configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()

    source_files = [REPO_ROOT / "moldisc.py", REPO_ROOT / "mainGPT.py"]
    source_files.extend(sorted((REPO_ROOT / "models").glob("*.py")))
    source_files.extend(sorted((REPO_ROOT / "SMILESX").glob("*.py")))
    source_sha256 = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_files}
    declaration_files = [
        "environment.yml",
        "environmentGPT.yml",
        "requirements.txt",
        "requirements-dev.txt",
        "requirementsGPT.txt",
        "requirementsGPT-dev.txt",
        "pyproject.toml",
    ]
    declaration_sha256 = {
        name: (_sha256(REPO_ROOT / name) if (REPO_ROOT / name).is_file() else None)
        for name in declaration_files
    }
    scientific_assets = {
        "models/fpscores.pkl.gz": _sha256(REPO_ROOT / "models" / "fpscores.pkl.gz")
    }

    cleaned_digest = hashlib.sha256(
        input_labeled[["smiles", "property"]]
        .sort_values("smiles", kind="stable")
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    ).hexdigest()
    predictor_keys = [
        "data_name", "data_units", "data_label", "smiles_concat", "geomopt_mode",
        "bayopt_mode", "train_mode", "model_type", "duplicate_policy", "scale_output",
        "bs_bounds", "lr_bounds", "embed_bounds", "lstm_bounds", "tdense_bounds",
        "bs_ref", "lr_ref", "embed_ref", "lstm_ref", "tdense_ref", "dense_depth",
        "k_fold_number", "n_runs", "check_smiles", "augmentation",
        "augmentation_max_per_molecule", "bayopt_n_rounds", "bayopt_n_epochs",
        "bayopt_n_runs", "n_gpus", "n_epochs", "patience", "random_seed",
    ]
    predictor_specification = {
        "schema_version": 1,
        "labeled_sha256": validation_report["labeled_sha256"],
        "cleaned_labeled_sha256": cleaned_digest,
        "parameters": {
            key: scientific_configuration[key] for key in predictor_keys
        },
        "source_sha256": {
            key: value
            for key, value in source_sha256.items()
            if key == "moldisc.py" or key.startswith("SMILESX/")
        },
    }

    smilesx_outdir = smilesx_output_dir / "outputs" / "0"
    predictor_reusable, predictor_reuse_reason = _smilesx_artifacts_reusable(
        smilesx_outdir,
        data_name,
        augmentation,
        k_fold_number,
        n_runs,
        scale_output,
        predictor_specification,
    )
    can_reuse_smilesx = bool(reuse_smilesx_models) and train_mode != "finetune" and predictor_reusable

    manifest_path = project_dir / "run_manifest.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != config_fingerprint:
            raise ValueError("The existing run manifest does not match the requested configuration.")
        if manifest.get("source_sha256") != source_sha256:
            raise ValueError("Source files changed since this run was started; refusing an irreproducible resume.")
        manifest.update({"status": "running", "resumed_at": _utc_now()})
    elif resume:
        raise FileNotFoundError("resume=True requires an existing run_manifest.json.")
    else:
        archive_path = _archive_fresh_run_artifacts(project_dir, preserve_smilesx=can_reuse_smilesx)
        manifest = {
            "manifest_schema_version": 2,
            "status": "running",
            "started_at": _utc_now(),
            "config_fingerprint": config_fingerprint,
            "scientific_configuration": scientific_configuration,
            "operational_configuration": operational_configuration,
            "configuration": scientific_configuration,
            "main_environment": _package_versions(),
            "gpt_environment": _gpt_environment_info(gpt_python_executable, gpt_pretrained_model),
            "source_sha256": source_sha256,
            "environment_declaration_sha256": declaration_sha256,
            "scientific_asset_sha256": scientific_assets,
            "predictor_reuse": {
                "requested": bool(reuse_smilesx_models),
                "reused": can_reuse_smilesx,
                "reason": predictor_reuse_reason,
            },
            "previous_run_archive": str(archive_path) if archive_path else None,
        }

    for directory in (
        smilesx_output_dir,
        gpt_output_dir,
        gpt_log_dir,
        tmp_output_dir,
        final_output_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _write_json(project_dir / "data_validation.json", validation_report)
    _write_json(manifest_path, manifest)
    if can_reuse_smilesx:
        print(
            "Complete SMILES-X artifacts found; skipping redundant "
            "train/validation/test reevaluation."
        )
    else:
        main.main(
            data_smiles=input_labeled[["smiles"]],
            data_prop=input_labeled[["property"]],
            data_err=None,
            data_name=data_name,
            data_units=data_units,
            data_label=data_label,
            smiles_concat=smiles_concat,
            outdir=str(smilesx_outdir),
            geomopt_mode=geomopt_mode,
            bayopt_mode=bayopt_mode,
            train_mode=train_mode,
            model_type=model_type,
            scale_output=scale_output,
            bs_bounds=bs_bounds,
            lr_bounds=lr_bounds,
            embed_bounds=embed_bounds,
            lstm_bounds=lstm_bounds,
            tdense_bounds=tdense_bounds,
            bs_ref=bs_ref,
            lr_ref=lr_ref,
            embed_ref=embed_ref,
            lstm_ref=lstm_ref,
            tdense_ref=tdense_ref,
            dense_depth=dense_depth,
            k_fold_number=k_fold_number,
            n_runs=n_runs,
            check_smiles=check_smiles,
            augmentation=augmentation,
            augmentation_max_per_molecule=augmentation_max_per_molecule,
            bayopt_n_rounds=bayopt_n_rounds,
            bayopt_n_epochs=bayopt_n_epochs,
            bayopt_n_runs=bayopt_n_runs,
            n_gpus=n_gpus,
            patience=patience,
            n_epochs=n_epochs,
            log_verbose=log_verbose,
            train_verbose=train_verbose,
            random_seed=random_seed,
        )
        required_predictor_artifacts = _smilesx_required_artifacts(
            smilesx_outdir,
            data_name,
            augmentation,
            k_fold_number,
            n_runs,
            scale_output,
        )
        missing_predictor_artifacts = [
            str(path) for path in required_predictor_artifacts if not path.is_file()
        ]
        if missing_predictor_artifacts:
            raise RuntimeError(
                "SMILES-X training completed without all required artifacts: "
                + ", ".join(missing_predictor_artifacts[:3])
            )
        predictor_signature = {
            "specification": predictor_specification,
            "artifact_sha256": {
                str(path.relative_to(smilesx_outdir)).replace("\\", "/"): _sha256(path)
                for path in required_predictor_artifacts
            },
            "created_at": _utc_now(),
        }
        _write_json(
            _smilesx_signature_path(smilesx_outdir, data_name, augmentation),
            predictor_signature,
        )

    smilesx_model = loadmodel.LoadModel(
        data_name=data_name,
        outdir=str(smilesx_output_dir / "outputs" / "0"),
        augment=augmentation,
        gpu_ind=cuda,
        return_attention=False,
    )

    input_labeled_original = input_labeled.copy()
    state_path = project_dir / "run_state.pkl"
    all_candidates = pd.DataFrame(
        columns=["smiles", "property", "prediction_std", "sa_score", "cycle", "selected"]
    )
    all_selected = pd.DataFrame(
        columns=["smiles", "property", "prediction_std", "sa_score", "cycle"]
    )
    seen_smiles = set(input_labeled["smiles"]) | set(input_unlabeled["smiles"])
    cycle_records = []
    collapse_streak = 0
    cycle = 0

    if resume and state_path.exists():
        with state_path.open("rb") as stream:
            state = pickle.load(stream)
        if state.get("config_fingerprint") != config_fingerprint:
            raise ValueError("The saved run state does not match the requested configuration.")
        input_labeled = state["input_labeled"]
        all_candidates = state["all_candidates"]
        all_selected = state["all_selected"]
        seen_smiles = set(state["seen_smiles"])
        cycle_records = list(state["cycle_records"])
        collapse_streak = int(state["collapse_streak"])
        cycle = int(state["next_cycle"])
        print(f"Resuming at cycle {cycle + 1} with {len(all_candidates)} candidates.")

    gpt_script = str(REPO_ROOT / "mainGPT.py")
    stop_reason = None
    run_started = time.perf_counter()
    invocation_start_cycle = cycle

    while stop_reason is None:
        if cycles != -1 and cycle >= cycles:
            stop_reason = "cycle_limit"
            break
        if max_generation != -1 and len(all_selected) >= max_generation:
            stop_reason = "selected_molecule_limit"
            break
        if max_generated_molecules != -1 and len(all_candidates) >= max_generated_molecules:
            stop_reason = "generated_molecule_limit"
            break

        requested_this_cycle = (
            int(gpt_initial_generation)
            if cycle == 0 and gpt_initial_generation is not None
            else int(gpt_num_generation)
        )
        generation_targets = [requested_this_cycle]
        if max_generated_molecules != -1:
            generation_targets.append(max_generated_molecules - len(all_candidates))
        if max_generation != -1:
            remaining_selected = max_generation - len(all_selected)
            if model_type == "regression" and cutoff > 0:
                generation_targets.append(int(np.ceil(remaining_selected / cutoff)))
            else:
                generation_targets.append(remaining_selected)
        cycle_generation_target = max(0, min(generation_targets))
        if cycle_generation_target == 0:
            stop_reason = "molecule_limit"
            break

        cycle_started = time.perf_counter()
        training_smiles = (
            input_labeled["smiles"].tolist() + input_unlabeled["smiles"].tolist()
        )
        smiles_input_path = tmp_output_dir / f"smiles_in_{cycle}.pkl"
        excluded_path = tmp_output_dir / f"smiles_excluded_{cycle}.pkl"
        args_path = tmp_output_dir / f"args{cycle}"
        _write_pickle(smiles_input_path, training_smiles)
        _write_pickle(excluded_path, sorted(seen_smiles))

        cycle_gpt_epochs = (
            gpt_initial_epochs if cycle == 0 and gpt_initial_epochs is not None
            else gpt_cycle_epochs if cycle > 0 and gpt_cycle_epochs is not None
            else gpt_tr_epochs
        )
        gpt_args = [
            str(cycle),
            gpt_pretrained_model,
            str(gpt_augmentation),
            str(gpt_data_split),
            str(gpt_output_dir),
            str(cycle_gpt_epochs),
            str(gpt_tr_batch_size),
            str(gpt_eval_batch_size),
            str(gpt_warmup_steps),
            str(gpt_decay),
            str(gpt_log_dir),
            str(gpt_patience),
            str(gpt_device),
            str(cycle_generation_target),
            gpt_remove_ionic,
            str(gpt_num_attempts),
            str(project_folder_path),
            project_name,
            str(gpt_generation_batch_size),
            str(gpt_max_new_tokens),
            str(random_seed),
            str(excluded_path),
            str(gpt_min_carbon_atoms),
            str(gpt_min_heavy_atoms),
            str(gpt_max_heavy_atoms),
            str(gpt_max_molecular_weight),
            str(gpt_max_logp),
            gpt_allowed_elements,
        ]
        _write_pickle(args_path, gpt_args)

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = sub_GPU_ids
        if gpt_python_executable is not None:
            command = [gpt_python_executable, gpt_script]
        else:
            command = [conda_executable, "run", "-n", gpt_environment, "python", gpt_script]
        subprocess.run(
            command
            + [
                "--iter",
                str(cycle),
                "--args_project_folder",
                str(project_folder_path),
                "--args_project_name",
                project_name,
            ],
            env=environment,
            cwd=str(REPO_ROOT),
            check=True,
        )

        generated_path = tmp_output_dir / f"smiles_new_{cycle}.pkl"
        with generated_path.open("rb") as stream:
            generated_smiles = list(pickle.load(stream))
        stats_path = tmp_output_dir / f"generation_stats_{cycle}.json"
        generation_stats = (
            json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        )

        # Defense in depth: the child generator already canonicalizes and excludes
        # seen structures, but the parent enforces global uniqueness again.
        canonical_generated = []
        for smiles in generated_smiles:
            canonical = _canonicalize(smiles)
            if canonical is not None and canonical not in seen_smiles:
                canonical_generated.append(canonical)
                seen_smiles.add(canonical)
        canonical_generated = list(dict.fromkeys(canonical_generated))
        if max_generated_molecules != -1:
            remaining = max_generated_molecules - len(all_candidates)
            canonical_generated = canonical_generated[:remaining]

        if not canonical_generated:
            cycle_record = {
                "cycle": cycle + 1,
                "requested": cycle_generation_target,
                "generated": 0,
                "selected": 0,
                "elapsed_seconds": time.perf_counter() - cycle_started,
                **{f"generation_{key}": value for key, value in generation_stats.items() if not isinstance(value, dict)},
            }
            cycle_records.append(cycle_record)
            pd.DataFrame(cycle_records).to_csv(project_dir / "cycle_metrics.csv", index=False)
            stop_reason = "no_novel_valid_molecules"
            break

        sa_scores = np.asarray([SAscore(smiles) for smiles in canonical_generated], dtype=float)
        predictions = inference.infer(
            model=smilesx_model,
            data_smiles=canonical_generated,
            augment=smilesx_inference_augmentation,
            check_smiles=check_smiles,
            log_verbose=log_verbose,
            batch_size=smilesx_inference_batch_size,
            max_augmentations=smilesx_inference_max_augmentations,
        )
        means = np.asarray(predictions["mean"], dtype=float).reshape(-1)
        standard_deviations = np.asarray(
            predictions.get(
                "sigma", predictions.get("std", np.full(len(means), np.nan))
            ),
            dtype=float,
        ).reshape(-1)
        if len(means) != len(canonical_generated):
            raise RuntimeError("SMILES-X returned a different number of predictions than inputs.")

        cycle_frame = pd.DataFrame(
            {
                "smiles": canonical_generated,
                "property": means,
                "prediction_std": standard_deviations,
                "sa_score": sa_scores,
                "cycle": cycle + 1,
                "selected": False,
            }
        )

        sa_collapse = bool(sa_score is not None and float(sa_scores.mean()) > float(sa_score))
        if model_type == "regression":
            ordered = cycle_frame.sort_values(
                "property", ascending=(target_property_mode == "min")
            )
            selected_count = int(np.ceil(len(ordered) * cutoff))
            selected_indices = ordered.index[:selected_count]
        else:
            selected_indices = cycle_frame.index[
                cycle_frame["property"] >= classification_threshold
            ]
        if sa_collapse:
            selected_indices = []
        if max_generation != -1:
            remaining_selected = max_generation - len(all_selected)
            selected_indices = list(selected_indices)[:remaining_selected]
        cycle_frame.loc[selected_indices, "selected"] = True
        selected_frame = cycle_frame.loc[cycle_frame["selected"]].drop(columns="selected")
        feedback_frame = selected_frame.sort_values(
            "property", ascending=(target_property_mode == "min")
        )
        if feedback_max_per_cycle is not None:
            feedback_frame = feedback_frame.head(int(feedback_max_per_cycle))

        all_candidates = (
            cycle_frame.copy()
            if all_candidates.empty
            else pd.concat([all_candidates, cycle_frame], ignore_index=True)
        )
        if not selected_frame.empty:
            all_selected = (
                selected_frame.copy()
                if all_selected.empty
                else pd.concat([all_selected, selected_frame], ignore_index=True)
            )
            input_labeled = pd.concat(
                [input_labeled, feedback_frame[["smiles", "property"]]],
                ignore_index=True,
            )

        cycle_frame.to_csv(tmp_output_dir / f"candidates_cycle_{cycle}.csv", index=False)
        selected_frame.to_csv(tmp_output_dir / f"selected_cycle_{cycle}.csv", index=False)
        all_candidates.to_csv(final_output_dir / "all_generated_candidates.csv", index=False)
        all_selected.to_csv(final_output_dir / "generated_smiles.csv", index=False)

        collapse_flags = []
        collapse_rules = {
            "validity": (collapse_min_validity, generation_stats.get("validity_rate")),
            "uniqueness": (collapse_min_uniqueness, generation_stats.get("uniqueness_rate")),
            "novelty": (collapse_min_novelty, generation_stats.get("novelty_rate")),
            "yield": (collapse_min_yield, generation_stats.get("yield_rate")),
        }
        for name, (minimum, observed) in collapse_rules.items():
            if minimum is not None and observed is not None and float(observed) < float(minimum):
                collapse_flags.append(name)
        if collapse_flags:
            collapse_streak += 1
        else:
            collapse_streak = 0

        cycle_record = {
            "cycle": cycle + 1,
            "training_molecules": len(training_smiles),
            "requested": cycle_generation_target,
            "generated": len(cycle_frame),
            "cumulative_generated": len(all_candidates),
            "selected": len(selected_frame),
            "cumulative_selected": len(all_selected),
            "feedback_molecules": len(feedback_frame),
            "sa_mean": float(sa_scores.mean()),
            "sa_median": float(np.median(sa_scores)),
            "property_mean": float(means.mean()),
            "property_std": float(means.std()),
            "property_min": float(means.min()),
            "property_max": float(means.max()),
            "collapse_flags": ";".join(collapse_flags),
            "collapse_streak": collapse_streak,
            "elapsed_seconds": time.perf_counter() - cycle_started,
            **{
                f"generation_{key}": value
                for key, value in generation_stats.items()
                if not isinstance(value, dict)
            },
        }
        cycle_records.append(cycle_record)
        pd.DataFrame(cycle_records).to_csv(project_dir / "cycle_metrics.csv", index=False)

        next_cycle = cycle + 1
        _write_pickle(
            state_path,
            {
                "config_fingerprint": config_fingerprint,
                "input_labeled": input_labeled,
                "all_candidates": all_candidates,
                "all_selected": all_selected,
                "seen_smiles": seen_smiles,
                "cycle_records": cycle_records,
                "collapse_streak": collapse_streak,
                "next_cycle": next_cycle,
            },
        )

        if _target_reached(cycle_frame["property"], target_property_value, target_property_mode):
            stop_reason = "target_property_reached"
        elif max_generated_molecules != -1 and len(all_candidates) >= max_generated_molecules:
            stop_reason = "generated_molecule_limit"
        elif max_generation != -1 and len(all_selected) >= max_generation:
            stop_reason = "selected_molecule_limit"
        elif cycles != -1 and next_cycle >= cycles:
            stop_reason = "cycle_limit"
        elif sa_collapse:
            stop_reason = "sa_score_collapse"
        elif collapse_streak >= collapse_patience:
            stop_reason = "model_collapse:" + ",".join(collapse_flags)
        elif (
            pause_after_cycles is not None
            and next_cycle - invocation_start_cycle >= int(pause_after_cycles)
        ):
            stop_reason = "operator_pause"
        cycle = next_cycle

        if remove_tmp:
            for path in (args_path, smiles_input_path, excluded_path):
                path.unlink(missing_ok=True)

    ascending = target_property_mode == "min"
    all_candidates = all_candidates.sort_values("property", ascending=ascending, na_position="last")
    all_selected = all_selected.sort_values("property", ascending=ascending, na_position="last")
    all_candidates.to_csv(final_output_dir / "all_generated_candidates.csv", index=False)
    all_selected.to_csv(final_output_dir / "generated_smiles.csv", index=False)
    pd.DataFrame(cycle_records).to_csv(project_dir / "cycle_metrics.csv", index=False)

    if option_save and not all_selected.empty:
        plot_top = min(len(all_selected), int(num_mol_top))
        plot_bottom = min(len(all_selected), int(num_mol_bottom))
        top_image = Draw.MolsToGridImage(
            [Chem.MolFromSmiles(smiles) for smiles in all_selected["smiles"].head(plot_top)]
        )
        bottom_image = Draw.MolsToGridImage(
            [Chem.MolFromSmiles(smiles) for smiles in all_selected["smiles"].tail(plot_bottom)]
        )
        top_image.save(final_output_dir / f"top_{plot_top}_molecules.png")
        bottom_image.save(final_output_dir / f"bottom_{plot_bottom}_molecules.png")
        try:
            input_fingerprints = smiles_to_fingerprints(input_labeled_original["smiles"])
            generated_fingerprints = smiles_to_fingerprints(all_selected["smiles"])
            if len(input_fingerprints) < 2 or len(generated_fingerprints) < 2:
                raise ValueError("At least two molecules per set are required for Tanimoto histograms.")
            input_similarities = calculate_pairwise_tanimoto(
                input_fingerprints, max_pairs=tanimoto_max_pairs, random_seed=random_seed
            )
            generated_similarities = calculate_pairwise_tanimoto(
                generated_fingerprints, max_pairs=tanimoto_max_pairs, random_seed=random_seed
            )
            plot_and_save_combined_tanimoto_histogram(
                input_similarities,
                generated_similarities,
                str(final_output_dir / "tanimoto.jpg"),
                500,
                show=option_show,
            )
        except Exception as exc:
            print(f"Tanimoto report failed: {exc}")

    model_provenance_path = gpt_output_dir / "model_provenance.json"
    generator_model_provenance = (
        json.loads(model_provenance_path.read_text(encoding="utf-8"))
        if model_provenance_path.is_file()
        else None
    )
    manifest.update(
        {
            "status": "paused" if stop_reason == "operator_pause" else "completed",
            "completed_at": _utc_now(),
            "stop_reason": stop_reason,
            "completed_cycles": len(cycle_records),
            "generated_molecules": len(all_candidates),
            "selected_molecules": len(all_selected),
            "elapsed_seconds": time.perf_counter() - run_started,
            "generator_model_provenance": generator_model_provenance,
        }
    )
    _write_json(manifest_path, manifest)
    print(
        f"MolDisc stopped: {stop_reason}. Generated {len(all_candidates)} globally novel "
        f"valid molecules and selected {len(all_selected)}."
    )
    return all_selected
