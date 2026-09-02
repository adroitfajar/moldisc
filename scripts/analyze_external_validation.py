#!/usr/bin/env python3
"""Evaluate a trained MolDisc predictor on a connectivity-overlap-free parent set.

The external set is constructed only from ``data/full_source/<project>/labeled.csv``.
Every molecular connectivity present in either active input file is removed
before evaluation, including stereochemical records that differ only in their
isomeric SMILES. SMILES-X and fixed Morgan-fingerprint baselines are then
evaluated on the identical remaining structures. External labels are never used for
fitting, hyperparameter selection, threshold selection, or early stopping;
however, parent labels were used to stratify the active 100-molecule subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_EXPECTED_LABELLED = 100
DEFAULT_EXPECTED_UNLABELLED = 1000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _canonical_smiles(value: object) -> str | None:
    if pd.isna(value):
        return None
    try:
        molecule = Chem.MolFromSmiles(str(value).strip())
        if molecule is None:
            return None
        Chem.SanitizeMol(molecule)
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _connectivity_smiles(value: object) -> str | None:
    """Return canonical non-isomeric SMILES for strict connectivity exclusion."""

    if pd.isna(value):
        return None
    try:
        molecule = Chem.MolFromSmiles(str(value).strip())
        if molecule is None:
            return None
        Chem.SanitizeMol(molecule)
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def _clean_labeled(path: Path, model_type: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    if not {"smiles", "property"}.issubset(raw.columns):
        raise ValueError(f"{path} must contain smiles and property columns.")

    working = raw.loc[:, ["smiles", "property"]].copy()
    working["canonical_smiles"] = working["smiles"].map(_canonical_smiles)
    working["property"] = pd.to_numeric(working["property"], errors="coerce")
    invalid_mask = working["canonical_smiles"].isna() | ~np.isfinite(
        working["property"].to_numpy(dtype=float)
    )
    invalid_rows = int(invalid_mask.sum())
    working = working.loc[~invalid_mask, ["canonical_smiles", "property"]].copy()

    if model_type == "classification":
        if not set(working["property"].unique()).issubset({0, 1}):
            raise ValueError(f"Classification labels in {path} must be 0 or 1.")

    grouped = working.groupby("canonical_smiles", sort=True)["property"]
    conflicting = int(sum(group.nunique() > 1 for _, group in grouped))
    if model_type == "classification" and conflicting:
        raise ValueError(
            f"{path} contains {conflicting} canonical structures with conflicting labels."
        )

    if model_type == "regression":
        clean = grouped.agg(observed="mean", source_measurements="size").reset_index()
    else:
        clean = grouped.agg(observed="first", source_measurements="size").reset_index()
        clean["observed"] = clean["observed"].astype(int)

    clean = clean.rename(columns={"canonical_smiles": "smiles"})
    clean = clean.sort_values("smiles", kind="stable").reset_index(drop=True)
    audit = {
        "raw_rows": int(len(raw)),
        "invalid_or_nonfinite_rows": invalid_rows,
        "valid_rows": int(len(working)),
        "canonical_duplicate_rows_removed": int(len(working) - len(clean)),
        "conflicting_canonical_structures": conflicting,
        "clean_unique_structures": int(len(clean)),
        "duplicate_policy": "mean" if model_type == "regression" else "error",
    }
    return clean, audit


def _clean_unlabeled(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    if "smiles" not in raw.columns:
        raise ValueError(f"{path} must contain a smiles column.")
    canonical = raw["smiles"].map(_canonical_smiles)
    invalid = int(canonical.isna().sum())
    clean = pd.DataFrame({"smiles": canonical.dropna()})
    clean = clean.drop_duplicates("smiles", keep="first")
    clean = clean.sort_values("smiles", kind="stable").reset_index(drop=True)
    audit = {
        "raw_rows": int(len(raw)),
        "invalid_rows": invalid,
        "canonical_duplicate_rows_removed": int(len(raw) - invalid - len(clean)),
        "clean_unique_structures": int(len(clean)),
    }
    return clean, audit


def _fingerprints(smiles: pd.Series) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    matrix = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Canonical SMILES unexpectedly failed RDKit parsing: {value}")
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
    return matrix


def _fit_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_external: np.ndarray,
    model_type: str,
    seed: int,
) -> dict[str, np.ndarray]:
    if model_type == "regression":
        models = {
            "morgan_ridge": Ridge(alpha=1.0),
            "morgan_random_forest": RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            ),
        }
    else:
        models = {
            "morgan_logistic_regression": LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=seed,
            ),
            "morgan_random_forest": RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        }

    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        if model_type == "regression":
            scores = model.predict(x_external)
        else:
            scores = model.predict_proba(x_external)[:, 1]
        predictions[name] = np.asarray(scores, dtype=float)
    return predictions


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _regression_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_score))),
        "mae": float(mean_absolute_error(y_true, y_score)),
        "r2": float(r2_score(y_true, y_score)),
        "pearson_r": _safe_correlation(y_true, y_score),
        "spearman_r": _safe_correlation(rankdata(y_true), rankdata(y_score)),
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    clipped = np.clip(y_score, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }


def _bootstrap_intervals(
    y_true: np.ndarray,
    model_scores: dict[str, np.ndarray],
    metric_function: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    model_type: str,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return paired percentile intervals using identical resamples per model."""

    rng = np.random.Generator(np.random.PCG64(seed))
    collected: dict[str, dict[str, list[float]]] = {
        model: {} for model in model_scores
    }
    if model_type == "classification":
        class_indices = {
            class_value: np.flatnonzero(y_true == class_value) for class_value in (0, 1)
        }
        if any(len(indices) == 0 for indices in class_indices.values()):
            raise ValueError("External classification evaluation requires both classes.")

    for _ in range(iterations):
        if model_type == "classification":
            sampled_indices = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in class_indices.values()
                ]
            )
        else:
            sampled_indices = rng.integers(0, len(y_true), size=len(y_true))
        sampled_true = y_true[sampled_indices]
        for model, scores in model_scores.items():
            metrics = metric_function(sampled_true, scores[sampled_indices])
            for metric, value in metrics.items():
                if np.isfinite(value):
                    collected[model].setdefault(metric, []).append(float(value))

    output: dict[str, dict[str, dict[str, float]]] = {}
    for model, scores in model_scores.items():
        point_metrics = metric_function(y_true, scores)
        output[model] = {}
        for metric, estimate in point_metrics.items():
            samples = np.asarray(collected[model].get(metric, []), dtype=float)
            if len(samples) != iterations:
                raise RuntimeError(
                    f"Only {len(samples)} finite bootstrap values were obtained for "
                    f"{model}/{metric}; expected {iterations}."
                )
            output[model][metric] = {
                "estimate": float(estimate),
                "ci_lower_95": float(np.percentile(samples, 2.5)),
                "ci_upper_95": float(np.percentile(samples, 97.5)),
            }
    return output


def _model_artifact_paths(train_dir: Path, config: dict) -> list[Path]:
    data_name = config["data_name"]
    model_paths = sorted((train_dir / "Models").glob(f"{data_name}_Model_Fold_*_Run_*.hdf5"))
    expected_models = int(config["k_fold_number"]) * int(config["n_runs"])
    if len(model_paths) != expected_models:
        raise FileNotFoundError(
            f"Expected {expected_models} SMILES-X model files in {train_dir / 'Models'}, "
            f"found {len(model_paths)}."
        )

    vocabulary = train_dir / "Other" / f"{data_name}_Vocabulary.txt"
    if not vocabulary.is_file():
        raise FileNotFoundError(vocabulary)
    artifacts = model_paths + [vocabulary]

    scaler_dir = train_dir / "Other" / "Scalers"
    output_scalers = sorted(scaler_dir.glob(f"{data_name}_Scaler_Outputs_Fold_*.pkl"))
    if bool(config.get("scale_output", True)):
        if len(output_scalers) != int(config["k_fold_number"]):
            raise FileNotFoundError(
                f"Expected {config['k_fold_number']} output scalers, found {len(output_scalers)}."
            )
        artifacts.extend(output_scalers)
    elif output_scalers:
        raise ValueError("Output scalers exist although scale_output is false in the config.")

    extra_scalers = sorted(scaler_dir.glob(f"{data_name}_Scaler_Extra_Fold_*.pkl"))
    artifacts.extend(extra_scalers)
    return artifacts


def _environment_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "tensorflow",
        "rdkit",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return versions


def run_analysis(args: argparse.Namespace) -> Path:
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if model_type not in {"regression", "classification"}:
        raise ValueError("config model_type must be regression or classification.")
    seed = int(config.get("random_seed", 42))
    bootstrap_seed = seed if args.bootstrap_seed is None else int(args.bootstrap_seed)
    if args.bootstrap_iterations < 1:
        raise ValueError("bootstrap-iterations must be positive.")
    if args.expected_active_labeled < 1 or args.expected_active_unlabeled < 1:
        raise ValueError("Expected active counts must be positive.")

    project_name = config["project_name"]
    project_folder = Path(config["project_folder"]).expanduser()
    if not project_folder.is_absolute():
        project_folder = REPO_ROOT / project_folder
    project_dir = (project_folder / project_name).resolve()
    active_dir = REPO_ROOT / "data" / project_name
    active_labeled_path = active_dir / config.get("input_file_labeled", "labeled.csv")
    active_unlabeled_path = active_dir / config.get("input_file_unlabeled", "unlabeled.csv")
    full_labeled_path = REPO_ROOT / "data" / "full_source" / project_name / "labeled.csv"
    subset_manifest_path = REPO_ROOT / "data" / "subset_manifest.json"
    for path in (
        config_path,
        active_labeled_path,
        active_unlabeled_path,
        full_labeled_path,
        subset_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    subset_manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest = subset_manifest.get("datasets", {}).get(project_name)
    if dataset_manifest is None:
        raise ValueError(f"subset_manifest.json has no entry for {project_name}.")
    recorded_active = dataset_manifest.get("active_files", {})
    recorded_source = dataset_manifest.get("full_source_files", {})
    expected_hashes = {
        "labeled_sha256": _sha256(active_labeled_path),
        "unlabeled_sha256": _sha256(active_unlabeled_path),
    }
    for key, observed_hash in expected_hashes.items():
        if recorded_active.get(key) != observed_hash:
            raise ValueError(f"Active {key} does not match data/subset_manifest.json.")
    if recorded_source.get("labeled_sha256") != _sha256(full_labeled_path):
        raise ValueError("Full-source labeled hash does not match data/subset_manifest.json.")

    full_source_hash_before = _sha256(full_labeled_path)
    active_labeled, active_labeled_audit = _clean_labeled(active_labeled_path, model_type)
    active_unlabeled, active_unlabeled_audit = _clean_unlabeled(active_unlabeled_path)
    if len(active_labeled) != args.expected_active_labeled:
        raise ValueError(
            f"Expected {args.expected_active_labeled} unique active labeled structures; "
            f"found {len(active_labeled)}."
        )
    if len(active_unlabeled) != args.expected_active_unlabeled:
        raise ValueError(
            f"Expected {args.expected_active_unlabeled} unique active unlabeled structures; "
            f"found {len(active_unlabeled)}."
        )
    active_labeled_set = set(active_labeled["smiles"])
    active_unlabeled_set = set(active_unlabeled["smiles"])
    active_overlap = active_labeled_set & active_unlabeled_set
    if active_overlap:
        raise ValueError(
            f"Active labeled and unlabeled inputs overlap by {len(active_overlap)} structures."
        )
    active_union = active_labeled_set | active_unlabeled_set

    active_labeled_connectivity = {
        _connectivity_smiles(value) for value in active_labeled["smiles"]
    }
    active_unlabeled_connectivity = {
        _connectivity_smiles(value) for value in active_unlabeled["smiles"]
    }
    if None in active_labeled_connectivity or None in active_unlabeled_connectivity:
        raise RuntimeError("An active canonical SMILES failed connectivity normalization.")
    active_connectivity_union = (
        active_labeled_connectivity | active_unlabeled_connectivity
    )

    full_labeled, full_labeled_audit = _clean_labeled(full_labeled_path, model_type)
    full_set = set(full_labeled["smiles"])
    excluded_as_labeled = full_set & active_labeled_set
    excluded_as_unlabeled = full_set & active_unlabeled_set
    full_connectivity = full_labeled["smiles"].map(_connectivity_smiles)
    if full_connectivity.isna().any():
        raise RuntimeError("A cleaned full-source SMILES failed connectivity normalization.")
    exact_active_mask = full_labeled["smiles"].isin(active_union)
    connectivity_active_mask = full_connectivity.isin(active_connectivity_union)
    excluded_connectivity_only = int((connectivity_active_mask & ~exact_active_mask).sum())
    external = full_labeled.loc[~connectivity_active_mask].copy()
    external = external.sort_values("smiles", kind="stable").reset_index(drop=True)
    if external.empty:
        raise ValueError("No external structures remain after excluding the active inputs.")
    if set(external["smiles"]) & active_union:
        raise RuntimeError("Leakage check failed: an active structure remains in the external set.")
    external_connectivity = {
        _connectivity_smiles(value) for value in external["smiles"]
    }
    if external_connectivity & active_connectivity_union:
        raise RuntimeError(
            "Leakage check failed: an active molecular connectivity remains in the external set."
        )

    y_train = active_labeled["observed"].to_numpy(
        dtype=int if model_type == "classification" else float
    )
    y_external = external["observed"].to_numpy(
        dtype=int if model_type == "classification" else float
    )
    if model_type == "classification":
        threshold = float(config.get("classification_threshold", 0.5))
        if not 0 <= threshold <= 1:
            raise ValueError("classification_threshold must be between 0 and 1.")
        if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_external)) != {0, 1}:
            raise ValueError("Both active training and external data must contain classes 0 and 1.")

    train_dir = (
        project_dir
        / "SMILESX"
        / "outputs"
        / "0"
        / config["data_name"]
        / ("Augm" if config.get("augmentation", True) else "Can")
        / "Train"
    )
    artifact_paths = _model_artifact_paths(train_dir, config)
    artifact_hashes_before = {_display_path(path): _sha256(path) for path in artifact_paths}

    # Import TensorFlow/SMILES-X only after CLI and input validation so --help is lightweight.
    sys.path.insert(0, str(REPO_ROOT))
    from SMILESX import inference, loadmodel

    smilesx_model = loadmodel.LoadModel(
        data_name=config["data_name"],
        outdir=str(project_dir / "SMILESX" / "outputs" / "0"),
        augment=bool(config.get("augmentation", True)),
        use_cpu=int(config.get("n_gpus", 0)) == 0,
        gpu_ind=int(config.get("cuda", 0)),
        log_verbose=False,
        return_attention=False,
    )
    if smilesx_model.k_fold_number != int(config["k_fold_number"]):
        raise ValueError("Loaded SMILES-X fold count does not match the config.")
    if smilesx_model.n_runs != int(config["n_runs"]):
        raise ValueError("Loaded SMILES-X run count does not match the config.")
    if bool(smilesx_model.scale_output) != bool(config.get("scale_output", True)):
        raise ValueError("Loaded SMILES-X output-scaling state does not match the config.")

    smilesx_output = inference.infer(
        model=smilesx_model,
        data_smiles=external["smiles"].tolist(),
        augment=bool(config.get("smilesx_inference_augmentation", False)),
        check_smiles=bool(config.get("check_smiles", True)),
        log_verbose=False,
        batch_size=int(config.get("smilesx_inference_batch_size", 512)),
        max_augmentations=config.get("smilesx_inference_max_augmentations"),
    )
    smilesx_scores = np.asarray(smilesx_output["mean"], dtype=float).reshape(-1)
    smilesx_sigma = np.asarray(smilesx_output["sigma"], dtype=float).reshape(-1)
    if smilesx_output["SMILES"].astype(str).tolist() != external["smiles"].tolist():
        raise RuntimeError("SMILES-X returned external predictions in an unexpected order.")
    if (
        len(smilesx_scores) != len(external)
        or not np.isfinite(smilesx_scores).all()
        or not np.isfinite(smilesx_sigma).all()
        or np.any(smilesx_sigma < 0)
    ):
        raise RuntimeError("SMILES-X returned invalid or incomplete external predictions.")

    x_train = _fingerprints(active_labeled["smiles"])
    x_external = _fingerprints(external["smiles"])
    baseline_scores = _fit_baselines(
        x_train=x_train,
        y_train=y_train,
        x_external=x_external,
        model_type=model_type,
        seed=seed,
    )
    model_scores = {"smilesx": smilesx_scores, **baseline_scores}
    for model, scores in model_scores.items():
        if len(scores) != len(external) or not np.isfinite(scores).all():
            raise ValueError(f"{model} produced invalid or incomplete external predictions.")
    if model_type == "classification":
        for model, scores in model_scores.items():
            if np.any((scores < 0) | (scores > 1)):
                raise ValueError(f"{model} produced values outside the probability interval [0, 1].")
        metric_function = lambda truth, score: _classification_metrics(
            truth, score, threshold
        )
        bootstrap_method = "class-stratified molecule bootstrap"
    else:
        threshold = None
        metric_function = _regression_metrics
        bootstrap_method = "ordinary molecule bootstrap"

    metric_results = _bootstrap_intervals(
        y_true=y_external,
        model_scores=model_scores,
        metric_function=metric_function,
        model_type=model_type,
        iterations=int(args.bootstrap_iterations),
        seed=bootstrap_seed,
    )

    predictions = external.rename(columns={"observed": "observed_value"}).copy()
    predictions["smilesx_score"] = smilesx_scores
    predictions["smilesx_sigma"] = smilesx_sigma
    for model, scores in baseline_scores.items():
        predictions[f"{model}_score"] = scores
    if model_type == "classification":
        for model, scores in model_scores.items():
            predictions[f"{model}_predicted_class"] = (scores >= threshold).astype(int)

    metric_rows = []
    for model, metrics in metric_results.items():
        for metric, values in metrics.items():
            metric_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    **values,
                    "n_external": int(len(external)),
                    "bootstrap_iterations": int(args.bootstrap_iterations),
                    "bootstrap_seed": bootstrap_seed,
                    "classification_threshold": threshold,
                }
            )
    metrics_frame = pd.DataFrame(metric_rows)

    analysis_dir = project_dir / "analysis" / "external_validation"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = analysis_dir / "external_predictions.csv"
    metrics_csv_path = analysis_dir / "external_metrics.csv"
    metrics_json_path = analysis_dir / "external_metrics.json"
    manifest_path = analysis_dir / "analysis_manifest.json"
    _write_csv_atomic(predictions_path, predictions)
    _write_csv_atomic(metrics_csv_path, metrics_frame)

    metrics_payload = {
        "schema_version": 1,
        "task": model_type,
        "project_name": project_name,
        "n_external": int(len(external)),
        "classification_threshold": threshold,
        "bootstrap": {
            "method": bootstrap_method,
            "unit": "canonical molecule",
            "iterations": int(args.bootstrap_iterations),
            "seed": bootstrap_seed,
            "confidence_interval": "two-sided 95% percentile interval",
            "paired_resamples_across_models": True,
            "uncertainty_scope": (
                "Evaluation-set sampling uncertainty conditional on the fixed fitted models; "
                "training-set and optimization uncertainty are not included."
            ),
        },
        "models": metric_results,
    }
    _write_json_atomic(metrics_json_path, metrics_payload)

    # The source labels and trained artifacts must remain byte-for-byte untouched.
    if _sha256(full_labeled_path) != full_source_hash_before:
        raise RuntimeError("The full-source labeled file changed during external validation.")
    artifact_hashes_after = {_display_path(path): _sha256(path) for path in artifact_paths}
    if artifact_hashes_after != artifact_hashes_before:
        raise RuntimeError("A trained SMILES-X artifact changed during external validation.")

    source_paths = [Path(__file__).resolve(), *sorted((REPO_ROOT / "SMILESX").glob("*.py"))]
    source_hashes = {_display_path(path): _sha256(path) for path in source_paths}
    input_hashes = {
        _display_path(config_path): _sha256(config_path),
        _display_path(active_labeled_path): _sha256(active_labeled_path),
        _display_path(active_unlabeled_path): _sha256(active_unlabeled_path),
        _display_path(full_labeled_path): full_source_hash_before,
        _display_path(subset_manifest_path): _sha256(subset_manifest_path),
    }
    output_hashes = {
        _display_path(predictions_path): _sha256(predictions_path),
        _display_path(metrics_csv_path): _sha256(metrics_csv_path),
        _display_path(metrics_json_path): _sha256(metrics_json_path),
    }
    external_summary = (
        {
            "minimum": float(external["observed"].min()),
            "maximum": float(external["observed"].max()),
            "mean": float(external["observed"].mean()),
            "standard_deviation": float(external["observed"].std(ddof=1)),
        }
        if model_type == "regression"
        else {
            "class_counts": {
                str(int(key)): int(value)
                for key, value in external["observed"].value_counts().sort_index().items()
            }
        }
    )
    manifest = {
        "schema_version": 1,
        "analysis": "connectivity_overlap_free_withheld_parent_set_evaluation",
        "created_at": _utc_now(),
        "task": model_type,
        "project_name": project_name,
        "project_directory": _display_path(project_dir),
        "config": config,
        "input_hashes_sha256": input_hashes,
        "source_hashes_sha256": source_hashes,
        "trained_smilesx_artifact_hashes_sha256": artifact_hashes_before,
        "output_hashes_sha256": output_hashes,
        "environment": _environment_versions(),
        "counts": {
            "active_labeled_unique": int(len(active_labeled)),
            "active_unlabeled_unique": int(len(active_unlabeled)),
            "active_union_unique": int(len(active_union)),
            "active_union_connectivity_unique": int(len(active_connectivity_union)),
            "full_source_clean_unique": int(len(full_labeled)),
            "excluded_as_active_labeled": int(len(excluded_as_labeled)),
            "excluded_as_active_unlabeled": int(len(excluded_as_unlabeled)),
            "excluded_active_union": int(exact_active_mask.sum()),
            "excluded_active_connectivity_union": int(connectivity_active_mask.sum()),
            "excluded_connectivity_only": excluded_connectivity_only,
            "external_unique": int(len(external)),
            "external_connectivity_unique": int(len(external_connectivity)),
        },
        "cleaning_audit": {
            "active_labeled": active_labeled_audit,
            "active_unlabeled": active_unlabeled_audit,
            "full_source_labeled": full_labeled_audit,
        },
        "external_target_summary": external_summary,
        "evaluation_design": {
            "type": "molecular-connectivity-overlap-free withheld parent-set evaluation",
            "active_subset_selection": subset_manifest.get("labelled_selection"),
            "parent_labels_used_for_active_subset_construction": True,
            "external_labels_used_for_model_fitting": False,
            "external_labels_used_for_hyperparameter_or_threshold_selection": False,
            "note": (
                "Labels attached to the withheld structures were not used for model fitting, "
                "threshold selection, or generator candidate selection. The active labeled subset "
                "itself was target/class stratified from the parent table, so this is not an "
                "independent-dataset validation."
            ),
        },
        "leakage_control": {
            "canonicalization": "RDKit canonical isomeric SMILES",
            "connectivity_key": "RDKit canonical non-isomeric SMILES",
            "excluded_sets": ["active labeled", "active unlabeled"],
            "external_active_overlap_after_exclusion": 0,
            "external_active_connectivity_overlap_after_exclusion": 0,
            "external_labels_used_for_model_training": False,
        },
        "smilesx_inference": {
            "ensemble_folds": int(smilesx_model.k_fold_number),
            "ensemble_runs_per_fold": int(smilesx_model.n_runs),
            "augmentation": bool(config.get("smilesx_inference_augmentation", False)),
            "max_augmentations": config.get("smilesx_inference_max_augmentations"),
            "batch_size": int(config.get("smilesx_inference_batch_size", 512)),
        },
        "baselines": {
            "fingerprint": "Morgan radius 2, 2048 bits, no external-label fitting",
            "training_structures": int(len(active_labeled)),
            "regression": {
                "morgan_ridge": {"alpha": 1.0},
                "morgan_random_forest": {
                    "n_estimators": 500,
                    "min_samples_leaf": 2,
                    "random_state": seed,
                },
            }
            if model_type == "regression"
            else None,
            "classification": {
                "morgan_logistic_regression": {
                    "max_iter": 5000,
                    "class_weight": "balanced",
                    "random_state": seed,
                },
                "morgan_random_forest": {
                    "n_estimators": 500,
                    "min_samples_leaf": 2,
                    "class_weight": "balanced",
                    "random_state": seed,
                },
            }
            if model_type == "classification"
            else None,
        },
        "bootstrap": metrics_payload["bootstrap"],
    }
    _write_json_atomic(manifest_path, manifest)

    print(f"Withheld parent-set evaluation complete: {len(external)} canonical molecules.")
    print(f"Predictions: {_display_path(predictions_path)}")
    print(f"Metrics: {_display_path(metrics_csv_path)}")
    print(f"Manifest: {_display_path(manifest_path)}")
    return analysis_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained SMILES-X ensemble and fixed Morgan baselines on "
            "full-source labels whose molecular connectivity is absent from both active input files."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Campaign JSON config.")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
        help=f"Deterministic molecule-bootstrap iterations (default: {DEFAULT_BOOTSTRAP_ITERATIONS}).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        help="Bootstrap seed; defaults to random_seed from the campaign config.",
    )
    parser.add_argument(
        "--expected-active-labeled",
        type=int,
        default=DEFAULT_EXPECTED_LABELLED,
        help=f"Required canonical active labeled count (default: {DEFAULT_EXPECTED_LABELLED}).",
    )
    parser.add_argument(
        "--expected-active-unlabeled",
        type=int,
        default=DEFAULT_EXPECTED_UNLABELLED,
        help=f"Required canonical active unlabeled count (default: {DEFAULT_EXPECTED_UNLABELLED}).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
