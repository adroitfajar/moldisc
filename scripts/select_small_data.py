#!/usr/bin/env python3
"""Build the deterministic 100-labelled/1,000-unlabelled MolDisc datasets.

The full user-supplied tables are preserved under ``data/full_source``.  The
active CSV files are canonicalized small-data subsets suitable for the paper
and public tutorials.  Property values are used only to stratify the labelled
benchmark; the selected unlabelled files contain SMILES alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LABELLED_SIZE = 100
UNLABELLED_SIZE = 1000
ALLOWED_ELEMENTS = {"B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_smiles(value: object) -> str | None:
    if pd.isna(value):
        return None
    molecule = Chem.MolFromSmiles(str(value).strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def clean_labelled(path: Path, task: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    required = {"smiles", "property"}
    if not required.issubset(raw.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    raw = raw[["smiles", "property"]].copy()
    raw["smiles"] = raw["smiles"].map(canonical_smiles)
    raw["property"] = pd.to_numeric(raw["property"], errors="coerce")
    invalid = int(raw[["smiles", "property"]].isna().any(axis=1).sum())
    raw = raw.dropna(subset=["smiles", "property"])

    grouped = raw.groupby("smiles", sort=True)["property"]
    conflicting = int(sum(group.nunique() > 1 for _, group in grouped))
    if task == "classification" and conflicting:
        raise ValueError(f"Conflicting classification labels found in {path}")
    if task == "regression":
        clean = grouped.mean().reset_index()
    else:
        clean = grouped.first().reset_index()
        clean["property"] = clean["property"].astype(int)
        if not set(clean["property"]).issubset({0, 1}):
            raise ValueError(f"Classification targets in {path} must be binary")
    audit = {
        "raw_rows": int(len(pd.read_csv(path))),
        "invalid_rows": invalid,
        "duplicate_rows_removed": int(len(raw) - len(clean)),
        "conflicting_canonical_structures": conflicting,
        "clean_unique_rows": int(len(clean)),
    }
    return clean.sort_values("smiles", kind="stable").reset_index(drop=True), audit


def regression_labelled_subset(clean: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    ordered = clean.sort_values(["property", "smiles"], kind="stable").reset_index(drop=True)
    ordered["_target_rank_bin"] = np.arange(len(ordered)) * 10 // len(ordered)
    rng = np.random.Generator(np.random.PCG64(seed))
    chosen = []
    for bin_number in range(10):
        bin_indices = np.flatnonzero(ordered["_target_rank_bin"].to_numpy() == bin_number)
        chosen.extend(rng.choice(bin_indices, size=10, replace=False).tolist())
    subset = ordered.iloc[chosen].drop(columns="_target_rank_bin")
    subset = subset.sort_values("smiles", kind="stable").reset_index(drop=True)
    return subset, {"target_rank_bins": 10, "selected_per_bin": [10] * 10}


def largest_remainder_alloc(counts: dict[int, int], total: int) -> dict[int, int]:
    population = sum(counts.values())
    exact = {key: total * value / population for key, value in counts.items()}
    allocation = {key: int(np.floor(value)) for key, value in exact.items()}
    remainder = total - sum(allocation.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remainder]:
        allocation[key] += 1
    return allocation


def classification_labelled_subset(clean: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    counts = {int(key): int(value) for key, value in clean["property"].value_counts().items()}
    allocation = largest_remainder_alloc(counts, LABELLED_SIZE)
    rng = np.random.Generator(np.random.PCG64(seed))
    parts = []
    for class_value in sorted(allocation):
        group = clean.loc[clean["property"] == class_value].sort_values("smiles", kind="stable")
        selected_positions = rng.choice(len(group), size=allocation[class_value], replace=False)
        parts.append(group.iloc[np.sort(selected_positions)])
    subset = pd.concat(parts, ignore_index=True).sort_values("smiles", kind="stable")
    return subset.reset_index(drop=True), {
        "allocation_method": "largest remainder proportional to parent class counts",
        "selected_class_counts": {str(key): value for key, value in sorted(allocation.items())},
    }


def clean_unlabelled_sources(
    path: Path,
    source_blocks: list[tuple[str, int]],
) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    if "smiles" not in raw.columns:
        raise ValueError(f"{path} must contain a smiles column")
    if sum(size for _, size in source_blocks) != len(raw):
        raise ValueError(f"Source block sizes do not match {path}: {len(raw)} rows")

    pieces = []
    start = 0
    invalid_by_source: Counter[str] = Counter()
    for source, size in source_blocks:
        block = raw.iloc[start : start + size][["smiles"]].copy()
        block["source"] = source
        block["source_row"] = np.arange(size)
        block["smiles"] = block["smiles"].map(canonical_smiles)
        invalid_by_source[source] += int(block["smiles"].isna().sum())
        pieces.append(block.dropna(subset=["smiles"]))
        start += size

    combined = pd.concat(pieces, ignore_index=True)
    # Stable source precedence follows the order in which the source datasets
    # were assembled into the user-supplied table.
    combined = combined.drop_duplicates("smiles", keep="first")
    audit = {
        "raw_rows": int(len(raw)),
        "invalid_rows_by_source": dict(sorted(invalid_by_source.items())),
        "duplicate_rows_removed": int(len(raw) - sum(invalid_by_source.values()) - len(combined)),
        "clean_unique_rows": int(len(combined)),
        "source_precedence": [name for name, _ in source_blocks],
    }
    return combined.reset_index(drop=True), audit


def apply_generation_domain_filter(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Match the public classification pool to MolDisc's generation domain."""
    keep = []
    disconnected = 0
    disallowed_counts: Counter[str] = Counter()
    disallowed_rows = 0
    for value in clean["smiles"]:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            keep.append(False)
            continue
        if len(Chem.GetMolFrags(molecule)) != 1:
            disconnected += 1
            keep.append(False)
            continue
        disallowed = sorted(
            {atom.GetSymbol() for atom in molecule.GetAtoms()} - ALLOWED_ELEMENTS
        )
        if disallowed:
            disallowed_rows += 1
            disallowed_counts.update(disallowed)
            keep.append(False)
            continue
        keep.append(True)
    filtered = clean.loc[keep].reset_index(drop=True)
    return filtered, {
        "rule": "single connected component and atoms limited to the disclosed allowed-element set",
        "allowed_elements": sorted(ALLOWED_ELEMENTS),
        "disconnected_rows_excluded": disconnected,
        "disallowed_element_rows_excluded": disallowed_rows,
        "disallowed_element_counts": dict(sorted(disallowed_counts.items())),
        "eligible_rows_after_domain_filter": int(len(filtered)),
    }


def unlabelled_subset(
    clean: pd.DataFrame,
    selected_labelled: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    labelled_set = set(selected_labelled["smiles"])
    eligible = clean.loc[~clean["smiles"].isin(labelled_set)].copy()
    source_counts = {str(key): int(value) for key, value in eligible["source"].value_counts().items()}
    allocation = largest_remainder_alloc(source_counts, UNLABELLED_SIZE)
    rng = np.random.Generator(np.random.PCG64(seed))
    parts = []
    for source in sorted(allocation):
        group = eligible.loc[eligible["source"] == source].sort_values("smiles", kind="stable")
        selected_positions = rng.choice(len(group), size=allocation[source], replace=False)
        parts.append(group.iloc[np.sort(selected_positions)])
    selected = pd.concat(parts, ignore_index=True).sort_values("smiles", kind="stable")
    selected = selected[["smiles"]].reset_index(drop=True)
    audit = {
        "eligible_unique_rows": int(len(eligible)),
        "selected_source_counts": {key: int(value) for key, value in sorted(allocation.items())},
        "selected_labelled_overlap_removed": int(len(clean) - len(eligible)),
    }
    return selected, audit


def scaffold_count(smiles: pd.Series) -> int:
    scaffolds = set()
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        # Acyclic compounds share the conventional empty Murcko scaffold.
        scaffolds.add(scaffold)
    return len(scaffolds)


def numeric_summary(values: pd.Series) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def build(data_root: Path, source_root: Path, seed: int) -> dict:
    specifications = {
        "data_1": {
            "task": "regression",
            "labelled_rows": 1128,
            "sources": [("ESOL", 1128), ("FreeSolv", 642), ("Lipophilicity", 4200)],
        },
        "data_2": {
            "task": "classification",
            "labelled_rows": 1513,
            "sources": [("BACE", 1513), ("Tox21", 7831)],
        },
    }
    missing_sources = [
        source_root / dataset / filename
        for dataset in specifications
        for filename in ("labeled.csv", "unlabeled.csv")
        if not (source_root / dataset / filename).is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Full source tables are required before subset selection; no active files "
            "were changed. Missing: " + ", ".join(map(str, missing_sources))
        )
    for dataset, specification in specifications.items():
        observed_labelled = len(pd.read_csv(source_root / dataset / "labeled.csv"))
        if observed_labelled != specification["labelled_rows"]:
            raise ValueError(
                f"Unexpected full-source row count for {dataset}/labeled.csv: "
                f"{observed_labelled}; expected {specification['labelled_rows']}. "
                "No active files were changed."
            )
        expected_unlabelled = sum(size for _, size in specification["sources"])
        observed_unlabelled = len(pd.read_csv(source_root / dataset / "unlabeled.csv"))
        if observed_unlabelled != expected_unlabelled:
            raise ValueError(
                f"Unexpected full-source row count for {dataset}/unlabeled.csv: "
                f"{observed_unlabelled}; expected {expected_unlabelled}. No active files were changed."
            )
    manifest = {
        "schema_version": 2,
        "selection_seed": seed,
        "random_generator": "NumPy PCG64",
        "selection_purpose": "illustrative small-data benchmark",
        "labelled_selection": (
            "target-rank-stratified for regression and proportionally class-stratified "
            "for classification; properties were used only for labelled-subset construction"
        ),
        "unlabelled_selection": (
            "source-stratified random selection after canonicalization and global deduplication; "
            "the selected 100 labelled structures were excluded and no property values were retained; "
            "the classification pool additionally used the disclosed connected-component "
            "and allowed-element prefilter"
        ),
        "datasets": {},
    }

    for dataset_index, (dataset, specification) in enumerate(specifications.items()):
        labelled_source = source_root / dataset / "labeled.csv"
        unlabelled_source = source_root / dataset / "unlabeled.csv"
        cleaned_labelled, labelled_audit = clean_labelled(labelled_source, specification["task"])
        if specification["task"] == "regression":
            selected_labelled, selection_details = regression_labelled_subset(cleaned_labelled, seed)
        else:
            selected_labelled, selection_details = classification_labelled_subset(cleaned_labelled, seed)
        clean_unlabelled, unlabelled_audit = clean_unlabelled_sources(
            unlabelled_source, specification["sources"]
        )
        domain_filter_audit = None
        if dataset == "data_2":
            clean_unlabelled, domain_filter_audit = apply_generation_domain_filter(
                clean_unlabelled
            )
        selected_unlabelled, unlabelled_selection = unlabelled_subset(
            clean_unlabelled, selected_labelled, seed
        )

        if len(selected_labelled) != LABELLED_SIZE or len(selected_unlabelled) != UNLABELLED_SIZE:
            raise RuntimeError(f"Unexpected subset sizes for {dataset}")
        if set(selected_labelled["smiles"]) & set(selected_unlabelled["smiles"]):
            raise RuntimeError(f"Labelled/unlabelled overlap remains in {dataset}")

        active_dir = data_root / dataset
        active_dir.mkdir(parents=True, exist_ok=True)
        labelled_output = active_dir / "labeled.csv"
        unlabelled_output = active_dir / "unlabeled.csv"
        selected_labelled.to_csv(labelled_output, index=False, lineterminator="\n")
        selected_unlabelled.to_csv(unlabelled_output, index=False, lineterminator="\n")

        dataset_manifest = {
            "task": specification["task"],
            "full_source_files": {
                "labeled_sha256": sha256(labelled_source),
                "unlabeled_sha256": sha256(unlabelled_source),
            },
            "full_labelled_cleaning": labelled_audit,
            "full_unlabelled_cleaning": unlabelled_audit,
            "labelled_selection": selection_details,
            "unlabelled_selection": unlabelled_selection,
            "active_files": {
                "labeled_rows": len(selected_labelled),
                "unlabeled_rows": len(selected_unlabelled),
                "labeled_sha256": sha256(labelled_output),
                "unlabeled_sha256": sha256(unlabelled_output),
                "labeled_unique_scaffolds": scaffold_count(selected_labelled["smiles"]),
                "unlabeled_unique_scaffolds": scaffold_count(selected_unlabelled["smiles"]),
            },
        }
        if domain_filter_audit is not None:
            dataset_manifest["unlabelled_generation_domain_filter"] = domain_filter_audit
        if specification["task"] == "regression":
            dataset_manifest["full_target_summary"] = numeric_summary(cleaned_labelled["property"])
            dataset_manifest["selected_target_summary"] = numeric_summary(selected_labelled["property"])
        else:
            dataset_manifest["full_class_counts"] = {
                str(key): int(value)
                for key, value in sorted(cleaned_labelled["property"].value_counts().items())
            }
            dataset_manifest["selected_class_counts"] = {
                str(key): int(value)
                for key, value in sorted(selected_labelled["property"].value_counts().items())
            }
        manifest["datasets"][dataset] = dataset_manifest

    manifest_path = data_root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    source_root = (args.source_root or data_root / "full_source").resolve()
    manifest = build(data_root, source_root, args.seed)
    for dataset, details in manifest["datasets"].items():
        active = details["active_files"]
        print(
            f"{dataset}: {active['labeled_rows']} labelled, "
            f"{active['unlabeled_rows']} unlabelled, "
            f"{active['labeled_unique_scaffolds']} labelled scaffolds"
        )


if __name__ == "__main__":
    main()
