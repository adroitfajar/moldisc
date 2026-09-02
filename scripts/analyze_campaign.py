#!/usr/bin/env python3
"""Create publication tables and figures from a completed/paused campaign."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from moldisc import _clean_input_data
from models.utilsReport import calculate_pairwise_tanimoto


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(series):
    """Parse saved selection flags without treating the string 'False' as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def connectivity_key(smiles):
    """Return a canonical non-isomeric SMILES key for connectivity-level audits."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Cannot derive a connectivity key from invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def describe_molecule(smiles, fingerprint_generator):
    molecule = Chem.MolFromSmiles(smiles)
    fingerprint = fingerprint_generator.GetFingerprint(molecule)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule, includeChirality=False
    )
    return {
        "molecular_weight": Descriptors.MolWt(molecule),
        "logp": Crippen.MolLogP(molecule),
        "tpsa": Descriptors.TPSA(molecule),
        "h_bond_donors": Lipinski.NumHDonors(molecule),
        "h_bond_acceptors": Lipinski.NumHAcceptors(molecule),
        "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
        "ring_count": Lipinski.RingCount(molecule),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "formal_charge": sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
        "fragment_count": len(Chem.GetMolFrags(molecule)),
        "qed": QED.qed(molecule),
        "murcko_scaffold": scaffold,
        "_fingerprint": fingerprint,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    project_folder = Path(config["project_folder"])
    if not project_folder.is_absolute():
        project_folder = REPO_ROOT / project_folder
    project_dir = project_folder / config["project_name"]
    candidates_path = project_dir / "final" / "all_generated_candidates.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)
    candidates = pd.read_csv(candidates_path)
    if candidates.empty:
        raise ValueError("The campaign has no generated candidates to analyze.")

    labeled = pd.read_csv(REPO_ROOT / "data" / config["project_name"] / "labeled.csv")
    unlabeled = pd.read_csv(REPO_ROOT / "data" / config["project_name"] / "unlabeled.csv")
    labeled, unlabeled, _ = _clean_input_data(labeled, unlabeled, config["model_type"])
    reference_smiles = set(labeled["smiles"]) | set(unlabeled["smiles"])
    overlap = reference_smiles & set(candidates["smiles"])
    if overlap:
        raise ValueError(f"Novelty violation: {len(overlap)} generated structures occur in the inputs.")
    if candidates["smiles"].duplicated().any():
        raise ValueError("Global uniqueness violation in all_generated_candidates.csv.")
    reference_connectivity = {connectivity_key(smiles) for smiles in reference_smiles}
    candidate_connectivity = candidates["smiles"].map(connectivity_key)
    input_connectivity_overlap = int(candidate_connectivity.isin(reference_connectivity).sum())
    unique_connectivity_keys = int(candidate_connectivity.nunique())

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    )
    reference_fingerprints = [
        fingerprint_generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in sorted(reference_smiles)
    ]
    descriptors = []
    candidate_fingerprints = []
    for smiles in candidates["smiles"]:
        row = describe_molecule(smiles, fingerprint_generator)
        fingerprint = row.pop("_fingerprint")
        row["nearest_input_tanimoto"] = max(
            DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fingerprints)
        )
        descriptors.append(row)
        candidate_fingerprints.append(fingerprint)
    descriptor_frame = pd.DataFrame(descriptors)
    enriched = pd.concat([candidates.reset_index(drop=True), descriptor_frame], axis=1)

    analysis_dir = project_dir / "analysis" / "campaign"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(
        analysis_dir / "candidate_descriptors.csv.gz",
        index=False,
        compression="gzip",
    )

    numeric_columns = [
        "property",
        "prediction_std",
        "sa_score",
        "molecular_weight",
        "logp",
        "tpsa",
        "qed",
        "nearest_input_tanimoto",
    ]
    summary_rows = []
    for cycle, frame in enriched.groupby("cycle"):
        record = {
            "cycle": int(cycle),
            "generated": len(frame),
            "selected": int(_as_bool(frame["selected"]).sum()),
            "unique_scaffolds": int(frame["murcko_scaffold"].nunique()),
            "new_scaffold_fraction": float(
                (~frame["murcko_scaffold"].isin({
                    MurckoScaffold.MurckoScaffoldSmiles(
                        mol=Chem.MolFromSmiles(smiles), includeChirality=False
                    )
                    for smiles in reference_smiles
                })).mean()
            ),
        }
        for column in numeric_columns:
            record[f"{column}_mean"] = float(frame[column].mean())
            record[f"{column}_median"] = float(frame[column].median())
            record[f"{column}_q05"] = float(frame[column].quantile(0.05))
            record[f"{column}_q95"] = float(frame[column].quantile(0.95))
        summary_rows.append(record)
    cycle_summary = pd.DataFrame(summary_rows)
    cycle_summary.to_csv(analysis_dir / "cycle_chemical_summary.csv", index=False)

    top_ascending = config.get("target_property_mode", "max") == "min"
    enriched.sort_values("property", ascending=top_ascending).head(20).to_csv(
        analysis_dir / "top_20_candidates.csv", index=False
    )
    internal_similarities = calculate_pairwise_tanimoto(
        candidate_fingerprints,
        max_pairs=int(config.get("tanimoto_max_pairs", 200000)),
        random_seed=int(config.get("random_seed", 42)),
    )
    overall = {
        "generated": len(enriched),
        "selected": int(_as_bool(enriched["selected"]).sum()),
        "cycles": int(enriched["cycle"].nunique()),
        "validity": 1.0,
        "global_uniqueness": 1.0,
        "input_novelty": 1.0,
        "input_connectivity_overlap": input_connectivity_overlap,
        "unique_connectivity_keys": unique_connectivity_keys,
        "connectivity_duplicate_rows": int(len(enriched) - unique_connectivity_keys),
        "unique_scaffolds": int(enriched["murcko_scaffold"].nunique()),
        "median_nearest_input_tanimoto": float(enriched["nearest_input_tanimoto"].median()),
        "mean_internal_pairwise_tanimoto": float(np.mean(internal_similarities)),
        "mean_sa_score": float(enriched["sa_score"].mean()),
        "mean_qed": float(enriched["qed"].mean()),
    }
    (analysis_dir / "campaign_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )

    cycles = sorted(enriched["cycle"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].boxplot(
        [enriched.loc[enriched["cycle"] == cycle, "property"] for cycle in cycles],
        labels=[str(cycle) for cycle in cycles],
        showfliers=False,
    )
    axes[0].axhline(labeled["property"].median(), color="black", linestyle="--", linewidth=1)
    units = config.get("data_units", "").strip()
    property_label = config.get("data_label", "Predicted property")
    if units:
        property_label = f"Predicted {property_label} ({units})"
    axes[0].set(xlabel="Discovery cycle", ylabel=property_label)
    axes[0].text(
        0.02,
        0.04,
        "Dashed: input median",
        transform=axes[0].transAxes,
        fontsize=8,
    )
    similarity_axis = axes[1]
    sa_axis = similarity_axis.twinx()
    similarity_axis.plot(
        cycle_summary["cycle"],
        cycle_summary["nearest_input_tanimoto_median"],
        marker="o",
        color="#1f77b4",
        label="Nearest-input Tanimoto",
    )
    sa_axis.plot(
        cycle_summary["cycle"],
        cycle_summary["sa_score_median"],
        marker="s",
        color="#ff7f0e",
        label="SA score",
    )
    similarity_axis.set(
        xlabel="Discovery cycle",
        ylabel="Median nearest-input Tanimoto",
        ylim=(0, 1),
    )
    sa_axis.set_ylabel("Median SA score")
    handles_left, labels_left = similarity_axis.get_legend_handles_labels()
    handles_right, labels_right = sa_axis.get_legend_handles_labels()
    similarity_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        loc="best",
    )
    figure.tight_layout()
    figure.savefig(analysis_dir / "campaign_property_and_quality.png", dpi=600)
    figure.savefig(analysis_dir / "campaign_property_and_quality.pdf")
    plt.close(figure)

    metrics_path = project_dir / "cycle_metrics.csv"
    if metrics_path.exists() and len(pd.read_csv(metrics_path)):
        metrics = pd.read_csv(metrics_path)
        figure, axis = plt.subplots(figsize=(6.4, 4.4))
        for column, label in (
            ("generation_validity_rate", "Validity"),
            ("generation_uniqueness_rate", "Uniqueness"),
            ("generation_novelty_rate", "Novelty"),
            ("generation_yield_rate", "Generation completion"),
        ):
            if column in metrics:
                axis.plot(metrics["cycle"], metrics[column], marker="o", label=label)
        axis.set(xlabel="Discovery cycle", ylabel="Fraction", ylim=(0, 1.05))
        axis.legend(frameon=False, ncol=2)
        figure.tight_layout()
        figure.savefig(analysis_dir / "generation_quality_by_cycle.png", dpi=600)
        figure.savefig(analysis_dir / "generation_quality_by_cycle.pdf")
        plt.close(figure)

    output_files = [
        analysis_dir / "candidate_descriptors.csv.gz",
        analysis_dir / "cycle_chemical_summary.csv",
        analysis_dir / "top_20_candidates.csv",
        analysis_dir / "campaign_summary.json",
        analysis_dir / "campaign_property_and_quality.png",
        analysis_dir / "campaign_property_and_quality.pdf",
        analysis_dir / "generation_quality_by_cycle.png",
        analysis_dir / "generation_quality_by_cycle.pdf",
    ]
    generator_model_dir = project_dir / "GPT" / "latest_model"
    generator_artifacts = sorted(
        path for path in generator_model_dir.rglob("*") if path.is_file()
    )
    if not generator_artifacts:
        raise FileNotFoundError(
            f"No final GPT generator artifacts were found in {generator_model_dir}."
        )
    analysis_manifest = {
        "schema_version": 1,
        "analysis": "campaign_chemistry_and_quality",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": config["model_type"],
        "counts": {
            "generated": int(len(enriched)),
            "cycles": int(enriched["cycle"].nunique()),
            "active_reference": int(len(reference_smiles)),
            "input_overlap": 0,
            "input_connectivity_overlap": input_connectivity_overlap,
            "unique_connectivity_keys": unique_connectivity_keys,
            "connectivity_duplicate_rows": int(len(enriched) - unique_connectivity_keys),
            "global_duplicates": 0,
        },
        "input_sha256": {
            "config": sha256(args.config.resolve()),
            "all_generated_candidates": sha256(candidates_path),
            "active_labeled": sha256(REPO_ROOT / "data" / config["project_name"] / "labeled.csv"),
            "active_unlabeled": sha256(REPO_ROOT / "data" / config["project_name"] / "unlabeled.csv"),
            "cycle_metrics": sha256(metrics_path) if metrics_path.is_file() else None,
        },
        "source_sha256": {"scripts/analyze_campaign.py": sha256(Path(__file__).resolve())},
        "generator_artifact_sha256": {
            str(path.relative_to(project_dir)).replace("\\", "/"): sha256(path)
            for path in generator_artifacts
        },
        "output_sha256": {
            path.name: sha256(path) for path in output_files if path.is_file()
        },
    }
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
