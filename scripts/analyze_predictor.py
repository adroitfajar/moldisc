#!/usr/bin/env python3
"""Generate paper-ready property-model and fingerprint-baseline metrics."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from moldisc import _clean_input_data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "spearman_r": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
    }


def classification_metrics(y_true, y_score, threshold=0.5):
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "threshold": float(threshold),
    }


def bootstrap_intervals(y_true, y_pred, metric_function, seed, iterations=2000):
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    size = len(y_true)
    for _ in range(iterations):
        indices = rng.integers(0, size, size=size)
        sampled_true = np.asarray(y_true)[indices]
        sampled_pred = np.asarray(y_pred)[indices]
        if len(np.unique(sampled_true)) < 2 and set(np.unique(y_true)).issubset({0, 1}):
            continue
        try:
            metrics = metric_function(sampled_true, sampled_pred)
        except ValueError:
            continue
        for key, value in metrics.items():
            if key != "threshold" and np.isfinite(value):
                values[key].append(value)
    return {
        key: {
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
        }
        for key, samples in values.items()
        if samples
    }


def fingerprints_and_scaffolds(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    arrays = []
    scaffolds = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        fingerprint = generator.GetFingerprint(molecule)
        array = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        fingerprints.append(fingerprint)
        arrays.append(array)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        scaffolds.append(scaffold or Chem.MolToSmiles(molecule, canonical=True))
    return np.asarray(arrays), np.asarray(scaffolds)


def repeated_predictions(x, y, model_type, seed, scaffold_groups=None):
    predictions = defaultdict(list)
    if scaffold_groups is None:
        splitter = (
            RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
            if model_type == "regression"
            else RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
        )
        splits = splitter.split(x, y)
    else:
        split_list = []
        for repeat in range(3):
            splitter = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=seed + repeat
            )
            stratify_y = y if model_type == "classification" else pd.qcut(
                y, q=min(5, len(np.unique(y))), labels=False, duplicates="drop"
            )
            split_list.extend(splitter.split(x, stratify_y, scaffold_groups))
        splits = split_list

    for split_number, (train_indices, test_indices) in enumerate(splits):
        if model_type == "regression":
            models = {
                "Morgan-Ridge": Ridge(alpha=1.0),
                "Morgan-RF": RandomForestRegressor(
                    n_estimators=500,
                    random_state=seed + split_number,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            }
        else:
            models = {
                "Morgan-LogReg": LogisticRegression(
                    max_iter=5000, class_weight="balanced", random_state=seed
                ),
                "Morgan-RF": RandomForestClassifier(
                    n_estimators=500,
                    random_state=seed + split_number,
                    n_jobs=-1,
                    min_samples_leaf=2,
                    class_weight="balanced",
                ),
            }
        for name, model in models.items():
            model.fit(x[train_indices], y[train_indices])
            if model_type == "regression":
                scores = model.predict(x[test_indices])
            else:
                scores = model.predict_proba(x[test_indices])[:, 1]
            for index, score in zip(test_indices, scores):
                predictions[name, int(index)].append(float(score))

    output = {}
    for name in sorted({key[0] for key in predictions}):
        output[name] = np.asarray(
            [np.mean(predictions[name, index]) for index in range(len(y))], dtype=float
        )
    return output


def shared_fold_predictions(x, y, folds, model_type, seed, rf_replicates=3):
    """Evaluate fingerprint baselines on the exact SMILES-X fold assignment."""
    folds = np.asarray(folds, dtype=int)
    output = defaultdict(list)
    for fold in sorted(np.unique(folds)):
        train_indices = np.flatnonzero(folds != fold)
        test_indices = np.flatnonzero(folds == fold)
        if model_type == "regression":
            linear = Ridge(alpha=1.0)
            linear.fit(x[train_indices], y[train_indices])
            output["Morgan-Ridge"].extend(
                zip(test_indices, linear.predict(x[test_indices]))
            )
            replicate_scores = []
            for replicate in range(int(rf_replicates)):
                model = RandomForestRegressor(
                    n_estimators=500,
                    random_state=seed + 100 * int(fold) + replicate,
                    n_jobs=-1,
                    min_samples_leaf=2,
                )
                model.fit(x[train_indices], y[train_indices])
                replicate_scores.append(model.predict(x[test_indices]))
            output["Morgan-RF"].extend(
                zip(test_indices, np.mean(replicate_scores, axis=0))
            )
        else:
            linear = LogisticRegression(
                max_iter=5000, class_weight="balanced", random_state=seed
            )
            linear.fit(x[train_indices], y[train_indices])
            output["Morgan-LogReg"].extend(
                zip(test_indices, linear.predict_proba(x[test_indices])[:, 1])
            )
            replicate_scores = []
            for replicate in range(int(rf_replicates)):
                model = RandomForestClassifier(
                    n_estimators=500,
                    random_state=seed + 100 * int(fold) + replicate,
                    n_jobs=-1,
                    min_samples_leaf=2,
                    class_weight="balanced",
                )
                model.fit(x[train_indices], y[train_indices])
                replicate_scores.append(model.predict_proba(x[test_indices])[:, 1])
            output["Morgan-RF"].extend(
                zip(test_indices, np.mean(replicate_scores, axis=0))
            )
    predictions = {}
    for name, pairs in output.items():
        values = np.full(len(y), np.nan, dtype=float)
        for index, score in pairs:
            values[int(index)] = float(score)
        if np.isnan(values).any():
            raise RuntimeError(f"Incomplete shared-fold predictions for {name}.")
        predictions[name] = values
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    seed = int(config.get("random_seed", 42))
    model_type = config["model_type"]
    project_folder = Path(config["project_folder"])
    if not project_folder.is_absolute():
        project_folder = REPO_ROOT / project_folder
    project_dir = project_folder / config["project_name"]
    train_dir = (
        project_dir
        / "SMILESX"
        / "outputs"
        / "0"
        / config["data_name"]
        / ("Augm" if config.get("augmentation", True) else "Can")
        / "Train"
    )
    prediction_path = train_dir / f"{config['data_name']}_Predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)

    predictions = pd.read_csv(prediction_path)
    excluded_columns = {"SMILES", "Fold", "Mean", "Standard deviation"}
    target_columns = [column for column in predictions.columns if column not in excluded_columns]
    if len(target_columns) != 1:
        raise ValueError(f"Could not identify the target column in {prediction_path}.")
    target_column = target_columns[0]
    y_true = predictions[target_column].to_numpy(dtype=float)
    y_smilesx = predictions["Mean"].to_numpy(dtype=float)

    labeled = pd.read_csv(REPO_ROOT / "data" / config["project_name"] / "labeled.csv")
    unlabeled = pd.read_csv(REPO_ROOT / "data" / config["project_name"] / "unlabeled.csv")
    labeled, _, _ = _clean_input_data(labeled, unlabeled, model_type)
    target_by_smiles = labeled.set_index("smiles")["property"]
    clean_targets = np.asarray([target_by_smiles[s] for s in predictions["SMILES"]], dtype=float)
    if not np.allclose(clean_targets, y_true):
        raise ValueError("Prediction targets do not match the cleaned labeled dataset.")

    x, scaffolds = fingerprints_and_scaffolds(predictions["SMILES"])
    fold_assignments = predictions["Fold"].to_numpy(dtype=int)
    random_baselines = shared_fold_predictions(
        x, y_true, fold_assignments, model_type, seed, rf_replicates=3
    )
    scaffold_baselines = repeated_predictions(
        x, y_true, model_type, seed, scaffold_groups=scaffolds
    )

    analysis_dir = project_dir / "analysis" / "predictor"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    model_predictions = {"SMILES-X": y_smilesx, **random_baselines}
    rows = []
    detailed = {}
    detailed["validation_protocol"] = {
        "primary": "One shared five-fold partition; SMILES-X predictions average three independently initialized runs within each held-out fold, and Morgan baselines use the identical fold assignments.",
        "secondary": "Morgan baselines only: three repeated scaffold-group five-fold partitions.",
        "random_seed": seed,
    }
    metric_function = regression_metrics if model_type == "regression" else classification_metrics
    for validation, prediction_sets in (
        ("shared 5-fold CV (3 SMILES-X runs per fold)", model_predictions),
        ("repeated scaffold-group CV", scaffold_baselines),
    ):
        for model_name, values in prediction_sets.items():
            metrics = metric_function(y_true, values)
            intervals = bootstrap_intervals(y_true, values, metric_function, seed)
            detailed[f"{validation}::{model_name}"] = {
                "metrics": metrics,
                "bootstrap_95_ci": intervals,
            }
            for metric_name, value in metrics.items():
                if metric_name == "threshold":
                    continue
                interval = intervals.get(metric_name, {})
                rows.append(
                    {
                        "validation": validation,
                        "model": model_name,
                        "metric": metric_name,
                        "value": value,
                        "lower_95": interval.get("lower_95", np.nan),
                        "upper_95": interval.get("upper_95", np.nan),
                    }
                )

    if model_type == "classification":
        precision, recall, thresholds = precision_recall_curve(y_true, y_smilesx)
        f_scores = np.divide(
            2 * precision[:-1] * recall[:-1],
            precision[:-1] + recall[:-1],
            out=np.zeros_like(thresholds),
            where=(precision[:-1] + recall[:-1]) != 0,
        )
        best_index = int(np.argmax(f_scores))
        detailed["exploratory_oof_threshold"] = {
            "threshold": float(thresholds[best_index]),
            "f1_on_same_oof_predictions": float(f_scores[best_index]),
            "warning": "Exploratory only; selecting and evaluating a threshold on the same OOF predictions is optimistic.",
        }

    pd.DataFrame(rows).to_csv(analysis_dir / "predictor_metrics.csv", index=False)
    (analysis_dir / "predictor_metrics.json").write_text(
        json.dumps(detailed, indent=2), encoding="utf-8"
    )
    output_predictions = predictions[["SMILES", target_column]].copy()
    output_predictions["SMILES-X"] = y_smilesx
    for name, values in random_baselines.items():
        output_predictions[name] = values
    output_predictions.to_csv(analysis_dir / "oof_predictions.csv", index=False)

    model_artifacts = sorted((train_dir / "Models").glob("*.hdf5"))
    analysis_manifest = {
        "schema_version": 1,
        "analysis": "predictor_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": model_type,
        "protocol": detailed["validation_protocol"],
        "counts": {
            "molecules": int(len(y_true)),
            "folds": int(len(np.unique(fold_assignments))),
            "smilesx_runs_per_fold": int(config["n_runs"]),
        },
        "input_sha256": {
            "config": sha256(args.config.resolve()),
            "smilesx_predictions": sha256(prediction_path),
            "active_labeled": sha256(REPO_ROOT / "data" / config["project_name"] / "labeled.csv"),
            "active_unlabeled": sha256(REPO_ROOT / "data" / config["project_name"] / "unlabeled.csv"),
        },
        "source_sha256": {"scripts/analyze_predictor.py": sha256(Path(__file__).resolve())},
        "trained_model_sha256": {
            str(path.relative_to(project_dir)).replace("\\", "/"): sha256(path)
            for path in model_artifacts
        },
    }
    if model_type == "regression":
        figure, axis = plt.subplots(figsize=(5.2, 5.0))
        axis.scatter(y_true, y_smilesx, s=24, alpha=0.75, edgecolor="none")
        limits = [min(y_true.min(), y_smilesx.min()), max(y_true.max(), y_smilesx.max())]
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Observed logS", ylabel="Out-of-fold predicted logS")
    else:
        figure, axis = plt.subplots(figsize=(5.2, 5.0))
        false_positive, true_positive, _ = roc_curve(y_true, y_smilesx)
        axis.plot(false_positive, true_positive, linewidth=2, label="SMILES-X")
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="False-positive rate", ylabel="True-positive rate")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(analysis_dir / "predictor_performance.png", dpi=600)
    figure.savefig(analysis_dir / "predictor_performance.pdf")
    plt.close(figure)
    analysis_manifest["output_sha256"] = {
        filename: sha256(analysis_dir / filename)
        for filename in (
            "predictor_metrics.csv",
            "predictor_metrics.json",
            "oof_predictions.csv",
            "predictor_performance.png",
            "predictor_performance.pdf",
        )
    }
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
