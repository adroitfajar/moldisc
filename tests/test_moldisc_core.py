import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from moldisc import (
    _clean_input_data,
    _normalize_allowed_elements,
    _smilesx_artifacts_complete,
    _target_reached,
    moldisc,
)
from SMILESX.augm import augmentation as augment_smiles


def test_clean_input_data_canonicalizes_and_removes_overlap():
    labeled = pd.DataFrame({"smiles": ["CCO", "OCC"], "property": [1.0, 1.0]})
    unlabeled = pd.DataFrame({"smiles": ["CCO", "CCN", "not_smiles"]})

    clean_labeled, clean_unlabeled, report = _clean_input_data(
        labeled, unlabeled, "regression"
    )

    assert clean_labeled["smiles"].tolist() == ["CCO"]
    assert clean_unlabeled["smiles"].tolist() == ["CCN"]
    assert report["canonical_labeled_duplicates_removed"] == 1
    assert report["invalid_unlabeled_count"] == 1
    assert report["labeled_unlabeled_overlap_removed_from_unlabeled_count"] == 1


def test_clean_input_data_rejects_conflicting_duplicate_labels():
    labeled = pd.DataFrame({"smiles": ["CCO", "OCC"], "property": [0, 1]})
    unlabeled = pd.DataFrame({"smiles": ["CCN"]})
    with pytest.raises(ValueError, match="Conflicting property"):
        _clean_input_data(labeled, unlabeled, "classification")


def test_clean_input_data_averages_regression_replicates():
    labeled = pd.DataFrame({"smiles": ["CCO", "OCC"], "property": [1.0, 3.0]})
    unlabeled = pd.DataFrame({"smiles": ["CCN"]})

    clean_labeled, _, report = _clean_input_data(labeled, unlabeled, "regression")

    assert clean_labeled.to_dict("records") == [{"property": 2.0, "smiles": "CCO"}]
    assert report["conflicting_labeled_duplicate_count"] == 1
    assert report["conflicting_labeled_duplicate_policy"] == "mean"


def test_clean_input_data_can_drop_conflicting_regression_replicates():
    labeled = pd.DataFrame(
        {"smiles": ["CCO", "OCC", "CCN"], "property": [1.0, 3.0, 2.0]}
    )
    unlabeled = pd.DataFrame({"smiles": ["CCC"]})

    clean_labeled, _, report = _clean_input_data(
        labeled, unlabeled, "regression", duplicate_policy="drop"
    )

    assert clean_labeled["smiles"].tolist() == ["CCN"]
    assert report["conflicting_labeled_duplicate_policy"] == "drop"


def test_target_property_modes():
    values = pd.Series([-2.0, -1.0, 0.2])
    assert _target_reached(values, 0.0, "max")
    assert _target_reached(values, -1.5, "min")
    assert not _target_reached(values, 1.0, "max")


def test_allowed_elements_are_validated_and_deduplicated():
    assert _normalize_allowed_elements(["C", "N", "Cl", "C"]) == ["C", "N", "Cl"]
    with pytest.raises(ValueError, match="Unknown"):
        _normalize_allowed_elements(["C", "chlorine"])
    with pytest.raises(ValueError, match="at least one"):
        _normalize_allowed_elements([])


def test_smiles_augmentation_cap_is_deterministic():
    smiles = pd.DataFrame({"smiles": ["CCCCCCCCCCCC"]}).to_numpy()
    first = augment_smiles(
        smiles, [0], check_smiles=True, augment=True, max_augmentations=4
    )[0]
    second = augment_smiles(
        smiles, [0], check_smiles=True, augment=True, max_augmentations=4
    )[0]

    assert first == second
    assert len(first) == 4


def test_smilesx_artifact_reuse_requires_complete_ensemble(tmp_path):
    train_dir = tmp_path / "demo" / "Augm" / "Train"
    (train_dir / "Other" / "Scalers").mkdir(parents=True)
    (train_dir / "Models").mkdir()
    (train_dir / "Other" / "demo_Vocabulary.txt").write_text("C\n", encoding="utf-8")
    for fold in range(2):
        (train_dir / "Other" / "Scalers" / f"demo_Scaler_Outputs_Fold_{fold}.pkl").touch()
        for run in range(2):
            (train_dir / "Models" / f"demo_Model_Fold_{fold}_Run_{run}.hdf5").touch()

    assert _smilesx_artifacts_complete(tmp_path, "demo", True, 2, 2, True)
    (train_dir / "Models" / "demo_Model_Fold_1_Run_1.hdf5").unlink()
    assert not _smilesx_artifacts_complete(tmp_path, "demo", True, 2, 2, True)


def test_paper_configs_only_use_public_api_parameters():
    root = Path(__file__).resolve().parents[1]
    parameters = set(inspect.signature(moldisc).parameters)
    for filename in ("paper_regression.json", "paper_classification.json"):
        config = json.loads((root / "configs" / filename).read_text(encoding="utf-8"))
        assert set(config).issubset(parameters)
