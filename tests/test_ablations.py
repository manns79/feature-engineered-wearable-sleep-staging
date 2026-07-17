import json

import pandas as pd
import pytest
from src.models.ablations import (
    PRIMARY_ABLATION_SPECS,
    AblationSpec,
    default_ablation_specs,
    run_ablation_experiments,
    signal_group_ablation_specs,
    validate_manifest_matches_features,
)


def _manifest():
    return pd.DataFrame(
        [
            {
                "feature": "BVP_mean",
                "feature_family": "basic_statistical",
                "signal_group": "cardiovascular",
                "source_signal": "BVP",
            },
            {
                "feature": "HR_abs_diff_mean",
                "feature_family": "signal_specific",
                "signal_group": "cardiovascular",
                "source_signal": "HR",
            },
            {
                "feature": "ACC_MAG_range",
                "feature_family": "basic_statistical",
                "signal_group": "movement",
                "source_signal": "ACC_MAG",
            },
            {
                "feature": "ACC_MAG_std_roll3_mean",
                "feature_family": "rolling_context",
                "signal_group": "movement",
                "source_signal": "ACC_MAG",
            },
        ]
    )


def _feature_frame(split, *, n_participants=6):
    rows = []
    labels = ["Wake", "Non-REM", "REM"]
    for participant_index in range(n_participants):
        participant_id = f"S{participant_index + 1:03d}"
        for epoch_index, label in enumerate(labels):
            label_index = labels.index(label)
            rows.append(
                {
                    "participant_id": participant_id,
                    "epoch_id": epoch_index,
                    "split": split,
                    "label": label,
                    "BVP_mean": float(label_index),
                    "HR_abs_diff_mean": float(label_index + 1),
                    "ACC_MAG_range": float(label_index * 2),
                    "ACC_MAG_std_roll3_mean": float(label_index * 3),
                }
            )
    return pd.DataFrame(rows)


def _small_param_grids():
    return {
        "elastic_net_logistic_regression": {
            "classifier__C": [1.0],
            "classifier__l1_ratio": [0.5],
        },
        "random_forest": {
            "classifier__n_estimators": [2],
            "classifier__max_depth": [None],
            "classifier__min_samples_leaf": [1],
            "classifier__max_features": ["sqrt"],
        },
    }


def test_primary_ablation_specs_select_cumulative_feature_families():
    manifest = _manifest()
    specs = {spec.name: spec for spec in PRIMARY_ABLATION_SPECS}

    assert specs["basic_statistical"].select_features(manifest) == [
        "BVP_mean",
        "ACC_MAG_range",
    ]
    assert specs["basic_plus_signal_specific"].select_features(manifest) == [
        "BVP_mean",
        "HR_abs_diff_mean",
        "ACC_MAG_range",
    ]
    assert specs["basic_signal_specific_rolling"].select_features(manifest) == [
        "BVP_mean",
        "HR_abs_diff_mean",
        "ACC_MAG_range",
        "ACC_MAG_std_roll3_mean",
    ]


def test_ablation_spec_accepts_single_string_filters():
    spec = AblationSpec(
        name="cardiovascular",
        description="Cardiovascular features.",
        signal_groups="cardiovascular",
    )

    assert spec.signal_groups == ("cardiovascular",)
    assert spec.select_features(_manifest()) == ["BVP_mean", "HR_abs_diff_mean"]


def test_signal_group_specs_are_manifest_driven():
    manifest = _manifest()
    specs = {spec.name: spec for spec in signal_group_ablation_specs(manifest)}

    assert sorted(specs) == ["signal_group_cardiovascular", "signal_group_movement"]
    assert specs["signal_group_movement"].select_features(manifest) == [
        "ACC_MAG_range",
        "ACC_MAG_std_roll3_mean",
    ]


def test_validate_manifest_matches_features_rejects_stale_manifest():
    with pytest.raises(ValueError, match="Missing from manifest"):
        validate_manifest_matches_features(_manifest(), ["BVP_mean", "TEMP_mean"])


def test_default_ablation_specs_include_primary_and_signal_groups():
    names = [spec.name for spec in default_ablation_specs(_manifest())]

    assert names[:5] == [spec.name for spec in PRIMARY_ABLATION_SPECS]
    assert "signal_group_movement" in names


def test_run_ablation_experiments_writes_combined_outputs_without_test_table(tmp_path):
    train_path = tmp_path / "features_train.csv"
    val_path = tmp_path / "features_val.csv"
    manifest_path = tmp_path / "feature_manifest.csv"
    _feature_frame("train").to_csv(train_path, index=False)
    _feature_frame("validation", n_participants=3).to_csv(val_path, index=False)
    _manifest().to_csv(manifest_path, index=False)

    outputs = run_ablation_experiments(
        feature_paths={
            "train": train_path,
            "validation": val_path,
        },
        manifest_path=manifest_path,
        output_dir=tmp_path / "outputs",
        ablation_names=("basic_plus_signal_specific", "signal_group_movement"),
        include_xgboost=False,
        cv_splits=3,
        param_grids=_small_param_grids(),
        run_id="test_run",
    )

    metrics = pd.read_csv(outputs.metrics_path)
    best_params = pd.read_csv(outputs.best_params_path)
    feature_sets = pd.read_csv(outputs.feature_sets_path)
    status = pd.read_csv(outputs.status_path)
    run_config = json.loads(outputs.run_config_path.read_text())
    assert set(outputs.run_outputs) == {
        "basic_plus_signal_specific",
        "signal_group_movement",
    }
    assert outputs.run_dir == tmp_path / "outputs" / "runs" / "test_run"
    assert outputs.metrics_path.parent == outputs.run_dir / "metrics"
    assert outputs.log_path.exists()
    assert outputs.status_path.exists()
    assert outputs.run_config_path.exists()
    assert run_config["run_id"] == "test_run"
    assert run_config["resolved_ablations"] == [
        "basic_plus_signal_specific",
        "signal_group_movement",
    ]
    assert set(metrics["ablation"]) == set(outputs.run_outputs)
    assert set(best_params["ablation"]) == set(outputs.run_outputs)
    assert set(feature_sets["ablation"]) == set(outputs.run_outputs)
    assert {"ablation_start", "model_start", "model_completed"}.issubset(
        set(status["event"])
    )
    assert "selected_features" in feature_sets.columns
    for ablation, run_output in outputs.run_outputs.items():
        assert run_output.metrics_path.parent == (
            outputs.run_dir / "ablations" / ablation / "metrics"
        )
        assert all(
            path.parent == outputs.run_dir / "ablations" / ablation / "models"
            for path in run_output.model_paths.values()
        )
    assert not (tmp_path / "features_test.csv").exists()
