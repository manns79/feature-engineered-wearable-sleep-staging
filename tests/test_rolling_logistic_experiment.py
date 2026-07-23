import pandas as pd
from src.models.locked_test_evaluation import run_locked_test_evaluation
from src.models.rolling_logistic_experiment import run_rolling_logistic_experiment


def _manifest():
    return pd.DataFrame(
        [
            {
                "feature": "HR_mean_roll3_mean",
                "feature_family": "rolling_context",
                "signal_group": "cardiovascular",
                "source_signal": "HR",
            },
            {
                "feature": "HR_mean_roll9_mean",
                "feature_family": "rolling_context",
                "signal_group": "cardiovascular",
                "source_signal": "HR",
            },
            {
                "feature": "ACC_MAG_std_roll9_mean",
                "feature_family": "rolling_context",
                "signal_group": "movement",
                "source_signal": "ACC_MAG",
            },
        ]
    )


def _feature_frame(split, *, n_participants):
    rows = []
    labels = ["Wake", "Non-REM", "REM"]
    for participant_index in range(n_participants):
        participant_id = f"S{participant_index + 1:03d}"
        for epoch_id, label in enumerate(labels):
            label_index = labels.index(label)
            value = float(label_index + participant_index * 0.01)
            rows.append(
                {
                    "participant_id": participant_id,
                    "epoch_id": epoch_id,
                    "split": split,
                    "label": label,
                    "HR_mean_roll3_mean": value,
                    "HR_mean_roll9_mean": value * 2,
                    "ACC_MAG_std_roll9_mean": float(label_index == 0),
                }
            )
    return pd.DataFrame(rows)


def test_rolling_logistic_experiment_writes_validation_only_outputs(tmp_path):
    train_path = tmp_path / "features_train.csv"
    val_path = tmp_path / "features_val.csv"
    manifest_path = tmp_path / "feature_manifest.csv"
    _feature_frame("train", n_participants=6).to_csv(train_path, index=False)
    _feature_frame("validation", n_participants=3).to_csv(val_path, index=False)
    _manifest().to_csv(manifest_path, index=False)

    outputs = run_rolling_logistic_experiment(
        train_features_path=train_path,
        validation_features_path=val_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "outputs",
        run_id="rolling_test",
        cv_splits=3,
        param_grid={
            "classifier__C": [1.0],
            "classifier__l1_ratio": [0.5],
        },
    )

    metrics = pd.read_csv(outputs.metrics_path)
    selected = pd.read_csv(outputs.selected_features_path)
    assert outputs.run_dir == tmp_path / "outputs" / "runs" / "rolling_test"
    assert set(metrics["model"]) == {
        "logistic_raw",
        "logistic_platt",
        "logistic_platt_viterbi",
    }
    assert "HR_mean_roll9_mean" in selected["feature"].tolist()
    assert "HR_mean_roll3_mean" not in selected["feature"].tolist()
    assert not (tmp_path / "features_test.csv").exists()


def test_locked_test_evaluation_is_separate_from_validation_experiment(tmp_path):
    train_path = tmp_path / "features_train.csv"
    val_path = tmp_path / "features_val.csv"
    test_path = tmp_path / "features_test.csv"
    manifest_path = tmp_path / "feature_manifest.csv"
    _feature_frame("train", n_participants=6).to_csv(train_path, index=False)
    _feature_frame("validation", n_participants=3).to_csv(val_path, index=False)
    _feature_frame("test", n_participants=3).to_csv(test_path, index=False)
    _manifest().to_csv(manifest_path, index=False)
    experiment = run_rolling_logistic_experiment(
        train_features_path=train_path,
        validation_features_path=val_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "outputs",
        run_id="rolling_test",
        cv_splits=3,
        param_grid={
            "classifier__C": [1.0],
            "classifier__l1_ratio": [0.5],
        },
    )

    outputs = run_locked_test_evaluation(
        rolling_run_dir=experiment.run_dir,
        test_features_path=test_path,
    )

    metrics = pd.read_csv(outputs.metrics_path)
    assert set(metrics["split"]) == {"test"}
    assert set(metrics["model"]) == {
        "logistic_raw",
        "logistic_platt",
        "logistic_platt_viterbi",
    }
