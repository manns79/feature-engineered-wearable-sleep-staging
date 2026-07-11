import pandas as pd
from src.models.training import _group_kfold, train_and_evaluate_model_set


def _feature_frame(split, *, n_participants=6):
    rows = []
    labels = ["Wake", "Non-REM", "REM"]
    feature_lookup = {
        "Wake": (0.0, 70.0),
        "Non-REM": (1.0, 60.0),
        "REM": (2.0, 75.0),
    }
    for participant_index in range(n_participants):
        participant_id = f"S{participant_index + 1:03d}"
        for epoch_index, label in enumerate(labels):
            bvp_mean, hr_mean = feature_lookup[label]
            rows.append(
                {
                    "participant_id": participant_id,
                    "epoch_id": epoch_index,
                    "split": split,
                    "label": label,
                    "BVP_mean": bvp_mean + participant_index * 0.01,
                    "HR_mean": hr_mean,
                    "TEMP_slope": 0.1 if label == "REM" else None,
                }
            )
    return pd.DataFrame(rows)


def test_group_kfold_keeps_participants_out_of_both_train_and_validation():
    frame = _feature_frame("train", n_participants=6)
    groups = frame["participant_id"]
    cv = _group_kfold(groups, requested_splits=3)

    for train_indices, validation_indices in cv.split(frame, frame["label"], groups):
        train_groups = set(groups.iloc[train_indices])
        validation_groups = set(groups.iloc[validation_indices])
        assert train_groups.isdisjoint(validation_groups)


def test_train_and_evaluate_model_set_writes_validation_only_outputs(tmp_path):
    train_path = tmp_path / "features_train.csv"
    val_path = tmp_path / "features_val.csv"
    _feature_frame("train", n_participants=6).to_csv(train_path, index=False)
    _feature_frame("validation", n_participants=3).to_csv(val_path, index=False)

    outputs = train_and_evaluate_model_set(
        feature_paths={
            "train": train_path,
            "validation": val_path,
        },
        output_dir=tmp_path / "outputs",
        include_xgboost=False,
        cv_splits=3,
        param_grids={
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
        },
    )

    metrics = pd.read_csv(outputs.metrics_path)
    best_params = pd.read_csv(outputs.best_params_path)
    cv_results = pd.read_csv(outputs.cv_results_path)
    assert set(metrics["model"]) == {
        "majority_class",
        "stratified_random",
        "elastic_net_logistic_regression",
        "random_forest",
    }
    assert set(metrics["split"]) == {"validation"}
    assert set(best_params["model"]) == set(metrics["model"])
    assert {"elastic_net_logistic_regression", "random_forest"}.issubset(
        set(cv_results["model"])
    )
    assert all(key.startswith("validation:") for key in outputs.prediction_paths)
    assert all(key.startswith("validation:") for key in outputs.confusion_paths)
    assert all("_test_" not in path.name for path in outputs.prediction_paths.values())
    assert all("_test_" not in path.name for path in outputs.confusion_paths.values())
    assert len(outputs.model_paths) == 4
