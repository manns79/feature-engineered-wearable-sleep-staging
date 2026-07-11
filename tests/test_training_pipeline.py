import pandas as pd
from src.models.train_models import train_and_evaluate_model_set


def _feature_frame(split):
    labels = ["Wake", "Non-REM", "REM"] * 3
    return pd.DataFrame(
        {
            "participant_id": [f"S{index:03d}" for index in range(len(labels))],
            "epoch_id": list(range(len(labels))),
            "split": [split] * len(labels),
            "label": labels,
            "BVP_mean": [0.0, 1.0, 2.0] * 3,
            "HR_mean": [70.0, 60.0, 75.0] * 3,
            "TEMP_slope": [0.0, None, 0.1] * 3,
        }
    )


def test_train_and_evaluate_model_set_writes_metrics_without_xgboost(tmp_path):
    train_path = tmp_path / "features_train.csv"
    val_path = tmp_path / "features_val.csv"
    test_path = tmp_path / "features_test.csv"
    _feature_frame("train").to_csv(train_path, index=False)
    _feature_frame("validation").to_csv(val_path, index=False)
    _feature_frame("test").to_csv(test_path, index=False)

    outputs = train_and_evaluate_model_set(
        feature_paths={
            "train": train_path,
            "validation": val_path,
            "test": test_path,
        },
        output_dir=tmp_path / "outputs",
        include_xgboost=False,
    )

    metrics = pd.read_csv(outputs.metrics_path)
    assert set(metrics["model"]) == {
        "majority_class",
        "stratified_random",
        "elastic_net_logistic_regression",
        "random_forest",
    }
    assert set(metrics["split"]) == {"validation", "test"}
    assert len(outputs.model_paths) == 4
