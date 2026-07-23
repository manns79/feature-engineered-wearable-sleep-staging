import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.models.error_analysis import (
    add_transition_context,
    load_validation_features,
    permutation_importance_frame,
    run_validation_error_analysis,
    select_validation_model_runs,
)


def _write_run_artifacts(tmp_path, *, include_random_forest=True):
    run_dir = tmp_path / "outputs" / "runs" / "ablation_run"
    metrics_dir = run_dir / "metrics"
    ablation_metrics_dir = run_dir / "ablations" / "basic_statistical" / "metrics"
    models_dir = run_dir / "ablations" / "basic_statistical" / "models"
    metrics_dir.mkdir(parents=True)
    ablation_metrics_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": "ablation_run",
                "resolved_ablations": ["basic_statistical"],
                "include_xgboost": False,
            }
        )
        + "\n"
    )
    pd.DataFrame(
        [
            {
                "ablation": "basic_statistical",
                "selected_features": "BVP_mean|HR_mean",
                "selected_feature_families": "basic_statistical",
            }
        ]
    ).to_csv(metrics_dir / "ablation_feature_sets.csv", index=False)

    rows = [
        _status_row(
            run_dir,
            ablation_metrics_dir,
            models_dir,
            model="elastic_net_logistic_regression",
            validation_macro_f1=0.50,
        )
    ]
    if include_random_forest:
        rows.append(
            _status_row(
                run_dir,
                ablation_metrics_dir,
                models_dir,
                model="random_forest",
                validation_macro_f1=0.55,
            )
        )
    pd.DataFrame(rows).to_csv(run_dir / "run_status.csv", index=False)
    return run_dir


def _status_row(run_dir, metrics_dir, models_dir, *, model, validation_macro_f1):
    prediction_path = (
        metrics_dir / f"ablation_basic_statistical_validation_{model}_predictions.csv"
    )
    confusion_path = (
        metrics_dir / f"ablation_basic_statistical_validation_{model}_confusion.csv"
    )
    model_path = models_dir / f"ablation_basic_statistical_{model}.joblib"
    prediction_path.write_text("participant_id,epoch_id,split,true_label,pred_label\n")
    confusion_path.write_text(",pred_Wake,pred_Non-REM,pred_REM\n")
    model_path.write_text("placeholder")
    return {
        "event": "model_completed",
        "status": "completed",
        "ablation": "basic_statistical",
        "feature_set": "ablation_basic_statistical",
        "model": model,
        "n_features": 2,
        "best_cv_macro_f1": 0.4,
        "validation_macro_f1": validation_macro_f1,
        "validation_accuracy": 0.6,
        "elapsed_seconds": 1.0,
        "prediction_path": str(prediction_path),
        "confusion_path": str(confusion_path),
        "model_path": str(model_path),
        "output_dir": str(run_dir / "ablations" / "basic_statistical"),
    }


def test_select_validation_model_runs_selects_requested_models_per_ablation(tmp_path):
    run_dir = _write_run_artifacts(tmp_path)

    selected = select_validation_model_runs(
        run_dir,
        models=("elastic_net_logistic_regression", "random_forest"),
    )

    assert selected["model"].tolist() == [
        "elastic_net_logistic_regression",
        "random_forest",
    ]
    assert selected["ablation"].tolist() == [
        "basic_statistical",
        "basic_statistical",
    ]
    assert selected["selected_features"].tolist() == [
        "BVP_mean|HR_mean",
        "BVP_mean|HR_mean",
    ]


def test_select_validation_model_runs_rejects_incomplete_selection(tmp_path):
    run_dir = _write_run_artifacts(tmp_path, include_random_forest=False)

    with pytest.raises(ValueError, match="incomplete"):
        select_validation_model_runs(
            run_dir,
            models=("elastic_net_logistic_regression", "random_forest"),
        )


def test_load_validation_features_rejects_non_validation_split(tmp_path):
    path = tmp_path / "features_test.csv"
    pd.DataFrame(
        {
            "participant_id": ["S001"],
            "epoch_id": [0],
            "split": ["test"],
            "label": ["Wake"],
            "BVP_mean": [1.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="validation feature table only"):
        load_validation_features(path)


def test_add_transition_context_stays_within_participant_boundaries():
    predictions = pd.DataFrame(
        {
            "ablation": ["a"] * 6,
            "model": ["m"] * 6,
            "feature_set": ["f"] * 6,
            "participant_id": ["S001", "S001", "S001", "S002", "S002", "S002"],
            "epoch_id": [0, 1, 2, 0, 1, 2],
            "split": ["validation"] * 6,
            "true_label": ["Wake", "Wake", "REM", "Non-REM", "Non-REM", "Non-REM"],
            "pred_label": ["Wake", "REM", "REM", "Non-REM", "Wake", "Non-REM"],
        }
    )

    output = add_transition_context(predictions)

    s001 = output[output["participant_id"] == "S001"].sort_values("epoch_id")
    s002 = output[output["participant_id"] == "S002"].sort_values("epoch_id")
    assert s001["distance_to_transition_epochs"].tolist() == [1, 0, 0]
    assert s001["transition_distance_bin"].astype(str).tolist() == ["1", "0", "0"]
    assert s002["distance_to_transition_epochs"].isna().all()
    assert s002["transition_distance_bin"].astype(str).tolist() == [
        "no_transition_in_participant",
        "no_transition_in_participant",
        "no_transition_in_participant",
    ]


def test_run_validation_error_analysis_writes_validation_only_outputs(tmp_path):
    run_dir = _write_run_artifacts(tmp_path, include_random_forest=False)
    features_path = tmp_path / "features_val.csv"
    predictions_path = (
        run_dir
        / "ablations"
        / "basic_statistical"
        / "metrics"
        / (
            "ablation_basic_statistical_validation_"
            "elastic_net_logistic_regression_predictions.csv"
        )
    )
    model_path = (
        run_dir
        / "ablations"
        / "basic_statistical"
        / "models"
        / "ablation_basic_statistical_elastic_net_logistic_regression.joblib"
    )
    features = pd.DataFrame(
        {
            "participant_id": ["S001", "S001", "S001", "S002", "S002", "S002"],
            "epoch_id": [0, 1, 2, 0, 1, 2],
            "split": ["validation"] * 6,
            "label": ["Wake", "Non-REM", "REM", "Wake", "Non-REM", "REM"],
            "BVP_mean": [0.0, 1.0, 2.0, 0.1, 1.1, 2.1],
            "HR_mean": [70.0, 60.0, 75.0, 71.0, 61.0, 76.0],
        }
    )
    features.to_csv(features_path, index=False)
    predictions = features[["participant_id", "epoch_id", "split"]].copy()
    predictions["true_label"] = features["label"]
    predictions["pred_label"] = features["label"]
    predictions.to_csv(predictions_path, index=False)
    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )
    estimator.fit(features[["BVP_mean", "HR_mean"]], features["label"])
    joblib.dump(estimator, model_path)

    outputs = run_validation_error_analysis(
        run_dir=run_dir,
        validation_features_path=features_path,
        manifest_path=None,
        epoch_index_path=None,
        models=("elastic_net_logistic_regression",),
        compute_permutation=False,
        compute_shap=False,
        create_plots=True,
    )

    assert outputs.selected_models_path.exists()
    assert outputs.per_class_metrics_path.exists()
    assert outputs.transition_metrics_path.exists()
    assert pd.read_csv(outputs.predictions_path)["split"].unique().tolist() == [
        "validation"
    ]
    artifact_names = {path.name for path in outputs.artifact_index["path"].map(Path)}
    assert (
        "basic_statistical_elastic_net_logistic_regression_confusion_counts.png"
        not in artifact_names
    )
    assert (
        "basic_statistical_elastic_net_logistic_regression_confusion_true_normalized.png"
        in artifact_names
    )
    assert (
        "basic_statistical_elastic_net_logistic_regression_feature_importance.png"
        not in artifact_names
    )


def test_permutation_importance_supports_encoded_label_model_artifacts(tmp_path):
    features = pd.DataFrame(
        {
            "participant_id": ["S001", "S001", "S001", "S002", "S002", "S002"],
            "epoch_id": [0, 1, 2, 0, 1, 2],
            "split": ["validation"] * 6,
            "label": ["Wake", "Non-REM", "REM", "Wake", "Non-REM", "REM"],
            "BVP_mean": [0.0, 1.0, 2.0, 0.1, 1.1, 2.1],
            "HR_mean": [70.0, 60.0, 75.0, 71.0, 61.0, 76.0],
        }
    )
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(features["label"])
    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )
    estimator.fit(features[["BVP_mean", "HR_mean"]], encoded_labels)
    model_path = tmp_path / "encoded_model.joblib"
    joblib.dump({"estimator": estimator, "label_encoder": encoder}, model_path)
    selected_models = pd.DataFrame(
        [
            {
                "ablation": "basic_statistical",
                "model": "xgboost",
                "model_path": str(model_path),
                "selected_features": "BVP_mean|HR_mean",
            }
        ]
    )

    importance = permutation_importance_frame(
        selected_models=selected_models,
        validation_features=features,
        n_repeats=1,
        random_state=42,
    )

    assert set(importance["feature"]) == {"BVP_mean", "HR_mean"}
