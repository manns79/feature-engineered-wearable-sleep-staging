import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.models.locked_test_evaluation import (
    FinalPriorModelSpec,
    _fit_prior_postprocessors,
    _prior_ablation_predictions,
    _resolve_prior_model_runs,
)


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
                }
            )
    return pd.DataFrame(rows)


def _write_prior_run(tmp_path, model_path):
    run_dir = tmp_path / "outputs" / "runs" / "full_ablation"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ablation": "basic_statistical",
                "selected_features": "BVP_mean|HR_mean",
            }
        ]
    ).to_csv(metrics_dir / "ablation_feature_sets.csv", index=False)
    pd.DataFrame(
        [
            {
                "event": "model_completed",
                "status": "completed",
                "ablation": "basic_statistical",
                "feature_set": "ablation_basic_statistical",
                "model": "elastic_net_logistic_regression",
                "model_path": str(model_path),
            }
        ]
    ).to_csv(run_dir / "run_status.csv", index=False)
    return run_dir


def test_final_prior_model_selection_resolves_frozen_artifacts(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_text("placeholder")
    run_dir = _write_prior_run(tmp_path, model_path)

    runs = _resolve_prior_model_runs(
        run_dir,
        prior_model_specs=(
            FinalPriorModelSpec(
                candidate="statistical_summary_only",
                ablation="basic_statistical",
                model="elastic_net_logistic_regression",
                apply_postprocessing=True,
            ),
        ),
    )

    assert len(runs) == 1
    assert runs[0].spec.candidate == "statistical_summary_only"
    assert runs[0].feature_set == "ablation_basic_statistical"
    assert runs[0].selected_features == ("BVP_mean", "HR_mean")


def test_prior_postprocessing_uses_train_oof_and_applies_to_split(tmp_path):
    train = _feature_frame("train", n_participants=6)
    test = _feature_frame("test", n_participants=2)
    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )
    estimator.fit(train[["BVP_mean", "HR_mean"]], train["label"])
    model_path = tmp_path / "model.joblib"
    joblib.dump(estimator, model_path)
    run_dir = _write_prior_run(tmp_path, model_path)
    runs = _resolve_prior_model_runs(
        run_dir,
        prior_model_specs=(
            FinalPriorModelSpec(
                candidate="statistical_summary_only",
                ablation="basic_statistical",
                model="elastic_net_logistic_regression",
                apply_postprocessing=True,
            ),
        ),
    )

    artifacts = _fit_prior_postprocessors(
        runs,
        train,
        output_dir=tmp_path / "postprocessing",
        threshold_grid=(0.2, 0.5, 0.8),
        smoothing_windows=(3,),
    )
    predictions = _prior_ablation_predictions(
        runs,
        test,
        prior_postprocessors=artifacts,
    )
    variants = {frame["variant"].iloc[0] for frame in predictions}

    assert variants == {
        "raw",
        "raw_threshold_tuned",
        "platt",
        "platt_threshold_tuned",
        "platt_smoothed",
        "platt_smoothed_threshold_tuned",
    }
    assert (tmp_path / "postprocessing/statistical_summary_only").exists()
    assert all(set(frame["split"]) == {"test"} for frame in predictions)
    metadata = pd.read_csv(
        tmp_path
        / "postprocessing"
        / "statistical_summary_only"
        / "postprocessing_metadata.csv"
    )
    assert metadata["postprocessing_tuning_source"].tolist() == ["train_out_of_fold"]
