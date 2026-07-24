"""Rolling-context logistic experiment with train-only correlation pruning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from src.config import TARGET_LABELS
from src.features.build_features import FEATURE_ID_COLUMNS
from src.models.calibration import (
    DEFAULT_THRESHOLD_GRID,
    OneVsRestPlattCalibrator,
    align_probabilities,
    probability_frame,
    threshold_prediction_frame,
    tune_class_thresholds,
)
from src.models.evaluate import classification_metrics, confusion_matrix_frame
from src.models.feature_selection import (
    DEFAULT_CORRELATION_THRESHOLD,
    correlation_prune_features,
    select_manifest_candidates,
)
from src.models.sequence_postprocessing import (
    DEFAULT_SMOOTHING_WINDOWS,
    apply_viterbi_by_participant,
    estimate_transition_model,
    smooth_probabilities_by_participant,
    tune_smoothing_window,
)
from src.models.train_baselines import elastic_net_logistic_regression
from src.models.training import MACRO_F1_SCORER, _group_kfold
from src.models.tune_models import LOGISTIC_PARAM_GRID


@dataclass(frozen=True)
class RollingLogisticExperimentOutputs:
    """Paths written by the rolling logistic experiment."""

    run_dir: Path
    config_path: Path
    selected_features_path: Path
    feature_metadata_path: Path
    dropped_features_path: Path
    correlation_edges_path: Path
    cv_results_path: Path
    best_params_path: Path
    train_oof_predictions_path: Path
    metrics_path: Path
    raw_predictions_path: Path
    raw_threshold_predictions_path: Path
    calibrated_predictions_path: Path
    decoded_predictions_path: Path
    threshold_predictions_path: Path
    smoothed_predictions_path: Path
    smoothed_threshold_predictions_path: Path
    raw_confusion_path: Path
    raw_threshold_confusion_path: Path
    calibrated_confusion_path: Path
    decoded_confusion_path: Path
    threshold_confusion_path: Path
    smoothed_confusion_path: Path
    smoothed_threshold_confusion_path: Path
    threshold_tuning_path: Path
    raw_threshold_tuning_path: Path
    smoothing_tuning_path: Path
    smoothed_threshold_tuning_path: Path
    model_path: Path
    calibrator_path: Path
    raw_threshold_rule_path: Path
    threshold_rule_path: Path
    smoothed_threshold_rule_path: Path
    transition_model_path: Path


def run_rolling_logistic_experiment(
    *,
    train_features_path: str | Path = "data/processed/features_train.csv",
    validation_features_path: str | Path = "data/processed/features_val.csv",
    manifest_path: str | Path = "data/processed/feature_manifest.csv",
    output_dir: str | Path = "outputs",
    run_id: str | None = None,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    random_state: int = 42,
    cv_splits: int = 5,
    n_jobs: int | None = None,
    verbose_search: int = 0,
    param_grid: dict[str, list[Any]] | None = None,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
    smoothing_windows: tuple[int, ...] = DEFAULT_SMOOTHING_WINDOWS,
) -> RollingLogisticExperimentOutputs:
    """Run the constrained validation-only rolling logistic experiment."""
    train_features = _load_split_features(train_features_path, expected_split="train")
    validation_features = _load_split_features(
        validation_features_path, expected_split="validation"
    )
    manifest = pd.read_csv(manifest_path)

    candidate_features = select_manifest_candidates(manifest)
    pruning = correlation_prune_features(
        train_features,
        manifest,
        candidate_features=candidate_features,
        threshold=correlation_threshold,
    )
    selected_features = pruning.selected_features

    run_root = _create_run_root(output_dir, run_id)
    metrics_dir = run_root / "metrics"
    models_dir = run_root / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    selected_features_path = metrics_dir / "selected_features.csv"
    feature_metadata_path = metrics_dir / "feature_selection_metadata.csv"
    dropped_features_path = metrics_dir / "dropped_correlated_features.csv"
    correlation_edges_path = metrics_dir / "high_correlation_edges.csv"
    _write_selected_features(selected_features, selected_features_path)
    pruning.feature_metadata.to_csv(feature_metadata_path, index=False)
    pruning.dropped_features.to_csv(dropped_features_path, index=False)
    pruning.correlation_edges.to_csv(correlation_edges_path, index=False)

    fitted, cv_results, best_params = _fit_logistic_grid(
        train_features,
        selected_features,
        random_state=random_state,
        cv_splits=cv_splits,
        n_jobs=n_jobs,
        verbose_search=verbose_search,
        param_grid=LOGISTIC_PARAM_GRID if param_grid is None else param_grid,
    )
    model_path = models_dir / "rolling_logistic_model.joblib"
    joblib.dump(fitted, model_path)

    cv_results_path = metrics_dir / "rolling_logistic_cv_results.csv"
    best_params_path = metrics_dir / "rolling_logistic_best_params.csv"
    cv_results.to_csv(cv_results_path, index=False)
    pd.DataFrame(
        [
            {
                "model": "elastic_net_logistic_regression",
                "feature_set": "rolling_context_corr_pruned",
                "best_params": repr(best_params),
                "best_cv_macro_f1": float(cv_results.iloc[0]["mean_test_score"]),
                "cv_n_splits": _group_kfold(
                    train_features["participant_id"], cv_splits
                ).get_n_splits(),
            }
        ]
    ).to_csv(best_params_path, index=False)

    oof_raw_probabilities = _out_of_fold_raw_probabilities(
        train_features,
        selected_features,
        best_params,
        random_state=random_state,
        cv_splits=cv_splits,
    )
    train_oof_predictions = probability_frame(train_features, oof_raw_probabilities)
    train_oof_predictions_path = metrics_dir / "train_oof_raw_predictions.csv"
    train_oof_predictions.to_csv(train_oof_predictions_path, index=False)

    raw_probabilities = align_probabilities(
        fitted.predict_proba(validation_features[selected_features]),
        fitted.classes_,
        TARGET_LABELS,
    )
    raw_predictions = probability_frame(validation_features, raw_probabilities)
    raw_predictions_path = metrics_dir / "validation_raw_predictions.csv"
    raw_predictions.to_csv(raw_predictions_path, index=False)

    raw_threshold_tuning = tune_class_thresholds(
        oof_raw_probabilities,
        train_features["label"],
        threshold_grid=threshold_grid,
    )
    raw_threshold_tuning_path = (
        metrics_dir / "train_oof_raw_threshold_tuning_results.csv"
    )
    raw_threshold_tuning.results.to_csv(raw_threshold_tuning_path, index=False)
    raw_threshold_rule_path = models_dir / "raw_threshold_rule.joblib"
    joblib.dump(raw_threshold_tuning.rule, raw_threshold_rule_path)
    raw_threshold_predictions = threshold_prediction_frame(
        validation_features, raw_probabilities, raw_threshold_tuning.rule
    )
    raw_threshold_predictions_path = (
        metrics_dir / "validation_raw_threshold_predictions.csv"
    )
    raw_threshold_predictions.to_csv(raw_threshold_predictions_path, index=False)

    calibrator = OneVsRestPlattCalibrator(classes=TARGET_LABELS).fit(
        oof_raw_probabilities, train_features["label"]
    )
    calibrator_path = models_dir / "platt_calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)
    oof_calibrated_probabilities = calibrator.predict_proba(oof_raw_probabilities)
    calibrated_probabilities = calibrator.predict_proba(raw_probabilities)
    calibrated_predictions = probability_frame(
        validation_features, calibrated_probabilities
    )
    calibrated_predictions_path = metrics_dir / "validation_calibrated_predictions.csv"
    calibrated_predictions.to_csv(calibrated_predictions_path, index=False)

    threshold_tuning = tune_class_thresholds(
        oof_calibrated_probabilities,
        train_features["label"],
        threshold_grid=threshold_grid,
    )
    threshold_tuning_path = metrics_dir / "train_oof_threshold_tuning_results.csv"
    threshold_tuning.results.to_csv(threshold_tuning_path, index=False)
    threshold_rule_path = models_dir / "threshold_rule.joblib"
    joblib.dump(threshold_tuning.rule, threshold_rule_path)
    threshold_predictions = threshold_prediction_frame(
        validation_features, calibrated_probabilities, threshold_tuning.rule
    )
    threshold_predictions_path = (
        metrics_dir / "validation_calibrated_threshold_predictions.csv"
    )
    threshold_predictions.to_csv(threshold_predictions_path, index=False)

    smoothing_selection = tune_smoothing_window(
        train_features,
        oof_calibrated_probabilities,
        train_features["label"],
        windows=smoothing_windows,
    )
    smoothing_tuning_path = metrics_dir / "train_oof_smoothing_tuning_results.csv"
    smoothing_selection.results.to_csv(smoothing_tuning_path, index=False)
    smoothed_probabilities = smooth_probabilities_by_participant(
        validation_features,
        calibrated_probabilities,
        window_epochs=smoothing_selection.window_epochs,
    )
    smoothed_predictions = probability_frame(
        validation_features, smoothed_probabilities
    )
    smoothed_predictions_path = (
        metrics_dir / "validation_calibrated_smoothed_predictions.csv"
    )
    smoothed_predictions.to_csv(smoothed_predictions_path, index=False)

    oof_smoothed_probabilities = smooth_probabilities_by_participant(
        train_features,
        oof_calibrated_probabilities,
        window_epochs=smoothing_selection.window_epochs,
    )
    smoothed_threshold_tuning = tune_class_thresholds(
        oof_smoothed_probabilities,
        train_features["label"],
        threshold_grid=threshold_grid,
    )
    smoothed_threshold_tuning_path = (
        metrics_dir / "train_oof_smoothed_threshold_tuning_results.csv"
    )
    smoothed_threshold_tuning.results.to_csv(
        smoothed_threshold_tuning_path, index=False
    )
    smoothed_threshold_rule_path = models_dir / "smoothed_threshold_rule.joblib"
    joblib.dump(smoothed_threshold_tuning.rule, smoothed_threshold_rule_path)
    smoothed_threshold_predictions = threshold_prediction_frame(
        validation_features, smoothed_probabilities, smoothed_threshold_tuning.rule
    )
    smoothed_threshold_predictions_path = (
        metrics_dir / "validation_calibrated_smoothed_threshold_predictions.csv"
    )
    smoothed_threshold_predictions.to_csv(
        smoothed_threshold_predictions_path, index=False
    )

    transition_model = estimate_transition_model(
        train_features[["participant_id", "epoch_id", "label"]]
    )
    transition_model_path = models_dir / "transition_model.joblib"
    joblib.dump(transition_model, transition_model_path)
    decoded_predictions = apply_viterbi_by_participant(
        calibrated_predictions, transition_model
    )
    decoded_predictions_path = (
        metrics_dir / "validation_calibrated_viterbi_predictions.csv"
    )
    decoded_predictions.to_csv(decoded_predictions_path, index=False)

    metrics_path = metrics_dir / "validation_metrics.csv"
    raw_confusion_path = metrics_dir / "validation_raw_confusion.csv"
    raw_threshold_confusion_path = (
        metrics_dir / "validation_raw_threshold_confusion.csv"
    )
    calibrated_confusion_path = metrics_dir / "validation_calibrated_confusion.csv"
    decoded_confusion_path = metrics_dir / "validation_calibrated_viterbi_confusion.csv"
    threshold_confusion_path = (
        metrics_dir / "validation_calibrated_threshold_confusion.csv"
    )
    smoothed_confusion_path = (
        metrics_dir / "validation_calibrated_smoothed_confusion.csv"
    )
    smoothed_threshold_confusion_path = (
        metrics_dir / "validation_calibrated_smoothed_threshold_confusion.csv"
    )
    _write_comparison_metrics(
        [
            ("logistic_raw", raw_predictions, raw_confusion_path),
            (
                "logistic_threshold_tuned",
                raw_threshold_predictions,
                raw_threshold_confusion_path,
            ),
            ("logistic_platt", calibrated_predictions, calibrated_confusion_path),
            ("logistic_platt_viterbi", decoded_predictions, decoded_confusion_path),
            (
                "logistic_platt_threshold_tuned",
                threshold_predictions,
                threshold_confusion_path,
            ),
            ("logistic_platt_smoothed", smoothed_predictions, smoothed_confusion_path),
            (
                "logistic_platt_smoothed_threshold_tuned",
                smoothed_threshold_predictions,
                smoothed_threshold_confusion_path,
            ),
        ],
        metrics_path,
    )

    config_path = run_root / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "train_features_path": str(train_features_path),
                "validation_features_path": str(validation_features_path),
                "manifest_path": str(manifest_path),
                "correlation_threshold": correlation_threshold,
                "candidate_feature_count": len(candidate_features),
                "selected_feature_count": len(selected_features),
                "target_labels": list(TARGET_LABELS),
                "random_state": random_state,
                "cv_splits": cv_splits,
                "n_jobs": n_jobs,
                "param_grid": LOGISTIC_PARAM_GRID if param_grid is None else param_grid,
                "threshold_grid": threshold_grid,
                "smoothing_windows": smoothing_windows,
                "postprocessing_tuning_source": "train_out_of_fold",
                "selected_smoothing_window": smoothing_selection.window_epochs,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    return RollingLogisticExperimentOutputs(
        run_dir=run_root,
        config_path=config_path,
        selected_features_path=selected_features_path,
        feature_metadata_path=feature_metadata_path,
        dropped_features_path=dropped_features_path,
        correlation_edges_path=correlation_edges_path,
        cv_results_path=cv_results_path,
        best_params_path=best_params_path,
        train_oof_predictions_path=train_oof_predictions_path,
        metrics_path=metrics_path,
        raw_predictions_path=raw_predictions_path,
        raw_threshold_predictions_path=raw_threshold_predictions_path,
        calibrated_predictions_path=calibrated_predictions_path,
        decoded_predictions_path=decoded_predictions_path,
        threshold_predictions_path=threshold_predictions_path,
        smoothed_predictions_path=smoothed_predictions_path,
        smoothed_threshold_predictions_path=smoothed_threshold_predictions_path,
        raw_confusion_path=raw_confusion_path,
        raw_threshold_confusion_path=raw_threshold_confusion_path,
        calibrated_confusion_path=calibrated_confusion_path,
        decoded_confusion_path=decoded_confusion_path,
        threshold_confusion_path=threshold_confusion_path,
        smoothed_confusion_path=smoothed_confusion_path,
        smoothed_threshold_confusion_path=smoothed_threshold_confusion_path,
        threshold_tuning_path=threshold_tuning_path,
        raw_threshold_tuning_path=raw_threshold_tuning_path,
        smoothing_tuning_path=smoothing_tuning_path,
        smoothed_threshold_tuning_path=smoothed_threshold_tuning_path,
        model_path=model_path,
        calibrator_path=calibrator_path,
        raw_threshold_rule_path=raw_threshold_rule_path,
        threshold_rule_path=threshold_rule_path,
        smoothed_threshold_rule_path=smoothed_threshold_rule_path,
        transition_model_path=transition_model_path,
    )


def _fit_logistic_grid(
    train_features: pd.DataFrame,
    selected_features: list[str],
    *,
    random_state: int,
    cv_splits: int,
    n_jobs: int | None,
    verbose_search: int,
    param_grid: dict[str, list[Any]],
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    estimator = elastic_net_logistic_regression(random_state=random_state).estimator
    cv = _group_kfold(train_features["participant_id"], cv_splits)
    search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        scoring=MACRO_F1_SCORER,
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=verbose_search,
    )
    search.fit(
        train_features[selected_features],
        train_features["label"],
        groups=train_features["participant_id"],
    )
    cv_results = _cv_results_frame(search)
    return search.best_estimator_, cv_results, search.best_params_


def _out_of_fold_raw_probabilities(
    train_features: pd.DataFrame,
    selected_features: list[str],
    best_params: dict[str, Any],
    *,
    random_state: int,
    cv_splits: int,
) -> np.ndarray:
    """Return train-set probabilities from participant-held-out fold models."""
    estimator = elastic_net_logistic_regression(random_state=random_state).estimator
    estimator.set_params(**best_params)
    cv = _group_kfold(train_features["participant_id"], cv_splits)
    probabilities = np.zeros((len(train_features), len(TARGET_LABELS)), dtype=float)
    X = train_features[selected_features]
    y = train_features["label"]
    groups = train_features["participant_id"]

    for train_indices, holdout_indices in cv.split(X, y, groups):
        fold_estimator = clone(estimator)
        fold_estimator.fit(X.iloc[train_indices], y.iloc[train_indices])
        probabilities[holdout_indices] = align_probabilities(
            fold_estimator.predict_proba(X.iloc[holdout_indices]),
            fold_estimator.classes_,
            TARGET_LABELS,
        )
    return probabilities


def _cv_results_frame(search: GridSearchCV) -> pd.DataFrame:
    columns = [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_train_score",
        "std_train_score",
    ]
    frame = pd.DataFrame(search.cv_results_)
    frame["params"] = frame["params"].map(repr)
    return frame[columns].sort_values("rank_test_score")


def _write_comparison_metrics(
    prediction_sets: list[tuple[str, pd.DataFrame, Path]], metrics_path: Path
) -> None:
    rows: list[dict[str, Any]] = []
    for model_name, predictions, confusion_path in prediction_sets:
        metrics = classification_metrics(
            predictions["true_label"], predictions["pred_label"], labels=TARGET_LABELS
        )
        rows.append(
            {
                "model": model_name,
                "split": predictions["split"].iloc[0],
                "n_epochs": len(predictions),
                **metrics,
            }
        )
        confusion_matrix_frame(
            predictions["true_label"], predictions["pred_label"], labels=TARGET_LABELS
        ).to_csv(confusion_path)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)


def _write_selected_features(features: list[str], path: Path) -> None:
    pd.DataFrame(
        [
            {"feature": feature, "selection_order": index}
            for index, feature in enumerate(features)
        ]
    ).to_csv(path, index=False)


def _load_split_features(path: str | Path, *, expected_split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"participant_id": str})
    missing = sorted(set(FEATURE_ID_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    observed_splits = set(frame["split"])
    if observed_splits != {expected_split}:
        raise ValueError(
            f"{path} must contain only split={expected_split!r}; "
            f"observed split value(s): {sorted(observed_splits)}"
        )
    return frame


def _create_run_root(output_dir: str | Path, run_id: str | None) -> Path:
    root = Path(output_dir)
    resolved_run_id = run_id or f"rolling_logistic_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    run_root = root / "runs" / resolved_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root
