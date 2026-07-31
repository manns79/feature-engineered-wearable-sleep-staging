"""Locked held-out test evaluation for finalized model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

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
from src.models.sequence_postprocessing import (
    DEFAULT_SMOOTHING_WINDOWS,
    smooth_probabilities_by_participant,
    tune_smoothing_window,
)
from src.models.train_baselines import balanced_sample_weights
from src.models.training import _group_kfold


@dataclass(frozen=True)
class FinalPriorModelSpec:
    """Frozen prior ablation model selected for final evaluation."""

    candidate: str
    ablation: str
    model: str
    apply_postprocessing: bool = False


DEFAULT_FINAL_PRIOR_MODEL_SPECS = (
    FinalPriorModelSpec(
        candidate="best_original_ablation",
        ablation="basic_signal_specific_rolling_subject_norm",
        model="elastic_net_logistic_regression",
        apply_postprocessing=True,
    ),
    FinalPriorModelSpec(
        candidate="statistical_summary_only",
        ablation="basic_statistical",
        model="elastic_net_logistic_regression",
        apply_postprocessing=True,
    ),
    FinalPriorModelSpec(
        candidate="majority_class_baseline",
        ablation="basic_statistical",
        model="majority_class",
    ),
    FinalPriorModelSpec(
        candidate="stratified_class_baseline",
        ablation="basic_statistical",
        model="stratified_random",
    ),
)


@dataclass(frozen=True)
class LockedTestEvaluationOutputs:
    """Paths written by the locked test evaluation."""

    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    confusion_path: Path
    validation_metrics_path: Path | None
    validation_predictions_path: Path | None
    validation_confusion_path: Path | None
    comparison_path: Path | None
    protocol_path: Path


@dataclass(frozen=True)
class PriorModelRun:
    """Resolved ablation model artifact plus its selected features."""

    spec: FinalPriorModelSpec
    feature_set: str
    model_path: Path
    selected_features: tuple[str, ...]


@dataclass(frozen=True)
class PriorPostprocessingArtifacts:
    """Train-only post-processing rules for one prior model."""

    calibrator: OneVsRestPlattCalibrator
    raw_threshold_rule: Any
    threshold_rule: Any
    smoothing_window: int
    smoothed_threshold_rule: Any


def run_locked_test_evaluation(
    *,
    rolling_run_dir: str | Path,
    train_features_path: str | Path = "data/processed/features_train.csv",
    validation_features_path: str | Path | None = "data/processed/features_val.csv",
    test_features_path: str | Path = "data/processed/features_test.csv",
    prior_ablation_run_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    prior_model_specs: tuple[
        FinalPriorModelSpec, ...
    ] = DEFAULT_FINAL_PRIOR_MODEL_SPECS,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
    smoothing_windows: tuple[int, ...] = DEFAULT_SMOOTHING_WINDOWS,
) -> LockedTestEvaluationOutputs:
    """Evaluate finalized models on the held-out test split exactly once."""
    rolling_root = Path(rolling_run_dir)
    test_features = _load_split_features(test_features_path, expected_split="test")
    validation_features = (
        _load_split_features(validation_features_path, expected_split="validation")
        if validation_features_path is not None
        else None
    )
    evaluation_dir = (
        Path(output_dir) if output_dir is not None else rolling_root / "locked_test"
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    prior_runs: list[PriorModelRun] = []
    prior_postprocessors: dict[tuple[str, str, str], PriorPostprocessingArtifacts] = {}
    if prior_ablation_run_dir is not None:
        prior_runs = _resolve_prior_model_runs(
            Path(prior_ablation_run_dir),
            prior_model_specs=prior_model_specs,
        )
        postprocessed_runs = [
            run for run in prior_runs if run.spec.apply_postprocessing
        ]
        if postprocessed_runs:
            # Calibrators, smoothing windows, and thresholds are fit from
            # train-OOF evidence only; the held-out test split is read later.
            train_features = _load_split_features(
                train_features_path, expected_split="train"
            )
            prior_postprocessors = _fit_prior_postprocessors(
                postprocessed_runs,
                train_features,
                output_dir=evaluation_dir / "postprocessing",
                threshold_grid=threshold_grid,
                smoothing_windows=smoothing_windows,
            )

    prediction_frames = _prediction_frames_for_split(
        rolling_root=rolling_root,
        split_features=test_features,
        prior_runs=prior_runs,
        prior_postprocessors=prior_postprocessors,
    )
    validation_prediction_frames = (
        _prediction_frames_for_split(
            rolling_root=rolling_root,
            split_features=validation_features,
            prior_runs=prior_runs,
            prior_postprocessors=prior_postprocessors,
        )
        if validation_features is not None
        else []
    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = _metrics_frame(predictions)
    confusion = _confusion_frame(predictions)
    validation_predictions = (
        pd.concat(validation_prediction_frames, ignore_index=True)
        if validation_prediction_frames
        else pd.DataFrame()
    )
    validation_metrics = (
        _metrics_frame(validation_predictions)
        if not validation_predictions.empty
        else pd.DataFrame()
    )
    validation_confusion = (
        _confusion_frame(validation_predictions)
        if not validation_predictions.empty
        else pd.DataFrame()
    )
    comparison = (
        _comparison_frame(validation_metrics, metrics)
        if not validation_metrics.empty
        else pd.DataFrame()
    )

    metrics_path = evaluation_dir / "locked_test_metrics.csv"
    predictions_path = evaluation_dir / "locked_test_predictions.csv"
    confusion_path = evaluation_dir / "locked_test_confusion.csv"
    validation_metrics_path = (
        evaluation_dir / "final_validation_metrics.csv"
        if not validation_metrics.empty
        else None
    )
    validation_predictions_path = (
        evaluation_dir / "final_validation_predictions.csv"
        if not validation_predictions.empty
        else None
    )
    validation_confusion_path = (
        evaluation_dir / "final_validation_confusion.csv"
        if not validation_confusion.empty
        else None
    )
    comparison_path = (
        evaluation_dir / "final_comparison_table.csv"
        if not comparison.empty
        else None
    )
    protocol_path = evaluation_dir / "final_test_protocol.json"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    if validation_metrics_path is not None:
        validation_metrics.to_csv(validation_metrics_path, index=False)
    if validation_predictions_path is not None:
        validation_predictions.to_csv(validation_predictions_path, index=False)
    if validation_confusion_path is not None:
        validation_confusion.to_csv(validation_confusion_path, index=False)
    if comparison_path is not None:
        comparison.to_csv(comparison_path, index=False)
    _write_protocol(
        protocol_path,
        rolling_run_dir=rolling_root,
        train_features_path=train_features_path,
        validation_features_path=validation_features_path,
        test_features_path=test_features_path,
        prior_ablation_run_dir=prior_ablation_run_dir,
        prior_model_specs=prior_model_specs,
        threshold_grid=threshold_grid,
        smoothing_windows=smoothing_windows,
    )

    return LockedTestEvaluationOutputs(
        output_dir=evaluation_dir,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        confusion_path=confusion_path,
        validation_metrics_path=validation_metrics_path,
        validation_predictions_path=validation_predictions_path,
        validation_confusion_path=validation_confusion_path,
        comparison_path=comparison_path,
        protocol_path=protocol_path,
    )


def _prediction_frames_for_split(
    *,
    rolling_root: Path,
    split_features: pd.DataFrame,
    prior_runs: list[PriorModelRun],
    prior_postprocessors: dict[tuple[str, str, str], PriorPostprocessingArtifacts],
) -> list[pd.DataFrame]:
    frames = _rolling_logistic_predictions(rolling_root, split_features)
    frames.extend(
        _prior_ablation_predictions(
            prior_runs,
            split_features,
            prior_postprocessors=prior_postprocessors,
        )
    )
    return frames


def _rolling_logistic_predictions(
    rolling_root: Path, split_features: pd.DataFrame
) -> list[pd.DataFrame]:
    metrics_dir = rolling_root / "metrics"
    models_dir = rolling_root / "models"
    selected_features = pd.read_csv(metrics_dir / "selected_features.csv")[
        "feature"
    ].tolist()
    fitted = joblib.load(models_dir / "rolling_logistic_model.joblib")
    calibrator = joblib.load(models_dir / "platt_calibrator.joblib")
    raw_threshold_rule = joblib.load(models_dir / "raw_threshold_rule.joblib")
    threshold_rule = joblib.load(models_dir / "threshold_rule.joblib")
    smoothed_threshold_rule = joblib.load(models_dir / "smoothed_threshold_rule.joblib")
    run_config = json.loads((rolling_root / "run_config.json").read_text())

    raw_probabilities = _predict_aligned_probabilities(
        fitted, split_features[selected_features]
    )
    raw = probability_frame(split_features, raw_probabilities)
    _annotate_predictions(
        raw,
        candidate="interpretable_rolling_logistic",
        ablation="rolling_context_corr_pruned",
        feature_set="rolling_context_corr_pruned",
        base_model="elastic_net_logistic_regression",
        model="logistic_raw",
        variant="raw",
    )

    raw_threshold = threshold_prediction_frame(
        split_features, raw_probabilities, raw_threshold_rule
    )
    _annotate_predictions(
        raw_threshold,
        candidate="interpretable_rolling_logistic",
        ablation="rolling_context_corr_pruned",
        feature_set="rolling_context_corr_pruned",
        base_model="elastic_net_logistic_regression",
        model="logistic_threshold_tuned",
        variant="raw_threshold_tuned",
    )

    calibrated_probabilities = calibrator.predict_proba(raw_probabilities)
    calibrated = probability_frame(split_features, calibrated_probabilities)
    _annotate_predictions(
        calibrated,
        candidate="interpretable_rolling_logistic",
        ablation="rolling_context_corr_pruned",
        feature_set="rolling_context_corr_pruned",
        base_model="elastic_net_logistic_regression",
        model="logistic_platt",
        variant="platt",
    )

    threshold = threshold_prediction_frame(
        split_features, calibrated_probabilities, threshold_rule
    )
    _annotate_predictions(
        threshold,
        candidate="interpretable_rolling_logistic",
        ablation="rolling_context_corr_pruned",
        feature_set="rolling_context_corr_pruned",
        base_model="elastic_net_logistic_regression",
        model="logistic_platt_threshold_tuned",
        variant="platt_threshold_tuned",
    )

    smoothed_probabilities = smooth_probabilities_by_participant(
        split_features,
        calibrated_probabilities,
        window_epochs=int(run_config["selected_smoothing_window"]),
    )
    smoothed = probability_frame(split_features, smoothed_probabilities)
    _annotate_predictions(
        smoothed,
        candidate="interpretable_rolling_logistic",
        ablation="rolling_context_corr_pruned",
        feature_set="rolling_context_corr_pruned",
        base_model="elastic_net_logistic_regression",
        model="logistic_platt_smoothed",
        variant="platt_smoothed",
    )

    smoothed_threshold = threshold_prediction_frame(
        split_features, smoothed_probabilities, smoothed_threshold_rule
    )
    _annotate_predictions(
        smoothed_threshold,
        candidate="interpretable_rolling_logistic",
        ablation="rolling_context_corr_pruned",
        feature_set="rolling_context_corr_pruned",
        base_model="elastic_net_logistic_regression",
        model="logistic_platt_smoothed_threshold_tuned",
        variant="platt_smoothed_threshold_tuned",
    )

    return [
        raw,
        raw_threshold,
        calibrated,
        threshold,
        smoothed,
        smoothed_threshold,
    ]


def _resolve_prior_model_runs(
    prior_root: Path,
    *,
    prior_model_specs: tuple[FinalPriorModelSpec, ...],
) -> list[PriorModelRun]:
    status_path = prior_root / "run_status.csv"
    feature_sets_path = prior_root / "metrics" / "ablation_feature_sets.csv"
    if not status_path.exists():
        raise FileNotFoundError(f"Ablation status does not exist: {status_path}")
    if not feature_sets_path.exists():
        raise FileNotFoundError(
            f"Ablation feature-set table does not exist: {feature_sets_path}"
        )

    status = pd.read_csv(status_path)
    feature_sets = pd.read_csv(feature_sets_path)
    feature_lookup = dict(
        zip(feature_sets["ablation"], feature_sets["selected_features"], strict=True)
    )
    completed = status[
        (status["event"] == "model_completed") & (status["status"] == "completed")
    ].copy()

    runs: list[PriorModelRun] = []
    for spec in prior_model_specs:
        matches = completed[
            (completed["ablation"] == spec.ablation)
            & (completed["model"] == spec.model)
        ]
        if matches.empty:
            raise ValueError(
                "Final prior model artifact is missing: "
                f"{spec.ablation}:{spec.model}"
            )
        row = matches.iloc[-1]
        selected_features = str(feature_lookup[spec.ablation]).split("|")
        runs.append(
            PriorModelRun(
                spec=spec,
                feature_set=str(row.get("feature_set", f"ablation_{spec.ablation}")),
                model_path=Path(str(row["model_path"])),
                selected_features=tuple(selected_features),
            )
        )
    return runs


def _prior_ablation_predictions(
    prior_runs: list[PriorModelRun],
    split_features: pd.DataFrame,
    *,
    prior_postprocessors: dict[tuple[str, str, str], PriorPostprocessingArtifacts],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for run in prior_runs:
        fitted = joblib.load(run.model_path)
        raw_probabilities = _predict_aligned_probabilities(
            fitted, split_features[list(run.selected_features)]
        )
        raw = probability_frame(split_features, raw_probabilities)
        _annotate_predictions(
            raw,
            candidate=run.spec.candidate,
            ablation=run.spec.ablation,
            feature_set=run.feature_set,
            base_model=run.spec.model,
            model=run.spec.model,
            variant="raw",
        )
        frames.append(raw)

        artifacts = prior_postprocessors.get(_postprocessor_key(run))
        if artifacts is None:
            continue

        raw_threshold = threshold_prediction_frame(
            split_features, raw_probabilities, artifacts.raw_threshold_rule
        )
        _annotate_predictions(
            raw_threshold,
            candidate=run.spec.candidate,
            ablation=run.spec.ablation,
            feature_set=run.feature_set,
            base_model=run.spec.model,
            model=f"{run.spec.model}_threshold_tuned",
            variant="raw_threshold_tuned",
        )
        frames.append(raw_threshold)

        calibrated_probabilities = artifacts.calibrator.predict_proba(raw_probabilities)
        calibrated = probability_frame(split_features, calibrated_probabilities)
        _annotate_predictions(
            calibrated,
            candidate=run.spec.candidate,
            ablation=run.spec.ablation,
            feature_set=run.feature_set,
            base_model=run.spec.model,
            model=f"{run.spec.model}_platt",
            variant="platt",
        )
        frames.append(calibrated)

        threshold = threshold_prediction_frame(
            split_features, calibrated_probabilities, artifacts.threshold_rule
        )
        _annotate_predictions(
            threshold,
            candidate=run.spec.candidate,
            ablation=run.spec.ablation,
            feature_set=run.feature_set,
            base_model=run.spec.model,
            model=f"{run.spec.model}_platt_threshold_tuned",
            variant="platt_threshold_tuned",
        )
        frames.append(threshold)

        smoothed_probabilities = smooth_probabilities_by_participant(
            split_features,
            calibrated_probabilities,
            window_epochs=artifacts.smoothing_window,
        )
        smoothed = probability_frame(split_features, smoothed_probabilities)
        _annotate_predictions(
            smoothed,
            candidate=run.spec.candidate,
            ablation=run.spec.ablation,
            feature_set=run.feature_set,
            base_model=run.spec.model,
            model=f"{run.spec.model}_platt_smoothed",
            variant="platt_smoothed",
        )
        frames.append(smoothed)

        smoothed_threshold = threshold_prediction_frame(
            split_features,
            smoothed_probabilities,
            artifacts.smoothed_threshold_rule,
        )
        _annotate_predictions(
            smoothed_threshold,
            candidate=run.spec.candidate,
            ablation=run.spec.ablation,
            feature_set=run.feature_set,
            base_model=run.spec.model,
            model=f"{run.spec.model}_platt_smoothed_threshold_tuned",
            variant="platt_smoothed_threshold_tuned",
        )
        frames.append(smoothed_threshold)
    return frames


def _fit_prior_postprocessors(
    prior_runs: list[PriorModelRun],
    train_features: pd.DataFrame,
    *,
    output_dir: Path,
    threshold_grid: tuple[float, ...],
    smoothing_windows: tuple[int, ...],
) -> dict[tuple[str, str, str], PriorPostprocessingArtifacts]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[tuple[str, str, str], PriorPostprocessingArtifacts] = {}
    for run in prior_runs:
        fitted = joblib.load(run.model_path)
        run_dir = output_dir / _safe_path_name(run.spec.candidate)
        run_dir.mkdir(parents=True, exist_ok=True)
        oof_raw_probabilities = _out_of_fold_probabilities(
            fitted,
            train_features,
            selected_features=list(run.selected_features),
            model_name=run.spec.model,
        )
        train_oof_predictions = probability_frame(train_features, oof_raw_probabilities)
        train_oof_predictions.to_csv(
            run_dir / "train_oof_raw_predictions.csv", index=False
        )

        raw_threshold_tuning = tune_class_thresholds(
            oof_raw_probabilities,
            train_features["label"],
            threshold_grid=threshold_grid,
        )
        raw_threshold_tuning.results.to_csv(
            run_dir / "train_oof_raw_threshold_tuning_results.csv", index=False
        )
        calibrator = OneVsRestPlattCalibrator(classes=TARGET_LABELS).fit(
            oof_raw_probabilities, train_features["label"]
        )
        oof_calibrated_probabilities = calibrator.predict_proba(oof_raw_probabilities)
        threshold_tuning = tune_class_thresholds(
            oof_calibrated_probabilities,
            train_features["label"],
            threshold_grid=threshold_grid,
        )
        threshold_tuning.results.to_csv(
            run_dir / "train_oof_threshold_tuning_results.csv", index=False
        )
        smoothing_selection = tune_smoothing_window(
            train_features,
            oof_calibrated_probabilities,
            train_features["label"],
            windows=smoothing_windows,
        )
        smoothing_selection.results.to_csv(
            run_dir / "train_oof_smoothing_tuning_results.csv", index=False
        )
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
        smoothed_threshold_tuning.results.to_csv(
            run_dir / "train_oof_smoothed_threshold_tuning_results.csv", index=False
        )

        artifact = PriorPostprocessingArtifacts(
            calibrator=calibrator,
            raw_threshold_rule=raw_threshold_tuning.rule,
            threshold_rule=threshold_tuning.rule,
            smoothing_window=smoothing_selection.window_epochs,
            smoothed_threshold_rule=smoothed_threshold_tuning.rule,
        )
        joblib.dump(calibrator, run_dir / "platt_calibrator.joblib")
        joblib.dump(raw_threshold_tuning.rule, run_dir / "raw_threshold_rule.joblib")
        joblib.dump(threshold_tuning.rule, run_dir / "threshold_rule.joblib")
        joblib.dump(
            smoothed_threshold_tuning.rule,
            run_dir / "smoothed_threshold_rule.joblib",
        )
        pd.DataFrame(
            [
                {
                    "candidate": run.spec.candidate,
                    "ablation": run.spec.ablation,
                    "model": run.spec.model,
                    "selected_smoothing_window": smoothing_selection.window_epochs,
                    "postprocessing_tuning_source": "train_out_of_fold",
                }
            ]
        ).to_csv(run_dir / "postprocessing_metadata.csv", index=False)
        artifacts[_postprocessor_key(run)] = artifact
    return artifacts


def _out_of_fold_probabilities(
    fitted: Any,
    train_features: pd.DataFrame,
    *,
    selected_features: list[str],
    model_name: str,
) -> np.ndarray:
    cv = _group_kfold(train_features["participant_id"], requested_splits=5)
    X = train_features[selected_features]
    y = train_features["label"]
    probabilities = np.zeros((len(train_features), len(TARGET_LABELS)), dtype=float)

    for train_indices, holdout_indices in cv.split(
        X, y, train_features["participant_id"]
    ):
        if isinstance(fitted, dict) and "label_encoder" in fitted:
            base_estimator = clone(fitted["estimator"])
            encoder = LabelEncoder()
            encoded_train = encoder.fit_transform(y.iloc[train_indices])
            fit_kwargs = {}
            if model_name == "xgboost":
                fit_kwargs["sample_weight"] = balanced_sample_weights(encoded_train)
            base_estimator.fit(
                X.iloc[train_indices],
                encoded_train,
                **fit_kwargs,
            )
            probabilities[holdout_indices] = align_probabilities(
                base_estimator.predict_proba(X.iloc[holdout_indices]),
                encoder.classes_.tolist(),
                TARGET_LABELS,
            )
            continue

        fold_estimator = clone(fitted)
        fold_estimator.fit(X.iloc[train_indices], y.iloc[train_indices])
        probabilities[holdout_indices] = align_probabilities(
            fold_estimator.predict_proba(X.iloc[holdout_indices]),
            fold_estimator.classes_,
            TARGET_LABELS,
        )
    return probabilities


def _predict_aligned_probabilities(fitted: Any, features: pd.DataFrame) -> Any:
    if isinstance(fitted, dict) and "label_encoder" in fitted:
        estimator = fitted["estimator"]
        encoder = fitted["label_encoder"]
        return align_probabilities(
            estimator.predict_proba(features), encoder.classes_.tolist(), TARGET_LABELS
        )
    return align_probabilities(
        fitted.predict_proba(features), fitted.classes_, TARGET_LABELS
    )


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


def _metrics_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = _group_columns(predictions)
    for keys, group in predictions.groupby(group_columns, sort=False):
        key_values = (keys,) if len(group_columns) == 1 else keys
        rows.append(
            {
                **dict(zip(group_columns, key_values, strict=True)),
                "split": group["split"].iloc[0],
                "n_epochs": len(group),
                **classification_metrics(
                    group["true_label"], group["pred_label"], labels=TARGET_LABELS
                ),
            }
        )
    return pd.DataFrame(rows)


def _confusion_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    group_columns = _group_columns(predictions)
    for keys, group in predictions.groupby(group_columns, sort=False):
        key_values = (keys,) if len(group_columns) == 1 else keys
        metadata = dict(zip(group_columns, key_values, strict=True))
        confusion = confusion_matrix_frame(
            group["true_label"], group["pred_label"], labels=TARGET_LABELS
        )
        confusion.insert(0, "true_label", confusion.index)
        for column, value in reversed(metadata.items()):
            confusion.insert(0, column, value)
        frames.append(confusion.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def _comparison_frame(
    validation_metrics: pd.DataFrame, test_metrics: pd.DataFrame
) -> pd.DataFrame:
    merge_columns = [
        column
        for column in (
            "candidate",
            "ablation",
            "feature_set",
            "base_model",
            "model",
            "variant",
        )
        if column in test_metrics.columns and column in validation_metrics.columns
    ]
    validation = validation_metrics[
        [*merge_columns, "macro_f1"]
    ].rename(columns={"macro_f1": "validation_macro_f1"})
    test = test_metrics[
        [
            *merge_columns,
            "macro_f1",
            "Wake_f1",
            "Non_REM_f1",
            "REM_f1",
            "accuracy",
            "balanced_accuracy",
            "n_epochs",
        ]
    ].rename(
        columns={
            "macro_f1": "test_macro_f1",
            "Wake_f1": "test_Wake_f1",
            "Non_REM_f1": "test_Non_REM_f1",
            "REM_f1": "test_REM_f1",
            "accuracy": "test_accuracy",
            "balanced_accuracy": "test_balanced_accuracy",
            "n_epochs": "test_n_epochs",
        }
    )
    comparison = validation.merge(test, on=merge_columns, how="outer")
    return comparison.sort_values(
        ["candidate", "base_model", "model"], kind="stable"
    ).reset_index(drop=True)


def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in (
            "candidate",
            "ablation",
            "feature_set",
            "base_model",
            "model",
            "variant",
        )
        if column in frame.columns
    ]


def _annotate_predictions(
    frame: pd.DataFrame,
    *,
    candidate: str,
    ablation: str,
    feature_set: str,
    base_model: str,
    model: str,
    variant: str,
) -> None:
    for column, value in reversed(
        [
            ("candidate", candidate),
            ("ablation", ablation),
            ("feature_set", feature_set),
            ("base_model", base_model),
            ("model", model),
            ("variant", variant),
        ]
    ):
        frame.insert(0, column, value)


def _postprocessor_key(run: PriorModelRun) -> tuple[str, str, str]:
    return (run.spec.candidate, run.spec.ablation, run.spec.model)


def _safe_path_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _write_protocol(
    path: Path,
    *,
    rolling_run_dir: Path,
    train_features_path: str | Path,
    validation_features_path: str | Path | None,
    test_features_path: str | Path,
    prior_ablation_run_dir: str | Path | None,
    prior_model_specs: tuple[FinalPriorModelSpec, ...],
    threshold_grid: tuple[float, ...],
    smoothing_windows: tuple[int, ...],
) -> None:
    protocol = {
        "purpose": "Final locked held-out test evaluation.",
        "test_set_rule": (
            "The held-out test split is used only for final evaluation after the "
            "model roster and post-processing protocol are frozen."
        ),
        "postprocessing_rule": (
            "Platt calibration, class thresholds, and smoothing windows are "
            "learned from participant-held-out training OOF predictions only; "
            "validation and test labels are not used to fit post-processing."
        ),
        "baseline_rule": (
            "Majority-class and stratified-class dummy baselines are reported "
            "raw only so they remain sanity checks."
        ),
        "rolling_run_dir": str(rolling_run_dir),
        "train_features_path": str(train_features_path),
        "validation_features_path": (
            str(validation_features_path)
            if validation_features_path is not None
            else None
        ),
        "test_features_path": str(test_features_path),
        "prior_ablation_run_dir": (
            str(prior_ablation_run_dir)
            if prior_ablation_run_dir is not None
            else None
        ),
        "prior_model_specs": [
            {
                "candidate": spec.candidate,
                "ablation": spec.ablation,
                "model": spec.model,
                "apply_postprocessing": spec.apply_postprocessing,
            }
            for spec in prior_model_specs
        ],
        "threshold_grid": list(threshold_grid),
        "smoothing_windows": list(smoothing_windows),
        "reported_table": (
            "final_comparison_table.csv contains validation macro F1, test "
            "macro F1, and test Wake/Non-REM/REM F1."
        ),
    }
    path.write_text(json.dumps(protocol, indent=2) + "\n")
