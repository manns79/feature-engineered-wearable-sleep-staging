"""Locked held-out test evaluation for finalized model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import TARGET_LABELS
from src.features.build_features import FEATURE_ID_COLUMNS
from src.models.calibration import align_probabilities, probability_frame
from src.models.evaluate import classification_metrics, confusion_matrix_frame
from src.models.sequence_postprocessing import apply_viterbi_by_participant

DEFAULT_PRIOR_MODELS = (
    "elastic_net_logistic_regression",
    "random_forest",
    "xgboost",
)


@dataclass(frozen=True)
class LockedTestEvaluationOutputs:
    """Paths written by the locked test evaluation."""

    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    confusion_path: Path


def run_locked_test_evaluation(
    *,
    rolling_run_dir: str | Path,
    test_features_path: str | Path = "data/processed/features_test.csv",
    prior_ablation_run_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    prior_models: tuple[str, ...] = DEFAULT_PRIOR_MODELS,
) -> LockedTestEvaluationOutputs:
    """Evaluate finalized models on the held-out test split exactly once."""
    rolling_root = Path(rolling_run_dir)
    test_features = _load_test_features(test_features_path)
    evaluation_dir = (
        Path(output_dir) if output_dir is not None else rolling_root / "locked_test"
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = _rolling_logistic_test_predictions(rolling_root, test_features)
    if prior_ablation_run_dir is not None:
        prediction_frames.extend(
            _prior_ablation_test_predictions(
                Path(prior_ablation_run_dir),
                test_features,
                prior_models=prior_models,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = _metrics_frame(predictions)
    confusion = _confusion_frame(predictions)

    metrics_path = evaluation_dir / "locked_test_metrics.csv"
    predictions_path = evaluation_dir / "locked_test_predictions.csv"
    confusion_path = evaluation_dir / "locked_test_confusion.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    return LockedTestEvaluationOutputs(
        output_dir=evaluation_dir,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        confusion_path=confusion_path,
    )


def _rolling_logistic_test_predictions(
    rolling_root: Path, test_features: pd.DataFrame
) -> list[pd.DataFrame]:
    metrics_dir = rolling_root / "metrics"
    models_dir = rolling_root / "models"
    selected_features = pd.read_csv(metrics_dir / "selected_features.csv")[
        "feature"
    ].tolist()
    fitted = joblib.load(models_dir / "rolling_logistic_model.joblib")
    calibrator = joblib.load(models_dir / "platt_calibrator.joblib")
    transition_model = joblib.load(models_dir / "transition_model.joblib")

    raw_probabilities = _predict_aligned_probabilities(
        fitted, test_features[selected_features]
    )
    raw = probability_frame(test_features, raw_probabilities)
    raw.insert(0, "model", "logistic_raw")
    raw.insert(0, "ablation", "rolling_context_corr_pruned")

    calibrated_probabilities = calibrator.predict_proba(raw_probabilities)
    calibrated = probability_frame(test_features, calibrated_probabilities)
    calibrated.insert(0, "model", "logistic_platt")
    calibrated.insert(0, "ablation", "rolling_context_corr_pruned")

    decoded = apply_viterbi_by_participant(calibrated, transition_model)
    decoded["model"] = "logistic_platt_viterbi"
    decoded["ablation"] = "rolling_context_corr_pruned"
    return [raw, calibrated, decoded]


def _prior_ablation_test_predictions(
    prior_root: Path,
    test_features: pd.DataFrame,
    *,
    prior_models: tuple[str, ...],
) -> list[pd.DataFrame]:
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
        (status["event"] == "model_completed")
        & (status["status"] == "completed")
        & (status["model"].isin(prior_models))
    ].copy()

    frames: list[pd.DataFrame] = []
    for row in completed.itertuples(index=False):
        selected_features = str(feature_lookup[row.ablation]).split("|")
        fitted = joblib.load(row.model_path)
        probabilities = _predict_aligned_probabilities(
            fitted, test_features[selected_features]
        )
        predictions = probability_frame(test_features, probabilities)
        predictions.insert(0, "model", row.model)
        predictions.insert(0, "ablation", row.ablation)
        frames.append(predictions)
    return frames


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


def _load_test_features(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"participant_id": str})
    missing = sorted(set(FEATURE_ID_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    observed_splits = set(frame["split"])
    if observed_splits != {"test"}:
        raise ValueError(
            f"{path} must contain only split='test'; "
            f"observed split value(s): {sorted(observed_splits)}"
        )
    return frame


def _metrics_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["ablation", "model"], sort=False):
        ablation, model = keys
        rows.append(
            {
                "ablation": ablation,
                "model": model,
                "split": "test",
                "n_epochs": len(group),
                **classification_metrics(
                    group["true_label"], group["pred_label"], labels=TARGET_LABELS
                ),
            }
        )
    return pd.DataFrame(rows)


def _confusion_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for keys, group in predictions.groupby(["ablation", "model"], sort=False):
        ablation, model = keys
        confusion = confusion_matrix_frame(
            group["true_label"], group["pred_label"], labels=TARGET_LABELS
        )
        confusion.insert(0, "true_label", confusion.index)
        confusion.insert(0, "model", model)
        confusion.insert(0, "ablation", ablation)
        frames.append(confusion.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)
