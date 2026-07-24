"""Probability calibration helpers for multiclass sleep-stage probabilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import TARGET_LABELS

DEFAULT_THRESHOLD_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
DEFAULT_THRESHOLD_GRID = (*DEFAULT_THRESHOLD_GRID, 0.80, 0.90)


@dataclass
class OneVsRestPlattCalibrator:
    """Fit one sigmoid per class and renormalize calibrated probabilities."""

    classes: tuple[str, ...] = TARGET_LABELS
    epsilon: float = 1e-6

    def fit(self, probabilities: Any, labels: Any) -> OneVsRestPlattCalibrator:
        """Fit sigmoid calibrators from raw probabilities and true labels."""
        probs = _as_probability_array(probabilities, self.classes)
        labels_array = np.asarray(labels)
        calibrators: dict[str, LogisticRegression] = {}
        priors: dict[str, float] = {}
        for index, label in enumerate(self.classes):
            binary = (labels_array == label).astype(int)
            priors[label] = float(binary.mean())
            if binary.min() == binary.max():
                continue
            calibrator = LogisticRegression(solver="lbfgs")
            calibrator.fit(_logit(probs[:, index], self.epsilon).reshape(-1, 1), binary)
            calibrators[label] = calibrator
        self.calibrators_ = calibrators
        self.priors_ = priors
        return self

    def predict_proba(self, probabilities: Any) -> np.ndarray:
        """Return calibrated class probabilities in ``self.classes`` order."""
        _check_is_fitted(self)
        probs = _as_probability_array(probabilities, self.classes)
        calibrated = np.zeros_like(probs, dtype=float)
        for index, label in enumerate(self.classes):
            if label in self.calibrators_:
                calibrated[:, index] = self.calibrators_[label].predict_proba(
                    _logit(probs[:, index], self.epsilon).reshape(-1, 1)
                )[:, 1]
            else:
                calibrated[:, index] = self.priors_[label]
        row_sums = calibrated.sum(axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] == 0
        if zero_rows.any():
            calibrated[zero_rows, :] = 1.0 / len(self.classes)
            row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / row_sums


@dataclass(frozen=True)
class ClassThresholdRule:
    """Class-specific thresholds used as a multiclass decision rule."""

    classes: tuple[str, ...]
    thresholds: tuple[float, ...]
    macro_f1: float

    def predict(self, probabilities: Any) -> list[str]:
        """Predict labels from probabilities divided by class thresholds."""
        probs = _as_probability_array(probabilities, self.classes)
        thresholds = np.asarray(self.thresholds, dtype=float)
        if np.any(thresholds <= 0):
            raise ValueError("thresholds must all be greater than 0.")
        adjusted = probs / thresholds[np.newaxis, :]
        return [self.classes[index] for index in adjusted.argmax(axis=1)]

    def metadata(self) -> dict[str, float | str]:
        """Return a flat metadata row for CSV output."""
        row: dict[str, float | str] = {"macro_f1": self.macro_f1}
        for label, threshold in zip(self.classes, self.thresholds, strict=True):
            row[f"threshold_{_safe_label(label)}"] = threshold
        return row


@dataclass(frozen=True)
class ThresholdTuningResult:
    """Best threshold rule and full validation tuning trace."""

    rule: ClassThresholdRule
    results: pd.DataFrame


def tune_class_thresholds(
    probabilities: Any,
    labels: Any,
    *,
    classes: tuple[str, ...] = TARGET_LABELS,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> ThresholdTuningResult:
    """Tune class-specific thresholds to maximize validation macro F1."""
    probs = _as_probability_array(probabilities, classes)
    labels_array = np.asarray(labels)
    grid = tuple(float(value) for value in threshold_grid)
    if not grid:
        raise ValueError("threshold_grid must contain at least one value.")
    if any(value <= 0 for value in grid):
        raise ValueError("threshold_grid values must all be greater than 0.")

    rows: list[dict[str, float]] = []
    best_rule: ClassThresholdRule | None = None
    for thresholds in product(grid, repeat=len(classes)):
        rule = ClassThresholdRule(classes=classes, thresholds=thresholds, macro_f1=0.0)
        predictions = rule.predict(probs)
        macro_f1 = float(
            f1_score(
                labels_array,
                predictions,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        )
        row = {
            f"threshold_{_safe_label(label)}": threshold
            for label, threshold in zip(classes, thresholds, strict=True)
        }
        row["macro_f1"] = macro_f1
        rows.append(row)
        if best_rule is None or macro_f1 > best_rule.macro_f1:
            best_rule = ClassThresholdRule(
                classes=classes,
                thresholds=thresholds,
                macro_f1=macro_f1,
            )

    if best_rule is None:
        raise ValueError("No threshold combinations were evaluated.")
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    return ThresholdTuningResult(rule=best_rule, results=results.reset_index(drop=True))


def predicted_labels(
    probabilities: Any, classes: tuple[str, ...] = TARGET_LABELS
) -> list[str]:
    """Return maximum-probability labels for a probability matrix."""
    probs = _as_probability_array(probabilities, classes)
    return [classes[index] for index in probs.argmax(axis=1)]


def align_probabilities(
    probabilities: Any,
    source_classes: list[Any] | tuple[Any, ...],
    target_classes: tuple[str, ...] = TARGET_LABELS,
) -> np.ndarray:
    """Return probabilities reordered from source class order to target class order."""
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(source_classes):
        raise ValueError("probabilities must have one column per source class.")
    source_lookup = {str(label): index for index, label in enumerate(source_classes)}
    missing = [label for label in target_classes if label not in source_lookup]
    if missing:
        raise ValueError(f"source_classes is missing target class(es): {missing}")
    aligned = np.zeros((array.shape[0], len(target_classes)), dtype=float)
    for target_index, label in enumerate(target_classes):
        aligned[:, target_index] = array[:, source_lookup[label]]
    return aligned


def probability_frame(
    base_frame: pd.DataFrame,
    probabilities: Any,
    *,
    classes: tuple[str, ...] = TARGET_LABELS,
    pred_column: str = "pred_label",
) -> pd.DataFrame:
    """Build a prediction frame with class probability columns."""
    probs = _as_probability_array(probabilities, classes)
    output = base_frame[["participant_id", "epoch_id", "split"]].copy()
    if "label" in base_frame:
        output["true_label"] = base_frame["label"].tolist()
    output[pred_column] = predicted_labels(probs, classes)
    for index, label in enumerate(classes):
        output[f"prob_{_safe_label(label)}"] = probs[:, index]
    return output


def threshold_prediction_frame(
    base_frame: pd.DataFrame,
    probabilities: Any,
    rule: ClassThresholdRule,
) -> pd.DataFrame:
    """Build a probability frame whose predictions use a threshold rule."""
    output = probability_frame(base_frame, probabilities, classes=rule.classes)
    output["pred_label"] = rule.predict(probabilities)
    return output


def probability_columns(classes: tuple[str, ...] = TARGET_LABELS) -> list[str]:
    """Return class probability column names for ``classes``."""
    return [f"prob_{_safe_label(label)}" for label in classes]


def _as_probability_array(probabilities: Any, classes: tuple[str, ...]) -> np.ndarray:
    if isinstance(probabilities, pd.DataFrame):
        columns = probability_columns(classes)
        missing = sorted(set(columns) - set(probabilities.columns))
        if missing:
            raise ValueError(f"Probability frame is missing column(s): {missing}")
        array = probabilities[columns].to_numpy(dtype=float)
    else:
        array = np.asarray(probabilities, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(classes):
        raise ValueError(
            "probabilities must be a two-dimensional array with one column per class."
        )
    return array


def _logit(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped))


def _check_is_fitted(calibrator: OneVsRestPlattCalibrator) -> None:
    if not hasattr(calibrator, "calibrators_") or not hasattr(calibrator, "priors_"):
        raise ValueError("Calibrator has not been fitted.")


def _safe_label(label: str) -> str:
    return label.replace("-", "_").replace(" ", "_")
