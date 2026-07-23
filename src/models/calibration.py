"""Probability calibration helpers for multiclass sleep-stage probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import TARGET_LABELS


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
