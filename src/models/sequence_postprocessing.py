"""Sequence-level post-processing for sleep-stage predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import TARGET_LABELS
from src.models.calibration import probability_columns, probability_frame

DEFAULT_SMOOTHING_WINDOWS = (3, 5, 9, 15)


@dataclass(frozen=True)
class TransitionModel:
    """Initial and transition probabilities estimated from labeled sequences."""

    classes: tuple[str, ...]
    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray


@dataclass(frozen=True)
class SmoothingSelectionResult:
    """Best smoothing window and validation tuning trace."""

    window_epochs: int
    macro_f1: float
    results: pd.DataFrame


def estimate_transition_model(
    labels: pd.DataFrame,
    *,
    classes: tuple[str, ...] = TARGET_LABELS,
    smoothing: float = 1.0,
) -> TransitionModel:
    """Estimate participant-contained initial and transition probabilities."""
    required = {"participant_id", "epoch_id", "label"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels is missing column(s): {missing}")
    if smoothing <= 0:
        raise ValueError("smoothing must be greater than 0.")

    class_to_index = {label: index for index, label in enumerate(classes)}
    initial_counts = np.full(len(classes), smoothing, dtype=float)
    transition_counts = np.full((len(classes), len(classes)), smoothing, dtype=float)

    for _, participant in labels.groupby("participant_id", sort=False):
        ordered = participant.sort_values("epoch_id")
        sequence = [
            label for label in ordered["label"].tolist() if label in class_to_index
        ]
        if not sequence:
            continue
        initial_counts[class_to_index[sequence[0]]] += 1
        for previous, current in zip(sequence, sequence[1:], strict=False):
            transition_counts[class_to_index[previous], class_to_index[current]] += 1

    initial_probabilities = initial_counts / initial_counts.sum()
    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    return TransitionModel(
        classes=classes,
        initial_probabilities=initial_probabilities,
        transition_matrix=transition_matrix,
    )


def smooth_probabilities_by_participant(
    frame: pd.DataFrame,
    probabilities: Any,
    *,
    window_epochs: int,
    classes: tuple[str, ...] = TARGET_LABELS,
    center: bool = True,
) -> np.ndarray:
    """Smooth probabilities within each participant using a rolling mean."""
    if window_epochs < 1:
        raise ValueError("window_epochs must be at least 1.")
    required = {"participant_id", "epoch_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frame is missing column(s): {missing}")

    columns = probability_columns(classes)
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(classes):
        raise ValueError("probabilities must have one column per class.")
    working = frame[["participant_id", "epoch_id"]].copy()
    for index, column in enumerate(columns):
        working[column] = probs[:, index]
    working["_original_order"] = np.arange(len(working))
    working = working.sort_values(["participant_id", "epoch_id"])
    smoothed_parts: list[pd.DataFrame] = []
    for _, participant in working.groupby("participant_id", sort=False):
        smoothed = participant[columns].rolling(
            window=window_epochs,
            min_periods=1,
            center=center,
        ).mean()
        smoothed["_original_order"] = participant["_original_order"].to_numpy()
        smoothed_parts.append(smoothed)
    output = pd.concat(smoothed_parts, ignore_index=True)
    output = output.sort_values("_original_order")
    smoothed_array = output[columns].to_numpy(dtype=float)
    row_sums = smoothed_array.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] == 0
    if zero_rows.any():
        smoothed_array[zero_rows, :] = 1.0 / len(classes)
        row_sums = smoothed_array.sum(axis=1, keepdims=True)
    return smoothed_array / row_sums


def tune_smoothing_window(
    frame: pd.DataFrame,
    probabilities: Any,
    labels: Any,
    *,
    windows: tuple[int, ...] = DEFAULT_SMOOTHING_WINDOWS,
    classes: tuple[str, ...] = TARGET_LABELS,
) -> SmoothingSelectionResult:
    """Select the smoothing window that maximizes validation macro F1."""
    if not windows:
        raise ValueError("windows must contain at least one value.")
    rows: list[dict[str, float | int]] = []
    best_window: int | None = None
    best_score = -1.0
    labels_array = np.asarray(labels)
    for window in windows:
        smoothed = smooth_probabilities_by_participant(
            frame, probabilities, window_epochs=window, classes=classes
        )
        predictions = probability_frame(frame, smoothed, classes=classes)["pred_label"]
        macro_f1 = float(
            f1_score(
                labels_array,
                predictions,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        )
        rows.append({"window_epochs": window, "macro_f1": macro_f1})
        if macro_f1 > best_score:
            best_score = macro_f1
            best_window = window
    if best_window is None:
        raise ValueError("No smoothing windows were evaluated.")
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    return SmoothingSelectionResult(
        window_epochs=best_window,
        macro_f1=best_score,
        results=results.reset_index(drop=True),
    )


def viterbi_decode(
    probabilities: Any,
    transition_model: TransitionModel,
    *,
    epsilon: float = 1e-12,
) -> list[str]:
    """Return the most likely state path for one sequence of probabilities."""
    emissions = np.asarray(probabilities, dtype=float)
    if emissions.ndim != 2 or emissions.shape[1] != len(transition_model.classes):
        raise ValueError(
            "probabilities must have one column per transition-model class."
        )
    if emissions.shape[0] == 0:
        return []

    log_emissions = np.log(np.clip(emissions, epsilon, 1.0))
    log_initial = np.log(np.clip(transition_model.initial_probabilities, epsilon, 1.0))
    log_transition = np.log(np.clip(transition_model.transition_matrix, epsilon, 1.0))

    n_epochs, n_classes = log_emissions.shape
    scores = np.empty((n_epochs, n_classes), dtype=float)
    backpointers = np.zeros((n_epochs, n_classes), dtype=int)
    scores[0] = log_initial + log_emissions[0]

    for epoch_index in range(1, n_epochs):
        previous_scores = scores[epoch_index - 1][:, np.newaxis] + log_transition
        backpointers[epoch_index] = previous_scores.argmax(axis=0)
        scores[epoch_index] = previous_scores.max(axis=0) + log_emissions[epoch_index]

    path = np.empty(n_epochs, dtype=int)
    path[-1] = int(scores[-1].argmax())
    for epoch_index in range(n_epochs - 1, 0, -1):
        path[epoch_index - 1] = backpointers[epoch_index, path[epoch_index]]
    return [transition_model.classes[index] for index in path]


def apply_viterbi_by_participant(
    predictions: pd.DataFrame,
    transition_model: TransitionModel,
    *,
    output_column: str = "pred_label",
) -> pd.DataFrame:
    """Apply Viterbi decoding independently to each participant sequence."""
    required = {
        "participant_id",
        "epoch_id",
        *probability_columns(transition_model.classes),
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions is missing column(s): {missing}")

    output = predictions.sort_values(["participant_id", "epoch_id"]).copy()
    output[output_column] = ""
    columns = probability_columns(transition_model.classes)
    for _, participant in output.groupby("participant_id", sort=False):
        decoded = viterbi_decode(participant[columns].to_numpy(), transition_model)
        output.loc[participant.index, output_column] = decoded
    return output
