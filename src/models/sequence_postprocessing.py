"""Sequence-level post-processing for sleep-stage predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import TARGET_LABELS
from src.models.calibration import probability_columns


@dataclass(frozen=True)
class TransitionModel:
    """Initial and transition probabilities estimated from labeled sequences."""

    classes: tuple[str, ...]
    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray


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
