"""Whole-night normalization and temporal context features."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def add_participant_zscores(
    features: pd.DataFrame,
    feature_columns: Iterable[str],
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """Add whole-night participant-normalized z-score features."""
    output = features.copy()
    for column in feature_columns:
        grouped = output.groupby(participant_col)[column]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, pd.NA)
        output[f"{column}_participant_z"] = (output[column] - mean) / std
    return output


def add_centered_rolling_means(
    features: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    window_epochs: int,
    participant_col: str = "participant_id",
    epoch_col: str = "epoch_id",
) -> pd.DataFrame:
    """Add retrospective centered rolling means within each participant night."""
    if window_epochs <= 0:
        raise ValueError("window_epochs must be positive.")
    output = features.sort_values([participant_col, epoch_col]).copy()
    for column in feature_columns:
        output[f"{column}_roll{window_epochs}_mean"] = output.groupby(participant_col)[
            column
        ].transform(
            lambda values: values.rolling(
                window=window_epochs,
                min_periods=1,
                center=True,
            ).mean()
        )
    return output
