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
    new_columns: dict[str, pd.Series] = {}
    for column in feature_columns:
        grouped = output.groupby(participant_col, sort=False)[column]
        mean = grouped.transform("mean")
        std = grouped.transform("std").mask(lambda values: values == 0)
        new_columns[f"{column}_participant_z"] = (output[column] - mean) / std
    if not new_columns:
        return output
    return pd.concat([output, pd.DataFrame(new_columns, index=output.index)], axis=1)


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
    new_columns: dict[str, pd.Series] = {}
    for column in feature_columns:
        new_columns[f"{column}_roll{window_epochs}_mean"] = _rolling_transform(
            output,
            column,
            window_epochs=window_epochs,
            group_cols=[participant_col],
            epoch_col=epoch_col,
            statistic="mean",
        )
    if not new_columns:
        return output
    return pd.concat([output, pd.DataFrame(new_columns, index=output.index)], axis=1)


def add_centered_rolling_stats(
    features: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    window_epochs: int,
    statistics: Iterable[str] = ("mean", "std"),
    group_cols: Iterable[str] = ("participant_id",),
    epoch_col: str = "epoch_id",
) -> pd.DataFrame:
    """Add centered rolling context statistics within participant-contained groups."""
    if window_epochs <= 0:
        raise ValueError("window_epochs must be positive.")
    stats = tuple(statistics)
    unsupported = sorted(set(stats) - {"mean", "std"})
    if unsupported:
        raise ValueError(f"Unsupported rolling statistic(s): {unsupported}")

    groups = list(group_cols)
    output = features.sort_values([*groups, epoch_col]).copy()
    new_columns: dict[str, pd.Series] = {}
    for column in feature_columns:
        for statistic in stats:
            new_columns[f"{column}_roll{window_epochs}_{statistic}"] = (
                _rolling_transform(
                    output,
                    column,
                    window_epochs=window_epochs,
                    group_cols=groups,
                    epoch_col=epoch_col,
                    statistic=statistic,
                )
            )
    if not new_columns:
        return output
    return pd.concat([output, pd.DataFrame(new_columns, index=output.index)], axis=1)


def _rolling_transform(
    features: pd.DataFrame,
    column: str,
    *,
    window_epochs: int,
    group_cols: list[str],
    epoch_col: str,
    statistic: str,
) -> pd.Series:
    sorted_features = features.sort_values([*group_cols, epoch_col])
    grouped = sorted_features.groupby(group_cols, sort=False)[column]
    values = grouped.transform(
        lambda series: getattr(
            series.rolling(window=window_epochs, min_periods=1, center=True),
            statistic,
        )()
    )
    return values.reindex(features.index)
