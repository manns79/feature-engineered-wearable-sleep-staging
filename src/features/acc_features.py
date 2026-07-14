"""Accelerometry-specific feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ACC_AXIS_COLUMNS
from src.features.base_features import (
    summarize_signal,
)

ACC_MAGNITUDE_COLUMN = "ACC_MAG"


def accelerometer_magnitude(epoch_df: pd.DataFrame) -> pd.Series:
    """Compute triaxial accelerometer vector magnitude."""
    missing = [column for column in ACC_AXIS_COLUMNS if column not in epoch_df]
    if missing:
        raise ValueError(f"Cannot compute ACC magnitude; missing {missing}")
    axes = epoch_df[list(ACC_AXIS_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return np.sqrt((axes**2).sum(axis=1, min_count=len(ACC_AXIS_COLUMNS))).rename(
        ACC_MAGNITUDE_COLUMN
    )


def summarize_accelerometry(epoch_df: pd.DataFrame) -> dict[str, float]:
    """Return per-axis and vector-magnitude ACC features."""
    features: dict[str, float] = {}
    for column in ACC_AXIS_COLUMNS:
        if column in epoch_df:
            features.update(summarize_signal(epoch_df[column], column))
    if all(column in epoch_df for column in ACC_AXIS_COLUMNS):
        magnitude = accelerometer_magnitude(epoch_df)
        features.update(summarize_signal(magnitude, ACC_MAGNITUDE_COLUMN))
        features["ACC_still_fraction"] = float((magnitude < magnitude.median()).mean())
    return features


def summarize_accelerometry_specific(epoch_df: pd.DataFrame) -> dict[str, float]:
    """Return movement features that are not generic per-signal summaries."""
    features = {
        "ACC_still_fraction": np.nan,
        "ACC_motion_intensity": np.nan,
        "ACC_magnitude_energy": np.nan,
        "ACC_magnitude_abs_diff_mean": np.nan,
    }
    if not all(column in epoch_df for column in ACC_AXIS_COLUMNS):
        return features

    magnitude = accelerometer_magnitude(epoch_df)
    valid = magnitude.dropna()
    if valid.empty:
        return features

    diffs = valid.diff().dropna().abs()
    features.update(
        {
            "ACC_still_fraction": float((valid < valid.median()).mean()),
            "ACC_motion_intensity": float(diffs.mean()) if not diffs.empty else np.nan,
            "ACC_magnitude_energy": float(np.mean(np.square(valid))),
            "ACC_magnitude_abs_diff_mean": (
                float(diffs.mean()) if not diffs.empty else np.nan
            ),
        }
    )
    return features
