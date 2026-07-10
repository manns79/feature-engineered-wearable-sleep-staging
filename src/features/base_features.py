"""General-purpose epoch summary features."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

SUMMARY_STATS = (
    "mean",
    "std",
    "median",
    "iqr",
    "min",
    "max",
    "range",
    "skew",
    "kurtosis",
    "slope",
    "diff_mean",
    "diff_std",
    "valid_fraction",
)


def numeric_series(values: Iterable[object] | pd.Series) -> pd.Series:
    """Coerce values to numeric pandas Series."""
    return pd.to_numeric(pd.Series(values), errors="coerce")


def linear_slope(values: pd.Series) -> float:
    """Return least-squares slope over valid sample positions."""
    valid = values.dropna()
    if len(valid) < 2:
        return np.nan
    x = valid.index.to_numpy(dtype=float)
    if np.ptp(x) == 0:
        return np.nan
    y = valid.to_numpy(dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def summarize_signal(
    values: Iterable[object] | pd.Series, prefix: str
) -> dict[str, float]:
    """Compute deterministic summary features for one signal epoch."""
    series = numeric_series(values)
    valid = series.dropna()
    features = {f"{prefix}_{stat}": np.nan for stat in SUMMARY_STATS}
    features[f"{prefix}_valid_fraction"] = (
        np.nan if len(series) == 0 else float(series.notna().mean())
    )
    if valid.empty:
        return features

    q25, q75 = valid.quantile([0.25, 0.75])
    diffs = valid.diff().dropna()
    features.update(
        {
            f"{prefix}_mean": float(valid.mean()),
            f"{prefix}_std": float(valid.std(ddof=1)),
            f"{prefix}_median": float(valid.median()),
            f"{prefix}_iqr": float(q75 - q25),
            f"{prefix}_min": float(valid.min()),
            f"{prefix}_max": float(valid.max()),
            f"{prefix}_range": float(valid.max() - valid.min()),
            f"{prefix}_skew": float(valid.skew()),
            f"{prefix}_kurtosis": float(valid.kurtosis()),
            f"{prefix}_slope": linear_slope(series),
            f"{prefix}_diff_mean": float(diffs.mean()) if not diffs.empty else np.nan,
            f"{prefix}_diff_std": float(diffs.std(ddof=1))
            if len(diffs) > 1
            else np.nan,
        }
    )
    return features


def summarize_epoch(
    epoch_df: pd.DataFrame,
    signal_columns: Iterable[str],
) -> dict[str, float]:
    """Summarize all available signal columns in an epoch DataFrame."""
    features: dict[str, float] = {}
    for column in signal_columns:
        if column in epoch_df.columns:
            features.update(summarize_signal(epoch_df[column], column))
    return features
