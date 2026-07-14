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


def summarize_signal_array(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute summary features for one numeric signal array."""
    array = np.asarray(values, dtype=float)
    features = {f"{prefix}_{stat}": np.nan for stat in SUMMARY_STATS}
    features[f"{prefix}_valid_fraction"] = (
        np.nan if len(array) == 0 else float(np.isfinite(array).mean())
    )
    valid_mask = np.isfinite(array)
    valid = array[valid_mask]
    if valid.size == 0:
        return features

    diffs = np.diff(valid)
    q25, q75 = np.quantile(valid, [0.25, 0.75])
    features.update(
        {
            f"{prefix}_mean": float(np.mean(valid)),
            f"{prefix}_std": _nanstd(valid),
            f"{prefix}_median": float(np.median(valid)),
            f"{prefix}_iqr": float(q75 - q25),
            f"{prefix}_min": float(np.min(valid)),
            f"{prefix}_max": float(np.max(valid)),
            f"{prefix}_range": float(np.max(valid) - np.min(valid)),
            f"{prefix}_skew": _skew(valid),
            f"{prefix}_kurtosis": _kurtosis(valid),
            f"{prefix}_slope": _linear_slope_array(array, valid_mask),
            f"{prefix}_diff_mean": float(np.mean(diffs)) if diffs.size else np.nan,
            f"{prefix}_diff_std": _nanstd(diffs),
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


def _nanstd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def _linear_slope_array(values: np.ndarray, valid_mask: np.ndarray) -> float:
    if valid_mask.sum() < 2:
        return np.nan
    x = np.flatnonzero(valid_mask).astype(float)
    if np.ptp(x) == 0:
        return np.nan
    y = values[valid_mask].astype(float)
    return float(np.polyfit(x, y, deg=1)[0])


def _skew(values: np.ndarray) -> float:
    n = values.size
    if n < 3:
        return np.nan
    mean = np.mean(values)
    centered = values - mean
    m2 = np.mean(centered**2)
    if m2 == 0:
        return np.nan
    m3 = np.mean(centered**3)
    return float(np.sqrt(n * (n - 1)) / (n - 2) * m3 / (m2**1.5))


def _kurtosis(values: np.ndarray) -> float:
    n = values.size
    if n < 4:
        return np.nan
    mean = np.mean(values)
    centered = values - mean
    m2 = np.mean(centered**2)
    if m2 == 0:
        return np.nan
    m4 = np.mean(centered**4)
    raw = m4 / (m2**2)
    return float(((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * raw - 3 * (n - 1)))
