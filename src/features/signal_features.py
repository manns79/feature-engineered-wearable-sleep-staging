"""Signal-specific feature helpers for DREAMT epoch slices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.acc_features import summarize_accelerometry_specific
from src.features.base_features import numeric_series
from src.features.hrv_features import summarize_ibi


def summarize_signal_specific_epoch(epoch_df: pd.DataFrame) -> dict[str, float]:
    """Return deterministic signal-specific features for one epoch."""
    features: dict[str, float] = {}
    if all(column in epoch_df for column in ("ACC_X", "ACC_Y", "ACC_Z")):
        features.update(summarize_accelerometry_specific(epoch_df))
    if "BVP" in epoch_df:
        features.update(_pulse_morphology_features(epoch_df["BVP"], "BVP"))
    if "EDA" in epoch_df:
        features.update(_rise_fall_features(epoch_df["EDA"], "EDA"))
    if "TEMP" in epoch_df:
        features.update(_drift_features(epoch_df["TEMP"], "TEMP"))
    if "HR" in epoch_df:
        features.update(_drift_features(epoch_df["HR"], "HR"))
    if "IBI" in epoch_df:
        hrv = summarize_ibi(epoch_df["IBI"])
        # IBI_sdnn duplicates the generic IBI_std feature, so keep the
        # nonredundant short-window HRV proxies in the rich table.
        hrv.pop("IBI_sdnn", None)
        features.update(hrv)
    return features


def summarize_signal_specific_arrays(
    signal_arrays: dict[str, np.ndarray],
    start_row: int,
    end_row: int,
) -> dict[str, float]:
    """Return signal-specific features from preloaded numeric arrays."""
    features: dict[str, float] = {}
    if all(column in signal_arrays for column in ("ACC_X", "ACC_Y", "ACC_Z")):
        magnitude = _acc_magnitude(signal_arrays, start_row, end_row)
        features.update(_acc_specific_features(magnitude))
    if "BVP" in signal_arrays:
        features.update(
            _pulse_morphology_features_array(
                signal_arrays["BVP"][start_row:end_row], "BVP"
            )
        )
    if "EDA" in signal_arrays:
        features.update(
            _rise_fall_features_array(signal_arrays["EDA"][start_row:end_row], "EDA")
        )
    if "TEMP" in signal_arrays:
        features.update(
            _drift_features_array(signal_arrays["TEMP"][start_row:end_row], "TEMP")
        )
    if "HR" in signal_arrays:
        features.update(
            _drift_features_array(signal_arrays["HR"][start_row:end_row], "HR")
        )
    if "IBI" in signal_arrays:
        features.update(_ibi_features(signal_arrays["IBI"][start_row:end_row]))
    return features


def _abs_diff_mean(values: pd.Series) -> float:
    valid = numeric_series(values).dropna()
    diffs = valid.diff().dropna().abs()
    return float(diffs.mean()) if not diffs.empty else np.nan


def _drift_features(values: pd.Series, prefix: str) -> dict[str, float]:
    series = numeric_series(values)
    valid = series.dropna()
    features = {
        f"{prefix}_epoch_change": np.nan,
        f"{prefix}_abs_diff_mean": _abs_diff_mean(series),
    }
    if len(valid) >= 2:
        features[f"{prefix}_epoch_change"] = float(valid.iloc[-1] - valid.iloc[0])
    return features


def _rise_fall_features(values: pd.Series, prefix: str) -> dict[str, float]:
    series = numeric_series(values)
    valid = series.dropna()
    diffs = valid.diff().dropna()
    features = {
        f"{prefix}_epoch_change": np.nan,
        f"{prefix}_abs_diff_mean": _abs_diff_mean(series),
        f"{prefix}_rise_fraction": np.nan,
        f"{prefix}_fall_fraction": np.nan,
    }
    if len(valid) >= 2:
        features[f"{prefix}_epoch_change"] = float(valid.iloc[-1] - valid.iloc[0])
    if not diffs.empty:
        features[f"{prefix}_rise_fraction"] = float((diffs > 0).mean())
        features[f"{prefix}_fall_fraction"] = float((diffs < 0).mean())
    return features


def _pulse_morphology_features(values: pd.Series, prefix: str) -> dict[str, float]:
    series = numeric_series(values)
    valid = series.dropna()
    centered = valid - valid.mean() if not valid.empty else valid
    signs = np.sign(centered.to_numpy(dtype=float))
    nonzero = signs[signs != 0]
    if len(nonzero) >= 2:
        zero_crossing_rate = float((np.diff(nonzero) != 0).mean())
    else:
        zero_crossing_rate = np.nan
    features = _drift_features(series, prefix)
    features[f"{prefix}_zero_crossing_rate"] = zero_crossing_rate
    return features


def _valid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _acc_magnitude(
    signal_arrays: dict[str, np.ndarray], start_row: int, end_row: int
) -> np.ndarray:
    axes = np.vstack(
        [
            signal_arrays["ACC_X"][start_row:end_row],
            signal_arrays["ACC_Y"][start_row:end_row],
            signal_arrays["ACC_Z"][start_row:end_row],
        ]
    )
    valid = np.isfinite(axes).all(axis=0)
    magnitude = np.full(axes.shape[1], np.nan, dtype=float)
    magnitude[valid] = np.sqrt(np.square(axes[:, valid]).sum(axis=0))
    return magnitude


def _acc_specific_features(magnitude: np.ndarray) -> dict[str, float]:
    features = {
        "ACC_still_fraction": np.nan,
        "ACC_motion_intensity": np.nan,
        "ACC_magnitude_energy": np.nan,
        "ACC_magnitude_abs_diff_mean": np.nan,
    }
    valid = _valid(magnitude)
    if valid.size == 0:
        return features
    diffs = np.abs(np.diff(valid))
    diff_mean = float(np.mean(diffs)) if diffs.size else np.nan
    features.update(
        {
            "ACC_still_fraction": float(np.mean(valid < np.median(valid))),
            "ACC_motion_intensity": diff_mean,
            "ACC_magnitude_energy": float(np.mean(np.square(valid))),
            "ACC_magnitude_abs_diff_mean": diff_mean,
        }
    )
    return features


def _abs_diff_mean_array(values: np.ndarray) -> float:
    valid = _valid(values)
    diffs = np.abs(np.diff(valid))
    return float(np.mean(diffs)) if diffs.size else np.nan


def _drift_features_array(values: np.ndarray, prefix: str) -> dict[str, float]:
    valid = _valid(values)
    features = {
        f"{prefix}_epoch_change": np.nan,
        f"{prefix}_abs_diff_mean": _abs_diff_mean_array(values),
    }
    if valid.size >= 2:
        features[f"{prefix}_epoch_change"] = float(valid[-1] - valid[0])
    return features


def _rise_fall_features_array(values: np.ndarray, prefix: str) -> dict[str, float]:
    valid = _valid(values)
    diffs = np.diff(valid)
    features = {
        f"{prefix}_epoch_change": np.nan,
        f"{prefix}_abs_diff_mean": _abs_diff_mean_array(values),
        f"{prefix}_rise_fraction": np.nan,
        f"{prefix}_fall_fraction": np.nan,
    }
    if valid.size >= 2:
        features[f"{prefix}_epoch_change"] = float(valid[-1] - valid[0])
    if diffs.size:
        features[f"{prefix}_rise_fraction"] = float(np.mean(diffs > 0))
        features[f"{prefix}_fall_fraction"] = float(np.mean(diffs < 0))
    return features


def _pulse_morphology_features_array(
    values: np.ndarray, prefix: str
) -> dict[str, float]:
    valid = _valid(values)
    centered = valid - np.mean(valid) if valid.size else valid
    signs = np.sign(centered)
    nonzero = signs[signs != 0]
    if nonzero.size >= 2:
        zero_crossing_rate = float(np.mean(np.diff(nonzero) != 0))
    else:
        zero_crossing_rate = np.nan
    features = _drift_features_array(values, prefix)
    features[f"{prefix}_zero_crossing_rate"] = zero_crossing_rate
    return features


def _ibi_features(values: np.ndarray) -> dict[str, float]:
    valid = _valid(values)
    features = {
        "IBI_rmssd": np.nan,
        "IBI_pnn50": np.nan,
    }
    if valid.size < 2:
        return features
    diffs = np.diff(valid)
    features["IBI_rmssd"] = float(np.sqrt(np.mean(np.square(diffs))))
    features["IBI_pnn50"] = float(np.mean(np.abs(diffs) > 0.05))
    return features
