"""Feature-table manifest generation."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

FEATURE_ID_COLUMNS = ["participant_id", "epoch_id", "split", "label"]
SIGNAL_GROUPS = {
    "BVP": "cardiovascular",
    "HR": "cardiovascular",
    "IBI": "cardiovascular",
    "ACC_X": "movement",
    "ACC_Y": "movement",
    "ACC_Z": "movement",
    "ACC_MAG": "movement",
    "ACC": "movement",
    "EDA": "electrodermal",
    "TEMP": "temperature",
}
SIGNAL_PREFIXES = tuple(sorted(SIGNAL_GROUPS, key=len, reverse=True))
SIGNAL_SPECIFIC_SUFFIXES = (
    "abs_diff_mean",
    "epoch_change",
    "fall_fraction",
    "magnitude_abs_diff_mean",
    "magnitude_energy",
    "motion_intensity",
    "pnn50",
    "rise_fraction",
    "rmssd",
    "still_fraction",
    "zero_crossing_rate",
)


def feature_manifest_from_columns(columns: list[str]) -> pd.DataFrame:
    """Describe feature family and signal group for each non-identity column."""
    rows = [
        _manifest_row(column) for column in columns if column not in FEATURE_ID_COLUMNS
    ]
    return pd.DataFrame(
        rows,
        columns=["feature", "feature_family", "signal_group", "source_signal"],
    ).sort_values(["feature_family", "signal_group", "feature"], ignore_index=True)


def write_feature_manifest(features: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a CSV manifest for the final feature table columns."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_manifest_from_columns(list(features.columns)).to_csv(path, index=False)
    return path


def _manifest_row(feature: str) -> dict[str, str]:
    base_feature = _base_feature_name(feature)
    source_signal = _source_signal(base_feature)
    return {
        "feature": feature,
        "feature_family": _feature_family(feature, base_feature),
        "signal_group": SIGNAL_GROUPS.get(source_signal, "other"),
        "source_signal": source_signal,
    }


def _base_feature_name(feature: str) -> str:
    if feature.endswith("_participant_z"):
        return feature.removesuffix("_participant_z")
    rolling_match = re.match(r"(.+)_roll\d+_(mean|std)$", feature)
    if rolling_match:
        return rolling_match.group(1)
    return feature


def _feature_family(feature: str, base_feature: str) -> str:
    if feature.endswith("_participant_z"):
        return "whole_night_subject_normalized"
    if re.search(r"_roll\d+_(mean|std)$", feature):
        return "rolling_context"
    if base_feature.endswith(SIGNAL_SPECIFIC_SUFFIXES):
        return "signal_specific"
    return "basic_statistical"


def _source_signal(feature: str) -> str:
    for prefix in SIGNAL_PREFIXES:
        if feature == prefix or feature.startswith(f"{prefix}_"):
            return prefix
    return "other"
