"""CSV feature-table construction for DREAMT epoch summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_SIGNAL_COLUMNS, SPLIT_ORDER, TARGET_LABELS
from src.data.load_dreamt import extract_participant_id, list_participant_csvs
from src.features.acc_features import ACC_MAGNITUDE_COLUMN, accelerometer_magnitude
from src.features.base_features import (
    summarize_epoch,
    summarize_signal,
    summarize_signal_array,
)
from src.features.feature_manifest import write_feature_manifest
from src.features.rolling_features import (
    add_centered_rolling_stats,
    add_participant_zscores,
)
from src.features.signal_features import summarize_signal_specific_arrays

FEATURE_ID_COLUMNS = ["participant_id", "epoch_id", "split", "label"]
SPLIT_OUTPUT_NAMES = {"train": "train", "validation": "val", "test": "test"}
FEATURE_SETS = ("basic", "rich")
DEFAULT_ROLLING_WINDOWS = (3, 5, 15)
DEFAULT_CONTEXT_SOURCE_FEATURES = (
    "BVP_mean",
    "ACC_MAG_mean",
    "ACC_MAG_std",
    "TEMP_mean",
    "EDA_mean",
    "IBI_mean",
    "HR_mean",
    "IBI_rmssd",
    "IBI_pnn50",
    "HR_abs_diff_mean",
)
TEMPORARY_CONTEXT_COLUMNS = ("_segment_id",)
EPOCH_INDEX_STRING_COLUMNS = {
    "participant_id": str,
    "split": str,
    "raw_label": "string",
    "mapped_label": "string",
    "exclusion_reason": "string",
    "segmentation_reason": "string",
}


def build_basic_feature_table(
    raw_dir: str | Path = "data/raw",
    epoch_index_path: str | Path = "data/interim/epoch_index.csv",
) -> pd.DataFrame:
    """Build one basic statistical feature row per valid 30-second epoch.

    Raw DREAMT files are intentionally loaded one participant at a time. The full
    64 Hz dataset is large enough that keeping every participant CSV in memory at
    once can exceed RAM on a normal workstation.
    """
    epoch_index = load_valid_epoch_index(epoch_index_path)
    participant_paths = _participant_path_lookup(raw_dir)
    rows: list[dict[str, object]] = []

    for participant_id, participant_epochs in epoch_index.groupby(
        "participant_id", sort=True
    ):
        participant_id = str(participant_id)
        if participant_id not in participant_paths:
            raise ValueError(f"Raw CSV for {participant_id} is missing from {raw_dir}.")

        participant = _load_participant_signal_frame(participant_paths[participant_id])
        for epoch_row in participant_epochs.itertuples(index=False):
            epoch = participant.iloc[int(epoch_row.start_row) : int(epoch_row.end_row)]
            features = _basic_epoch_features(epoch)
            rows.append(
                {
                    "participant_id": participant_id,
                    "epoch_id": int(epoch_row.epoch_id),
                    "split": str(epoch_row.split),
                    "label": str(epoch_row.mapped_label),
                    **features,
                }
            )

    if not rows:
        return pd.DataFrame(columns=FEATURE_ID_COLUMNS)

    features = pd.DataFrame(rows)
    feature_columns = sorted(
        column for column in features.columns if column not in FEATURE_ID_COLUMNS
    )
    return features[[*FEATURE_ID_COLUMNS, *feature_columns]]


def build_rich_feature_table(
    raw_dir: str | Path = "data/raw",
    epoch_index_path: str | Path = "data/interim/epoch_index.csv",
    *,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    context_source_features: tuple[str, ...] = DEFAULT_CONTEXT_SOURCE_FEATURES,
) -> pd.DataFrame:
    """Build modular rich features for valid 30-second epochs.

    The rich table starts with deterministic per-epoch features, then adds
    participant-contained temporal context features and whole-night
    participant-normalized z-scores. No labels are used except to carry the
    target column forward.
    """
    epoch_features = _build_epoch_level_feature_table(
        raw_dir=raw_dir,
        epoch_index_path=epoch_index_path,
        include_signal_specific=True,
        include_context_metadata=True,
    )
    if epoch_features.empty:
        return pd.DataFrame(columns=FEATURE_ID_COLUMNS)

    per_epoch_feature_columns = _feature_columns(epoch_features)
    context_columns = [
        column for column in context_source_features if column in epoch_features.columns
    ]
    context_group_columns = _context_group_columns(epoch_features)
    features = epoch_features
    for window_epochs in rolling_windows:
        features = add_centered_rolling_stats(
            features,
            context_columns,
            window_epochs=window_epochs,
            statistics=("mean", "std"),
            group_cols=context_group_columns,
        )

    features = add_participant_zscores(features, per_epoch_feature_columns)
    features = features.drop(
        columns=[
            column for column in TEMPORARY_CONTEXT_COLUMNS if column in features.columns
        ]
    )
    return _ordered_feature_table(features)


def write_split_feature_tables(
    features: pd.DataFrame,
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Write train/validation/test feature CSVs using prior-project filenames."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for split in SPLIT_ORDER:
        split_rows = features[features["split"] == split].copy()
        filename = f"features_{SPLIT_OUTPUT_NAMES[split]}.csv"
        output_path = output_root / filename
        split_rows.to_csv(output_path, index=False)
        written[split] = output_path
    return written


def build_and_write_basic_feature_tables(
    raw_dir: str | Path = "data/raw",
    epoch_index_path: str | Path = "data/interim/epoch_index.csv",
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Build basic features and write split CSV files."""
    features = build_basic_feature_table(raw_dir, epoch_index_path)
    return write_split_feature_tables(features, output_dir)


def build_and_write_rich_feature_tables(
    raw_dir: str | Path = "data/raw",
    epoch_index_path: str | Path = "data/interim/epoch_index.csv",
    output_dir: str | Path = "data/processed",
    manifest_path: str | Path = "data/processed/feature_manifest.csv",
) -> dict[str, Path]:
    """Build rich features, write split CSV files, and write a CSV manifest."""
    features = build_rich_feature_table(raw_dir, epoch_index_path)
    written = write_split_feature_tables(features, output_dir)
    written["manifest"] = write_feature_manifest(features, manifest_path)
    return written


def build_and_write_feature_tables(
    raw_dir: str | Path = "data/raw",
    epoch_index_path: str | Path = "data/interim/epoch_index.csv",
    output_dir: str | Path = "data/processed",
    *,
    feature_set: str = "rich",
    manifest_path: str | Path = "data/processed/feature_manifest.csv",
) -> dict[str, Path]:
    """Build and write the requested feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {FEATURE_SETS}; got {feature_set}"
        )
    if feature_set == "basic":
        features = build_basic_feature_table(raw_dir, epoch_index_path)
        written = write_split_feature_tables(features, output_dir)
        written["manifest"] = write_feature_manifest(features, manifest_path)
        return written
    return build_and_write_rich_feature_tables(
        raw_dir=raw_dir,
        epoch_index_path=epoch_index_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )


def load_valid_epoch_index(epoch_index_path: str | Path) -> pd.DataFrame:
    """Load valid target-labeled epochs from the epoch-index CSV."""
    path = Path(epoch_index_path)
    if not path.exists():
        raise FileNotFoundError(f"Epoch index CSV does not exist: {path}")

    epoch_index = pd.read_csv(path, dtype=EPOCH_INDEX_STRING_COLUMNS)
    required = {
        "participant_id",
        "split",
        "epoch_id",
        "start_row",
        "end_row",
        "mapped_label",
        "is_valid_epoch",
    }
    missing = sorted(required - set(epoch_index.columns))
    if missing:
        raise ValueError(f"epoch index is missing column(s): {missing}")

    valid = epoch_index[epoch_index["is_valid_epoch"].astype(bool)].copy()
    valid = valid[valid["mapped_label"].isin(TARGET_LABELS)].copy()
    return valid.sort_values(["participant_id", "epoch_id"]).reset_index(drop=True)


def _participant_path_lookup(raw_dir: str | Path) -> dict[str, Path]:
    """Return participant CSV paths keyed by participant ID."""
    return {
        extract_participant_id(path): path for path in list_participant_csvs(raw_dir)
    }


def _load_participant_signal_frame(path: str | Path) -> pd.DataFrame:
    """Load only signal columns needed for feature extraction."""
    csv_path = Path(path)
    available_columns = pd.read_csv(csv_path, nrows=0).columns
    usecols = [column for column in RAW_SIGNAL_COLUMNS if column in available_columns]
    if not usecols:
        raise ValueError(f"{csv_path} does not contain any expected signal columns.")
    return pd.read_csv(csv_path, usecols=usecols)


def _basic_epoch_features(epoch: pd.DataFrame) -> dict[str, float]:
    """Compute first-milestone statistical features for one epoch."""
    features = summarize_epoch(epoch, RAW_SIGNAL_COLUMNS)
    if all(column in epoch.columns for column in ("ACC_X", "ACC_Y", "ACC_Z")):
        acc_mag = accelerometer_magnitude(epoch)
        features.update(summarize_signal(acc_mag, ACC_MAGNITUDE_COLUMN))
    return features


def _build_epoch_level_feature_table(
    *,
    raw_dir: str | Path,
    epoch_index_path: str | Path,
    include_signal_specific: bool,
    include_context_metadata: bool,
) -> pd.DataFrame:
    epoch_index = load_valid_epoch_index(epoch_index_path)
    participant_paths = _participant_path_lookup(raw_dir)
    rows: list[dict[str, object]] = []

    for participant_id, participant_epochs in epoch_index.groupby(
        "participant_id", sort=True
    ):
        participant_id = str(participant_id)
        if participant_id not in participant_paths:
            raise ValueError(f"Raw CSV for {participant_id} is missing from {raw_dir}.")

        participant = _load_participant_signal_frame(participant_paths[participant_id])
        signal_arrays = _participant_signal_arrays(participant)
        for epoch_row in participant_epochs.itertuples(index=False):
            start_row = int(epoch_row.start_row)
            end_row = int(epoch_row.end_row)
            features = _basic_epoch_features_from_arrays(
                signal_arrays, start_row, end_row
            )
            if include_signal_specific:
                features.update(
                    summarize_signal_specific_arrays(signal_arrays, start_row, end_row)
                )
            row = {
                "participant_id": participant_id,
                "epoch_id": int(epoch_row.epoch_id),
                "split": str(epoch_row.split),
                "label": str(epoch_row.mapped_label),
                **features,
            }
            if include_context_metadata and hasattr(epoch_row, "segment_id"):
                segment_id = epoch_row.segment_id
                row["_segment_id"] = int(segment_id) if pd.notna(segment_id) else -1
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=FEATURE_ID_COLUMNS)
    return _ordered_feature_table(pd.DataFrame(rows), include_temporary=True)


def _feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        column
        for column in features.columns
        if column not in FEATURE_ID_COLUMNS and column not in TEMPORARY_CONTEXT_COLUMNS
    ]


def _context_group_columns(features: pd.DataFrame) -> list[str]:
    columns = ["participant_id"]
    if "_segment_id" in features.columns:
        columns.append("_segment_id")
    return columns


def _ordered_feature_table(
    features: pd.DataFrame, *, include_temporary: bool = False
) -> pd.DataFrame:
    temporary_columns = [
        column for column in TEMPORARY_CONTEXT_COLUMNS if column in features.columns
    ]
    feature_columns = sorted(
        column
        for column in features.columns
        if column not in FEATURE_ID_COLUMNS
        and (include_temporary or column not in temporary_columns)
    )
    return features[[*FEATURE_ID_COLUMNS, *feature_columns]]


def _participant_signal_arrays(participant: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column: pd.to_numeric(participant[column], errors="coerce").to_numpy(
            dtype=float
        )
        for column in RAW_SIGNAL_COLUMNS
        if column in participant.columns
    }


def _basic_epoch_features_from_arrays(
    signal_arrays: dict[str, np.ndarray], start_row: int, end_row: int
) -> dict[str, float]:
    features: dict[str, float] = {}
    for column in RAW_SIGNAL_COLUMNS:
        if column in signal_arrays:
            features.update(
                summarize_signal_array(signal_arrays[column][start_row:end_row], column)
            )
    if all(column in signal_arrays for column in ("ACC_X", "ACC_Y", "ACC_Z")):
        axes = np.vstack(
            [
                signal_arrays[column][start_row:end_row]
                for column in ("ACC_X", "ACC_Y", "ACC_Z")
            ]
        )
        valid = np.isfinite(axes).all(axis=0)
        magnitude = np.full(axes.shape[1], np.nan, dtype=float)
        magnitude[valid] = np.sqrt(np.square(axes[:, valid]).sum(axis=0))
        features.update(summarize_signal_array(magnitude, ACC_MAGNITUDE_COLUMN))
    return features
