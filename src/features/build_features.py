"""CSV feature-table construction for basic DREAMT epoch summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_SIGNAL_COLUMNS, SPLIT_ORDER, TARGET_LABELS
from src.data.load_dreamt import extract_participant_id, list_participant_csvs
from src.features.acc_features import ACC_MAGNITUDE_COLUMN, accelerometer_magnitude
from src.features.base_features import summarize_epoch, summarize_signal

FEATURE_ID_COLUMNS = ["participant_id", "epoch_id", "split", "label"]
SPLIT_OUTPUT_NAMES = {"train": "train", "validation": "val", "test": "test"}
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
