"""Epoch-index construction for DREAMT participant CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    LABEL_COLUMN,
    RAW_SIGNAL_COLUMNS,
)
from src.data.label_mapping import map_sleep_stage
from src.data.load_dreamt import (
    list_participant_csvs,
    load_participant_csv,
)
from src.data.splits import load_split_assignments


def build_epoch_index(
    raw_dir: str | Path,
    split_assignments_path: str | Path,
    *,
    rows_per_epoch: int,
) -> pd.DataFrame:
    """Build a simple 30-second epoch index from fixed-row DREAMT CSVs."""
    if rows_per_epoch <= 0:
        raise ValueError("rows_per_epoch must be positive.")

    split_df = load_split_assignments(split_assignments_path)
    split_lookup = dict(zip(split_df["participant_id"], split_df["split"], strict=True))
    rows: list[dict[str, object]] = []

    for csv_path in list_participant_csvs(raw_dir):
        participant = load_participant_csv(csv_path)
        participant_id = str(participant["participant_id"].iloc[0])
        if participant_id not in split_lookup:
            raise ValueError(
                f"{participant_id} is missing from split assignments. "
                "Copy the previous project split before epoching."
            )
        if LABEL_COLUMN not in participant.columns:
            raise ValueError(f"{csv_path} is missing {LABEL_COLUMN!r}.")

        n_epochs = len(participant) // rows_per_epoch
        for epoch_id in range(n_epochs):
            start_row = epoch_id * rows_per_epoch
            end_row = start_row + rows_per_epoch
            epoch = participant.iloc[start_row:end_row]
            raw_label = epoch[LABEL_COLUMN].mode(dropna=True)
            raw_label_value = raw_label.iloc[0] if not raw_label.empty else None
            mapped_label = map_sleep_stage(raw_label_value)
            rows.append(
                {
                    "participant_id": participant_id,
                    "split": split_lookup[participant_id],
                    "epoch_id": epoch_id,
                    "start_row": start_row,
                    "end_row": end_row,
                    "n_rows": len(epoch),
                    "raw_label": raw_label_value,
                    "mapped_label": mapped_label,
                    "is_valid_epoch": mapped_label is not None,
                    **{
                        f"missingness_{column}": float(
                            pd.to_numeric(epoch[column], errors="coerce").isna().mean()
                        )
                        for column in RAW_SIGNAL_COLUMNS
                        if column in epoch.columns
                    },
                }
            )
    return pd.DataFrame(rows)
