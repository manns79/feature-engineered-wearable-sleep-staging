"""Participant-level split utilities anchored to the previous project."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pandas as pd

from src.config import (
    PREVIOUS_PROJECT_SPLIT_PATH,
    SPLIT_ORDER,
)

REQUIRED_SPLIT_COLUMNS = ("participant_id", "split")


def load_split_assignments(path: str | Path) -> pd.DataFrame:
    """Load and validate participant-level split assignments."""
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split assignment CSV does not exist: {split_path}")
    split_df = pd.read_csv(split_path, dtype={"participant_id": str})
    validate_split_assignments(split_df)
    return split_df


def validate_split_assignments(split_df: pd.DataFrame) -> None:
    """Validate required columns, split names, and participant uniqueness."""
    missing = [column for column in REQUIRED_SPLIT_COLUMNS if column not in split_df]
    if missing:
        raise ValueError(f"Split assignments are missing column(s): {missing}")

    cleaned = split_df.assign(
        participant_id=split_df["participant_id"].astype(str).str.strip(),
        split=split_df["split"].astype(str).str.strip(),
    )
    if cleaned["participant_id"].eq("").any():
        raise ValueError("Participant IDs must not be empty.")
    duplicate_ids = sorted(
        cleaned.loc[cleaned["participant_id"].duplicated(), "participant_id"].unique()
    )
    if duplicate_ids:
        raise ValueError(f"Participants assigned to multiple rows: {duplicate_ids}")

    unknown_splits = sorted(set(cleaned["split"]) - set(SPLIT_ORDER))
    if unknown_splits:
        raise ValueError(f"Unknown split label(s): {unknown_splits}")


def check_no_participant_overlap(split_df: pd.DataFrame) -> bool:
    """Return true when every participant appears in only one split."""
    validate_split_assignments(split_df)
    return True


def copy_previous_project_splits(
    output_path: str | Path = "data/interim/split_assignments.csv",
    source_path: str | Path = PREVIOUS_PROJECT_SPLIT_PATH,
) -> Path:
    """Copy the prior project's split assignments into this project."""
    source = Path(source_path)
    output = Path(output_path)
    split_df = load_split_assignments(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    copyfile(source, output)
    copied = load_split_assignments(output)
    assert_split_alignment(copied, split_df)
    return output


def assert_split_alignment(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
) -> None:
    """Require two split-assignment tables to contain identical assignments."""
    validate_split_assignments(candidate)
    validate_split_assignments(reference)
    candidate_map = _split_lookup(candidate)
    reference_map = _split_lookup(reference)
    if candidate_map != reference_map:
        missing = sorted(set(reference_map) - set(candidate_map))
        extra = sorted(set(candidate_map) - set(reference_map))
        changed = sorted(
            participant_id
            for participant_id in set(candidate_map) & set(reference_map)
            if candidate_map[participant_id] != reference_map[participant_id]
        )
        raise ValueError(
            "Split assignments do not match the previous project. "
            f"Missing={missing}, extra={extra}, changed={changed}"
        )


def _split_lookup(split_df: pd.DataFrame) -> dict[str, str]:
    cleaned = split_df.assign(
        participant_id=split_df["participant_id"].astype(str).str.strip(),
        split=split_df["split"].astype(str).str.strip(),
    )
    return dict(zip(cleaned["participant_id"], cleaned["split"], strict=True))
