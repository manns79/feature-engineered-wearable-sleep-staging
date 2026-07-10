"""DREAMT CSV loading helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import PARTICIPANT_PATTERN


def extract_participant_id(path: str | Path) -> str:
    """Extract participant ID such as S002 from a DREAMT participant filename."""
    name = Path(path).name
    match = re.search(r"(?i)\b(S\d+)(?=_whole_df\.csv$|[_\-.]|$)", name)
    if match is None:
        raise ValueError(f"Could not parse participant ID from {name!r}.")
    return match.group(1).upper()


def list_participant_csvs(
    raw_dir: str | Path,
    pattern: str = PARTICIPANT_PATTERN,
) -> list[Path]:
    """Return sorted DREAMT participant CSV files from a directory."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")
    if not raw_path.is_dir():
        raise NotADirectoryError(f"Raw data path is not a directory: {raw_path}")
    return sorted(path for path in raw_path.glob(pattern) if path.is_file())


def load_participant_csv(path: str | Path) -> pd.DataFrame:
    """Load one participant CSV with participant_id attached."""
    csv_path = Path(path)
    participant_id = extract_participant_id(csv_path)
    frame = pd.read_csv(csv_path)
    frame.insert(0, "participant_id", participant_id)
    return frame
