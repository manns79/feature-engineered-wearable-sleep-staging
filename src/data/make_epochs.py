"""Epoch-index construction for DREAMT participant CSV files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import EPOCH_SECONDS, LABEL_COLUMN, RAW_SIGNAL_COLUMNS, TIME_COLUMN
from src.data.label_mapping import map_sleep_stage, normalize_stage_text
from src.data.load_dreamt import list_participant_csvs, load_participant_csv
from src.data.splits import load_split_assignments

DEFAULT_SAMPLING_RATE_HZ = 64
DEFAULT_MISSINGNESS_THRESHOLD = 1.0
DEFAULT_MIN_TRANSITION_AGREEMENT = 0.80

EPOCH_INDEX_COLUMNS = [
    "participant_id",
    "split",
    "epoch_id",
    "start_row",
    "end_row",
    "n_rows",
    "expected_n_rows",
    "epoch_start_offset_rows",
    "start_time",
    "end_time",
    "raw_label",
    "mapped_label",
    "is_valid_label",
    "is_valid_epoch",
    "exclusion_reason",
]


def build_epoch_index(
    raw_dir: str | Path,
    split_assignments_path: str | Path,
    *,
    rows_per_epoch: int | None = None,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    epoch_seconds: int = EPOCH_SECONDS,
    missingness_threshold: float = DEFAULT_MISSINGNESS_THRESHOLD,
    infer_start_offset: bool = True,
    min_transition_agreement: float = DEFAULT_MIN_TRANSITION_AGREEMENT,
) -> pd.DataFrame:
    """Build a 30-second DREAMT epoch index from local participant CSVs.

    The output keeps complete fixed-row epochs, but marks preparation, mixed-label,
    and otherwise unmapped epochs as invalid so feature generation can exclude
    them explicitly. Splits are participant-level assignments copied from the
    prior project.
    """
    expected_n_rows = (
        rows_per_epoch
        if rows_per_epoch is not None
        else sampling_rate_hz * epoch_seconds
    )
    if expected_n_rows <= 0:
        raise ValueError("rows_per_epoch/sampling_rate_hz must imply positive rows.")
    if not 0 <= missingness_threshold <= 1:
        raise ValueError("missingness_threshold must be between 0 and 1.")
    if not 0 <= min_transition_agreement <= 1:
        raise ValueError("min_transition_agreement must be between 0 and 1.")

    split_df = load_split_assignments(split_assignments_path)
    split_lookup = dict(
        zip(
            split_df["participant_id"].astype(str).str.strip().str.upper(),
            split_df["split"].astype(str).str.strip(),
            strict=True,
        )
    )
    rows: list[dict[str, object]] = []
    missing_split_ids: list[str] = []

    for csv_path in list_participant_csvs(raw_dir):
        participant = load_participant_csv(csv_path)
        participant_id = str(participant["participant_id"].iloc[0]).upper()
        split = split_lookup.get(participant_id)
        if split is None:
            missing_split_ids.append(participant_id)
            continue
        if LABEL_COLUMN not in participant.columns:
            raise ValueError(f"{csv_path} is missing {LABEL_COLUMN!r}.")

        start_offset = (
            infer_epoch_start_offset_from_labels(
                participant[LABEL_COLUMN],
                expected_n_rows=expected_n_rows,
                min_transition_agreement=min_transition_agreement,
            )
            if infer_start_offset
            else 0
        )
        usable_rows = len(participant) - start_offset
        n_epochs = max(usable_rows, 0) // expected_n_rows
        for epoch_id in range(n_epochs):
            start_row = start_offset + epoch_id * expected_n_rows
            end_row = start_row + expected_n_rows
            epoch = participant.iloc[start_row:end_row]
            rows.append(
                _epoch_index_row(
                    epoch,
                    participant_id=participant_id,
                    split=split,
                    epoch_id=epoch_id,
                    start_row=start_row,
                    end_row=end_row,
                    expected_n_rows=expected_n_rows,
                    epoch_start_offset_rows=start_offset,
                    missingness_threshold=missingness_threshold,
                )
            )

    if missing_split_ids:
        raise ValueError(
            "Participant CSV file(s) missing from split assignments: "
            f"{sorted(missing_split_ids)}"
        )

    if not rows:
        return pd.DataFrame(columns=[*EPOCH_INDEX_COLUMNS, *_missingness_columns()])

    epoch_index = pd.DataFrame(rows)
    ordered = [
        *EPOCH_INDEX_COLUMNS,
        *[
            column
            for column in epoch_index.columns
            if column.startswith("missingness_")
        ],
    ]
    return epoch_index[ordered]


def save_epoch_index(
    raw_dir: str | Path = "data/raw",
    split_assignments_path: str | Path = "data/interim/split_assignments.csv",
    output_path: str | Path = "data/interim/epoch_index.csv",
    *,
    rows_per_epoch: int | None = None,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    epoch_seconds: int = EPOCH_SECONDS,
    missingness_threshold: float = DEFAULT_MISSINGNESS_THRESHOLD,
    infer_start_offset: bool = True,
    min_transition_agreement: float = DEFAULT_MIN_TRANSITION_AGREEMENT,
) -> Path:
    """Build and save the epoch index as CSV."""
    epoch_index = build_epoch_index(
        raw_dir,
        split_assignments_path,
        rows_per_epoch=rows_per_epoch,
        sampling_rate_hz=sampling_rate_hz,
        epoch_seconds=epoch_seconds,
        missingness_threshold=missingness_threshold,
        infer_start_offset=infer_start_offset,
        min_transition_agreement=min_transition_agreement,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    epoch_index.to_csv(output, index=False)
    return output


def infer_epoch_start_offset_from_labels(
    labels: pd.Series,
    *,
    expected_n_rows: int,
    min_transition_agreement: float = DEFAULT_MIN_TRANSITION_AGREEMENT,
) -> int:
    """Infer a row offset when label transitions share a stable epoch remainder."""
    if expected_n_rows <= 0:
        raise ValueError("expected_n_rows must be positive.")
    remainders: Counter[int] = Counter()
    previous_label: str | None = None
    has_previous = False
    for row_position, raw_label in enumerate(labels):
        label = normalize_stage_text(raw_label)
        if has_previous and previous_label is not None and label is not None:
            if label != previous_label:
                remainders[row_position % expected_n_rows] += 1
        previous_label = label
        has_previous = True

    if not remainders:
        return 0
    offset, count = remainders.most_common(1)[0]
    if offset == 0 or count / sum(remainders.values()) < min_transition_agreement:
        return 0
    return int(offset)


def _epoch_index_row(
    epoch: pd.DataFrame,
    *,
    participant_id: str,
    split: str,
    epoch_id: int,
    start_row: int,
    end_row: int,
    expected_n_rows: int,
    epoch_start_offset_rows: int,
    missingness_threshold: float,
) -> dict[str, object]:
    """Return one epoch-index row with label and missingness metadata."""
    label_info = _epoch_label_info(epoch[LABEL_COLUMN])
    missingness = _signal_missingness(epoch)
    max_missingness = max(missingness.values(), default=0.0)
    is_valid_epoch = bool(
        label_info["is_valid_label"] and max_missingness <= missingness_threshold
    )
    exclusion_reason = _exclusion_reason(
        label_issue=label_info["label_issue"],
        max_missingness=max_missingness,
        missingness_threshold=missingness_threshold,
    )

    return {
        "participant_id": participant_id,
        "split": split,
        "epoch_id": epoch_id,
        "start_row": start_row,
        "end_row": end_row,
        "n_rows": len(epoch),
        "expected_n_rows": expected_n_rows,
        "epoch_start_offset_rows": epoch_start_offset_rows,
        "start_time": _epoch_time_value(epoch, first=True),
        "end_time": _epoch_time_value(epoch, first=False),
        "raw_label": label_info["raw_label"],
        "mapped_label": label_info["mapped_label"],
        "is_valid_label": label_info["is_valid_label"],
        "is_valid_epoch": is_valid_epoch,
        "exclusion_reason": exclusion_reason,
        **{f"missingness_{key}": value for key, value in missingness.items()},
    }


def _epoch_label_info(labels: pd.Series) -> dict[str, object]:
    """Return strict epoch label metadata.

    A valid epoch must contain one and only one mapped target label across all
    rows. Preparation, missing, unknown, and mixed-label epochs are invalid.
    """
    raw_labels = sorted({str(label).strip() for label in labels.dropna().unique()})
    raw_label = "|".join(raw_labels) if raw_labels else None
    mapped_labels = [map_sleep_stage(label) for label in labels]
    unique_mapped = sorted({label for label in mapped_labels if label is not None})

    if not raw_labels:
        return {
            "raw_label": raw_label,
            "mapped_label": None,
            "is_valid_label": False,
            "label_issue": "missing_or_invalid_label",
        }
    if len(unique_mapped) != 1 or any(label is None for label in mapped_labels):
        issue = (
            "label_changed_within_epoch" if len(unique_mapped) > 1 else "excluded_label"
        )
        return {
            "raw_label": raw_label,
            "mapped_label": "|".join(unique_mapped) if unique_mapped else None,
            "is_valid_label": False,
            "label_issue": issue,
        }
    return {
        "raw_label": raw_label,
        "mapped_label": unique_mapped[0],
        "is_valid_label": True,
        "label_issue": None,
    }


def _epoch_time_value(epoch: pd.DataFrame, *, first: bool) -> object | None:
    """Return first or last timestamp when the DREAMT timestamp column exists."""
    if TIME_COLUMN not in epoch.columns or epoch.empty:
        return None
    values = epoch[TIME_COLUMN].dropna()
    if values.empty:
        return None
    return values.iloc[0] if first else values.iloc[-1]


def _signal_missingness(epoch: pd.DataFrame) -> dict[str, float]:
    """Return per-signal missingness fractions for columns present in the CSV."""
    return {
        column: float(pd.to_numeric(epoch[column], errors="coerce").isna().mean())
        for column in RAW_SIGNAL_COLUMNS
        if column in epoch.columns
    }


def _missingness_columns() -> list[str]:
    return [f"missingness_{column}" for column in RAW_SIGNAL_COLUMNS]


def _exclusion_reason(
    *,
    label_issue: object,
    max_missingness: float,
    missingness_threshold: float,
) -> str | None:
    if label_issue is not None:
        return str(label_issue)
    if max_missingness > missingness_threshold:
        return "excessive_missingness"
    return None
