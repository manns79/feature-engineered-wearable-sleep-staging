"""Epoch-index construction for DREAMT participant CSV files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import EPOCH_SECONDS, LABEL_COLUMN, RAW_SIGNAL_COLUMNS, TIME_COLUMN
from src.data.label_mapping import map_sleep_stage, normalize_stage_text
from src.data.load_dreamt import (
    extract_participant_id,
    list_participant_csvs,
    load_participant_csv,
)
from src.data.splits import load_split_assignments

DEFAULT_SAMPLING_RATE_HZ = 64
DEFAULT_MISSINGNESS_THRESHOLD = 1.0
DEFAULT_MIN_TRANSITION_AGREEMENT = 0.80
DEFAULT_SEGMENT_BOUNDARY_UNMAPPED_EPOCHS = 2.0

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

SEGMENT_METADATA_COLUMNS = [
    "segment_id",
    "segment_start_row",
    "segment_end_row",
    "segment_epoch_start_offset_rows",
    "segment_transition_count",
    "segment_transition_agreement",
    "segmentation_reason",
]


@dataclass(frozen=True)
class LabelAlignedSegment:
    """A label-aligned region bounded by long unmapped-label runs."""

    segment_id: int
    start_row: int
    end_row: int
    offset_rows: int
    transition_count: int
    transition_agreement: float | None
    segmentation_reason: str


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
    segmented_epoching: bool = False,
    min_segment_boundary_unmapped_epochs: float = (
        DEFAULT_SEGMENT_BOUNDARY_UNMAPPED_EPOCHS
    ),
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
    if min_segment_boundary_unmapped_epochs <= 0:
        raise ValueError("min_segment_boundary_unmapped_epochs must be positive.")

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

        if segmented_epoching:
            participant_rows = _segmented_epoch_index_rows(
                participant,
                participant_id=participant_id,
                split=split,
                expected_n_rows=expected_n_rows,
                missingness_threshold=missingness_threshold,
                infer_start_offset=infer_start_offset,
                min_transition_agreement=min_transition_agreement,
                min_segment_boundary_unmapped_rows=round(
                    min_segment_boundary_unmapped_epochs * expected_n_rows
                ),
            )
        else:
            participant_rows = _single_offset_epoch_index_rows(
                participant,
                participant_id=participant_id,
                split=split,
                expected_n_rows=expected_n_rows,
                missingness_threshold=missingness_threshold,
                infer_start_offset=infer_start_offset,
                min_transition_agreement=min_transition_agreement,
            )
        rows.extend(participant_rows)

    if missing_split_ids:
        raise ValueError(
            "Participant CSV file(s) missing from split assignments: "
            f"{sorted(missing_split_ids)}"
        )

    if not rows:
        return pd.DataFrame(columns=[*EPOCH_INDEX_COLUMNS, *_missingness_columns()])

    epoch_index = pd.DataFrame(rows)
    return epoch_index[_ordered_epoch_index_columns(epoch_index)]


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
    segmented_epoching: bool = False,
    min_segment_boundary_unmapped_epochs: float = (
        DEFAULT_SEGMENT_BOUNDARY_UNMAPPED_EPOCHS
    ),
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
        segmented_epoching=segmented_epoching,
        min_segment_boundary_unmapped_epochs=min_segment_boundary_unmapped_epochs,
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
    details = _offset_inference_details(
        _normalized_labels(labels),
        expected_n_rows=expected_n_rows,
        min_transition_agreement=min_transition_agreement,
        fallback_offset=0,
    )
    return int(details["inferred_offset"])


def audit_epoch_alignment(
    labels: pd.Series,
    *,
    expected_n_rows: int,
    min_transition_agreement: float = DEFAULT_MIN_TRANSITION_AGREEMENT,
    candidate_offset_step: int = 1,
) -> dict[str, object]:
    """Return transition-remainder diagnostics for one participant label series."""
    if expected_n_rows <= 0:
        raise ValueError("expected_n_rows must be positive.")
    if candidate_offset_step <= 0:
        raise ValueError("candidate_offset_step must be positive.")

    normalized = _normalized_labels(labels)
    details = _offset_inference_details(
        normalized,
        expected_n_rows=expected_n_rows,
        min_transition_agreement=min_transition_agreement,
        fallback_offset=0,
    )
    transitions = _label_transitions(normalized)
    inferred_offset = int(details["inferred_offset"])
    current_raw_mixed, current_mapped_mixed = _mixed_epoch_counts_from_transitions(
        transitions,
        n_rows=len(normalized),
        offset_rows=inferred_offset,
        expected_n_rows=expected_n_rows,
    )
    best_offset = inferred_offset
    best_raw_mixed = current_raw_mixed
    best_mapped_mixed = current_mapped_mixed
    for candidate_offset in range(0, expected_n_rows, candidate_offset_step):
        raw_mixed, mapped_mixed = _mixed_epoch_counts_from_transitions(
            transitions,
            n_rows=len(normalized),
            offset_rows=candidate_offset,
            expected_n_rows=expected_n_rows,
        )
        candidate = (mapped_mixed, raw_mixed, candidate_offset)
        best = (best_mapped_mixed, best_raw_mixed, best_offset)
        if candidate < best:
            best_mapped_mixed = mapped_mixed
            best_raw_mixed = raw_mixed
            best_offset = candidate_offset

    prominent = [
        {
            "remainder": int(remainder),
            "count": int(count),
            "share": count / details["transition_count"]
            if details["transition_count"]
            else 0.0,
        }
        for remainder, count in details["remainder_counts"].most_common()
        if details["transition_count"] and count / details["transition_count"] >= 0.10
    ]
    return {
        "transition_count": details["transition_count"],
        "top_remainder": details["top_remainder"],
        "top_transition_count": details["top_transition_count"],
        "top_transition_share": details["top_transition_share"],
        "second_remainder": details["second_remainder"],
        "second_transition_count": details["second_transition_count"],
        "inferred_offset": inferred_offset,
        "current_raw_mixed_epochs": current_raw_mixed,
        "current_mapped_mixed_epochs": current_mapped_mixed,
        "best_offset": best_offset,
        "best_raw_mixed_epochs": best_raw_mixed,
        "best_mapped_mixed_epochs": best_mapped_mixed,
        "prominent_remainders": prominent,
    }


def audit_epoch_alignment_for_raw_dir(
    raw_dir: str | Path,
    *,
    rows_per_epoch: int | None = None,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    epoch_seconds: int = EPOCH_SECONDS,
    min_transition_agreement: float = DEFAULT_MIN_TRANSITION_AGREEMENT,
    candidate_offset_step: int | None = None,
) -> pd.DataFrame:
    """Audit transition-remainder stability for every DREAMT CSV in a raw dir."""
    expected_n_rows = (
        rows_per_epoch
        if rows_per_epoch is not None
        else sampling_rate_hz * epoch_seconds
    )
    if expected_n_rows <= 0:
        raise ValueError("rows_per_epoch/sampling_rate_hz must imply positive rows.")
    step = (
        candidate_offset_step
        if candidate_offset_step is not None
        else max(1, sampling_rate_hz)
    )
    rows: list[dict[str, object]] = []
    for csv_path in list_participant_csvs(raw_dir):
        participant_id = extract_participant_id(csv_path)
        labels = pd.read_csv(csv_path, usecols=[LABEL_COLUMN])[LABEL_COLUMN]
        audit = audit_epoch_alignment(
            labels,
            expected_n_rows=expected_n_rows,
            min_transition_agreement=min_transition_agreement,
            candidate_offset_step=step,
        )
        rows.append(
            {
                "participant_id": participant_id,
                "n_rows": len(labels),
                **{
                    key: value
                    for key, value in audit.items()
                    if key != "prominent_remainders"
                },
                "prominent_remainders": _format_prominent_remainders(
                    audit["prominent_remainders"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)


def infer_label_aligned_segments(
    labels: pd.Series,
    *,
    expected_n_rows: int,
    min_segment_boundary_unmapped_rows: int,
    min_transition_agreement: float = DEFAULT_MIN_TRANSITION_AGREEMENT,
) -> list[LabelAlignedSegment]:
    """Infer label-aligned regions separated by long unmapped-label runs."""
    if expected_n_rows <= 0:
        raise ValueError("expected_n_rows must be positive.")
    if min_segment_boundary_unmapped_rows <= 0:
        raise ValueError("min_segment_boundary_unmapped_rows must be positive.")

    normalized = _normalized_labels(labels)
    mapped = [map_sleep_stage(label) for label in normalized]
    long_unmapped_runs = _long_unmapped_runs(
        mapped,
        min_run_rows=min_segment_boundary_unmapped_rows,
    )
    segments: list[LabelAlignedSegment] = []
    region_start = 0
    segment_id = 0
    for run_start, run_end in [*long_unmapped_runs, (len(normalized), len(normalized))]:
        trimmed = _trim_to_mapped_region(mapped, region_start, run_start)
        if trimmed is not None:
            segment_start, segment_end = trimmed
            if segment_end - segment_start >= expected_n_rows:
                details = _offset_inference_details(
                    normalized,
                    expected_n_rows=expected_n_rows,
                    min_transition_agreement=min_transition_agreement,
                    start_row=segment_start,
                    end_row=segment_end,
                    fallback_offset=segment_start % expected_n_rows,
                )
                segments.append(
                    LabelAlignedSegment(
                        segment_id=segment_id,
                        start_row=segment_start,
                        end_row=segment_end,
                        offset_rows=int(details["inferred_offset"]),
                        transition_count=int(details["transition_count"]),
                        transition_agreement=details["top_transition_share"],
                        segmentation_reason="long_unmapped_boundary",
                    )
                )
                segment_id += 1
        region_start = run_end
    return segments


def _single_offset_epoch_index_rows(
    participant: pd.DataFrame,
    *,
    participant_id: str,
    split: str,
    expected_n_rows: int,
    missingness_threshold: float,
    infer_start_offset: bool,
    min_transition_agreement: float,
) -> list[dict[str, object]]:
    start_offset = (
        infer_epoch_start_offset_from_labels(
            participant[LABEL_COLUMN],
            expected_n_rows=expected_n_rows,
            min_transition_agreement=min_transition_agreement,
        )
        if infer_start_offset
        else 0
    )
    rows: list[dict[str, object]] = []
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
    return rows


def _segmented_epoch_index_rows(
    participant: pd.DataFrame,
    *,
    participant_id: str,
    split: str,
    expected_n_rows: int,
    missingness_threshold: float,
    infer_start_offset: bool,
    min_transition_agreement: float,
    min_segment_boundary_unmapped_rows: int,
) -> list[dict[str, object]]:
    segments = infer_label_aligned_segments(
        participant[LABEL_COLUMN],
        expected_n_rows=expected_n_rows,
        min_segment_boundary_unmapped_rows=max(
            1,
            min_segment_boundary_unmapped_rows,
        ),
        min_transition_agreement=min_transition_agreement,
    )
    rows: list[dict[str, object]] = []
    epoch_id = 0
    for segment in segments:
        offset_rows = segment.offset_rows if infer_start_offset else 0
        first_start = _first_epoch_start_at_or_after(
            segment.start_row,
            offset_rows=offset_rows,
            expected_n_rows=expected_n_rows,
        )
        latest_start = segment.end_row - expected_n_rows
        for start_row in range(first_start, latest_start + 1, expected_n_rows):
            end_row = start_row + expected_n_rows
            epoch = participant.iloc[start_row:end_row]
            row = _epoch_index_row(
                epoch,
                participant_id=participant_id,
                split=split,
                epoch_id=epoch_id,
                start_row=start_row,
                end_row=end_row,
                expected_n_rows=expected_n_rows,
                epoch_start_offset_rows=offset_rows,
                missingness_threshold=missingness_threshold,
            )
            row.update(
                {
                    "segment_id": segment.segment_id,
                    "segment_start_row": segment.start_row,
                    "segment_end_row": segment.end_row,
                    "segment_epoch_start_offset_rows": offset_rows,
                    "segment_transition_count": segment.transition_count,
                    "segment_transition_agreement": segment.transition_agreement,
                    "segmentation_reason": segment.segmentation_reason,
                }
            )
            rows.append(row)
            epoch_id += 1
    return rows


def _offset_inference_details(
    labels: list[str | None],
    *,
    expected_n_rows: int,
    min_transition_agreement: float,
    fallback_offset: int,
    start_row: int = 0,
    end_row: int | None = None,
) -> dict[str, object]:
    transitions = _label_transitions(labels, start_row=start_row, end_row=end_row)
    remainders: Counter[int] = Counter(
        row_position % expected_n_rows for row_position, _, _ in transitions
    )
    if not remainders:
        return {
            "inferred_offset": int(fallback_offset),
            "transition_count": 0,
            "top_remainder": None,
            "top_transition_count": 0,
            "top_transition_share": None,
            "second_remainder": None,
            "second_transition_count": 0,
            "remainder_counts": remainders,
        }
    common = remainders.most_common()
    offset, count = common[0]
    share = count / sum(remainders.values())
    inferred_offset = (
        int(offset) if share >= min_transition_agreement else fallback_offset
    )
    return {
        "inferred_offset": int(inferred_offset),
        "transition_count": sum(remainders.values()),
        "top_remainder": int(offset),
        "top_transition_count": int(count),
        "top_transition_share": float(share),
        "second_remainder": int(common[1][0]) if len(common) > 1 else None,
        "second_transition_count": int(common[1][1]) if len(common) > 1 else 0,
        "remainder_counts": remainders,
    }


def _label_transitions(
    labels: list[str | None],
    *,
    start_row: int = 0,
    end_row: int | None = None,
) -> list[tuple[int, str, str]]:
    end = len(labels) if end_row is None else min(end_row, len(labels))
    transitions: list[tuple[int, str, str]] = []
    previous_label: str | None = None
    has_previous = False
    for row_position in range(start_row, end):
        label = labels[row_position]
        if has_previous and previous_label is not None and label is not None:
            if label != previous_label:
                transitions.append((row_position, previous_label, label))
        previous_label = label
        has_previous = True
    return transitions


def _mixed_epoch_counts_from_transitions(
    transitions: list[tuple[int, str, str]],
    *,
    n_rows: int,
    offset_rows: int,
    expected_n_rows: int,
) -> tuple[int, int]:
    n_epochs = max(n_rows - offset_rows, 0) // expected_n_rows
    end_row = offset_rows + n_epochs * expected_n_rows
    raw_mixed_epoch_ids: set[int] = set()
    mapped_mixed_epoch_ids: set[int] = set()
    for row_position, previous_label, label in transitions:
        if row_position < offset_rows or row_position >= end_row:
            continue
        relative_position = row_position - offset_rows
        if relative_position % expected_n_rows == 0:
            continue
        epoch_id = relative_position // expected_n_rows
        raw_mixed_epoch_ids.add(epoch_id)
        previous_mapped = map_sleep_stage(previous_label)
        mapped = map_sleep_stage(label)
        if (
            previous_mapped is not None
            and mapped is not None
            and previous_mapped != mapped
        ):
            mapped_mixed_epoch_ids.add(epoch_id)
    return len(raw_mixed_epoch_ids), len(mapped_mixed_epoch_ids)


def _normalized_labels(labels: pd.Series) -> list[str | None]:
    return [normalize_stage_text(label) for label in labels]


def _long_unmapped_runs(
    mapped_labels: list[str | None],
    *,
    min_run_rows: int,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for row_position, mapped_label in enumerate(mapped_labels):
        if mapped_label is None:
            if run_start is None:
                run_start = row_position
        elif run_start is not None:
            if row_position - run_start >= min_run_rows:
                runs.append((run_start, row_position))
            run_start = None
    if run_start is not None and len(mapped_labels) - run_start >= min_run_rows:
        runs.append((run_start, len(mapped_labels)))
    return runs


def _trim_to_mapped_region(
    mapped_labels: list[str | None],
    start_row: int,
    end_row: int,
) -> tuple[int, int] | None:
    mapped_positions = [
        row_position
        for row_position in range(start_row, end_row)
        if mapped_labels[row_position] is not None
    ]
    if not mapped_positions:
        return None
    return mapped_positions[0], mapped_positions[-1] + 1


def _first_epoch_start_at_or_after(
    start_row: int,
    *,
    offset_rows: int,
    expected_n_rows: int,
) -> int:
    remainder = (offset_rows - start_row) % expected_n_rows
    return start_row + remainder


def _format_prominent_remainders(prominent_remainders: object) -> str:
    if not prominent_remainders:
        return ""
    return ";".join(
        f"{item['remainder']}:{item['count']}:{item['share']:.3f}"
        for item in prominent_remainders
    )


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


def _ordered_epoch_index_columns(epoch_index: pd.DataFrame) -> list[str]:
    missingness_columns = [
        column for column in epoch_index.columns if column.startswith("missingness_")
    ]
    segment_columns = [
        column for column in SEGMENT_METADATA_COLUMNS if column in epoch_index.columns
    ]
    remaining_columns = [
        column
        for column in epoch_index.columns
        if column
        not in {
            *EPOCH_INDEX_COLUMNS,
            *SEGMENT_METADATA_COLUMNS,
            *missingness_columns,
        }
    ]
    return [
        *EPOCH_INDEX_COLUMNS,
        *segment_columns,
        *missingness_columns,
        *remaining_columns,
    ]


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
