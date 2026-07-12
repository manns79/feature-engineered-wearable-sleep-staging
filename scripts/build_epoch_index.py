"""Build the DREAMT epoch-index CSV from local raw participant files."""

from __future__ import annotations

import argparse

from src.data.make_epochs import save_epoch_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--split-assignments", default="data/interim/split_assignments.csv"
    )
    parser.add_argument("--output", default="data/interim/epoch_index.csv")
    parser.add_argument("--sampling-rate-hz", type=int, default=64)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--rows-per-epoch", type=int, default=None)
    parser.add_argument("--missingness-threshold", type=float, default=1.0)
    parser.add_argument("--no-infer-start-offset", action="store_true")
    parser.add_argument("--min-transition-agreement", type=float, default=0.80)
    parser.add_argument(
        "--segmented-epoching",
        action="store_true",
        help=(
            "Opt in to segment-wise epoch alignment around long unmapped-label "
            "runs. The default preserves the legacy single-offset behavior."
        ),
    )
    parser.add_argument(
        "--min-segment-boundary-unmapped-epochs",
        type=float,
        default=2.0,
        help=(
            "Minimum contiguous unmapped-label duration, in epochs, used as a "
            "segment boundary when --segmented-epoching is enabled."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = save_epoch_index(
        raw_dir=args.raw_dir,
        split_assignments_path=args.split_assignments,
        output_path=args.output,
        rows_per_epoch=args.rows_per_epoch,
        sampling_rate_hz=args.sampling_rate_hz,
        epoch_seconds=args.epoch_seconds,
        missingness_threshold=args.missingness_threshold,
        infer_start_offset=not args.no_infer_start_offset,
        min_transition_agreement=args.min_transition_agreement,
        segmented_epoching=args.segmented_epoching,
        min_segment_boundary_unmapped_epochs=(
            args.min_segment_boundary_unmapped_epochs
        ),
    )
    print(f"Wrote epoch index to {output}")


if __name__ == "__main__":
    main()
