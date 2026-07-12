"""Audit DREAMT label-transition alignment before building epoch indexes."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.make_epochs import audit_epoch_alignment_for_raw_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="outputs/metrics/epoch_alignment_audit.csv")
    parser.add_argument("--sampling-rate-hz", type=int, default=64)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--rows-per-epoch", type=int, default=None)
    parser.add_argument("--min-transition-agreement", type=float, default=0.80)
    parser.add_argument(
        "--candidate-offset-step",
        type=int,
        default=None,
        help=(
            "Grid step for best-offset diagnostics. Defaults to the sampling "
            "rate so 64 Hz data is checked at whole-second offsets."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_epoch_alignment_for_raw_dir(
        args.raw_dir,
        rows_per_epoch=args.rows_per_epoch,
        sampling_rate_hz=args.sampling_rate_hz,
        epoch_seconds=args.epoch_seconds,
        min_transition_agreement=args.min_transition_agreement,
        candidate_offset_step=args.candidate_offset_step,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)

    flagged = audit[
        (audit["top_transition_share"].fillna(1.0) < args.min_transition_agreement)
        | (audit["current_mapped_mixed_epochs"] > audit["best_mapped_mixed_epochs"])
    ]
    print(f"Wrote epoch alignment audit to {output}")
    print(f"Audited {len(audit)} participant file(s); flagged {len(flagged)}.")
    if not flagged.empty:
        columns = [
            "participant_id",
            "transition_count",
            "top_remainder",
            "top_transition_share",
            "second_remainder",
            "inferred_offset",
            "current_mapped_mixed_epochs",
            "best_offset",
            "best_mapped_mixed_epochs",
            "prominent_remainders",
        ]
        print(flagged[columns].to_string(index=False))


if __name__ == "__main__":
    main()
