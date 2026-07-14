"""Build feature CSVs from local DREAMT data."""

from __future__ import annotations

import argparse

from src.features.build_features import FEATURE_SETS, build_and_write_feature_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--epoch-index", default="data/interim/epoch_index.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default="rich",
        help="Feature family to build. Defaults to the richer engineered table.",
    )
    parser.add_argument(
        "--manifest-output",
        default="data/processed/feature_manifest.csv",
        help="CSV path for the feature manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = build_and_write_feature_tables(
        raw_dir=args.raw_dir,
        epoch_index_path=args.epoch_index,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        manifest_path=args.manifest_output,
    )
    for split, path in written.items():
        if split == "manifest":
            print(f"Wrote feature manifest to {path}")
        else:
            print(f"Wrote {split} {args.feature_set} features to {path}")


if __name__ == "__main__":
    main()
