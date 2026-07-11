"""Build first-milestone basic feature CSVs from local DREAMT data."""

from __future__ import annotations

import argparse

from src.features.build_features import build_and_write_basic_feature_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--epoch-index", default="data/interim/epoch_index.csv")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = build_and_write_basic_feature_tables(
        raw_dir=args.raw_dir,
        epoch_index_path=args.epoch_index,
        output_dir=args.output_dir,
    )
    for split, path in written.items():
        print(f"Wrote {split} features to {path}")


if __name__ == "__main__":
    main()
