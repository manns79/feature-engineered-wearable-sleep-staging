"""Print the first-milestone metrics table after model training."""

from __future__ import annotations

import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", default="outputs/metrics/basic_features_metrics.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    display_columns = ["model", "split", "macro_f1", "balanced_accuracy", "cohen_kappa"]
    print(metrics[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
