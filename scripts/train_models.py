"""Train and evaluate the first basic-feature model set."""

from __future__ import annotations

import argparse

from src.models.train_models import train_and_evaluate_model_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", default="data/processed/features_train.csv")
    parser.add_argument("--val-features", default="data/processed/features_val.csv")
    parser.add_argument("--test-features", default="data/processed/features_test.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-xgboost", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = train_and_evaluate_model_set(
        feature_paths={
            "train": args.train_features,
            "validation": args.val_features,
            "test": args.test_features,
        },
        output_dir=args.output_dir,
        include_xgboost=not args.skip_xgboost,
        random_state=args.random_state,
    )
    print(f"Wrote metrics to {outputs.metrics_path}")
    print(f"Wrote {len(outputs.model_paths)} fitted model artifact(s)")


if __name__ == "__main__":
    main()
