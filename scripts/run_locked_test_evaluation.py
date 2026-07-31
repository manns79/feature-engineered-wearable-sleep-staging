"""Run locked held-out test evaluation for finalized model artifacts."""

from __future__ import annotations

import argparse

from src.models.locked_test_evaluation import run_locked_test_evaluation


def parse_args() -> argparse.Namespace:
    """Parse command-line options for locked-test model evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rolling-run-dir",
        required=True,
        help="Completed rolling logistic experiment run directory.",
    )
    parser.add_argument("--train-features", default="data/processed/features_train.csv")
    parser.add_argument("--val-features", default="data/processed/features_val.csv")
    parser.add_argument("--test-features", default="data/processed/features_test.csv")
    parser.add_argument(
        "--prior-ablation-run-dir",
        default=None,
        help="Optional completed ablation run to evaluate on the locked test split.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to ROLLING_RUN_DIR/locked_test.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate finalized artifacts on the held-out test split."""
    args = parse_args()
    outputs = run_locked_test_evaluation(
        rolling_run_dir=args.rolling_run_dir,
        train_features_path=args.train_features,
        validation_features_path=args.val_features,
        test_features_path=args.test_features,
        prior_ablation_run_dir=args.prior_ablation_run_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote locked test metrics to {outputs.metrics_path}")
    print(f"Wrote locked test predictions to {outputs.predictions_path}")
    print(f"Wrote locked test confusion matrices to {outputs.confusion_path}")
    if outputs.comparison_path is not None:
        print(f"Wrote final comparison table to {outputs.comparison_path}")
    print(f"Wrote final test protocol to {outputs.protocol_path}")


if __name__ == "__main__":
    main()
