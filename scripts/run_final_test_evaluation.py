"""Generate frozen final test evaluation and compact test analysis artifacts."""

from __future__ import annotations

import argparse

from src.models.error_analysis import run_locked_test_error_analysis
from src.models.locked_test_evaluation import run_locked_test_evaluation


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the final locked-test workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rolling-run-dir",
        default="outputs/runs/rolling_logistic_train_oof_postprocessed_20260724",
        help="Completed interpretable rolling logistic run directory.",
    )
    parser.add_argument(
        "--prior-ablation-run-dir",
        default="outputs/runs/full_ablation_20260718",
        help="Completed original ablation run directory.",
    )
    parser.add_argument("--train-features", default="data/processed/features_train.csv")
    parser.add_argument("--val-features", default="data/processed/features_val.csv")
    parser.add_argument("--test-features", default="data/processed/features_test.csv")
    parser.add_argument(
        "--epoch-index",
        default="data/interim/epoch_index.csv",
        help="Optional epoch metadata for transition/timing context.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/runs/final_test_evaluation",
        help="Directory for final locked-test artifacts.",
    )
    parser.add_argument(
        "--analysis-output-dir",
        default=None,
        help="Defaults to OUTPUT_DIR/test_error_analysis.",
    )
    parser.add_argument(
        "--low-rem-threshold-epochs",
        type=int,
        default=30,
        help="Participant REM support threshold for low-support summaries.",
    )
    return parser.parse_args()


def main() -> None:
    """Run final model evaluation followed by compact test error analysis."""
    args = parse_args()
    evaluation = run_locked_test_evaluation(
        rolling_run_dir=args.rolling_run_dir,
        train_features_path=args.train_features,
        validation_features_path=args.val_features,
        test_features_path=args.test_features,
        prior_ablation_run_dir=args.prior_ablation_run_dir,
        output_dir=args.output_dir,
    )
    analysis = run_locked_test_error_analysis(
        predictions_path=evaluation.predictions_path,
        test_features_path=args.test_features,
        epoch_index_path=args.epoch_index,
        output_dir=args.analysis_output_dir,
        low_rem_threshold_epochs=args.low_rem_threshold_epochs,
    )

    print(f"Wrote final test protocol to {evaluation.protocol_path}")
    print(f"Wrote final comparison table to {evaluation.comparison_path}")
    print(f"Wrote locked test predictions to {evaluation.predictions_path}")
    print(f"Wrote locked test metrics to {evaluation.metrics_path}")
    print(f"Wrote locked test error analysis under {analysis.analysis_dir}")
    print(f"Wrote test analysis artifact index to {analysis.artifact_index_path}")


if __name__ == "__main__":
    main()
