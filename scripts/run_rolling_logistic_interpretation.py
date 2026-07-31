"""Generate interpretation artifacts for the finalized rolling logistic model."""

from __future__ import annotations

import argparse

from src.models.rolling_logistic_interpretation import (
    DEFAULT_PREDICTION_VARIANTS,
    run_rolling_logistic_interpretation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for rolling logistic interpretation outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rolling-run-dir",
        default="outputs/runs/rolling_logistic_train_oof_postprocessed_20260724",
        help="Completed interpretable rolling logistic run directory.",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/runs/final_test_evaluation/locked_test_predictions.csv",
        help="Final prediction CSV containing rolling logistic variants.",
    )
    parser.add_argument("--features", default="data/processed/features_test.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", default="data/processed/feature_manifest.csv")
    parser.add_argument(
        "--epoch-index",
        default="data/interim/epoch_index.csv",
        help="Optional epoch metadata for transition/timing context.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/runs/final_test_evaluation/"
            "rolling_logistic_interpretation"
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Rolling prediction variant to include. May be passed multiple "
            "times. Defaults to raw and threshold-focused final variants."
        ),
    )
    parser.add_argument(
        "--rem-error-variant",
        default="raw",
        help="Variant used for REM-focused error-group feature summaries.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV artifacts only.",
    )
    parser.add_argument("--max-plot-features", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Generate coefficient, error-group, and plot artifacts."""
    args = parse_args()
    outputs = run_rolling_logistic_interpretation(
        rolling_run_dir=args.rolling_run_dir,
        predictions_path=args.predictions,
        features_path=args.features,
        split=args.split,
        manifest_path=args.manifest,
        epoch_index_path=args.epoch_index,
        output_dir=args.output_dir,
        prediction_variants=tuple(args.variant) or DEFAULT_PREDICTION_VARIANTS,
        rem_error_variant=args.rem_error_variant,
        create_plots=not args.no_plots,
        max_plot_features=args.max_plot_features,
    )

    print(f"Wrote rolling logistic interpretation under {outputs.analysis_dir}")
    print(f"Wrote coefficient contrasts to {outputs.contrast_path}")
    print(
        "Wrote REM error feature summaries to "
        f"{outputs.rem_error_feature_summary_path}"
    )
    print(f"Wrote artifact index to {outputs.artifact_index_path}")


if __name__ == "__main__":
    main()
