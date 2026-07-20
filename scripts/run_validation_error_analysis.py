"""Run validation-only error analysis for completed ablation experiments."""

from __future__ import annotations

import argparse

from src.models.error_analysis import (
    DEFAULT_ERROR_ANALYSIS_MODELS,
    run_validation_error_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed ablation run directory, e.g. outputs/runs/full_ablation_...",
    )
    parser.add_argument(
        "--val-features",
        default="data/processed/features_val.csv",
        help="Validation feature table. The test table must not be used here.",
    )
    parser.add_argument("--manifest", default="data/processed/feature_manifest.csv")
    parser.add_argument(
        "--epoch-index",
        default="data/interim/epoch_index.csv",
        help="Optional epoch metadata for transition/timing context.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to RUN_DIR/error_analysis.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        nargs="+",
        help=(
            "Model name(s) to analyze for each ablation. May be passed multiple "
            "times. Defaults to logistic regression, random forest, and XGBoost."
        ),
    )
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        nargs="+",
        help=(
            "Ablation name(s) to analyze. May be passed multiple times. Defaults "
            "to every ablation listed in the run config."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Analyze completed selected ablation/model pairs even if others "
            "are missing."
        ),
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Skip validation permutation importance.",
    )
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip optional SHAP summary plots for selected XGBoost models.",
    )
    parser.add_argument("--max-shap-rows", type=int, default=1000)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV metrics only; the notebook can create or display plots later.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [model for group in args.model for model in group]
    ablations = [ablation for group in args.ablation for ablation in group]
    outputs = run_validation_error_analysis(
        run_dir=args.run_dir,
        validation_features_path=args.val_features,
        manifest_path=args.manifest,
        epoch_index_path=args.epoch_index,
        output_dir=args.output_dir,
        models=models or DEFAULT_ERROR_ANALYSIS_MODELS,
        ablations=ablations or None,
        allow_partial=args.allow_partial,
        compute_permutation=not args.skip_permutation,
        permutation_repeats=args.permutation_repeats,
        random_state=args.random_state,
        compute_shap=not args.skip_shap,
        max_shap_rows=args.max_shap_rows,
        create_plots=not args.no_plots,
    )
    print(f"Wrote validation error analysis under {outputs.analysis_dir}")
    print(f"Wrote selected model list to {outputs.selected_models_path}")
    print(f"Wrote artifact index to {outputs.artifact_index_path}")
    print(f"Wrote {len(outputs.artifact_index)} artifact record(s)")


if __name__ == "__main__":
    main()
