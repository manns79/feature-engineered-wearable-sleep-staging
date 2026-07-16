"""Run manifest-driven validation ablation experiments."""

from __future__ import annotations

import argparse

from src.models.ablations import run_ablation_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", default="data/processed/features_train.csv")
    parser.add_argument("--val-features", default="data/processed/features_val.csv")
    parser.add_argument("--manifest", default="data/processed/feature_manifest.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        nargs="+",
        help=(
            "Name one or more ablations to run. May be passed multiple times. "
            "Defaults to the full planned ablation set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation_names = [
        ablation_name
        for ablation_group in args.ablation
        for ablation_name in ablation_group
    ]
    outputs = run_ablation_experiments(
        feature_paths={
            "train": args.train_features,
            "validation": args.val_features,
        },
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        ablation_names=ablation_names,
        include_xgboost=not args.skip_xgboost,
        random_state=args.random_state,
        cv_splits=args.cv_splits,
        n_jobs=args.n_jobs,
    )
    print(f"Wrote combined validation metrics to {outputs.metrics_path}")
    print(f"Wrote combined grouped-CV results to {outputs.cv_results_path}")
    print(f"Wrote combined best parameters to {outputs.best_params_path}")
    print(f"Wrote ablation feature-set summary to {outputs.feature_sets_path}")
    print(f"Ran {len(outputs.run_outputs)} ablation(s)")


if __name__ == "__main__":
    main()
