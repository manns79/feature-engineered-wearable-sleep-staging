"""Run rolling-context logistic validation experiment with correlation pruning."""

from __future__ import annotations

import argparse

from src.models.feature_selection import DEFAULT_CORRELATION_THRESHOLD
from src.models.rolling_logistic_experiment import run_rolling_logistic_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", default="data/processed/features_train.csv")
    parser.add_argument("--val-features", default="data/processed/features_val.csv")
    parser.add_argument("--manifest", default="data/processed/feature_manifest.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=DEFAULT_CORRELATION_THRESHOLD,
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--verbose-search", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_rolling_logistic_experiment(
        train_features_path=args.train_features,
        validation_features_path=args.val_features,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        correlation_threshold=args.correlation_threshold,
        random_state=args.random_state,
        cv_splits=args.cv_splits,
        n_jobs=args.n_jobs,
        verbose_search=args.verbose_search,
    )
    print(f"Wrote run artifacts under {outputs.run_dir}")
    print(f"Wrote selected features to {outputs.selected_features_path}")
    print(f"Wrote validation metrics to {outputs.metrics_path}")
    print(f"Wrote fitted logistic model to {outputs.model_path}")
    print(f"Wrote Platt calibrator to {outputs.calibrator_path}")
    print(f"Wrote transition model to {outputs.transition_model_path}")


if __name__ == "__main__":
    main()
