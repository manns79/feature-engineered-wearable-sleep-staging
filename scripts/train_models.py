"""Placeholder entry point for traditional ML model training."""

from __future__ import annotations

from src.models.train_baselines import (
    default_model_specs,
)


def main() -> None:
    specs = default_model_specs(include_xgboost=False)
    names = ", ".join(spec.name for spec in specs)
    print(f"Training scaffold is ready for: {names}")


if __name__ == "__main__":
    main()
