"""Training and evaluation orchestration for first-milestone feature baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import TARGET_LABELS
from src.features.build_features import FEATURE_ID_COLUMNS
from src.models.evaluate import classification_metrics, confusion_matrix_frame
from src.models.train_baselines import (
    ModelSpec,
    balanced_sample_weights,
    default_model_specs,
)

DEFAULT_FEATURE_PATHS = {
    "train": Path("data/processed/features_train.csv"),
    "validation": Path("data/processed/features_val.csv"),
    "test": Path("data/processed/features_test.csv"),
}


@dataclass(frozen=True)
class TrainingOutputs:
    """Paths written by a training run."""

    metrics_path: Path
    prediction_paths: dict[str, Path]
    confusion_paths: dict[str, Path]
    model_paths: dict[str, Path]


def train_and_evaluate_model_set(
    feature_paths: dict[str, str | Path] | None = None,
    output_dir: str | Path = "outputs",
    *,
    include_xgboost: bool = True,
    random_state: int = 42,
    model_prefix: str = "basic_features",
) -> TrainingOutputs:
    """Train planned models on basic feature CSVs and evaluate val/test splits."""
    paths = _feature_paths(feature_paths)
    frames = {split: _load_feature_table(path) for split, path in paths.items()}
    feature_columns = _feature_columns(frames["train"])
    _validate_feature_columns(frames, feature_columns)

    output_root = Path(output_dir)
    metrics_dir = output_root / "metrics"
    models_dir = output_root / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    prediction_paths: dict[str, Path] = {}
    confusion_paths: dict[str, Path] = {}
    model_paths: dict[str, Path] = {}

    X_train = frames["train"][feature_columns]
    y_train = frames["train"]["label"]
    specs = default_model_specs(
        random_state=random_state, include_xgboost=include_xgboost
    )

    for spec in specs:
        fitted = _fit_model(spec, X_train, y_train)
        model_path = models_dir / f"{model_prefix}_{spec.name}.joblib"
        joblib.dump(fitted, model_path)
        model_paths[spec.name] = model_path

        for split in ("validation", "test"):
            split_frame = frames[split]
            X_split = split_frame[feature_columns]
            y_true = split_frame["label"]
            y_pred, probabilities, classes = _predict_model(fitted, X_split)
            metrics = classification_metrics(y_true, y_pred, labels=TARGET_LABELS)
            metric_rows.append(
                {
                    "model": spec.name,
                    "feature_set": model_prefix,
                    "split": split,
                    "n_epochs": len(split_frame),
                    **metrics,
                }
            )

            prediction_path = (
                metrics_dir / f"{model_prefix}_{split}_{spec.name}_predictions.csv"
            )
            _prediction_frame(
                split_frame, y_true, y_pred, probabilities, classes
            ).to_csv(prediction_path, index=False)
            prediction_paths[f"{split}:{spec.name}"] = prediction_path

            confusion_path = (
                metrics_dir / f"{model_prefix}_{split}_{spec.name}_confusion.csv"
            )
            confusion_matrix_frame(y_true, y_pred, labels=TARGET_LABELS).to_csv(
                confusion_path
            )
            confusion_paths[f"{split}:{spec.name}"] = confusion_path

    metrics_path = metrics_dir / f"{model_prefix}_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    return TrainingOutputs(
        metrics_path=metrics_path,
        prediction_paths=prediction_paths,
        confusion_paths=confusion_paths,
        model_paths=model_paths,
    )


def _feature_paths(
    feature_paths: dict[str, str | Path] | None,
) -> dict[str, Path]:
    paths = DEFAULT_FEATURE_PATHS if feature_paths is None else feature_paths
    required = {"train", "validation", "test"}
    missing = sorted(required - set(paths))
    if missing:
        raise ValueError(f"feature_paths is missing split(s): {missing}")
    return {split: Path(paths[split]) for split in required}


def _load_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV does not exist: {path}")
    frame = pd.read_csv(path, dtype={"participant_id": str})
    missing = sorted(set(FEATURE_ID_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    if frame.empty:
        raise ValueError(f"Feature CSV is empty: {path}")
    return frame


def _feature_columns(train_frame: pd.DataFrame) -> list[str]:
    columns = [
        column for column in train_frame.columns if column not in FEATURE_ID_COLUMNS
    ]
    if not columns:
        raise ValueError("No feature columns found in training feature table.")
    return columns


def _validate_feature_columns(
    frames: dict[str, pd.DataFrame], feature_columns: list[str]
) -> None:
    expected = [*FEATURE_ID_COLUMNS, *feature_columns]
    for split, frame in frames.items():
        missing = sorted(set(expected) - set(frame.columns))
        if missing:
            raise ValueError(f"{split} feature table is missing column(s): {missing}")


def _fit_model(spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    if spec.name == "xgboost":
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(y_train)
        sample_weight = balanced_sample_weights(encoded)
        spec.estimator.fit(X_train, encoded, sample_weight=sample_weight)
        return {"estimator": spec.estimator, "label_encoder": encoder}

    if spec.use_sample_weight:
        spec.estimator.fit(
            X_train, y_train, sample_weight=balanced_sample_weights(y_train)
        )
    else:
        spec.estimator.fit(X_train, y_train)
    return spec.estimator


def _predict_model(
    fitted: Any, X: pd.DataFrame
) -> tuple[list[object], Any, list[object]]:
    if isinstance(fitted, dict) and "label_encoder" in fitted:
        estimator = fitted["estimator"]
        encoder = fitted["label_encoder"]
        encoded_predictions = estimator.predict(X)
        predictions = encoder.inverse_transform(
            encoded_predictions.astype(int)
        ).tolist()
        probabilities = (
            estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None
        )
        classes = encoder.classes_.tolist()
        return predictions, probabilities, classes

    predictions = fitted.predict(X).tolist()
    probabilities = (
        fitted.predict_proba(X) if hasattr(fitted, "predict_proba") else None
    )
    classes = fitted.classes_.tolist() if hasattr(fitted, "classes_") else TARGET_LABELS
    return predictions, probabilities, list(classes)


def _prediction_frame(
    split_frame: pd.DataFrame,
    y_true: pd.Series,
    y_pred: list[object],
    probabilities: Any,
    classes: list[object],
) -> pd.DataFrame:
    output = split_frame[["participant_id", "epoch_id", "split"]].copy()
    output["true_label"] = list(y_true)
    output["pred_label"] = y_pred
    if probabilities is not None:
        for index, label in enumerate(classes):
            output[f"prob_{_safe_label(label)}"] = probabilities[:, index]
    return output


def _safe_label(label: object) -> str:
    return str(label).replace("-", "_").replace(" ", "_")
