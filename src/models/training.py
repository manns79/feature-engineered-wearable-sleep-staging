"""Training and validation orchestration for first-milestone feature baselines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder

from src.config import TARGET_LABELS
from src.features.build_features import FEATURE_ID_COLUMNS
from src.models.evaluate import classification_metrics, confusion_matrix_frame
from src.models.train_baselines import (
    ModelSpec,
    balanced_sample_weights,
    default_model_specs,
)
from src.models.tune_models import MODEL_PARAM_GRIDS

DEFAULT_FEATURE_PATHS = {
    "train": Path("data/processed/features_train.csv"),
    "validation": Path("data/processed/features_val.csv"),
}
MACRO_F1_SCORER = make_scorer(f1_score, average="macro", zero_division=0)


@dataclass(frozen=True)
class TrainingOutputs:
    """Paths written by a training run."""

    metrics_path: Path
    cv_results_path: Path
    best_params_path: Path
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
    cv_splits: int = 5,
    n_jobs: int | None = None,
    param_grids: dict[str, dict[str, list[Any]]] | None = None,
    feature_columns: Iterable[str] | None = None,
) -> TrainingOutputs:
    """Tune on train-only grouped CV, then evaluate best models on validation.

    The held-out test feature table is intentionally not loaded here. Model
    family comparison during development is based on validation predictions only.
    """
    paths = _feature_paths(feature_paths)
    frames = {split: _load_feature_table(path) for split, path in paths.items()}
    selected_feature_columns = _feature_columns(
        frames["train"], requested_columns=feature_columns
    )
    _validate_feature_columns(frames, selected_feature_columns)

    output_root = Path(output_dir)
    metrics_dir = output_root / "metrics"
    models_dir = output_root / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    cv_result_frames: list[pd.DataFrame] = []
    best_param_rows: list[dict[str, object]] = []
    prediction_paths: dict[str, Path] = {}
    confusion_paths: dict[str, Path] = {}
    model_paths: dict[str, Path] = {}

    X_train = frames["train"][selected_feature_columns]
    y_train = frames["train"]["label"]
    groups_train = frames["train"]["participant_id"]
    cv = _group_kfold(groups_train, requested_splits=cv_splits)
    grids = MODEL_PARAM_GRIDS if param_grids is None else param_grids
    specs = default_model_specs(
        random_state=random_state, include_xgboost=include_xgboost
    )

    for spec in specs:
        fitted, cv_results, best_params = _fit_best_model(
            spec,
            X_train,
            y_train,
            groups_train,
            cv=cv,
            param_grid=grids.get(spec.name, {}),
            n_jobs=n_jobs,
        )
        model_path = models_dir / f"{model_prefix}_{spec.name}.joblib"
        joblib.dump(fitted, model_path)
        model_paths[spec.name] = model_path

        if not cv_results.empty:
            cv_result_frames.append(cv_results.assign(model=spec.name))
        best_param_rows.append(
            {
                "model": spec.name,
                "feature_set": model_prefix,
                "best_params": best_params,
                "best_cv_macro_f1": _best_cv_score(cv_results),
                "cv_n_splits": cv.get_n_splits(),
            }
        )

        split = "validation"
        split_frame = frames[split]
        X_split = split_frame[selected_feature_columns]
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
        _prediction_frame(split_frame, y_true, y_pred, probabilities, classes).to_csv(
            prediction_path, index=False
        )
        prediction_paths[f"{split}:{spec.name}"] = prediction_path

        confusion_path = (
            metrics_dir / f"{model_prefix}_{split}_{spec.name}_confusion.csv"
        )
        confusion_matrix_frame(y_true, y_pred, labels=TARGET_LABELS).to_csv(
            confusion_path
        )
        confusion_paths[f"{split}:{spec.name}"] = confusion_path

    metrics_path = metrics_dir / f"{model_prefix}_validation_metrics.csv"
    cv_results_path = metrics_dir / f"{model_prefix}_cv_results.csv"
    best_params_path = metrics_dir / f"{model_prefix}_best_params.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    _combine_cv_results(cv_result_frames).to_csv(cv_results_path, index=False)
    pd.DataFrame(best_param_rows).to_csv(best_params_path, index=False)
    return TrainingOutputs(
        metrics_path=metrics_path,
        cv_results_path=cv_results_path,
        best_params_path=best_params_path,
        prediction_paths=prediction_paths,
        confusion_paths=confusion_paths,
        model_paths=model_paths,
    )


def _feature_paths(
    feature_paths: dict[str, str | Path] | None,
) -> dict[str, Path]:
    paths = DEFAULT_FEATURE_PATHS if feature_paths is None else feature_paths
    required = {"train", "validation"}
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


def _feature_columns(
    train_frame: pd.DataFrame, requested_columns: Iterable[str] | None = None
) -> list[str]:
    if requested_columns is None:
        columns = [
            column for column in train_frame.columns if column not in FEATURE_ID_COLUMNS
        ]
        if not columns:
            raise ValueError("No feature columns found in training feature table.")
        return columns

    columns = list(requested_columns)
    if not columns:
        raise ValueError("feature_columns must contain at least one feature.")

    id_columns = sorted(set(columns) & set(FEATURE_ID_COLUMNS))
    if id_columns:
        raise ValueError(f"feature_columns contains ID/target column(s): {id_columns}")

    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"feature_columns contains duplicate column(s): {duplicates}")

    missing = sorted(set(columns) - set(train_frame.columns))
    if missing:
        raise ValueError(
            f"training feature table is missing requested column(s): {missing}"
        )
    return columns


def _validate_feature_columns(
    frames: dict[str, pd.DataFrame], feature_columns: list[str]
) -> None:
    expected = [*FEATURE_ID_COLUMNS, *feature_columns]
    for split, frame in frames.items():
        missing = sorted(set(expected) - set(frame.columns))
        if missing:
            raise ValueError(f"{split} feature table is missing column(s): {missing}")


def _group_kfold(groups: pd.Series, requested_splits: int) -> GroupKFold:
    """Create a mandatory participant-grouped CV splitter."""
    if requested_splits < 2:
        raise ValueError("cv_splits must be at least 2 for grouped CV.")
    n_groups = groups.astype(str).nunique()
    if n_groups < 2:
        raise ValueError("Grouped CV requires at least two training participants.")
    return GroupKFold(n_splits=min(requested_splits, n_groups))


def _fit_best_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    *,
    cv: GroupKFold,
    param_grid: dict[str, list[Any]],
    n_jobs: int | None,
) -> tuple[Any, pd.DataFrame, str]:
    if not param_grid:
        estimator = clone(spec.estimator)
        estimator.fit(X_train, y_train)
        return estimator, _baseline_cv_result(spec.name), "{}"

    if spec.name == "xgboost":
        return _fit_best_xgboost(
            spec,
            X_train,
            y_train,
            groups_train,
            cv=cv,
            param_grid=param_grid,
            n_jobs=n_jobs,
        )

    search = GridSearchCV(
        estimator=clone(spec.estimator),
        param_grid=param_grid,
        scoring=MACRO_F1_SCORER,
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        return_train_score=True,
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search.best_estimator_, _cv_results_frame(search), repr(search.best_params_)


def _fit_best_xgboost(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    *,
    cv: GroupKFold,
    param_grid: dict[str, list[Any]],
    n_jobs: int | None,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y_train)
    sample_weight = balanced_sample_weights(encoded)
    search = GridSearchCV(
        estimator=clone(spec.estimator),
        param_grid=param_grid,
        scoring=MACRO_F1_SCORER,
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        return_train_score=True,
    )
    search.fit(X_train, encoded, groups=groups_train, sample_weight=sample_weight)
    fitted = {"estimator": search.best_estimator_, "label_encoder": encoder}
    return fitted, _cv_results_frame(search), repr(search.best_params_)


def _baseline_cv_result(model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "params": "{}",
                "mean_test_score": pd.NA,
                "std_test_score": pd.NA,
                "rank_test_score": pd.NA,
                "mean_train_score": pd.NA,
                "std_train_score": pd.NA,
                "model": model_name,
            }
        ]
    )


def _cv_results_frame(search: GridSearchCV) -> pd.DataFrame:
    columns = [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_train_score",
        "std_train_score",
    ]
    frame = pd.DataFrame(search.cv_results_)
    frame["params"] = frame["params"].map(repr)
    return frame[columns].sort_values("rank_test_score")


def _combine_cv_results(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=[
                "params",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
                "mean_train_score",
                "std_train_score",
                "model",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _best_cv_score(cv_results: pd.DataFrame) -> object:
    if cv_results.empty or cv_results["mean_test_score"].isna().all():
        return pd.NA
    return float(cv_results.sort_values("rank_test_score").iloc[0]["mean_test_score"])


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
