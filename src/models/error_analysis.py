"""Post-hoc validation error analysis for ablation experiment outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_fscore_support

from src.config import TARGET_LABELS
from src.features.build_features import FEATURE_ID_COLUMNS
from src.models.evaluate import classification_metrics, confusion_matrix_frame
from src.visualization.plots import (
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_model_family_comparison_plot,
    save_participant_metric_plot,
    save_per_class_metric_plot,
    save_transition_metric_plot,
    save_xgboost_shap_summary,
)

DEFAULT_ERROR_ANALYSIS_MODELS = (
    "elastic_net_logistic_regression",
    "random_forest",
    "xgboost",
)
DUMMY_MODELS = ("majority_class", "stratified_random")
TRANSITION_BIN_ORDER = (
    "0",
    "1",
    "2-3",
    "4-10",
    ">10",
    "no_transition_in_participant",
)


@dataclass(frozen=True)
class ValidationErrorAnalysisOutputs:
    """Paths written by a validation error-analysis run."""

    analysis_dir: Path
    metrics_dir: Path
    figures_dir: Path
    config_path: Path
    artifact_index_path: Path
    selected_models_path: Path
    predictions_path: Path
    per_class_metrics_path: Path
    per_participant_metrics_path: Path
    transition_metrics_path: Path
    model_family_comparison_path: Path
    feature_importance_path: Path
    permutation_importance_path: Path | None
    artifact_index: pd.DataFrame


def run_validation_error_analysis(
    *,
    run_dir: str | Path,
    validation_features_path: str | Path = "data/processed/features_val.csv",
    manifest_path: str | Path | None = "data/processed/feature_manifest.csv",
    epoch_index_path: str | Path | None = "data/interim/epoch_index.csv",
    output_dir: str | Path | None = None,
    models: Sequence[str] | None = DEFAULT_ERROR_ANALYSIS_MODELS,
    ablations: Sequence[str] | None = None,
    allow_partial: bool = False,
    compute_permutation: bool = True,
    permutation_repeats: int = 5,
    random_state: int = 42,
    compute_shap: bool = True,
    max_shap_rows: int = 1000,
    create_plots: bool = True,
) -> ValidationErrorAnalysisOutputs:
    """Analyze selected ablation models on validation predictions only."""
    run_root = Path(run_dir)
    analysis_dir = (
        Path(output_dir) if output_dir is not None else run_root / "error_analysis"
    )
    metrics_dir = analysis_dir / "metrics"
    figures_dir = analysis_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_run_config(run_root)
    selected_models = select_validation_model_runs(
        run_root,
        models=_resolve_default_models(run_config, models),
        ablations=ablations,
        allow_partial=allow_partial,
    )
    validation_features = load_validation_features(validation_features_path)
    manifest = load_manifest_if_present(manifest_path)
    epoch_index = load_epoch_index_if_present(epoch_index_path)

    predictions = load_selected_predictions(selected_models)
    predictions = add_prediction_context(
        predictions,
        validation_features=validation_features,
        epoch_index=epoch_index,
    )
    predictions = add_transition_context(predictions)

    per_class_metrics = per_class_metrics_frame(predictions)
    per_participant_metrics = per_participant_metrics_frame(predictions)
    transition_metrics = transition_metrics_frame(predictions)
    model_family_comparison = model_family_comparison_frame(selected_models)
    feature_importance = feature_importance_frame(
        selected_models=selected_models,
        validation_features=validation_features,
        manifest=manifest,
    )
    permutation_frame = (
        permutation_importance_frame(
            selected_models=selected_models,
            validation_features=validation_features,
            n_repeats=permutation_repeats,
            random_state=random_state,
        )
        if compute_permutation
        else pd.DataFrame()
    )

    artifact_rows: list[dict[str, object]] = []
    selected_models_path = metrics_dir / "selected_models.csv"
    predictions_path = metrics_dir / "validation_predictions_with_context.csv"
    per_class_metrics_path = metrics_dir / "per_model_per_class_metrics.csv"
    per_participant_metrics_path = metrics_dir / "per_participant_metrics.csv"
    transition_metrics_path = metrics_dir / "transition_distance_metrics.csv"
    model_family_comparison_path = metrics_dir / "model_family_comparison.csv"
    feature_importance_path = metrics_dir / "feature_importance.csv"
    permutation_importance_path = (
        metrics_dir / "permutation_importance.csv" if compute_permutation else None
    )
    config_path = analysis_dir / "error_analysis_config.json"
    artifact_index_path = analysis_dir / "artifact_index.csv"

    _write_csv(selected_models, selected_models_path, artifact_rows, "metrics")
    _write_csv(predictions, predictions_path, artifact_rows, "metrics")
    _write_csv(per_class_metrics, per_class_metrics_path, artifact_rows, "metrics")
    _write_csv(
        per_participant_metrics, per_participant_metrics_path, artifact_rows, "metrics"
    )
    _write_csv(transition_metrics, transition_metrics_path, artifact_rows, "metrics")
    _write_csv(
        model_family_comparison, model_family_comparison_path, artifact_rows, "metrics"
    )
    _write_csv(feature_importance, feature_importance_path, artifact_rows, "metrics")
    if permutation_importance_path is not None:
        _write_csv(
            permutation_frame,
            permutation_importance_path,
            artifact_rows,
            "metrics",
        )

    if create_plots:
        artifact_rows.extend(
            create_error_analysis_plots(
                selected_models=selected_models,
                predictions=predictions,
                per_class_metrics=per_class_metrics,
                per_participant_metrics=per_participant_metrics,
                transition_metrics=transition_metrics,
                model_family_comparison=model_family_comparison,
                feature_importance=feature_importance,
                permutation_importance=permutation_frame,
                validation_features=validation_features,
                figures_dir=figures_dir,
                compute_shap=compute_shap,
                max_shap_rows=max_shap_rows,
            )
        )

    config = {
        "run_dir": str(run_root),
        "validation_features_path": str(validation_features_path),
        "manifest_path": str(manifest_path),
        "epoch_index_path": str(epoch_index_path) if epoch_index_path else None,
        "output_dir": str(analysis_dir),
        "models": list(_resolve_default_models(run_config, models)),
        "ablations": list(ablations) if ablations is not None else None,
        "allow_partial": allow_partial,
        "compute_permutation": compute_permutation,
        "permutation_repeats": permutation_repeats,
        "random_state": random_state,
        "compute_shap": compute_shap,
        "max_shap_rows": max_shap_rows,
        "create_plots": create_plots,
        "source_run_config": run_config,
    }
    config_path.write_text(json.dumps(config, indent=2, default=str) + "\n")
    artifact_rows.append(_artifact_row(config_path, "config"))
    artifact_index = pd.DataFrame(artifact_rows)
    artifact_index.to_csv(artifact_index_path, index=False)

    return ValidationErrorAnalysisOutputs(
        analysis_dir=analysis_dir,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        config_path=config_path,
        artifact_index_path=artifact_index_path,
        selected_models_path=selected_models_path,
        predictions_path=predictions_path,
        per_class_metrics_path=per_class_metrics_path,
        per_participant_metrics_path=per_participant_metrics_path,
        transition_metrics_path=transition_metrics_path,
        model_family_comparison_path=model_family_comparison_path,
        feature_importance_path=feature_importance_path,
        permutation_importance_path=permutation_importance_path,
        artifact_index=artifact_index,
    )


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    """Load an ablation run config JSON."""
    path = Path(run_dir) / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Ablation run config does not exist: {path}")
    return json.loads(path.read_text())


def select_validation_model_runs(
    run_dir: str | Path,
    *,
    models: Sequence[str],
    ablations: Sequence[str] | None = None,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Return completed validation model artifacts selected for analysis."""
    run_root = Path(run_dir)
    run_config = load_run_config(run_root)
    status_path = run_root / "run_status.csv"
    if not status_path.exists():
        raise FileNotFoundError(f"Ablation run status does not exist: {status_path}")

    requested_models = tuple(models)
    requested_ablations = (
        tuple(ablations)
        if ablations is not None
        else tuple(run_config.get("resolved_ablations", ()))
    )
    if not requested_models:
        raise ValueError("At least one model must be selected for error analysis.")
    if not requested_ablations:
        raise ValueError("At least one ablation must be selected for error analysis.")

    status = pd.read_csv(status_path)
    completed = status[
        (status["event"] == "model_completed")
        & (status["status"] == "completed")
        & (status["ablation"].isin(requested_ablations))
        & (status["model"].isin(requested_models))
    ].copy()

    missing: list[str] = []
    rows: list[pd.Series] = []
    for ablation in requested_ablations:
        ablation_rows = completed[completed["ablation"] == ablation].copy()
        for model in requested_models:
            model_rows = ablation_rows[ablation_rows["model"] == model]
            if model_rows.empty:
                missing.append(f"{ablation}:{model}")
                continue
            rows.append(model_rows.iloc[-1])

    if missing and not allow_partial:
        raise ValueError(
            "Selected ablation/model outputs are incomplete. "
            f"Missing completed model(s): {missing}. "
            "Pass allow_partial=True to analyze only completed selected models."
        )
    if not rows:
        raise ValueError("No completed selected validation model outputs were found.")

    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected = selected.assign(
        model_order=selected["model"].map(
            {model: index for index, model in enumerate(requested_models)}
        ),
        ablation_order=selected["ablation"].map(
            {ablation: index for index, ablation in enumerate(requested_ablations)}
        ),
    )
    selected = selected.sort_values(["ablation_order", "model_order"]).drop(
        columns=["ablation_order", "model_order"]
    )
    selected = _attach_feature_set_metadata(selected, run_root)
    selected = _absolutize_artifact_paths(selected, run_root)
    _validate_selected_artifact_paths(selected)
    return selected.reset_index(drop=True)


def load_validation_features(path: str | Path) -> pd.DataFrame:
    """Load the validation feature table with stable participant IDs."""
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Validation feature CSV does not exist: {feature_path}"
        )
    frame = pd.read_csv(feature_path, dtype={"participant_id": str})
    required = set(FEATURE_ID_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Validation feature CSV is missing column(s): {missing}")
    non_validation = sorted(set(frame["split"]) - {"validation"})
    if non_validation:
        raise ValueError(
            "Validation error analysis must receive the validation feature table only; "
            f"found split value(s): {non_validation}"
        )
    return frame


def load_manifest_if_present(path: str | Path | None) -> pd.DataFrame | None:
    """Load a feature manifest when available."""
    if path is None:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    return pd.read_csv(manifest_path)


def load_epoch_index_if_present(path: str | Path | None) -> pd.DataFrame | None:
    """Load epoch metadata when available."""
    if path is None:
        return None
    epoch_path = Path(path)
    if not epoch_path.exists():
        return None
    return pd.read_csv(epoch_path, dtype={"participant_id": str})


def load_selected_predictions(selected_models: pd.DataFrame) -> pd.DataFrame:
    """Load and annotate prediction rows for selected validation models."""
    frames: list[pd.DataFrame] = []
    for row in selected_models.itertuples(index=False):
        prediction = pd.read_csv(row.prediction_path, dtype={"participant_id": str})
        required = {
            "participant_id",
            "epoch_id",
            "split",
            "true_label",
            "pred_label",
        }
        missing = sorted(required - set(prediction.columns))
        if missing:
            raise ValueError(
                f"Prediction CSV {row.prediction_path} is missing column(s): {missing}"
            )
        invalid_splits = sorted(set(prediction["split"]) - {"validation"})
        if invalid_splits:
            raise ValueError(
                "Validation error analysis only supports validation predictions; "
                f"{row.prediction_path} has split value(s): {invalid_splits}"
            )
        prediction.insert(0, "model", row.model)
        prediction.insert(0, "ablation", row.ablation)
        prediction.insert(2, "feature_set", row.feature_set)
        frames.append(prediction)
    return pd.concat(frames, ignore_index=True)


def add_prediction_context(
    predictions: pd.DataFrame,
    *,
    validation_features: pd.DataFrame,
    epoch_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach validation feature labels and optional epoch metadata."""
    keys = ["participant_id", "epoch_id", "split"]
    feature_context = validation_features[[*keys, "label"]].rename(
        columns={"label": "feature_label"}
    )
    output = predictions.merge(
        feature_context, on=keys, how="left", validate="many_to_one"
    )
    missing_labels = output["feature_label"].isna().sum()
    if missing_labels:
        raise ValueError(
            "Prediction rows did not match validation feature rows: "
            f"{missing_labels} row(s) missing."
        )
    mismatched = output[output["true_label"] != output["feature_label"]]
    if not mismatched.empty:
        raise ValueError(
            "Prediction true_label does not match validation feature label for "
            f"{len(mismatched)} row(s)."
        )

    if epoch_index is None:
        return output

    metadata_columns = [
        column
        for column in (
            "participant_id",
            "epoch_id",
            "split",
            "start_time",
            "end_time",
            "start_row",
            "end_row",
            "segment_id",
            "segment_transition_count",
            "segment_transition_agreement",
            "segmentation_reason",
        )
        if column in epoch_index.columns
    ]
    if set(keys).issubset(metadata_columns):
        metadata = epoch_index[metadata_columns].copy()
        metadata = metadata[metadata["split"] == "validation"]
        output = output.merge(metadata, on=keys, how="left", validate="many_to_one")
    return output


def add_transition_context(predictions: pd.DataFrame) -> pd.DataFrame:
    """Annotate rows with participant-contained sleep-stage transition proximity."""
    output = predictions.sort_values(
        ["ablation", "model", "participant_id", "epoch_id"]
    ).copy()
    pieces = []
    for (_, _), group in output.groupby(["ablation", "model"], sort=False):
        group = group.copy()
        group["previous_true_label"] = group.groupby("participant_id")[
            "true_label"
        ].shift(1)
        group["next_true_label"] = group.groupby("participant_id")[
            "true_label"
        ].shift(-1)
        group["is_transition_epoch"] = (
            (group["previous_true_label"].notna())
            & (group["previous_true_label"] != group["true_label"])
        ) | (
            (group["next_true_label"].notna())
            & (group["next_true_label"] != group["true_label"])
        )
        group["distance_to_transition_epochs"] = _transition_distances(group)
        group["transition_distance_bin"] = group["distance_to_transition_epochs"].map(
            transition_distance_bin
        )
        group["transition_distance_bin"] = pd.Categorical(
            group["transition_distance_bin"],
            categories=TRANSITION_BIN_ORDER,
            ordered=True,
        )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def transition_distance_bin(distance: object) -> str:
    """Return the reporting bin for a distance to nearest transition epoch."""
    if pd.isna(distance):
        return "no_transition_in_participant"
    value = int(distance)
    if value <= 1:
        return str(value)
    if value <= 3:
        return "2-3"
    if value <= 10:
        return "4-10"
    return ">10"


def per_class_metrics_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one precision/recall/F1 row per class and selected model."""
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(["ablation", "model", "feature_set"]):
        ablation, model, feature_set = keys
        precision, recall, f1, support = precision_recall_fscore_support(
            group["true_label"],
            group["pred_label"],
            labels=list(TARGET_LABELS),
            zero_division=0,
        )
        for label, p_value, r_value, f_value, support_value in zip(
            TARGET_LABELS, precision, recall, f1, support, strict=True
        ):
            rows.append(
                {
                    "ablation": ablation,
                    "model": model,
                    "feature_set": feature_set,
                    "label": label,
                    "precision": float(p_value),
                    "recall": float(r_value),
                    "f1": float(f_value),
                    "support": int(support_value),
                }
            )
    return pd.DataFrame(rows)


def per_participant_metrics_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return participant-level validation metrics for selected models."""
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(
        ["ablation", "model", "feature_set", "participant_id"]
    ):
        ablation, model, feature_set, participant_id = keys
        metrics = classification_metrics(
            group["true_label"], group["pred_label"], labels=TARGET_LABELS
        )
        rows.append(
            {
                "ablation": ablation,
                "model": model,
                "feature_set": feature_set,
                "participant_id": participant_id,
                "n_epochs": len(group),
                "REM_support": int((group["true_label"] == "REM").sum()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def transition_metrics_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return validation metrics by distance to the nearest true-label transition."""
    rows: list[dict[str, object]] = []
    grouped = predictions.groupby(
        ["ablation", "model", "feature_set", "transition_distance_bin"],
        observed=False,
    )
    for keys, group in grouped:
        if group.empty:
            continue
        ablation, model, feature_set, transition_bin = keys
        metrics = classification_metrics(
            group["true_label"], group["pred_label"], labels=TARGET_LABELS
        )
        rows.append(
            {
                "ablation": ablation,
                "model": model,
                "feature_set": feature_set,
                "transition_distance_bin": transition_bin,
                "n_epochs": len(group),
                "REM_support": int((group["true_label"] == "REM").sum()),
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["transition_distance_bin"] = pd.Categorical(
            frame["transition_distance_bin"],
            categories=TRANSITION_BIN_ORDER,
            ordered=True,
        )
        frame = frame.sort_values(
            ["ablation", "model", "transition_distance_bin"]
        ).reset_index(drop=True)
    return frame


def model_family_comparison_frame(selected_models: pd.DataFrame) -> pd.DataFrame:
    """Return compact model-family comparison rows."""
    columns = [
        "ablation",
        "feature_set",
        "model",
        "n_features",
        "validation_macro_f1",
        "validation_accuracy",
        "best_cv_macro_f1",
        "elapsed_seconds",
        "prediction_path",
        "confusion_path",
        "model_path",
    ]
    available = [column for column in columns if column in selected_models.columns]
    output = selected_models[available].copy()
    numeric_columns = [
        "n_features",
        "validation_macro_f1",
        "validation_accuracy",
        "best_cv_macro_f1",
        "elapsed_seconds",
    ]
    for column in numeric_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.sort_values(["ablation", "model"]).reset_index(drop=True)


def feature_importance_frame(
    *,
    selected_models: pd.DataFrame,
    validation_features: pd.DataFrame,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract model-native feature importances where available."""
    rows: list[dict[str, object]] = []
    manifest_lookup = _manifest_lookup(manifest)
    feature_sets = _selected_features_by_ablation(selected_models)
    for row in selected_models.itertuples(index=False):
        fitted = joblib.load(row.model_path)
        features = feature_sets.get(row.ablation) or _feature_columns(
            validation_features
        )
        importance = _native_feature_importance(fitted, features, row.model)
        for rank, (feature, value) in enumerate(
            sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True),
            start=1,
        ):
            metadata = manifest_lookup.get(feature, {})
            rows.append(
                {
                    "ablation": row.ablation,
                    "model": row.model,
                    "feature": feature,
                    "importance": float(value),
                    "abs_importance": float(abs(value)),
                    "rank": rank,
                    **metadata,
                }
            )
    return pd.DataFrame(rows)


def permutation_importance_frame(
    *,
    selected_models: pd.DataFrame,
    validation_features: pd.DataFrame,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute validation-only permutation importance for selected models."""
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")

    rows: list[dict[str, object]] = []
    feature_sets = _selected_features_by_ablation(selected_models)
    for row in selected_models.itertuples(index=False):
        features = feature_sets.get(row.ablation) or _feature_columns(
            validation_features
        )
        fitted = joblib.load(row.model_path)
        estimator = _prediction_estimator(fitted)
        result = permutation_importance(
            estimator,
            validation_features[features],
            validation_features["label"],
            scoring="f1_macro",
            n_repeats=n_repeats,
            random_state=random_state,
        )
        order = np.argsort(np.abs(result.importances_mean))[::-1]
        for rank, index in enumerate(order, start=1):
            rows.append(
                {
                    "ablation": row.ablation,
                    "model": row.model,
                    "feature": features[index],
                    "importance_mean": float(result.importances_mean[index]),
                    "importance_std": float(result.importances_std[index]),
                    "abs_importance_mean": float(abs(result.importances_mean[index])),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def create_error_analysis_plots(
    *,
    selected_models: pd.DataFrame,
    predictions: pd.DataFrame,
    per_class_metrics: pd.DataFrame,
    per_participant_metrics: pd.DataFrame,
    transition_metrics: pd.DataFrame,
    model_family_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    permutation_importance: pd.DataFrame,
    validation_features: pd.DataFrame,
    figures_dir: Path,
    compute_shap: bool,
    max_shap_rows: int,
) -> list[dict[str, object]]:
    """Create validation error-analysis figures and return artifact rows."""
    rows: list[dict[str, object]] = []
    comparison_path = figures_dir / "model_family_macro_f1.png"
    save_model_family_comparison_plot(
        model_family_comparison,
        comparison_path,
        metric="validation_macro_f1",
    )
    rows.append(_artifact_row(comparison_path, "figure"))

    feature_sets = _selected_features_by_ablation(selected_models)
    for row in selected_models.itertuples(index=False):
        subset = predictions[
            (predictions["ablation"] == row.ablation)
            & (predictions["model"] == row.model)
        ]
        stem = f"{row.ablation}_{row.model}"

        confusion = confusion_matrix_frame(
            subset["true_label"], subset["pred_label"], labels=TARGET_LABELS
        )
        confusion_path = figures_dir / f"{stem}_confusion_counts.png"
        save_confusion_matrix_plot(confusion, confusion_path)
        rows.append(_artifact_row(confusion_path, "figure", row.ablation, row.model))

        normalized_confusion = confusion.div(confusion.sum(axis=1), axis=0).fillna(0.0)
        normalized_path = figures_dir / f"{stem}_confusion_true_normalized.png"
        save_confusion_matrix_plot(
            normalized_confusion,
            normalized_path,
            fmt=".2f",
            cmap="YlGnBu",
        )
        rows.append(_artifact_row(normalized_path, "figure", row.ablation, row.model))

        per_class_path = figures_dir / f"{stem}_per_class_metrics.png"
        save_per_class_metric_plot(
            per_class_metrics[
                (per_class_metrics["ablation"] == row.ablation)
                & (per_class_metrics["model"] == row.model)
            ],
            per_class_path,
        )
        rows.append(_artifact_row(per_class_path, "figure", row.ablation, row.model))

        participant_path = figures_dir / f"{stem}_participant_macro_f1.png"
        save_participant_metric_plot(
            per_participant_metrics[
                (per_participant_metrics["ablation"] == row.ablation)
                & (per_participant_metrics["model"] == row.model)
            ],
            participant_path,
            metric="macro_f1",
        )
        rows.append(_artifact_row(participant_path, "figure", row.ablation, row.model))

        transition_path = figures_dir / f"{stem}_transition_macro_f1.png"
        save_transition_metric_plot(
            transition_metrics[
                (transition_metrics["ablation"] == row.ablation)
                & (transition_metrics["model"] == row.model)
            ],
            transition_path,
            metric="macro_f1",
        )
        rows.append(_artifact_row(transition_path, "figure", row.ablation, row.model))

        importance_subset = feature_importance[
            (feature_importance["ablation"] == row.ablation)
            & (feature_importance["model"] == row.model)
        ]
        if not importance_subset.empty:
            importance_path = figures_dir / f"{stem}_feature_importance.png"
            save_feature_importance_plot(
                importance_subset,
                importance_path,
                value_column="abs_importance",
                title="Native feature importance",
            )
            rows.append(
                _artifact_row(importance_path, "figure", row.ablation, row.model)
            )

        permutation_subset = permutation_importance[
            (permutation_importance.get("ablation") == row.ablation)
            & (permutation_importance.get("model") == row.model)
        ] if not permutation_importance.empty else pd.DataFrame()
        if not permutation_subset.empty:
            permutation_path = figures_dir / f"{stem}_permutation_importance.png"
            save_feature_importance_plot(
                permutation_subset,
                permutation_path,
                value_column="abs_importance_mean",
                title="Validation permutation importance",
            )
            rows.append(
                _artifact_row(permutation_path, "figure", row.ablation, row.model)
            )

        if compute_shap and row.model == "xgboost":
            shap_path = figures_dir / f"{stem}_xgboost_shap_summary.png"
            fitted = joblib.load(row.model_path)
            estimator = _xgboost_estimator(fitted)
            features = feature_sets.get(row.ablation) or _feature_columns(
                validation_features
            )
            shap_sample = validation_features[features].head(max_shap_rows)
            try:
                save_xgboost_shap_summary(estimator, shap_sample, shap_path)
            except ImportError:
                continue
            rows.append(_artifact_row(shap_path, "figure", row.ablation, row.model))
    return rows


def _resolve_default_models(
    run_config: dict[str, Any], models: Sequence[str] | None
) -> tuple[str, ...]:
    if models is not None:
        return tuple(models)
    if run_config.get("include_xgboost", True):
        return DEFAULT_ERROR_ANALYSIS_MODELS
    return tuple(model for model in DEFAULT_ERROR_ANALYSIS_MODELS if model != "xgboost")


def _absolutize_artifact_paths(frame: pd.DataFrame, run_root: Path) -> pd.DataFrame:
    output = frame.copy()
    for column in ("prediction_path", "confusion_path", "model_path"):
        output[column] = output[column].map(
            lambda value: str(_resolve_path(value, run_root))
        )
    return output


def _attach_feature_set_metadata(frame: pd.DataFrame, run_root: Path) -> pd.DataFrame:
    feature_sets_path = run_root / "metrics" / "ablation_feature_sets.csv"
    if not feature_sets_path.exists():
        return frame
    feature_sets = pd.read_csv(feature_sets_path)
    metadata_columns = [
        column
        for column in (
            "ablation",
            "description",
            "selected_feature_families",
            "selected_signal_groups",
            "selected_source_signals",
            "selected_features",
        )
        if column in feature_sets.columns
    ]
    if "ablation" not in metadata_columns:
        return frame
    return frame.merge(
        feature_sets[metadata_columns].drop_duplicates("ablation"),
        on="ablation",
        how="left",
    )


def _resolve_path(value: object, run_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = run_root / path
    if candidate.exists():
        return candidate
    return path


def _validate_selected_artifact_paths(selected: pd.DataFrame) -> None:
    for column in ("prediction_path", "confusion_path", "model_path"):
        missing = [
            path for path in selected[column].astype(str).map(Path) if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Selected model run artifact path(s) do not exist for {column}: "
                f"{missing}"
            )


def _transition_distances(group: pd.DataFrame) -> pd.Series:
    distances = pd.Series(pd.NA, index=group.index, dtype="Int64")
    for _, participant in group.groupby("participant_id", sort=False):
        ordered = participant.sort_values("epoch_id")
        transition_positions = np.flatnonzero(ordered["is_transition_epoch"].to_numpy())
        if len(transition_positions) == 0:
            continue
        positions = np.arange(len(ordered))
        distance_values = np.min(
            np.abs(positions[:, np.newaxis] - transition_positions[np.newaxis, :]),
            axis=1,
        )
        distances.loc[ordered.index] = distance_values
    return distances


def _selected_features_by_ablation(
    selected_models: pd.DataFrame,
) -> dict[str, list[str]]:
    if "selected_features" not in selected_models.columns:
        return {}
    output: dict[str, list[str]] = {}
    for ablation, group in selected_models.groupby("ablation"):
        value = group["selected_features"].dropna()
        if not value.empty:
            output[str(ablation)] = str(value.iloc[0]).split("|")
    return output


def _feature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column not in FEATURE_ID_COLUMNS]


def _native_feature_importance(
    fitted: Any, features: Sequence[str], model_name: str
) -> dict[str, float]:
    estimator = fitted.get("estimator") if isinstance(fitted, dict) else fitted
    if model_name == "elastic_net_logistic_regression" and hasattr(
        estimator, "named_steps"
    ):
        classifier = estimator.named_steps["classifier"]
        coefficients = np.asarray(classifier.coef_)
        values = np.mean(np.abs(coefficients), axis=0)
        return dict(zip(features, values, strict=True))
    if hasattr(estimator, "named_steps"):
        classifier = estimator.named_steps.get("classifier")
    else:
        classifier = estimator
    if classifier is not None and hasattr(classifier, "feature_importances_"):
        return dict(zip(features, classifier.feature_importances_, strict=True))
    return {}


def _prediction_estimator(fitted: Any) -> Any:
    if isinstance(fitted, dict) and {"estimator", "label_encoder"}.issubset(fitted):
        return _EncodedLabelEstimator(fitted["estimator"], fitted["label_encoder"])
    return fitted


def _xgboost_estimator(fitted: Any) -> Any:
    if isinstance(fitted, dict) and "estimator" in fitted:
        return fitted["estimator"]
    return fitted


class _EncodedLabelEstimator:
    """Adapter exposing decoded labels for encoded XGBoost validation scoring."""

    def __init__(self, estimator: Any, label_encoder: Any) -> None:
        self.estimator = estimator
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self.estimator.predict(X)
        return self.label_encoder.inverse_transform(encoded.astype(int))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)


def _manifest_lookup(manifest: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if manifest is None or manifest.empty or "feature" not in manifest.columns:
        return {}
    metadata_columns = [
        column
        for column in ("feature_family", "signal_group", "source_signal")
        if column in manifest.columns
    ]
    rows = {}
    for row in manifest[["feature", *metadata_columns]].itertuples(index=False):
        feature = row[0]
        rows[str(feature)] = {
            column: getattr(row, column) for column in metadata_columns
        }
    return rows


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    artifact_rows: list[dict[str, object]],
    artifact_type: str,
) -> None:
    frame.to_csv(path, index=False)
    artifact_rows.append(_artifact_row(path, artifact_type))


def _artifact_row(
    path: Path,
    artifact_type: str,
    ablation: str | None = None,
    model: str | None = None,
) -> dict[str, object]:
    return {
        "artifact_type": artifact_type,
        "ablation": ablation,
        "model": model,
        "path": str(path),
    }
