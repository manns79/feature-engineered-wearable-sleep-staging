"""Interpretation artifacts for the finalized rolling logistic model."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.error_analysis import (
    add_prediction_context,
    add_transition_context,
    load_epoch_index_if_present,
    load_manifest_if_present,
    load_prediction_file,
    load_split_features,
    per_participant_metrics_frame,
    transition_metrics_frame,
)
from src.visualization.plots import (
    save_participant_metric_plot,
    save_signed_feature_contrast_plot,
    save_transition_metric_plot,
)

DEFAULT_CONTRASTS = (
    ("REM", "Non-REM"),
    ("Wake", "Non-REM"),
    ("REM", "Wake"),
)
DEFAULT_PREDICTION_VARIANTS = (
    "raw",
    "raw_threshold_tuned",
    "platt_threshold_tuned",
    "platt_smoothed_threshold_tuned",
)
ROLLING_CANDIDATE = "interpretable_rolling_logistic"


@dataclass(frozen=True)
class RollingLogisticInterpretationOutputs:
    """Paths written by rolling logistic interpretation."""

    analysis_dir: Path
    metrics_dir: Path
    figures_dir: Path
    config_path: Path
    artifact_index_path: Path
    coefficient_path: Path
    contrast_path: Path
    contrast_summary_path: Path
    rolling_predictions_path: Path
    participant_metrics_path: Path
    transition_metrics_path: Path
    rem_error_predictions_path: Path
    rem_error_counts_path: Path
    rem_error_feature_summary_path: Path
    rem_error_transition_summary_path: Path
    artifact_index: pd.DataFrame


def run_rolling_logistic_interpretation(
    *,
    rolling_run_dir: str | Path = (
        "outputs/runs/rolling_logistic_train_oof_postprocessed_20260724"
    ),
    predictions_path: str | Path = (
        "outputs/runs/final_test_evaluation/locked_test_predictions.csv"
    ),
    features_path: str | Path = "data/processed/features_test.csv",
    split: str = "test",
    manifest_path: str | Path | None = "data/processed/feature_manifest.csv",
    epoch_index_path: str | Path | None = "data/interim/epoch_index.csv",
    output_dir: str | Path = (
        "outputs/runs/final_test_evaluation/rolling_logistic_interpretation"
    ),
    contrasts: Sequence[tuple[str, str]] = DEFAULT_CONTRASTS,
    prediction_variants: Sequence[str] = DEFAULT_PREDICTION_VARIANTS,
    rem_error_variant: str = "raw",
    create_plots: bool = True,
    max_plot_features: int = 20,
) -> RollingLogisticInterpretationOutputs:
    """Generate coefficient and error-analysis artifacts for rolling logistic."""
    rolling_root = Path(rolling_run_dir)
    analysis_dir = Path(output_dir)
    metrics_dir = analysis_dir / "metrics"
    figures_dir = analysis_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest_if_present(manifest_path)
    selected_features = load_selected_features(rolling_root)
    coefficients = coefficient_frame(
        model_path=rolling_root / "models" / "rolling_logistic_model.joblib",
        selected_features=selected_features,
        manifest=manifest,
    )
    contrast_table = coefficient_contrast_frame(coefficients, contrasts=contrasts)
    contrast_summary = contrast_summary_frame(contrast_table)

    split_features = load_split_features(features_path, expected_split=split)
    epoch_index = load_epoch_index_if_present(epoch_index_path)
    rolling_predictions = load_rolling_predictions(
        predictions_path,
        split=split,
        variants=tuple(prediction_variants),
    )
    rolling_predictions = add_prediction_context(
        rolling_predictions,
        validation_features=split_features,
        epoch_index=epoch_index,
    )
    rolling_predictions = add_transition_context(rolling_predictions)
    participant_metrics = per_participant_metrics_frame(rolling_predictions)
    transition_metrics = transition_metrics_frame(rolling_predictions)

    rem_error_predictions = rem_error_prediction_frame(
        rolling_predictions,
        split_features=split_features,
        selected_features=selected_features,
        variant=rem_error_variant,
    )
    rem_error_counts = rem_error_counts_frame(rem_error_predictions)
    rem_error_feature_summary = rem_error_feature_summary_frame(
        rem_error_predictions,
        selected_features=selected_features,
        manifest=manifest,
    )
    rem_error_transition_summary = rem_error_transition_summary_frame(
        rem_error_predictions
    )

    artifact_rows: list[dict[str, object]] = []
    coefficient_path = metrics_dir / "standardized_coefficients.csv"
    contrast_path = metrics_dir / "coefficient_contrasts.csv"
    contrast_summary_path = metrics_dir / "coefficient_contrast_summary.csv"
    rolling_predictions_path = metrics_dir / "rolling_predictions_with_context.csv"
    participant_metrics_path = metrics_dir / "per_participant_metrics.csv"
    transition_metrics_path = metrics_dir / "transition_distance_metrics.csv"
    rem_error_predictions_path = metrics_dir / "rem_error_predictions.csv"
    rem_error_counts_path = metrics_dir / "rem_error_counts.csv"
    rem_error_feature_summary_path = metrics_dir / "rem_error_feature_summary.csv"
    rem_error_transition_summary_path = metrics_dir / "rem_error_transition_summary.csv"
    config_path = analysis_dir / "rolling_logistic_interpretation_config.json"
    artifact_index_path = analysis_dir / "artifact_index.csv"

    write_csv(coefficients, coefficient_path, artifact_rows, "metrics")
    write_csv(contrast_table, contrast_path, artifact_rows, "metrics")
    write_csv(contrast_summary, contrast_summary_path, artifact_rows, "metrics")
    write_csv(rolling_predictions, rolling_predictions_path, artifact_rows, "metrics")
    write_csv(participant_metrics, participant_metrics_path, artifact_rows, "metrics")
    write_csv(transition_metrics, transition_metrics_path, artifact_rows, "metrics")
    write_csv(
        rem_error_predictions,
        rem_error_predictions_path,
        artifact_rows,
        "metrics",
    )
    write_csv(rem_error_counts, rem_error_counts_path, artifact_rows, "metrics")
    write_csv(
        rem_error_feature_summary,
        rem_error_feature_summary_path,
        artifact_rows,
        "metrics",
    )
    write_csv(
        rem_error_transition_summary,
        rem_error_transition_summary_path,
        artifact_rows,
        "metrics",
    )

    if create_plots:
        artifact_rows.extend(
            create_interpretation_plots(
                contrast_table=contrast_table,
                participant_metrics=participant_metrics,
                transition_metrics=transition_metrics,
                figures_dir=figures_dir,
                max_plot_features=max_plot_features,
            )
        )

    config = {
        "rolling_run_dir": str(rolling_root),
        "predictions_path": str(predictions_path),
        "features_path": str(features_path),
        "split": split,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "epoch_index_path": str(epoch_index_path) if epoch_index_path else None,
        "output_dir": str(analysis_dir),
        "contrasts": [list(contrast) for contrast in contrasts],
        "prediction_variants": list(prediction_variants),
        "rem_error_variant": rem_error_variant,
        "create_plots": create_plots,
        "interpretation_note": (
            "Coefficients are interpreted for the raw rolling logistic model. "
            "Post-processing variants are summarized as operating-point changes."
        ),
    }
    config_path.write_text(json.dumps(config, indent=2, default=str) + "\n")
    artifact_rows.append(artifact_row(config_path, "config"))
    artifact_index = pd.DataFrame(artifact_rows)
    artifact_index.to_csv(artifact_index_path, index=False)

    return RollingLogisticInterpretationOutputs(
        analysis_dir=analysis_dir,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        config_path=config_path,
        artifact_index_path=artifact_index_path,
        coefficient_path=coefficient_path,
        contrast_path=contrast_path,
        contrast_summary_path=contrast_summary_path,
        rolling_predictions_path=rolling_predictions_path,
        participant_metrics_path=participant_metrics_path,
        transition_metrics_path=transition_metrics_path,
        rem_error_predictions_path=rem_error_predictions_path,
        rem_error_counts_path=rem_error_counts_path,
        rem_error_feature_summary_path=rem_error_feature_summary_path,
        rem_error_transition_summary_path=rem_error_transition_summary_path,
        artifact_index=artifact_index,
    )


def load_selected_features(rolling_run_dir: str | Path) -> list[str]:
    """Load the frozen rolling-logistic feature list."""
    path = Path(rolling_run_dir) / "metrics" / "selected_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Selected feature CSV does not exist: {path}")
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        raise ValueError(f"{path} is missing required column: feature")
    features = frame["feature"].astype(str).tolist()
    if not features:
        raise ValueError(f"{path} does not contain any selected features.")
    return features


def coefficient_frame(
    *,
    model_path: str | Path,
    selected_features: Sequence[str],
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one standardized coefficient row per class and feature."""
    fitted = joblib.load(model_path)
    classifier = logistic_classifier(fitted)
    coefficients = np.asarray(classifier.coef_, dtype=float)
    classes = [str(label) for label in classifier.classes_]
    if coefficients.shape != (len(classes), len(selected_features)):
        raise ValueError(
            "Coefficient shape does not match class and selected-feature counts."
        )

    metadata = manifest_metadata(manifest)
    rows: list[dict[str, object]] = []
    for class_index, label in enumerate(classes):
        intercept = float(classifier.intercept_[class_index])
        for feature_index, feature in enumerate(selected_features):
            rows.append(
                {
                    "class_label": label,
                    "feature": feature,
                    "coefficient": float(coefficients[class_index, feature_index]),
                    "abs_coefficient": float(
                        abs(coefficients[class_index, feature_index])
                    ),
                    "class_intercept": intercept,
                    **metadata.get(str(feature), {}),
                }
            )
    output = pd.DataFrame(rows)
    output["class_rank"] = output.groupby("class_label")["abs_coefficient"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    return output.sort_values(["class_label", "class_rank"]).reset_index(drop=True)


def coefficient_contrast_frame(
    coefficients: pd.DataFrame,
    *,
    contrasts: Sequence[tuple[str, str]] = DEFAULT_CONTRASTS,
) -> pd.DataFrame:
    """Return signed class-contrast coefficients."""
    required = {"class_label", "feature", "coefficient"}
    missing = sorted(required - set(coefficients.columns))
    if missing:
        raise ValueError(f"Coefficient frame is missing column(s): {missing}")

    index = coefficients.set_index(["class_label", "feature"])
    features = coefficients["feature"].drop_duplicates().tolist()
    metadata_columns = [
        column
        for column in ("feature_family", "signal_group", "source_signal")
        if column in coefficients.columns
    ]
    metadata = (
        coefficients.drop_duplicates("feature")
        .set_index("feature")[metadata_columns]
        .to_dict("index")
        if metadata_columns
        else {}
    )

    rows: list[dict[str, object]] = []
    for positive_class, negative_class in contrasts:
        for feature in features:
            positive = float(index.loc[(positive_class, feature), "coefficient"])
            negative = float(index.loc[(negative_class, feature), "coefficient"])
            contrast = positive - negative
            rows.append(
                {
                    "contrast": f"{positive_class}_vs_{negative_class}",
                    "positive_class": positive_class,
                    "negative_class": negative_class,
                    "feature": feature,
                    "positive_coefficient": positive,
                    "negative_coefficient": negative,
                    "contrast_coefficient": contrast,
                    "abs_contrast": abs(contrast),
                    **metadata.get(str(feature), {}),
                }
            )
    output = pd.DataFrame(rows)
    output["rank"] = output.groupby("contrast")["abs_contrast"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    return output.sort_values(["contrast", "rank"]).reset_index(drop=True)


def contrast_summary_frame(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Summarize coefficient contrast magnitude by feature metadata."""
    group_columns = [
        column
        for column in ("contrast", "feature_family", "signal_group", "source_signal")
        if column in contrasts.columns
    ]
    rows: list[dict[str, object]] = []
    for keys, group in contrasts.groupby(group_columns, dropna=False):
        key_values = (keys,) if len(group_columns) == 1 else keys
        top = group.sort_values("abs_contrast", ascending=False).iloc[0]
        rows.append(
            {
                **dict(zip(group_columns, key_values, strict=True)),
                "n_features": len(group),
                "mean_abs_contrast": float(group["abs_contrast"].mean()),
                "sum_abs_contrast": float(group["abs_contrast"].sum()),
                "top_feature": top["feature"],
                "top_abs_contrast": float(top["abs_contrast"]),
                "top_signed_contrast": float(top["contrast_coefficient"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["contrast", "sum_abs_contrast"], ascending=[True, False]
    )


def load_rolling_predictions(
    path: str | Path,
    *,
    split: str,
    variants: tuple[str, ...],
) -> pd.DataFrame:
    """Load rolling-logistic prediction variants from final prediction artifacts."""
    predictions = load_prediction_file(path, expected_split=split)
    required = {"candidate", "variant"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing column(s): {missing}")
    selected = predictions[
        (predictions["candidate"] == ROLLING_CANDIDATE)
        & (predictions["variant"].isin(variants))
    ].copy()
    if selected.empty:
        raise ValueError(
            "No rolling-logistic prediction rows matched "
            f"candidate={ROLLING_CANDIDATE!r} and variants={variants}."
        )
    return selected.reset_index(drop=True)


def rem_error_prediction_frame(
    predictions: pd.DataFrame,
    *,
    split_features: pd.DataFrame,
    selected_features: Sequence[str],
    variant: str,
) -> pd.DataFrame:
    """Return one prediction frame with REM-focused error groups and features."""
    subset = predictions[predictions["variant"] == variant].copy()
    if subset.empty:
        raise ValueError(f"No prediction rows matched variant={variant!r}.")
    keys = ["participant_id", "epoch_id", "split"]
    feature_columns = [*keys, *selected_features]
    missing = sorted(set(feature_columns) - set(split_features.columns))
    if missing:
        raise ValueError(f"Feature table is missing column(s): {missing}")
    subset = subset.merge(
        split_features[feature_columns],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    subset["rem_error_group"] = [
        rem_error_group(true, pred)
        for true, pred in zip(subset["true_label"], subset["pred_label"], strict=True)
    ]
    return subset


def rem_error_group(true_label: object, pred_label: object) -> str:
    """Return a REM-focused error group for one epoch."""
    true_text = str(true_label)
    pred_text = str(pred_label)
    if true_text == "REM" and pred_text == "REM":
        return "true_REM_pred_REM"
    if true_text == "REM":
        return f"true_REM_pred_{safe_label(pred_text)}"
    if pred_text == "REM":
        return f"true_{safe_label(true_text)}_pred_REM"
    return "not_REM_related"


def rem_error_counts_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Count REM error groups overall and by transition-distance bin."""
    group_columns = ["rem_error_group"]
    if "transition_distance_bin" in predictions.columns:
        group_columns.append("transition_distance_bin")
    counts = (
        predictions.groupby(group_columns, observed=False)
        .size()
        .reset_index(name="n_epochs")
    )
    total = len(predictions)
    counts["epoch_fraction"] = counts["n_epochs"] / total if total else 0.0
    return counts


def rem_error_feature_summary_frame(
    predictions: pd.DataFrame,
    *,
    selected_features: Sequence[str],
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize selected-feature distributions by REM error group."""
    metadata = manifest_metadata(manifest)
    rows: list[dict[str, object]] = []
    for group_name, group in predictions.groupby("rem_error_group"):
        for feature in selected_features:
            values = pd.to_numeric(group[feature], errors="coerce")
            rows.append(
                {
                    "rem_error_group": group_name,
                    "feature": feature,
                    "n_epochs": len(group),
                    "n_observed": int(values.notna().sum()),
                    "missing_rate": float(values.isna().mean()),
                    "mean": float(values.mean()) if values.notna().any() else np.nan,
                    "std": float(values.std()) if values.notna().sum() > 1 else np.nan,
                    "median": (
                        float(values.median()) if values.notna().any() else np.nan
                    ),
                    "q25": (
                        float(values.quantile(0.25))
                        if values.notna().any()
                        else np.nan
                    ),
                    "q75": (
                        float(values.quantile(0.75))
                        if values.notna().any()
                        else np.nan
                    ),
                    **metadata.get(str(feature), {}),
                }
            )
    return pd.DataFrame(rows)


def rem_error_transition_summary_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize REM error groups by transition proximity."""
    if "transition_distance_bin" not in predictions.columns:
        return pd.DataFrame()
    return (
        predictions.groupby(
            ["rem_error_group", "transition_distance_bin"],
            observed=False,
        )
        .size()
        .reset_index(name="n_epochs")
    )


def create_interpretation_plots(
    *,
    contrast_table: pd.DataFrame,
    participant_metrics: pd.DataFrame,
    transition_metrics: pd.DataFrame,
    figures_dir: Path,
    max_plot_features: int,
) -> list[dict[str, object]]:
    """Create compact interpretation figures."""
    rows: list[dict[str, object]] = []
    for contrast, subset in contrast_table.groupby("contrast"):
        path = figures_dir / f"{safe_path_name(contrast)}_coefficient_contrast.png"
        save_signed_feature_contrast_plot(
            subset,
            path,
            title=contrast.replace("_", " "),
            max_features=max_plot_features,
        )
        rows.append(artifact_row(path, "figure"))

    for variant, subset in participant_metrics.groupby("variant"):
        path = figures_dir / f"{safe_path_name(variant)}_participant_macro_f1.png"
        save_participant_metric_plot(subset, path, metric="macro_f1")
        rows.append(artifact_row(path, "figure"))

    for variant, subset in transition_metrics.groupby("variant"):
        path = figures_dir / f"{safe_path_name(variant)}_transition_macro_f1.png"
        save_transition_metric_plot(subset, path, metric="macro_f1")
        rows.append(artifact_row(path, "figure"))
    return rows


def logistic_classifier(fitted: Any) -> Any:
    """Return the logistic-regression classifier from a fitted pipeline."""
    estimator = fitted.get("estimator") if isinstance(fitted, dict) else fitted
    if not hasattr(estimator, "named_steps"):
        raise ValueError("Rolling logistic model must be a fitted sklearn Pipeline.")
    if "classifier" not in estimator.named_steps:
        raise ValueError("Rolling logistic pipeline is missing a classifier step.")
    classifier = estimator.named_steps["classifier"]
    for attribute in ("coef_", "classes_", "intercept_"):
        if not hasattr(classifier, attribute):
            raise ValueError(f"Classifier is missing fitted attribute: {attribute}")
    return classifier


def manifest_metadata(manifest: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    """Return feature metadata keyed by feature name."""
    if manifest is None or manifest.empty or "feature" not in manifest.columns:
        return {}
    metadata_columns = [
        column
        for column in ("feature_family", "signal_group", "source_signal")
        if column in manifest.columns
    ]
    if not metadata_columns:
        return {}
    frame = manifest[["feature", *metadata_columns]].drop_duplicates("feature")
    return frame.set_index("feature")[metadata_columns].to_dict("index")


def write_csv(
    frame: pd.DataFrame,
    path: Path,
    artifact_rows: list[dict[str, object]],
    artifact_type: str,
) -> None:
    """Write a CSV and add it to the artifact index."""
    frame.to_csv(path, index=False)
    artifact_rows.append(artifact_row(path, artifact_type))


def artifact_row(path: Path, artifact_type: str) -> dict[str, object]:
    """Return one artifact-index row."""
    return {"artifact_type": artifact_type, "path": str(path)}


def safe_label(value: object) -> str:
    """Return a label safe for compact categorical identifiers."""
    return str(value).replace("-", "_").replace(" ", "_")


def safe_path_name(value: object) -> str:
    """Return a path-safe identifier."""
    return "".join(
        character if character.isalnum() else "_" for character in str(value)
    ).strip("_")
