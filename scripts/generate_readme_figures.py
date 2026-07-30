"""Curate README summary CSVs and figures from saved experiment artifacts."""

# ruff: noqa: E402, I001

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary"
FIGURES = RESULTS / "figures"

FINAL = ROOT / "outputs" / "runs" / "final_test_evaluation"
ABLATION = ROOT / "outputs" / "runs" / "full_ablation_20260718"
ROLLING = FINAL / "rolling_logistic_interpretation"
PREVIOUS = ROOT.parent / "dreamt-wearable-sleep-staging" / "results" / "summary"


DISPLAY_NAMES = {
    ("majority_class_baseline", "raw"): "Majority-class baseline",
    ("stratified_class_baseline", "raw"): "Stratified-random baseline",
    (
        "statistical_summary_only",
        "platt_smoothed_threshold_tuned",
    ): "Statistical-summary elastic-net logistic, post-processed",
    (
        "best_original_ablation",
        "platt_smoothed_threshold_tuned",
    ): "Engineered-feature elastic-net logistic, post-processed",
    ("interpretable_rolling_logistic", "raw"): "Interpretable rolling logistic, raw",
    (
        "interpretable_rolling_logistic",
        "raw_threshold_tuned",
    ): "Interpretable rolling logistic, threshold-tuned",
}


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    key_results = write_key_results()
    write_ablation_summary()
    write_interpretation_summary()
    make_postprocessing_tradeoff_figure()
    make_transition_distance_figure()
    make_coefficient_contrast_figure()
    write_artifact_manifest(key_results)


def write_key_results() -> pd.DataFrame:
    comparison = pd.read_csv(FINAL / "final_comparison_table.csv")
    previous = pd.read_csv(PREVIOUS / "key_results.csv")
    rows: list[dict[str, object]] = []

    for order, ((candidate, variant), display_name) in enumerate(
        DISPLAY_NAMES.items(), start=1
    ):
        match = comparison[
            (comparison["candidate"] == candidate) & (comparison["variant"] == variant)
        ]
        if match.empty:
            raise ValueError(f"Missing final result for {candidate}:{variant}")
        row = match.iloc[0]
        rows.append(
            {
                "readme_order": order,
                "readme_display_name": display_name,
                "source_project": "feature-engineered-wearable-sleep-staging",
                "source_artifact": (
                    "outputs/runs/final_test_evaluation/"
                    "final_comparison_table.csv"
                ),
                "candidate": candidate,
                "variant": variant,
                "validation_macro_f1": row["validation_macro_f1"],
                "test_macro_f1": row["test_macro_f1"],
                "test_wake_f1": row["test_Wake_f1"],
                "test_non_rem_f1": row["test_Non_REM_f1"],
                "test_rem_f1": row["test_REM_f1"],
            }
        )

    deep = previous[
        previous["readme_display_name"]
        == "Transition-regularized 61-epoch MSResCNN-MLP-TCN"
    ].iloc[0]
    rows.append(
        {
            "readme_order": len(rows) + 1,
            "readme_display_name": (
                "Previous project: transition-regularized "
                "61-epoch MSResCNN-MLP-TCN"
            ),
            "source_project": "dreamt-wearable-sleep-staging",
            "source_artifact": "results/summary/key_results.csv",
            "candidate": deep["artifact_model_name"],
            "variant": "external_comparison",
            "validation_macro_f1": deep["validation_macro_f1"],
            "test_macro_f1": deep["test_macro_f1"],
            "test_wake_f1": deep["test_wake_f1"],
            "test_non_rem_f1": deep["test_non_rem_f1"],
            "test_rem_f1": deep["test_rem_f1"],
        }
    )
    output = pd.DataFrame(rows)
    output.to_csv(SUMMARY / "key_results.csv", index=False)
    return output


def write_ablation_summary() -> pd.DataFrame:
    metrics = pd.read_csv(ABLATION / "metrics" / "ablation_validation_metrics.csv")
    ablations = [
        "basic_statistical",
        "basic_plus_signal_specific",
        "basic_signal_specific_rolling",
        "basic_signal_specific_rolling_subject_norm",
        "signal_group_movement",
        "signal_group_cardiovascular",
        "signal_group_temperature",
        "signal_group_electrodermal",
    ]
    labels = {
        "basic_statistical": "Basic statistical summaries",
        "basic_plus_signal_specific": "Basic + signal-specific",
        "basic_signal_specific_rolling": "Basic + signal-specific + rolling context",
        "basic_signal_specific_rolling_subject_norm": (
            "All engineered feature families"
        ),
        "signal_group_movement": "Movement signal group",
        "signal_group_cardiovascular": "Cardiovascular signal group",
        "signal_group_temperature": "Temperature signal group",
        "signal_group_electrodermal": "Electrodermal signal group",
    }
    rows = []
    for ablation in ablations:
        subset = metrics[metrics["ablation"] == ablation]
        best = subset.sort_values("macro_f1", ascending=False).iloc[0]
        rows.append(
            {
                "ablation": ablation,
                "display_name": labels[ablation],
                "best_model": best["model"],
                "n_features": best["n_features"],
                "validation_macro_f1": best["macro_f1"],
                "validation_accuracy": best["accuracy"],
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(SUMMARY / "validation_ablation_summary.csv", index=False)
    return output


def write_interpretation_summary() -> None:
    files = {
        "coefficient_contrast_summary.csv": (
            ROLLING / "metrics" / "coefficient_contrast_summary.csv"
        ),
        "rolling_rem_error_counts.csv": ROLLING / "metrics" / "rem_error_counts.csv",
        "rolling_transition_distance_metrics.csv": (
            ROLLING / "metrics" / "transition_distance_metrics.csv"
        ),
    }
    for output_name, source in files.items():
        pd.read_csv(source).to_csv(SUMMARY / output_name, index=False)


def make_postprocessing_tradeoff_figure() -> None:
    metrics = pd.read_csv(FINAL / "locked_test_metrics.csv")
    selected = metrics[
        (
            (metrics["candidate"] == "interpretable_rolling_logistic")
            & metrics["variant"].isin(
                [
                    "raw",
                    "raw_threshold_tuned",
                    "platt_threshold_tuned",
                    "platt_smoothed_threshold_tuned",
                ]
            )
        )
        | (
            (metrics["candidate"] == "best_original_ablation")
            & metrics["variant"].isin(["raw", "platt_smoothed_threshold_tuned"])
        )
    ].copy()
    selected["display_name"] = [
        _variant_label(candidate, variant)
        for candidate, variant in zip(
            selected["candidate"], selected["variant"], strict=True
        )
    ]
    plot_frame = selected.melt(
        id_vars=["display_name"],
        value_vars=["macro_f1", "REM_f1"],
        var_name="metric",
        value_name="score",
    )
    metric_names = {"macro_f1": "Test macro F1", "REM_f1": "Test REM F1"}
    plot_frame["metric"] = plot_frame["metric"].map(metric_names)

    plt.figure(figsize=(10, 4.8))
    ax = sns.barplot(
        data=plot_frame,
        x="score",
        y="display_name",
        hue="metric",
        palette=["#4c78a8", "#f58518"],
    )
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_title("Post-processing changes macro F1 and REM sensitivity")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(FIGURES / "postprocessing_tradeoff.png", dpi=160)
    plt.close()


def make_transition_distance_figure() -> None:
    metrics = pd.read_csv(
        FINAL / "test_error_analysis" / "metrics" / "transition_distance_metrics.csv"
    )
    selected = metrics[
        (
            (metrics["candidate"] == "best_original_ablation")
            & (metrics["variant"] == "platt_smoothed_threshold_tuned")
        )
        | (
            (metrics["candidate"] == "interpretable_rolling_logistic")
            & (metrics["variant"] == "raw_threshold_tuned")
        )
    ].copy()
    selected["model"] = [
        _variant_label(candidate, variant)
        for candidate, variant in zip(
            selected["candidate"], selected["variant"], strict=True
        )
    ]
    order = ["0", "1", "2-3", "4-10", ">10"]
    selected = selected[selected["transition_distance_bin"].isin(order)].copy()
    selected["transition_distance_bin"] = pd.Categorical(
        selected["transition_distance_bin"], categories=order, ordered=True
    )

    plt.figure(figsize=(8.2, 4.6))
    ax = sns.lineplot(
        data=selected,
        x="transition_distance_bin",
        y="macro_f1",
        hue="model",
        marker="o",
        sort=False,
    )
    ax.set_ylim(0.3, 0.58)
    ax.set_xlabel("Epochs from nearest true sleep-stage transition")
    ax.set_ylabel("Test macro F1")
    ax.set_title("Stable sleep-stage regions were easier than boundaries")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(FIGURES / "transition_distance_macro_f1.png", dpi=160)
    plt.close()


def make_coefficient_contrast_figure() -> None:
    contrasts = pd.read_csv(ROLLING / "metrics" / "coefficient_contrasts.csv")
    subset = (
        contrasts[contrasts["contrast"] == "REM_vs_Non-REM"]
        .sort_values("abs_contrast", ascending=False)
        .head(14)
        .sort_values("contrast_coefficient")
    )
    plt.figure(figsize=(8, 5.2))
    colors = ["#4c78a8" if value < 0 else "#f58518" for value in subset[
        "contrast_coefficient"
    ]]
    ax = plt.barh(subset["feature"], subset["contrast_coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Standardized coefficient contrast")
    plt.ylabel("")
    plt.title("Rolling logistic REM vs Non-REM associations")
    for bar in ax:
        width = bar.get_width()
        plt.text(
            width + (0.015 if width >= 0 else -0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left" if width >= 0 else "right",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(FIGURES / "rolling_logistic_rem_contrast.png", dpi=160)
    plt.close()


def write_artifact_manifest(key_results: pd.DataFrame) -> None:
    manifest = pd.DataFrame(
        [
            {
                "artifact": "summary/key_results.csv",
                "source": (
                    "outputs/runs/final_test_evaluation/"
                    "final_comparison_table.csv and sibling "
                    "dreamt-wearable-sleep-staging/results/summary/key_results.csv"
                ),
                "metric_scope": "locked test plus external comparison",
                "supports": "README Key Results table",
            },
            {
                "artifact": "summary/validation_ablation_summary.csv",
                "source": (
                    "outputs/runs/full_ablation_20260718/metrics/"
                    "ablation_validation_metrics.csv"
                ),
                "metric_scope": "validation",
                "supports": "Validation ablation findings",
            },
            {
                "artifact": "summary/coefficient_contrast_summary.csv",
                "source": (
                    "outputs/runs/final_test_evaluation/"
                    "rolling_logistic_interpretation/metrics/"
                    "coefficient_contrast_summary.csv"
                ),
                "metric_scope": "locked-test interpretation",
                "supports": "Rolling logistic interpretation",
            },
            {
                "artifact": "figures/postprocessing_tradeoff.png",
                "source": (
                    "outputs/runs/final_test_evaluation/locked_test_metrics.csv"
                ),
                "metric_scope": "locked test",
                "supports": "Post-processing operating-point tradeoff",
            },
            {
                "artifact": "figures/transition_distance_macro_f1.png",
                "source": (
                    "outputs/runs/final_test_evaluation/test_error_analysis/"
                    "metrics/transition_distance_metrics.csv"
                ),
                "metric_scope": "locked-test error analysis",
                "supports": "Transition-distance failure analysis",
            },
            {
                "artifact": "figures/rolling_logistic_rem_contrast.png",
                "source": (
                    "outputs/runs/final_test_evaluation/"
                    "rolling_logistic_interpretation/metrics/"
                    "coefficient_contrasts.csv"
                ),
                "metric_scope": "locked-test interpretation",
                "supports": "Interpretable rolling logistic coefficient contrasts",
            },
        ]
    )
    manifest.to_csv(SUMMARY / "artifact_manifest.csv", index=False)
    (RESULTS / "MANIFEST.md").write_text(manifest_markdown(key_results) + "\n")


def manifest_markdown(key_results: pd.DataFrame) -> str:
    return """# Results Manifest

This directory contains a compact, tracked subset of result artifacts used by
the README. Large generated outputs, trained model files, per-epoch prediction
CSVs, local DREAMT data, and intermediate feature tables remain outside the
curated evidence set.

## Summary Files

- `summary/key_results.csv` contains the README Key Results table, including
  the external deep-learning comparison row from the sibling repository.
- `summary/validation_ablation_summary.csv` summarizes validation-only
  feature-family and signal-group ablation findings.
- `summary/coefficient_contrast_summary.csv`,
  `summary/rolling_rem_error_counts.csv`, and
  `summary/rolling_transition_distance_metrics.csv` support the rolling
  logistic interpretation discussion.
- `summary/artifact_manifest.csv` maps each curated file to its source
  artifact, metric scope, and README claim.

## Figures

- `figures/postprocessing_tradeoff.png` compares test macro F1 and REM F1
  across selected raw and post-processed variants.
- `figures/transition_distance_macro_f1.png` summarizes locked-test macro F1 by
  distance to the nearest true sleep-stage transition.
- `figures/rolling_logistic_rem_contrast.png` shows the largest standardized
  rolling-logistic coefficient contrasts for `REM` versus `Non-REM`.

## Model-Name Crosswalk

The README uses human-readable model names. Internal candidate and variant
identifiers are preserved in `summary/key_results.csv`:

| README name | Internal candidate | Variant |
| --- | --- | --- |
""" + "\n".join(
        f"| {row.readme_display_name} | {row.candidate} | {row.variant} |"
        for row in key_results.itertuples(index=False)
    ) + """

## Regeneration

Run:

```bash
python scripts/generate_readme_figures.py
```

The script reads saved summary artifacts only. It does not train models, tune
models, alter the final held-out test protocol, or inspect raw DREAMT data.
"""


def _variant_label(candidate: str, variant: str) -> str:
    labels = {
        ("best_original_ablation", "raw"): "Engineered logistic, raw",
        (
            "best_original_ablation",
            "platt_smoothed_threshold_tuned",
        ): "Engineered logistic, post-processed",
        ("interpretable_rolling_logistic", "raw"): "Rolling logistic, raw",
        (
            "interpretable_rolling_logistic",
            "raw_threshold_tuned",
        ): "Rolling logistic, threshold-tuned",
        (
            "interpretable_rolling_logistic",
            "platt_threshold_tuned",
        ): "Rolling logistic, Platt + threshold",
        (
            "interpretable_rolling_logistic",
            "platt_smoothed_threshold_tuned",
        ): "Rolling logistic, Platt + smoothing + threshold",
    }
    return labels[(candidate, variant)]


if __name__ == "__main__":
    main()
