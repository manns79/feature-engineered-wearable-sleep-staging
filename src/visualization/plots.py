"""Visualization helpers for model evaluation and interpretation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_confusion_matrix_plot(
    matrix: pd.DataFrame,
    path: str | Path,
    *,
    fmt: str = "d",
    cmap: str = "Blues",
) -> Path:
    """Save a compact confusion-matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    display_matrix = matrix.rename(
        index=lambda label: str(label).removeprefix("true_"),
        columns=lambda label: str(label).removeprefix("pred_"),
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(display_matrix, annot=True, fmt=fmt, cmap=cmap, cbar=False, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def save_per_class_metric_plot(
    metrics: pd.DataFrame,
    path: str | Path,
    *,
    metric_columns: tuple[str, ...] = ("precision", "recall", "f1"),
) -> Path:
    """Save per-class precision/recall/F1 bars for one model."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    long = metrics.melt(
        id_vars=["label"],
        value_vars=list(metric_columns),
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=long, x="label", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("True label")
    ax.set_ylabel("Score")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def save_participant_metric_plot(
    metrics: pd.DataFrame,
    path: str | Path,
    *,
    metric: str = "macro_f1",
) -> Path:
    """Save participant-level metric values sorted from weakest to strongest."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = metrics.sort_values(metric).copy()
    fig_height = max(4, min(12, 0.25 * len(ordered)))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    sns.scatterplot(data=ordered, x=metric, y="participant_id", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Participant")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def save_transition_metric_plot(
    metrics: pd.DataFrame,
    path: str | Path,
    *,
    metric: str = "macro_f1",
) -> Path:
    """Save metric values by distance to nearest sleep-stage transition."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = metrics.sort_values("transition_distance_bin").copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(
        data=ordered,
        x="transition_distance_bin",
        y=metric,
        marker="o",
        sort=False,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epochs from nearest true-label transition")
    ax.set_ylabel(metric.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def save_model_family_comparison_plot(
    metrics: pd.DataFrame,
    path: str | Path,
    *,
    metric: str = "validation_macro_f1",
) -> Path:
    """Save a cross-ablation comparison for selected model families."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(4, 0.45 * metrics["ablation"].nunique())
    fig, ax = plt.subplots(figsize=(9, fig_height))
    sns.barplot(data=metrics, x=metric, y="ablation", hue="model", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Ablation")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def save_feature_importance_plot(
    importance: pd.DataFrame,
    path: str | Path,
    *,
    value_column: str,
    title: str,
    max_features: int = 25,
) -> Path:
    """Save a horizontal top-feature importance chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    top = importance.sort_values(value_column, ascending=False).head(max_features)
    top = top.sort_values(value_column)
    fig_height = max(4, 0.28 * len(top))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    sns.barplot(data=top, x=value_column, y="feature", ax=ax)
    ax.set_xlabel(value_column.replace("_", " ").title())
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def save_xgboost_shap_summary(
    model: Any,
    X: pd.DataFrame,
    path: str | Path,
    *,
    max_display: int = 25,
) -> Path:
    """Save a SHAP summary plot for an XGBoost model when SHAP is installed."""
    import matplotlib.pyplot as plt
    import shap

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    return output
