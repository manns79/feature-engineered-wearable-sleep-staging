"""Visualization helpers for model evaluation and interpretation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_confusion_matrix_plot(matrix: pd.DataFrame, path: str | Path) -> Path:
    """Save a compact confusion-matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
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
