"""Evaluation metrics for three-class sleep staging."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import TARGET_LABELS


def safe_label(label: str) -> str:
    """Return label text suitable for metric column names."""
    return label.replace("-", "_").replace(" ", "_")


def classification_metrics(
    y_true: Iterable[object],
    y_pred: Iterable[object],
    labels: Iterable[str] = TARGET_LABELS,
) -> dict[str, float]:
    """Compute overall and per-class classification metrics."""
    label_list = list(labels)
    true = list(y_true)
    pred = list(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=label_list,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(sum(recall) / len(recall)),
        "macro_f1": float(
            f1_score(true, pred, labels=label_list, average="macro", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(true, pred, labels=label_list)),
    }
    for label, p_value, r_value, f_value in zip(
        label_list,
        precision,
        recall,
        f1,
        strict=True,
    ):
        name = safe_label(label)
        metrics[f"{name}_precision"] = float(p_value)
        metrics[f"{name}_recall"] = float(r_value)
        metrics[f"{name}_f1"] = float(f_value)
    return metrics


def confusion_matrix_frame(
    y_true: Iterable[object],
    y_pred: Iterable[object],
    labels: Iterable[str] = TARGET_LABELS,
) -> pd.DataFrame:
    """Return a labeled confusion-matrix DataFrame."""
    label_list = list(labels)
    matrix = confusion_matrix(list(y_true), list(y_pred), labels=label_list)
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in label_list],
        columns=[f"pred_{label}" for label in label_list],
    )
