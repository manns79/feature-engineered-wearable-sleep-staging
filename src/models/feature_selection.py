"""Train-only feature selection helpers for interpretable logistic models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

DEFAULT_CORRELATION_THRESHOLD = 0.85
DEFAULT_FEATURE_FAMILIES = ("rolling_context",)
DEFAULT_SIGNAL_GROUPS = ("cardiovascular", "movement")
ROLLING_FEATURE_PATTERN = re.compile(
    r"(?P<base>.+)_roll(?P<window>\d+)_(?P<stat>mean|std)$"
)
MANIFEST_COLUMNS = ("feature", "feature_family", "signal_group", "source_signal")


@dataclass(frozen=True)
class RollingFeatureParts:
    """Parsed pieces of a rolling-context feature name."""

    feature: str
    base_feature: str
    window_epochs: int
    rolling_stat: str

    @property
    def family_key(self) -> str:
        """Return the redundant-family key used for within-family tie breaks."""
        return f"{self.base_feature}__{self.rolling_stat}"


@dataclass(frozen=True)
class CorrelationPruningResult:
    """Selected features and audit tables from correlation pruning."""

    selected_features: list[str]
    candidate_features: list[str]
    feature_metadata: pd.DataFrame
    dropped_features: pd.DataFrame
    correlation_edges: pd.DataFrame


def select_manifest_candidates(
    manifest: pd.DataFrame,
    *,
    feature_families: tuple[str, ...] = DEFAULT_FEATURE_FAMILIES,
    signal_groups: tuple[str, ...] = DEFAULT_SIGNAL_GROUPS,
) -> list[str]:
    """Return manifest features matching the planned family/group filters."""
    _validate_manifest(manifest)
    selected = manifest[
        manifest["feature_family"].isin(feature_families)
        & manifest["signal_group"].isin(signal_groups)
    ]
    features = selected["feature"].tolist()
    if not features:
        raise ValueError(
            "No candidate features matched feature_families="
            f"{feature_families} and signal_groups={signal_groups}."
        )
    return features


def parse_rolling_feature_name(feature: str) -> RollingFeatureParts:
    """Parse a rolling-context feature name such as ``HR_mean_roll9_mean``."""
    match = ROLLING_FEATURE_PATTERN.fullmatch(feature)
    if match is None:
        raise ValueError(f"Feature is not a supported rolling feature name: {feature}")
    return RollingFeatureParts(
        feature=feature,
        base_feature=match.group("base"),
        window_epochs=int(match.group("window")),
        rolling_stat=match.group("stat"),
    )


def correlation_prune_features(
    train_features: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    candidate_features: list[str] | None = None,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> CorrelationPruningResult:
    """Prune train-only features until retained absolute correlations are bounded.

    The removal rule is deterministic. For a highly correlated pair from the same
    rolling family, the shorter rolling window is dropped first. Otherwise the
    feature with more missing training values is dropped, followed by shorter
    window and later manifest order as tie breakers.
    """
    if not 0 <= threshold < 1:
        raise ValueError(
            "threshold must be greater than or equal to 0 and less than 1."
        )

    _validate_manifest(manifest)
    candidates = (
        select_manifest_candidates(manifest)
        if candidate_features is None
        else list(candidate_features)
    )
    if not candidates:
        raise ValueError("candidate_features must contain at least one feature.")

    missing = sorted(set(candidates) - set(train_features.columns))
    if missing:
        raise ValueError(
            f"Training features are missing candidate column(s): {missing}"
        )

    metadata = _feature_metadata(train_features, manifest, candidates)
    correlations = _absolute_correlations(train_features[candidates])
    retained = set(candidates)
    dropped_rows: list[dict[str, Any]] = []

    while True:
        edge = _strongest_retained_edge(correlations, retained, threshold)
        if edge is None:
            break
        feature_a, feature_b, abs_correlation = edge
        drop_feature, keep_feature, reason = _choose_feature_to_drop(
            feature_a, feature_b, metadata
        )
        retained.remove(drop_feature)
        dropped_rows.append(
            {
                "dropped_feature": drop_feature,
                "kept_feature": keep_feature,
                "abs_correlation": abs_correlation,
                "reason": reason,
            }
        )

    selected = [feature for feature in candidates if feature in retained]
    annotated_metadata = metadata.copy()
    annotated_metadata["selected"] = annotated_metadata["feature"].isin(retained)
    dropped = pd.DataFrame(
        dropped_rows,
        columns=["dropped_feature", "kept_feature", "abs_correlation", "reason"],
    )
    edges = _correlation_edges(correlations, threshold)
    return CorrelationPruningResult(
        selected_features=selected,
        candidate_features=candidates,
        feature_metadata=annotated_metadata,
        dropped_features=dropped,
        correlation_edges=edges,
    )


def _validate_manifest(manifest: pd.DataFrame) -> None:
    missing = sorted(set(MANIFEST_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(f"Feature manifest is missing column(s): {missing}")
    if manifest["feature"].duplicated().any():
        duplicates = sorted(manifest.loc[manifest["feature"].duplicated(), "feature"])
        raise ValueError(f"Feature manifest has duplicate feature(s): {duplicates}")


def _feature_metadata(
    train_features: pd.DataFrame, manifest: pd.DataFrame, candidates: list[str]
) -> pd.DataFrame:
    manifest_order = {
        feature: order for order, feature in enumerate(manifest["feature"])
    }
    manifest_lookup = manifest.set_index("feature")
    rows: list[dict[str, Any]] = []
    for feature in candidates:
        parts = parse_rolling_feature_name(feature)
        row = manifest_lookup.loc[feature]
        rows.append(
            {
                "feature": feature,
                "feature_family": row["feature_family"],
                "signal_group": row["signal_group"],
                "source_signal": row["source_signal"],
                "base_feature": parts.base_feature,
                "rolling_stat": parts.rolling_stat,
                "rolling_family": parts.family_key,
                "window_epochs": parts.window_epochs,
                "missing_rate": float(train_features[feature].isna().mean()),
                "manifest_order": manifest_order[feature],
            }
        )
    return pd.DataFrame(rows)


def _absolute_correlations(features: pd.DataFrame) -> pd.DataFrame:
    numeric = features.apply(pd.to_numeric, errors="coerce")
    return numeric.corr().abs().fillna(0.0)


def _strongest_retained_edge(
    correlations: pd.DataFrame, retained: set[str], threshold: float
) -> tuple[str, str, float] | None:
    retained_order = [
        feature for feature in correlations.columns if feature in retained
    ]
    best: tuple[str, str, float] | None = None
    for index, feature_a in enumerate(retained_order):
        for feature_b in retained_order[index + 1 :]:
            value = float(correlations.loc[feature_a, feature_b])
            if value <= threshold:
                continue
            if best is None or value > best[2]:
                best = (feature_a, feature_b, value)
    return best


def _choose_feature_to_drop(
    feature_a: str, feature_b: str, metadata: pd.DataFrame
) -> tuple[str, str, str]:
    rows = metadata.set_index("feature")
    row_a = rows.loc[feature_a]
    row_b = rows.loc[feature_b]
    if row_a["rolling_family"] == row_b["rolling_family"]:
        if row_a["window_epochs"] != row_b["window_epochs"]:
            if row_a["window_epochs"] < row_b["window_epochs"]:
                return feature_a, feature_b, "same_rolling_family_shorter_window"
            return feature_b, feature_a, "same_rolling_family_shorter_window"

    key_a = _general_drop_key(row_a)
    key_b = _general_drop_key(row_b)
    if key_a >= key_b:
        return feature_a, feature_b, "higher_missingness_or_shorter_window"
    return feature_b, feature_a, "higher_missingness_or_shorter_window"


def _general_drop_key(row: pd.Series) -> tuple[float, int, int]:
    return (
        float(row["missing_rate"]),
        -int(row["window_epochs"]),
        int(row["manifest_order"]),
    )


def _correlation_edges(correlations: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = list(correlations.columns)
    for index, feature_a in enumerate(columns):
        for feature_b in columns[index + 1 :]:
            value = float(correlations.loc[feature_a, feature_b])
            if value > threshold:
                rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "abs_correlation": value,
                    }
                )
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "abs_correlation"])
