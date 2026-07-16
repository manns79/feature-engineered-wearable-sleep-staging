"""Manifest-driven feature ablation definitions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MANIFEST_REQUIRED_COLUMNS = (
    "feature",
    "feature_family",
    "signal_group",
    "source_signal",
)
FEATURE_FAMILY_ORDER = (
    "basic_statistical",
    "signal_specific",
    "rolling_context",
    "whole_night_subject_normalized",
)


@dataclass(frozen=True)
class AblationSpec:
    """A named feature subset selected from the feature manifest."""

    name: str
    description: str
    feature_families: tuple[str, ...] = ()
    signal_groups: tuple[str, ...] = ()
    source_signals: tuple[str, ...] = ()
    include_all: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_families", _as_tuple(self.feature_families))
        object.__setattr__(self, "signal_groups", _as_tuple(self.signal_groups))
        object.__setattr__(self, "source_signals", _as_tuple(self.source_signals))

        if not self.name:
            raise ValueError("AblationSpec.name must be non-empty.")
        if self.include_all and (
            self.feature_families or self.signal_groups or self.source_signals
        ):
            raise ValueError("include_all cannot be combined with manifest filters.")
        if not self.include_all and not (
            self.feature_families or self.signal_groups or self.source_signals
        ):
            raise ValueError(
                "AblationSpec must set include_all or at least one manifest filter."
            )

    def select_features(self, manifest: pd.DataFrame) -> list[str]:
        """Return manifest features selected by this ablation."""
        return select_features_from_manifest(
            manifest,
            feature_families=self.feature_families,
            signal_groups=self.signal_groups,
            source_signals=self.source_signals,
            include_all=self.include_all,
        )

    def metadata(self, n_features: int | None = None) -> dict[str, object]:
        """Return flat metadata suitable for CSV output."""
        row: dict[str, object] = {
            "ablation": self.name,
            "description": self.description,
            "feature_families": "|".join(self.feature_families),
            "signal_groups": "|".join(self.signal_groups),
            "source_signals": "|".join(self.source_signals),
            "include_all": self.include_all,
        }
        if n_features is not None:
            row["n_features"] = n_features
        return row


PRIMARY_ABLATION_SPECS = (
    AblationSpec(
        name="basic_statistical",
        description="Basic statistical epoch summaries only.",
        feature_families=("basic_statistical",),
    ),
    AblationSpec(
        name="basic_plus_signal_specific",
        description=(
            "Basic statistical summaries plus signal-specific engineered features."
        ),
        feature_families=("basic_statistical", "signal_specific"),
    ),
    AblationSpec(
        name="basic_signal_specific_rolling",
        description=(
            "Basic statistical, signal-specific, and participant-contained "
            "rolling/context features."
        ),
        feature_families=(
            "basic_statistical",
            "signal_specific",
            "rolling_context",
        ),
    ),
    AblationSpec(
        name="basic_signal_specific_rolling_subject_norm",
        description=(
            "Basic statistical, signal-specific, rolling/context, and "
            "whole-night subject-normalized features."
        ),
        feature_families=FEATURE_FAMILY_ORDER,
    ),
    AblationSpec(
        name="all_engineered",
        description="All engineered features present in the manifest.",
        include_all=True,
    ),
)


def load_feature_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate a feature manifest CSV."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Feature manifest does not exist: {manifest_path}")
    return validate_feature_manifest(pd.read_csv(manifest_path))


def validate_feature_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate manifest columns and feature uniqueness."""
    missing = sorted(set(MANIFEST_REQUIRED_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(f"Feature manifest is missing column(s): {missing}")
    if manifest.empty:
        raise ValueError("Feature manifest is empty.")

    validated = manifest.loc[:, MANIFEST_REQUIRED_COLUMNS].copy()
    for column in MANIFEST_REQUIRED_COLUMNS:
        if validated[column].isna().any():
            raise ValueError(f"Feature manifest column has missing values: {column}")
        validated[column] = validated[column].astype(str)

    duplicates = sorted(
        validated.loc[validated["feature"].duplicated(), "feature"].unique()
    )
    if duplicates:
        raise ValueError(f"Feature manifest has duplicate feature(s): {duplicates}")
    return validated


def validate_manifest_matches_features(
    manifest: pd.DataFrame, feature_columns: Iterable[str]
) -> None:
    """Raise when manifest features differ from a feature table's columns."""
    validated = validate_feature_manifest(manifest)
    manifest_features = set(validated["feature"])
    table_features = set(feature_columns)
    missing_from_manifest = sorted(table_features - manifest_features)
    missing_from_table = sorted(manifest_features - table_features)
    if missing_from_manifest or missing_from_table:
        raise ValueError(
            "Feature manifest does not match feature table columns. "
            f"Missing from manifest: {missing_from_manifest}; "
            f"missing from table: {missing_from_table}"
        )


def select_features_from_manifest(
    manifest: pd.DataFrame,
    *,
    feature_families: Iterable[str] = (),
    signal_groups: Iterable[str] = (),
    source_signals: Iterable[str] = (),
    include_all: bool = False,
) -> list[str]:
    """Select features by manifest metadata, preserving manifest row order."""
    validated = validate_feature_manifest(manifest)
    families = _as_tuple(feature_families)
    groups = _as_tuple(signal_groups)
    signals = _as_tuple(source_signals)

    if include_all and (families or groups or signals):
        raise ValueError("include_all cannot be combined with manifest filters.")
    if not include_all and not (families or groups or signals):
        raise ValueError("At least one manifest filter is required.")

    selected = validated
    if not include_all:
        if families:
            selected = selected[selected["feature_family"].isin(families)]
        if groups:
            selected = selected[selected["signal_group"].isin(groups)]
        if signals:
            selected = selected[selected["source_signal"].isin(signals)]

    features = selected["feature"].tolist()
    if not features:
        filters = {
            "feature_families": families,
            "signal_groups": groups,
            "source_signals": signals,
        }
        raise ValueError(f"Ablation selector matched no features: {filters}")
    return features


def signal_group_ablation_specs(manifest: pd.DataFrame) -> tuple[AblationSpec, ...]:
    """Return one all-family ablation per signal group present in the manifest."""
    validated = validate_feature_manifest(manifest)
    return tuple(
        AblationSpec(
            name=f"signal_group_{_slug(signal_group)}",
            description=f"All engineered features for the {signal_group} signal group.",
            signal_groups=(signal_group,),
        )
        for signal_group in sorted(validated["signal_group"].unique())
    )


def source_signal_ablation_specs(manifest: pd.DataFrame) -> tuple[AblationSpec, ...]:
    """Return one all-family ablation per source signal present in the manifest."""
    validated = validate_feature_manifest(manifest)
    return tuple(
        AblationSpec(
            name=f"source_signal_{_slug(source_signal)}",
            description=f"All engineered features derived from {source_signal}.",
            source_signals=(source_signal,),
        )
        for source_signal in sorted(validated["source_signal"].unique())
    )


def default_ablation_specs(manifest: pd.DataFrame) -> tuple[AblationSpec, ...]:
    """Return the planned cumulative and per-signal-group ablation specs."""
    return (*PRIMARY_ABLATION_SPECS, *signal_group_ablation_specs(manifest))


def selected_feature_summary(
    manifest: pd.DataFrame, spec: AblationSpec
) -> dict[str, object]:
    """Return compact feature-count metadata for one ablation."""
    selected = spec.select_features(manifest)
    selected_manifest = validate_feature_manifest(manifest)
    selected_manifest = selected_manifest[selected_manifest["feature"].isin(selected)]
    row = spec.metadata(n_features=len(selected))
    row["selected_feature_families"] = "|".join(
        sorted(selected_manifest["feature_family"].unique())
    )
    row["selected_signal_groups"] = "|".join(
        sorted(selected_manifest["signal_group"].unique())
    )
    row["selected_source_signals"] = "|".join(
        sorted(selected_manifest["source_signal"].unique())
    )
    return row


def _as_tuple(values: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def _slug(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
