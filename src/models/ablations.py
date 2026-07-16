"""Manifest-driven feature ablation definitions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.build_features import FEATURE_ID_COLUMNS
from src.models.training import TrainingOutputs, train_and_evaluate_model_set

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
DEFAULT_ABLATION_FEATURE_PATHS = {
    "train": Path("data/processed/features_train.csv"),
    "validation": Path("data/processed/features_val.csv"),
}


@dataclass(frozen=True)
class AblationExperimentOutputs:
    """Paths written by an ablation experiment run."""

    metrics_path: Path
    cv_results_path: Path
    best_params_path: Path
    feature_sets_path: Path
    run_outputs: dict[str, TrainingOutputs]


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


def resolve_ablation_specs(
    manifest: pd.DataFrame, ablation_names: Iterable[str] = ()
) -> tuple[AblationSpec, ...]:
    """Return default specs or the named subset requested by a caller."""
    specs = default_ablation_specs(manifest)
    names = tuple(ablation_names)
    if not names:
        return specs

    specs_by_name = {spec.name: spec for spec in specs}
    missing = sorted(set(names) - set(specs_by_name))
    if missing:
        available = sorted(specs_by_name)
        raise ValueError(
            f"Unknown ablation name(s): {missing}. Available ablations: {available}"
        )
    return tuple(specs_by_name[name] for name in names)


def run_ablation_experiments(
    feature_paths: dict[str, str | Path] | None = None,
    manifest_path: str | Path = "data/processed/feature_manifest.csv",
    output_dir: str | Path = "outputs",
    *,
    ablation_names: Iterable[str] = (),
    include_xgboost: bool = True,
    random_state: int = 42,
    cv_splits: int = 5,
    n_jobs: int | None = None,
    param_grids: dict[str, dict[str, list[Any]]] | None = None,
) -> AblationExperimentOutputs:
    """Run manifest-defined ablations through the training pipeline."""
    paths = _ablation_feature_paths(feature_paths)
    manifest = load_feature_manifest(manifest_path)
    _validate_manifest_against_feature_tables(manifest, paths)
    specs = resolve_ablation_specs(manifest, ablation_names)

    output_root = Path(output_dir)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_outputs: dict[str, TrainingOutputs] = {}
    metric_frames: list[pd.DataFrame] = []
    cv_result_frames: list[pd.DataFrame] = []
    best_param_frames: list[pd.DataFrame] = []
    feature_set_rows: list[dict[str, object]] = []

    for spec in specs:
        selected_features = spec.select_features(manifest)
        summary = selected_feature_summary(manifest, spec)
        output_metadata = _compact_output_metadata(summary)
        feature_set_rows.append(summary)
        outputs = train_and_evaluate_model_set(
            feature_paths=paths,
            output_dir=output_root,
            include_xgboost=include_xgboost,
            random_state=random_state,
            model_prefix=f"ablation_{spec.name}",
            cv_splits=cv_splits,
            n_jobs=n_jobs,
            param_grids=param_grids,
            feature_columns=selected_features,
        )
        run_outputs[spec.name] = outputs

        metric_frames.append(_annotated_frame(outputs.metrics_path, output_metadata))
        cv_result_frames.append(
            _annotated_frame(outputs.cv_results_path, output_metadata)
        )
        best_param_frames.append(
            _annotated_frame(outputs.best_params_path, output_metadata)
        )

    metrics_path = metrics_dir / "ablation_validation_metrics.csv"
    cv_results_path = metrics_dir / "ablation_cv_results.csv"
    best_params_path = metrics_dir / "ablation_best_params.csv"
    feature_sets_path = metrics_dir / "ablation_feature_sets.csv"
    _combine_frames(metric_frames).to_csv(metrics_path, index=False)
    _combine_frames(cv_result_frames).to_csv(cv_results_path, index=False)
    _combine_frames(best_param_frames).to_csv(best_params_path, index=False)
    pd.DataFrame(feature_set_rows).to_csv(feature_sets_path, index=False)

    return AblationExperimentOutputs(
        metrics_path=metrics_path,
        cv_results_path=cv_results_path,
        best_params_path=best_params_path,
        feature_sets_path=feature_sets_path,
        run_outputs=run_outputs,
    )


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
    row["selected_features"] = "|".join(selected)
    return row


def _ablation_feature_paths(
    feature_paths: dict[str, str | Path] | None,
) -> dict[str, Path]:
    paths = DEFAULT_ABLATION_FEATURE_PATHS if feature_paths is None else feature_paths
    required = {"train", "validation"}
    missing = sorted(required - set(paths))
    if missing:
        raise ValueError(f"feature_paths is missing split(s): {missing}")
    return {split: Path(paths[split]) for split in required}


def _validate_manifest_against_feature_tables(
    manifest: pd.DataFrame, feature_paths: dict[str, Path]
) -> None:
    for split, path in feature_paths.items():
        feature_columns = _feature_table_columns(path)
        try:
            validate_manifest_matches_features(manifest, feature_columns)
        except ValueError as exc:
            raise ValueError(f"{split} feature table mismatch: {exc}") from exc


def _feature_table_columns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV does not exist: {path}")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(FEATURE_ID_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    feature_columns = [
        column for column in columns if column not in FEATURE_ID_COLUMNS
    ]
    if not feature_columns:
        raise ValueError(f"{path} does not contain any feature columns.")
    return feature_columns


def _annotated_frame(path: Path, metadata: dict[str, object]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column, value in reversed(metadata.items()):
        frame.insert(0, column, value)
    return frame


def _compact_output_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        column: value
        for column, value in metadata.items()
        if column != "selected_features"
    }


def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
