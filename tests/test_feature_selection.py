import pandas as pd
from src.models.feature_selection import (
    DEFAULT_CORRELATION_THRESHOLD,
    correlation_prune_features,
    parse_rolling_feature_name,
    select_manifest_candidates,
)


def _manifest():
    return pd.DataFrame(
        [
            {
                "feature": "HR_mean_roll3_mean",
                "feature_family": "rolling_context",
                "signal_group": "cardiovascular",
                "source_signal": "HR",
            },
            {
                "feature": "HR_mean_roll9_mean",
                "feature_family": "rolling_context",
                "signal_group": "cardiovascular",
                "source_signal": "HR",
            },
            {
                "feature": "ACC_MAG_std_roll3_mean",
                "feature_family": "rolling_context",
                "signal_group": "movement",
                "source_signal": "ACC_MAG",
            },
            {
                "feature": "TEMP_mean_roll9_mean",
                "feature_family": "rolling_context",
                "signal_group": "temperature",
                "source_signal": "TEMP",
            },
        ]
    )


def test_default_correlation_threshold_is_085():
    assert DEFAULT_CORRELATION_THRESHOLD == 0.85


def test_select_manifest_candidates_uses_rolling_cardio_and_movement_only():
    assert select_manifest_candidates(_manifest()) == [
        "HR_mean_roll3_mean",
        "HR_mean_roll9_mean",
        "ACC_MAG_std_roll3_mean",
    ]


def test_parse_rolling_feature_name_returns_family_parts():
    parts = parse_rolling_feature_name("HR_mean_roll9_mean")

    assert parts.base_feature == "HR_mean"
    assert parts.window_epochs == 9
    assert parts.rolling_stat == "mean"
    assert parts.family_key == "HR_mean__mean"


def test_correlation_pruning_prefers_longer_window_within_redundant_family():
    train_features = pd.DataFrame(
        {
            "HR_mean_roll3_mean": [1, 2, 3, 4, 5, 6],
            "HR_mean_roll9_mean": [2, 4, 6, 8, 10, 12],
            "ACC_MAG_std_roll3_mean": [1, 1, 2, 3, 5, 8],
        }
    )

    result = correlation_prune_features(train_features, _manifest())

    assert "HR_mean_roll9_mean" in result.selected_features
    assert "HR_mean_roll3_mean" not in result.selected_features
    assert result.dropped_features.loc[0, "reason"] == (
        "same_rolling_family_shorter_window"
    )
