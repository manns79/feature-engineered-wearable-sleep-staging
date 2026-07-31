import pandas as pd
from src.models.rolling_logistic_interpretation import (
    coefficient_contrast_frame,
    contrast_summary_frame,
    rem_error_counts_frame,
    rem_error_group,
)


def test_coefficient_contrast_frame_ranks_signed_class_differences():
    coefficients = pd.DataFrame(
        [
            {
                "class_label": "REM",
                "feature": "HR_mean_roll5_std",
                "coefficient": 0.8,
                "feature_family": "rolling_context",
                "signal_group": "cardiovascular",
                "source_signal": "HR",
            },
            {
                "class_label": "Non-REM",
                "feature": "HR_mean_roll5_std",
                "coefficient": -0.2,
                "feature_family": "rolling_context",
                "signal_group": "cardiovascular",
                "source_signal": "HR",
            },
            {
                "class_label": "REM",
                "feature": "ACC_MAG_std_roll5_std",
                "coefficient": -0.1,
                "feature_family": "rolling_context",
                "signal_group": "movement",
                "source_signal": "ACC_MAG",
            },
            {
                "class_label": "Non-REM",
                "feature": "ACC_MAG_std_roll5_std",
                "coefficient": 0.2,
                "feature_family": "rolling_context",
                "signal_group": "movement",
                "source_signal": "ACC_MAG",
            },
        ]
    )

    contrasts = coefficient_contrast_frame(
        coefficients,
        contrasts=(("REM", "Non-REM"),),
    )

    assert contrasts["feature"].tolist() == [
        "HR_mean_roll5_std",
        "ACC_MAG_std_roll5_std",
    ]
    assert contrasts["contrast_coefficient"].round(3).tolist() == [1.0, -0.3]
    assert contrasts["rank"].tolist() == [1, 2]


def test_contrast_summary_groups_by_signal_metadata():
    contrasts = pd.DataFrame(
        [
            {
                "contrast": "REM_vs_Non-REM",
                "feature": "HR_mean_roll5_std",
                "abs_contrast": 1.0,
                "contrast_coefficient": 1.0,
                "feature_family": "rolling_context",
                "signal_group": "cardiovascular",
                "source_signal": "HR",
            },
            {
                "contrast": "REM_vs_Non-REM",
                "feature": "IBI_rmssd_roll5_std",
                "abs_contrast": 0.5,
                "contrast_coefficient": -0.5,
                "feature_family": "rolling_context",
                "signal_group": "cardiovascular",
                "source_signal": "IBI",
            },
        ]
    )

    summary = contrast_summary_frame(contrasts)

    assert set(summary["source_signal"]) == {"HR", "IBI"}
    assert summary["top_feature"].tolist() == [
        "HR_mean_roll5_std",
        "IBI_rmssd_roll5_std",
    ]


def test_rem_error_group_labels_rem_hits_misses_and_false_positives():
    assert rem_error_group("REM", "REM") == "true_REM_pred_REM"
    assert rem_error_group("REM", "Non-REM") == "true_REM_pred_Non_REM"
    assert rem_error_group("Wake", "REM") == "true_Wake_pred_REM"
    assert rem_error_group("Wake", "Non-REM") == "not_REM_related"


def test_rem_error_counts_includes_transition_bins_when_present():
    predictions = pd.DataFrame(
        {
            "rem_error_group": [
                "true_REM_pred_REM",
                "true_REM_pred_REM",
                "true_Non_REM_pred_REM",
            ],
            "transition_distance_bin": ["0", "0", ">10"],
        }
    )

    counts = rem_error_counts_frame(predictions)

    assert counts["n_epochs"].sum() == 3
    assert {"rem_error_group", "transition_distance_bin", "epoch_fraction"}.issubset(
        counts.columns
    )
