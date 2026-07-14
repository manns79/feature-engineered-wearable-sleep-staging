import pandas as pd
import pytest
from src.features.rolling_features import add_centered_rolling_stats


def test_centered_rolling_stats_stay_inside_context_groups():
    features = pd.DataFrame(
        {
            "participant_id": ["S001", "S001", "S001", "S001"],
            "_segment_id": [0, 0, 1, 1],
            "epoch_id": [0, 1, 2, 3],
            "BVP_mean": [1.0, 3.0, 100.0, 104.0],
        }
    )

    output = add_centered_rolling_stats(
        features,
        ["BVP_mean"],
        window_epochs=3,
        group_cols=["participant_id", "_segment_id"],
    )

    assert output.loc[0, "BVP_mean_roll3_mean"] == pytest.approx(2.0)
    assert output.loc[1, "BVP_mean_roll3_mean"] == pytest.approx(2.0)
    assert output.loc[2, "BVP_mean_roll3_mean"] == pytest.approx(102.0)
    assert output.loc[3, "BVP_mean_roll3_mean"] == pytest.approx(102.0)
