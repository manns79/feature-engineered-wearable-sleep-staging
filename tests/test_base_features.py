import pytest
from src.features.base_features import (
    summarize_signal,
)


def test_summarize_signal_computes_expected_basic_statistics():
    features = summarize_signal([1, 2, 3, 4], "BVP")

    assert features["BVP_mean"] == pytest.approx(2.5)
    assert features["BVP_median"] == pytest.approx(2.5)
    assert features["BVP_iqr"] == pytest.approx(1.5)
    assert features["BVP_range"] == pytest.approx(3.0)
    assert features["BVP_valid_fraction"] == pytest.approx(1.0)
    assert features["BVP_slope"] == pytest.approx(1.0)
