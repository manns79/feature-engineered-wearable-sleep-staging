import pandas as pd
import pytest
from src.data.splits import (
    assert_split_alignment,
    validate_split_assignments,
)


def test_validate_split_assignments_rejects_duplicate_participant():
    split_df = pd.DataFrame(
        {
            "participant_id": ["S001", "S001"],
            "split": ["train", "validation"],
        }
    )

    with pytest.raises(ValueError, match="multiple rows"):
        validate_split_assignments(split_df)


def test_assert_split_alignment_rejects_changed_assignment():
    reference = pd.DataFrame(
        {"participant_id": ["S001", "S002"], "split": ["train", "test"]}
    )
    candidate = pd.DataFrame(
        {"participant_id": ["S001", "S002"], "split": ["validation", "test"]}
    )

    with pytest.raises(ValueError, match="do not match"):
        assert_split_alignment(candidate, reference)
