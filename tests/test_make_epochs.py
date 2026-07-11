import pandas as pd
from src.data.make_epochs import build_epoch_index


def _write_participant(path, labels):
    n_rows = len(labels)
    pd.DataFrame(
        {
            "TIMESTAMP": list(range(n_rows)),
            "BVP": range(n_rows),
            "ACC_X": range(n_rows),
            "ACC_Y": range(n_rows),
            "ACC_Z": range(n_rows),
            "TEMP": [30.0] * n_rows,
            "EDA": [0.1] * n_rows,
            "IBI": [0.8] * n_rows,
            "HR": [70.0] * n_rows,
            "Sleep_Stage": labels,
        }
    ).to_csv(path, index=False)


def test_build_epoch_index_maps_labels_and_excludes_preparation(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(
        raw_dir / "S001_whole_df.csv", ["P", "P", "W", "W", "N2", "N2", "REM", "REM"]
    )
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame({"participant_id": ["S001"], "split": ["train"]}).to_csv(
        split_path, index=False
    )

    epoch_index = build_epoch_index(raw_dir, split_path, rows_per_epoch=2)

    assert len(epoch_index) == 4
    assert not bool(epoch_index.loc[0, "is_valid_epoch"])
    assert epoch_index.loc[0, "exclusion_reason"] == "excluded_label"
    assert epoch_index.loc[1:, "mapped_label"].tolist() == ["Wake", "Non-REM", "REM"]
    assert set(epoch_index["split"]) == {"train"}


def test_build_epoch_index_rejects_participants_missing_from_split(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(raw_dir / "S001_whole_df.csv", ["W", "W"])
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame({"participant_id": ["S002"], "split": ["train"]}).to_csv(
        split_path, index=False
    )

    try:
        build_epoch_index(raw_dir, split_path, rows_per_epoch=2)
    except ValueError as exc:
        assert "missing from split assignments" in str(exc)
    else:
        raise AssertionError("Expected missing split assignment to raise ValueError")


def test_build_epoch_index_can_infer_nonzero_start_offset(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(
        raw_dir / "S001_whole_df.csv",
        ["P", "W", "W", "N2", "N2", "REM", "REM"],
    )
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame({"participant_id": ["S001"], "split": ["train"]}).to_csv(
        split_path, index=False
    )

    epoch_index = build_epoch_index(raw_dir, split_path, rows_per_epoch=2)

    assert epoch_index["epoch_start_offset_rows"].unique().tolist() == [1]
    assert epoch_index["mapped_label"].tolist() == ["Wake", "Non-REM", "REM"]


def test_build_epoch_index_marks_mixed_label_epochs_invalid(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(raw_dir / "S001_whole_df.csv", ["W", "N2"])
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame({"participant_id": ["S001"], "split": ["train"]}).to_csv(
        split_path, index=False
    )

    epoch_index = build_epoch_index(
        raw_dir, split_path, rows_per_epoch=2, infer_start_offset=False
    )

    assert not bool(epoch_index.loc[0, "is_valid_epoch"])
    assert epoch_index.loc[0, "exclusion_reason"] == "label_changed_within_epoch"
