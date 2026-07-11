import pandas as pd
from src.data.make_epochs import build_epoch_index
from src.features.build_features import (
    build_basic_feature_table,
    write_split_feature_tables,
)


def _write_participant(path, labels, offset):
    n_rows = len(labels)
    values = [offset + index for index in range(n_rows)]
    pd.DataFrame(
        {
            "TIMESTAMP": list(range(n_rows)),
            "BVP": values,
            "ACC_X": values,
            "ACC_Y": values,
            "ACC_Z": values,
            "TEMP": [30.0 + offset] * n_rows,
            "EDA": [0.1 + offset] * n_rows,
            "IBI": [0.8] * n_rows,
            "HR": [70.0] * n_rows,
            "Sleep_Stage": labels,
        }
    ).to_csv(path, index=False)


def test_build_and_write_basic_feature_tables_keep_split_rows_separate(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_participant(raw_dir / "S001_whole_df.csv", ["W", "W", "P", "P"], 0)
    _write_participant(raw_dir / "S002_whole_df.csv", ["N2", "N2"], 10)
    _write_participant(raw_dir / "S003_whole_df.csv", ["REM", "REM"], 20)
    split_path = tmp_path / "split_assignments.csv"
    pd.DataFrame(
        {
            "participant_id": ["S001", "S002", "S003"],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(split_path, index=False)
    epoch_index = build_epoch_index(raw_dir, split_path, rows_per_epoch=2)
    epoch_index_path = tmp_path / "epoch_index.csv"
    epoch_index.to_csv(epoch_index_path, index=False)

    features = build_basic_feature_table(raw_dir, epoch_index_path)
    written = write_split_feature_tables(features, tmp_path / "processed")

    assert set(features["label"]) == {"Wake", "Non-REM", "REM"}
    assert "BVP_mean" in features.columns
    assert "ACC_MAG_mean" in features.columns
    assert pd.read_csv(written["train"])["split"].tolist() == ["train"]
    assert pd.read_csv(written["validation"])["split"].tolist() == ["validation"]
    assert pd.read_csv(written["test"])["split"].tolist() == ["test"]
