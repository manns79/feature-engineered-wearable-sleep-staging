# Data Directory

Raw DREAMT files are not committed. Place local participant files such as
`S002_whole_df.csv` under `data/raw/`.

Canonical generated artifacts should be CSV files:

- `data/interim/split_assignments.csv`
- `data/interim/epoch_index.csv`
- `data/processed/features_train.csv`
- `data/processed/features_val.csv`
- `data/processed/features_test.csv`

The split assignments should match the earlier project unless an explicit,
documented analysis requires otherwise.

