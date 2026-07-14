# Data Directory

Raw DREAMT files are not committed. Place local participant files such as
`S002_whole_df.csv` under `data/raw/`.

Canonical generated artifacts should be CSV files:

- `data/interim/split_assignments.csv`
- `data/interim/epoch_index.csv`
- `data/processed/features_train.csv`
- `data/processed/features_val.csv`
- `data/processed/features_test.csv`
- `data/processed/feature_manifest.csv`

The split assignments should match the earlier project unless an explicit,
documented analysis requires otherwise.

Feature CSVs are generated for all three splits, but the held-out test feature
table is for final evaluation only. During feature exploration and model
selection, use the training table for predictive EDA and the validation table
for model-family comparison.
