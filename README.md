# Feature-Engineered Machine Learning for Wearable Sleep Stage Classification

This project uses wearable signals from DREAMT to classify 30-second sleep epochs
as `Wake`, `Non-REM`, or `REM`. It follows the label mapping and evaluation
principles from the earlier `dreamt-wearable-sleep-staging` project while
shifting the modeling emphasis from deep learning toward traditional machine
learning with richer engineered features.

## Project Contract

- Dataset: DREAMT wearable physiological signals.
- Signals: `BVP`, `ACC_X`, `ACC_Y`, `ACC_Z`, `TEMP`, `EDA`, `IBI`, and `HR`.
- Labels: exclude preparation epochs; map `N1`, `N2`, and `N3` to `Non-REM`.
- Splits: reuse the participant-level train/validation/test assignments from
  `/home/manns79/dreamt-wearable-sleep-staging/data/interim/split_assignments.csv`.
- Data artifacts: use CSV files by default.
- Target setting: within-night retrospective staging, so whole-night
  participant normalization is allowed when documented.
- Models: majority-class dummy classifier, stratified random dummy classifier,
  multinomial elastic-net logistic regression, random forest, and XGBoost.
- Imbalance handling: use class weights or balanced sample weights for
  nontrivial models.
- Interpretability: include feature importance and SHAP summary plots for
  XGBoost when the optional dependency is available.

## Repository Layout

```text
data/
  raw/          # local DREAMT files; ignored by git
  interim/      # split assignments and epoch indexes; mostly ignored by git
  processed/    # generated CSV feature tables; ignored by git
notebooks/
  01_data_exploration.ipynb
  02_feature_exploration.ipynb
  03_model_results.ipynb
src/
  config.py
  data/
  features/
  models/
  visualization/
scripts/
tests/
outputs/
  figures/
  metrics/
  models/
```

The numbered notebooks are added as each exploratory pass begins.
`notebooks/01_data_exploration.ipynb` contains the non-predictive data
exploration pass for participant counts, epoch counts, missingness, recording
durations, label balance, excluded epochs, and signal schema checks. Reusable
logic belongs under `src/`, and tests should be added alongside each
nontrivial behavior.

## First Milestone

1. Validate or copy the prior project split assignments.
2. Build a DREAMT epoch index with preparation epochs excluded.
3. Extract basic statistical features into split CSV tables.
4. Tune baseline and traditional ML models with participant-grouped cross-validation on the training split only.
5. Evaluate the best fitted model from each family on validation data only.
6. Reserve the held-out test split for final evaluation after choices are frozen.

## Rich Feature Engineering

The default feature-build command now writes the richer engineered feature set:

```bash
python scripts/build_features.py
```

This generates train, validation, and test feature CSVs plus
`data/processed/feature_manifest.csv`. The test feature table is generated only
so it is available after final model choices are frozen; model selection,
predictive EDA, and validation comparisons should not load or inspect test
labels, metrics, or predictions.

The rich feature set is assembled in code from modular feature families:

- basic statistical epoch summaries
- signal-specific features such as movement intensity and short-window IBI/HRV proxies
- participant-contained rolling context features
- whole-night participant-normalized z-score features
- CSV manifest metadata for signal groups and feature families

Whole-night participant normalization is intentional for this within-night
retrospective staging target. Learned preprocessing in model pipelines should
still be fit on the training split only.

To rebuild the original first-milestone statistical table, pass:

```bash
python scripts/build_features.py --feature-set basic
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,interpretability]"
pytest
python scripts/copy_previous_split.py
```

Place local DREAMT participant CSVs such as `S002_whole_df.csv` under
`data/raw/`. These files are ignored by git and should not be committed.

Then run the first milestone pipeline:

```bash
python scripts/build_epoch_index.py
python scripts/build_features.py
python scripts/train_models.py
```

To audit whether participant sleep-stage labels follow a stable 30-second epoch
phase before rebuilding artifacts, run:

```bash
python scripts/audit_epoch_alignment.py
```

The audit writes `outputs/metrics/epoch_alignment_audit.csv` and is useful for
finding recordings like S077, where one global epoch offset is not a good fit.
An opt-in segmented epoching mode is available for experiments around long
unmapped-label gaps:

```bash
python scripts/build_epoch_index.py \
  --segmented-epoching \
  --output data/interim/epoch_index_segmented.csv
```

The default epoch-index command still uses the original single-offset behavior.
When segmented epoching is enabled, long unmapped-label regions such as
preparation (`P`) blocks and long `Missing` spans are treated as segment
boundaries rather than ordinary indexed epochs. As a result, descriptive
notebooks may no longer show `P` in raw-label distributions, and missingness
summaries may drop missing signal values that occurred only inside those
excluded unmapped regions.

For descriptive data-quality exploration, run
`notebooks/01_data_exploration.ipynb` after `data/interim/epoch_index.csv` has
been generated. The notebook uses the full dataset only for non-predictive
summaries and should not be used to select features, tune models, or guide
held-out test decisions.

`scripts/train_models.py` uses `GroupKFold` on training participants to select
hyperparameters by macro F1, refits the best model from each family on the full
training split, and writes validation-only metrics and predictions. The test
feature table is intentionally not loaded during this milestone.

If your DREAMT CSVs are not sampled at 64 Hz, pass the correct sampling rate or
explicit rows per 30-second epoch:

```bash
python scripts/build_epoch_index.py --sampling-rate-hz 100
# or
python scripts/build_epoch_index.py --rows-per-epoch 3000
```
