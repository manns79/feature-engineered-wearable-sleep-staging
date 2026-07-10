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
  interim/      # CSV split assignments and epoch indexes; ignored by git
  processed/    # CSV feature tables; ignored by git
  synthetic/    # optional small DREAMT-compatible smoke-test files
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

The numbered notebooks are intentionally not created yet; they should be added
when the first exploratory pass begins. Reusable logic belongs under `src/`, and
tests should be added alongside each nontrivial behavior.

## First Milestone

1. Copy or validate the prior project split assignments.
2. Build a DREAMT epoch index with preparation epochs excluded.
3. Extract basic and advanced engineered features into CSV tables.
4. Train baseline and traditional ML models using validation data for model
   selection.
5. Reserve the held-out test split for final evaluation after choices are frozen.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,interpretability]"
pytest
python scripts/copy_previous_split.py
```

