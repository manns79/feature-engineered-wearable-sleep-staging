# Results Manifest

This directory contains a compact, tracked subset of result artifacts used by
the README. Large generated outputs, trained model files, per-epoch prediction
CSVs, local DREAMT data, and intermediate feature tables remain outside the
curated evidence set.

## Summary Files

- `summary/key_results.csv` contains the README Key Results table, including
  the external deep-learning comparison row from the sibling repository.
- `summary/validation_ablation_summary.csv` summarizes validation-only
  feature-family and signal-group ablation findings.
- `summary/coefficient_contrast_summary.csv`,
  `summary/rolling_rem_error_counts.csv`, and
  `summary/rolling_transition_distance_metrics.csv` support the rolling
  logistic interpretation discussion.
- `summary/artifact_manifest.csv` maps each curated file to its source
  artifact, metric scope, and README claim.

## Figures

- `figures/postprocessing_tradeoff.png` compares test macro F1 and REM F1
  across selected raw and post-processed variants.
- `figures/transition_distance_macro_f1.png` summarizes locked-test macro F1 by
  distance to the nearest true sleep-stage transition.
- `figures/rolling_logistic_rem_contrast.png` shows the largest standardized
  rolling-logistic coefficient contrasts for `REM` versus `Non-REM`.

## Model-Name Crosswalk

The README uses human-readable model names. Internal candidate and variant
identifiers are preserved in `summary/key_results.csv`:

| README name | Internal candidate | Variant |
| --- | --- | --- |
| Majority-class baseline | majority_class_baseline | raw |
| Stratified-random baseline | stratified_class_baseline | raw |
| Statistical-summary elastic-net logistic, post-processed | statistical_summary_only | platt_smoothed_threshold_tuned |
| Engineered-feature elastic-net logistic, post-processed | best_original_ablation | platt_smoothed_threshold_tuned |
| Interpretable rolling logistic, raw | interpretable_rolling_logistic | raw |
| Interpretable rolling logistic, threshold-tuned | interpretable_rolling_logistic | raw_threshold_tuned |
| Previous project: transition-regularized 61-epoch MSResCNN-MLP-TCN | stage19_best_equal_weight_seed_ensemble_lambda_0.05 | external_comparison |

## Regeneration

Run:

```bash
python scripts/generate_readme_figures.py
```

The script reads saved summary artifacts only. It does not train models, tune
models, alter the final held-out test protocol, or inspect raw DREAMT data.

