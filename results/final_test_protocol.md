# Final Locked Test Evaluation Protocol

This protocol freezes the final held-out test evaluation before inspecting test
results.

## Model Roster

The final evaluation includes:

- Interpretable rolling-context elastic-net logistic regression from
  `outputs/runs/rolling_logistic_train_oof_postprocessed_20260724`.
- Best original ablation model:
  `basic_signal_specific_rolling_subject_norm` +
  `elastic_net_logistic_regression`.
- Best statistical-summary-only model:
  `basic_statistical` + `elastic_net_logistic_regression`.
- Raw majority-class dummy baseline from `basic_statistical`.
- Raw stratified-class dummy baseline from `basic_statistical`.

## Post-Processing Rule

For non-dummy finalized models, Platt calibration, class-threshold tuning, and
probability-smoothing window selection must be fit from participant-held-out
training OOF predictions only. Validation and test labels must not be used to
fit any post-processing rule.

The dummy baselines are reported raw only so they remain sanity checks rather
than tuned classifiers.

## Reporting

The primary comparison table reports validation macro F1, test macro F1, and
test Wake/Non-REM/REM F1. Compact locked-test error analysis reports per-class,
per-participant, transition-distance, and low-REM-support summaries.

## Command

```bash
python scripts/run_final_test_evaluation.py
```

By default this writes final artifacts under
`outputs/runs/final_test_evaluation/`.

## Rolling Logistic Interpretation

After final test artifacts exist, generate script-only interpretation artifacts
for the interpretable rolling logistic model with:

```bash
python scripts/run_rolling_logistic_interpretation.py
```

This interprets raw-model standardized coefficients separately from
post-processing variants, which are summarized as operating-point changes.
