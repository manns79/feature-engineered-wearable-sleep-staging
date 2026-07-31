# Feature-Engineered Machine Learning for Wearable Sleep Staging

This repository extends the [`dreamt-wearable-sleep-staging`](https://github.com/manns79/dreamt-wearable-sleep-staging) project. Refer to the previous project's `README.md` for details about the DREAMT dataset. 

## Executive Summary

- **Research question:** Can classical ML with richer engineered wearable
  features match the previous deep-learning benchmark closely enough to offer a
  simpler and more interpretable alternative?
- **Task:** Three-class retrospective sleep staging on DREAMT wearable signals:
  `Wake`, `Non-REM`, and `REM`.
- **Evaluation design:** Fixed participant-level train/validation/test split
  reused from the earlier project, participant-grouped cross-validation on
  training participants, validation-only model selection, and one frozen
  held-out test evaluation.
- **Feature families:** Within-epoch statistical summaries, signal-specific
  physiological features, centered rolling temporal-context features, and
  whole-night participant-normalized features.
- **Model families:** Majority and stratified dummy baselines, elastic-net
  multinomial logistic regression, random forest, and XGBoost.
- **Best performing model:** Feature-engineered elastic-net logistic regression
  with train-OOF Platt calibration, probability smoothing, and class-threshold
  tuning reached **0.497** test macro F1.
- **Comparison to DL benchmark:** The previous project's transition-regularized
  61-epoch MSResCNN-MLP-TCN reached **0.501** test macro F1 on the same label
  mapping and participant-level split.
- **Most interpretable model:** The pruned rolling elastic-net logistic
  model remained close to the best feature-engineered model after threshold
  tuning, **0.492** test macro F1, while exposing readable coefficient-level
  associations.

## Key Results

The table below summarizes model performance. Per-class F1 scores are reported on the held-out test set. The last row in this table, transition-regularized 61-epoch MSResCNN-MLP-TCN, is the best performing model from [`dreamt-wearable-sleep-staging`](https://github.com/manns79/dreamt-wearable-sleep-staging). The value shown in bold in each column denotes the highest obtained value of the corresponding F1 score.

| Model | Validation macro F1 | Test macro F1 | Wake F1 | Non-REM F1 | REM F1 |
| ----- | ------------------: | ------------: | ------: | ---------: | -----: |
| Majority-class baseline | 0.276 | 0.266 | 0.000 | **0.797** | 0.000 |
| Stratified-random baseline | 0.326 | 0.337 | 0.248 | 0.649 | 0.114 |
| Statistical-summary elastic-net logistic, post-processed | 0.438 | 0.445 | 0.542 | 0.689 | 0.105 |
| Engineered-feature elastic-net logistic, post-processed | **0.520** | 0.497 | 0.549 | 0.775 | 0.167 |
| Interpretable rolling logistic, raw | 0.410 | 0.468 | 0.556 | 0.589 | **0.259** |
| Interpretable rolling logistic, threshold-tuned | 0.481 | 0.492 | 0.553 | 0.727 | 0.196 |
| transition-regularized 61-epoch MSResCNN-MLP-TCN | 0.510 | **0.501** | **0.564** | 0.793 | 0.146 |

Interpretation:
- The best performing traditional ML model, which used all feature and signal families, achieved performance comparable to the DL benchmark.
- After threshold tuning, an intepretable elastic-net logistic model that only used rolling context features from the movement and cardiovascular signal families also achieved performance comparable to the DL benchmark (see the next-to-last row).
- Generally, Platt scaling improved probability calibration but not necessarily macro F1; probability smoothing had little to no effect; and per-class threshold tuning improved performance by reducing overprediction of REM.  


Using the intepretable elastic-net logistic model, the figure below further illustrates the consequences of the different post-processing techniques used. 

![Post-processing tradeoff](results/figures/postprocessing_tradeoff.png)


## Main Scientific Findings


### Stable Epochs Were Easier Than Transition Epochs

Errors were worse near true sleep-stage transitions. For the best final
feature-engineered model, locked-test macro F1 improved from 0.395 on transition
epochs to 0.461 one epoch away, 0.501 two to three epochs away, and 0.544 four
to ten epochs away.

![Transition-distance macro F1](results/figures/transition_distance_macro_f1.png)

*Performance generally improved away from true sleep-stage boundaries. This
supports the interpretation that stable sleep regions are easier to classify
than physiologically ambiguous transition regions.*

### REM Remained The Central Failure Mode

REM was difficult even though the test split contained 1,318 REM epochs. Low
participant-level support still matters: participant `S081` had zero REM epochs
and `S050` had only 12. Their participant-level REM metrics are therefore
unstable or undefined in a practical sense, but the global REM weakness cannot
be dismissed as a support artifact.

The raw rolling logistic model's REM error analysis makes the limitation clear:
it correctly identified 708 REM epochs, but also predicted REM for 3,038 true
`Non-REM` epochs and 407 true `Wake` epochs. The wearable features carried REM
signal, but they did not cleanly separate REM from quiet `Non-REM`.

## Interpretable Rolling Logistic Model

The final interpretation phase focused on the raw rolling logistic model because
its coefficients describe the fitted classifier directly. Post-processing
variants are better understood as operating-point changes, not as feature-level
associations.

The selected rolling model uses 25 train-pruned features from cardiovascular and
movement signal groups. Candidate features were defined from the feature
manifest, correlations were computed on training data only, and redundant
rolling-window features were pruned deterministically before grouped
cross-validation tuning. Post-processing choices were fit from participant-held-
out training predictions, not validation or test labels.

### Coefficient-Level Interpretation

The strongest coefficient contrasts were movement-context features, especially
`ACC_MAG_std_roll15_mean`:

- `REM` vs `Non-REM`: -0.949
- `Wake` vs `Non-REM`: +0.767
- `REM` vs `Wake`: -1.716

This suggests that the model's largest axis is sustained movement variability:
more movement-like context pushes toward `Wake`, while low movement variability
pushes toward sleep-like states, including REM. This is an association learned
by a standardized logistic model, not a causal physiological claim.

Autonomic rolling features also contributed to `REM` versus `Non-REM`
separation. Positive `REM` contrasts included `HR_mean_roll5_std` (+0.527),
`IBI_mean_roll5_std` (+0.315), and `IBI_pnn50_roll15_std` (+0.276). In contrast,
`IBI_pnn50_roll15_mean` pushed away from both `Wake` and `REM` toward
`Non-REM`, consistent with the model using sustained HRV-like context as a
Non-REM signal.

![Rolling logistic REM contrast](results/figures/rolling_logistic_rem_contrast.png)

*The rolling logistic model mainly distinguished REM from Non-REM through low
movement variability plus short-window HR/IBI variability. These are model
associations, not causal physiological effects.*

### REM Error Interpretation

The raw rolling model often treated quiet `Non-REM` as REM. For example,
`ACC_MAG_std_roll15_mean` was almost identical for true REM predicted REM
(0.2975) and true `Non-REM` predicted REM (0.2990), while non-REM-related rows
had a much higher mean value (0.8217). This supports a conservative conclusion:
rolling wearable features contain REM-relevant information, but quiet `Non-REM`
can look REM-like in this feature space.

## Validation Ablation Findings

Ablations were selected and interpreted on validation data only. They were not
used to retrospectively choose models after seeing the locked test set.

Adding signal-specific features to basic statistical summaries did not improve
the best validation macro F1 by itself: 0.385 for basic summaries versus 0.385
for basic plus signal-specific features. Rolling temporal context produced the
larger jump: the best cumulative rolling-context model reached 0.452. Adding
whole-night participant-normalized features produced the strongest validation
model, an elastic-net logistic regression at 0.470.

Among individual signal groups, movement was strongest on validation
(0.456), followed by cardiovascular features (0.425), temperature (0.385), and
electrodermal features (0.381). Random forest was competitive in several
ablations, but the best overall validation model and the final locked-test
winner both used elastic-net logistic regression. XGBoost was not meaningfully
better than logistic regression in the final feature-family comparisons.



## End-To-End Workflow

The repository implements an end-to-end applied ML workflow:

1. data validation and PSG epoch construction;
2. modular physiological feature engineering;
3. participant-grouped model tuning;
4. feature-family and signal-group ablations;
5. interpretable rolling-feature selection;
6. calibration, threshold, smoothing, and sequence post-processing experiments;
7. frozen held-out test evaluation;
8. transition, participant-level, and REM-focused failure analysis;
9. curated result artifacts and reproducible README visualizations.

## Repository Structure

```text
feature-engineered-wearable-sleep-staging/
  README.md
  pyproject.toml
  data/
    README.md
    raw/          # local DREAMT files; ignored
    interim/      # split assignments and epoch indexes
    processed/    # generated feature tables; ignored
  docs/
    final_test_protocol.md
  notebooks/
    01_data_exploration.ipynb
    02_feature_exploration.ipynb
    03_validation_error_analysis.ipynb
    04_locked_test_evaluation.ipynb
  results/
    summary/      # compact tracked metrics
    figures/      # README figures
  scripts/
  src/
    data/
    features/
    models/
    visualization/
  tests/
```

Raw DREAMT files, generated feature tables, per-epoch predictions, trained
models, and large run outputs are not committed.

## Reproducibility

Set up the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,interpretability,notebooks]"
```

Run tests and linting:

```bash
pytest
ruff check .
```

Prepare local data and splits:

```bash
python scripts/copy_previous_split.py
python scripts/build_epoch_index.py
python scripts/build_features.py
```

`scripts/copy_previous_split.py` reuses the fixed split assignments from the
sibling project. If the sibling repository is not adjacent to this one, pass the
appropriate source path to the script or place a compatible
`data/interim/split_assignments.csv` file locally.

Run the main validation-stage experiments:

```bash
python scripts/run_ablation_experiments.py --run-id full_ablation_YYYYMMDD
python scripts/run_validation_error_analysis.py \
  --run-dir outputs/runs/full_ablation_YYYYMMDD
python scripts/run_rolling_logistic_experiment.py \
  --run-id rolling_logistic_train_oof_postprocessed_YYYYMMDD
```

The final test protocol has already been spent for the results in this README.
Do not rerun final-test commands merely while editing documentation. To review
or reproduce the frozen final artifacts intentionally:

```bash
python scripts/run_final_test_evaluation.py
python scripts/run_rolling_logistic_interpretation.py
```

## Limitations And Future Work

- REM performance remains weak, and REM behavior is sensitive to calibration
  and thresholding.
- Errors are concentrated near sleep-stage transitions.
- Participant-level performance is heterogeneous, and participant-level REM
  metrics are unstable when REM support is very low.
- Centered rolling features and whole-night normalization make the primary
  workflow retrospective rather than real-time.
- Results are from one dataset and one fixed participant-level split.
- Wearable signals are indirect proxies for PSG-defined sleep stage.

Future work could include external validation on another wearable sleep dataset,
participant-specific calibration or adaptation, methods designed specifically to
distinguish REM from quiet `Non-REM`, streaming-compatible alternatives to
centered rolling and whole-night normalization, and uncertainty estimates around
class-level and participant-level metrics.