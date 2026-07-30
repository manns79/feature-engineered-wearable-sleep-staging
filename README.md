# Feature-Engineered Machine Learning for Wearable Sleep Staging

This repository extends the sibling
[`dreamt-wearable-sleep-staging`](https://github.com/manns79/dreamt-wearable-sleep-staging)
project on the same DREAMT wearable sleep-staging task: classifying each
30-second epoch as `Wake`, `Non-REM`, or `REM`. The earlier project compared
traditional baselines with deep learning models. This project asks whether
richer physiological feature engineering and classical machine-learning models
can approach that performance while remaining simpler and, for a pruned rolling
logistic model, substantially more interpretable.

The answer is nuanced. A feature-engineered elastic-net logistic model reached
nearly the same held-out test macro F1 as the previous best deep model, but REM
classification remained weak and highly sensitive to post-processing choices.
Macro F1 is the average of the per-class F1 scores, so it gives equal weight to
`Wake`, `Non-REM`, and the less frequent `REM` class.

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
- **Best test macro F1:** Feature-engineered elastic-net logistic regression
  with train-OOF Platt calibration, probability smoothing, and class-threshold
  tuning reached **0.497** test macro F1.
- **External comparison:** The previous project's transition-regularized
  61-epoch MSResCNN-MLP-TCN reached **0.501** test macro F1 on the same label
  mapping and participant-level split.
- **Best REM sensitivity:** The raw interpretable rolling logistic model had
  lower test macro F1, **0.468**, but the strongest REM F1 among the main final
  candidates, **0.259**, with REM recall **0.537**.
- **Preferred interpretation target:** The pruned rolling elastic-net logistic
  model remained close to the best feature-engineered model after threshold
  tuning, **0.492** test macro F1, while exposing readable coefficient-level
  associations.
- **Central limitation:** REM remained the principal failure mode; calibration
  and thresholding changed the REM operating point substantially.

## Key Results

These rows summarize the final held-out test comparison plus the previous
project's best deep model. The deep model was trained in the sibling repository,
not here, but the comparison is meaningful because both projects reuse the same
three-class label mapping and participant-level split. Differences of only a few
thousandths should not be interpreted as statistically significant.

| Model | Validation macro F1 | Test macro F1 | Wake F1 | Non-REM F1 | REM F1 |
| ----- | ------------------: | ------------: | ------: | ---------: | -----: |
| Majority-class baseline | 0.276 | 0.266 | 0.000 | **0.797** | 0.000 |
| Stratified-random baseline | 0.326 | 0.337 | 0.248 | 0.649 | 0.114 |
| Statistical-summary elastic-net logistic, post-processed | 0.438 | 0.445 | 0.542 | 0.689 | 0.105 |
| Engineered-feature elastic-net logistic, post-processed | **0.520** | 0.497 | 0.549 | 0.775 | 0.167 |
| Interpretable rolling logistic, raw | 0.410 | 0.468 | 0.556 | 0.589 | **0.259** |
| Interpretable rolling logistic, threshold-tuned | 0.481 | 0.492 | 0.553 | 0.727 | 0.196 |
| Previous project: transition-regularized 61-epoch MSResCNN-MLP-TCN | 0.510 | **0.501** | **0.564** | 0.793 | 0.146 |

The best feature-engineered model reached approximately 0.497 test macro F1,
within 0.004 of the earlier deep-learning benchmark. The interpretable rolling
logistic model reached approximately 0.492 after threshold tuning. Its raw
variant sacrificed macro F1 but preserved more REM signal than the stronger
macro-F1 operating points.

![Post-processing tradeoff](results/figures/postprocessing_tradeoff.png)

*Post-processing changed the operating point, especially for REM. The raw
rolling logistic model retained more REM F1, while thresholded variants improved
macro F1 by reducing false REM predictions.*

## Main Scientific Findings

### Classical ML Approached The Deep-Learning Benchmark

The central positive result is that a classical feature-engineered model came
very close to the previous deep-learning benchmark on the same split. The
engineered-feature elastic-net logistic model reached 0.497 test macro F1, while
the previous transition-regularized MSResCNN-MLP-TCN reached 0.501.

This does not prove that classical ML is universally preferable. It does show
that careful temporal and physiological feature engineering can produce a strong
performance-versus-complexity tradeoff. The comparison is useful because the
classical model is architecturally simple, and the logistic variants support
direct feature-level interpretation that is much harder to obtain from the deep
sequence model.

### The Interpretable Model Remained Competitive

The pruned rolling elastic-net logistic model was designed for interpretation:
it uses a constrained set of rolling cardiovascular and movement-context
features, deterministic train-only correlation pruning, and elastic-net
regularization. It did not produce the highest final macro F1, but it remained
close to the best feature-engineered model after threshold tuning.

The model is scientifically useful because it exposes a real tradeoff. Its raw
predictions had the strongest REM F1 among the main final candidates, but this
came from high REM recall and many false REM predictions, particularly among
true `Non-REM` epochs. Threshold tuning improved macro F1 by moving the model to
a more conservative REM operating point.

### Post-Processing Changed The Class Tradeoff

Post-processing was not cosmetic. It changed which scientific conclusion one
would draw from the same base model.

Platt calibration by itself often increased accuracy or shifted probabilities
toward `Non-REM` while sharply suppressing REM predictions. For the rolling
logistic model, Platt-only test accuracy was 0.709, but REM F1 was 0.000. The
best engineered-feature ablation showed a similar pattern: Platt-only test
accuracy was 0.715, but REM F1 was only 0.011.

Class-threshold tuning was the main source of macro-F1 improvement. The raw
rolling logistic model moved from 0.468 test macro F1 to 0.492 with raw
threshold tuning. For the best engineered-feature ablation, Platt plus threshold
tuning reached 0.493, and adding probability smoothing produced a modest
increase to 0.497.

This distinction matters:

- probability calibration changes the probability scale;
- class-threshold tuning changes the decision rule;
- temporal smoothing changes output probabilities across neighboring epochs;
- rolling input features provide temporal context before the classifier makes
  predictions.

Viterbi decoding was explored during validation for the rolling logistic model,
but it collapsed REM recall in the saved validation artifacts and was not part
of the final locked-test roster.

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

## Technical Approach

The engineered feature table combines:

- basic within-epoch statistical summaries;
- signal-specific physiological and movement features;
- centered rolling temporal-context features over neighboring epochs;
- whole-night participant-normalized features.

Concrete examples include accelerometer motion intensity and magnitude energy,
BVP summary behavior, EDA rise/fall features, temperature and heart-rate drift,
short-window IBI/HRV proxies such as RMSSD and pNN50, and rolling means and
standard deviations around each epoch.

The scope is retrospective within-night sleep staging. Centered rolling features
use both earlier and later neighboring epochs, and whole-night normalization uses
the participant's complete recording. Those decisions are appropriate for this
analysis target but are not streaming-compatible real-time inference.

Evaluated models included sanity-check dummy baselines, elastic-net multinomial
logistic regression, random forest, and XGBoost. Elastic-net logistic regression
became the most important family because it produced the best final result and
supported the most useful interpretation.

## Dataset And Evaluation Design

This project uses [DREAMT](https://physionet.org/content/dreamt/2.2.0/)
wearable physiological signals, including `BVP`, accelerometry, temperature,
EDA, HR, and IBI. PSG labels `N1`, `N2`, and `N3` are mapped to `Non-REM`,
preparation epochs are excluded, and the final task is `Wake` versus `Non-REM`
versus `REM`. The earlier repository contains the fuller dataset and cohort
description.

Evaluation safeguards:

- fixed participant-level train/validation/test split reused from the earlier
  project;
- participant-grouped cross-validation on training participants;
- validation-only model, ablation, and exploratory error-analysis decisions;
- train-OOF fitting of Platt calibration, class thresholds, and smoothing
  choices;
- one locked held-out test evaluation after the final protocol was frozen;
- macro F1 as the primary metric because accuracy can obscure minority-class
  failure.

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

## Skills Demonstrated

**Machine learning and statistical modeling**

- Elastic-net multinomial logistic regression, random forest, XGBoost, and
  sanity-check baselines.
- Participant-grouped cross-validation and class-imbalance-aware macro-F1
  evaluation.
- Feature-family and signal-group ablations.
- Probability calibration, class-threshold tuning, temporal smoothing, and
  validation-only sequence decoding experiments.
- Coefficient interpretation, native feature importance, permutation
  importance, and optional SHAP summaries.

**Biomedical time-series analysis**

- Processing multimodal wearable signals from DREAMT.
- Physiological feature engineering for movement, cardiovascular,
  electrodermal, and temperature signals.
- Temporal-context construction and participant-level normalization.
- Sleep-stage transition analysis and participant-level failure analysis.
- REM-focused error grouping for physiological interpretation.

**Machine-learning engineering**

- Modular Python source code rather than notebook-only analysis.
- Reproducible command-line experiment scripts.
- Manifest-driven feature and ablation definitions.
- Train-only preprocessing and post-processing selection.
- Frozen held-out test protocol.
- Automated tests, linting support, structured experiment outputs, curated
  result artifacts, and reproducible visualizations.

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

## Project Takeaway

Carefully engineered temporal and physiological features allowed classical ML to
approach the earlier deep-learning benchmark on the same wearable sleep-staging
task. The interpretable rolling logistic model remained competitive and
preserved more REM sensitivity, but reliable REM classification remains the main
open challenge.

## Curated Results Artifacts

The tracked `results/` directory contains compact artifacts that support this
README:

- [results/summary/key_results.csv](results/summary/key_results.csv) contains
  the Key Results table and model-name crosswalk fields.
- [results/summary/validation_ablation_summary.csv](results/summary/validation_ablation_summary.csv)
  summarizes validation-only ablation findings.
- [results/summary/coefficient_contrast_summary.csv](results/summary/coefficient_contrast_summary.csv)
  supports the rolling logistic interpretation.
- [results/MANIFEST.md](results/MANIFEST.md) documents each curated file and
  its source artifact.

Figures are regenerated from saved summary artifacts with:

```bash
python scripts/generate_readme_figures.py
```

The script does not train models, tune hyperparameters, or reopen the locked
held-out test protocol.

## Repository Entry Points

- [src/features/](src/features/) contains the physiological feature-engineering
  modules.
- [src/models/ablations.py](src/models/ablations.py) defines manifest-driven
  ablation experiments.
- [src/models/rolling_logistic_experiment.py](src/models/rolling_logistic_experiment.py)
  implements the pruned interpretable rolling logistic experiment.
- [src/models/calibration.py](src/models/calibration.py) and
  [src/models/sequence_postprocessing.py](src/models/sequence_postprocessing.py)
  implement calibration, thresholds, smoothing, and sequence post-processing.
- [src/models/locked_test_evaluation.py](src/models/locked_test_evaluation.py)
  contains the frozen final evaluation logic.
- [src/models/rolling_logistic_interpretation.py](src/models/rolling_logistic_interpretation.py)
  generates the final interpretation artifacts.
- [notebooks/03_validation_error_analysis.ipynb](notebooks/03_validation_error_analysis.ipynb)
  reviews validation-stage failure analysis.
- [notebooks/04_locked_test_evaluation.ipynb](notebooks/04_locked_test_evaluation.ipynb)
  reviews the already-frozen locked-test artifacts.

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
