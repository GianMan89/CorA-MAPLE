# Experimental protocol

## Scope

The analysis is offline: one class prediction is produced from each complete
60-sample observation. There is no renewed flood detection, online prefix averaging
or early-decision metric. Values beyond the first hour are absent from the bundled
data. All methods use the same channels, observation window, class labels and
perturbed test inputs.

The default identifier is `cora_maple_cv10x5_v1`. Its two studies are `primary`
(`configs/repeated_cv.yaml`) and `pool_sizes` (`configs/pool_sizes.yaml`).

## Repetition and splitting

`RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=2027)` independently
randomizes five-fold partitions in each repetition. Every source is tested once
per repetition. Splitting occurs at complete-recording level before excerpts,
label selection, model fitting or test-corruption generation. Exact duplicate
inputs are kept together if supplied; conflicting labels for duplicate inputs are
rejected. The packaged TEP data contain 1,000 unique first-hour observations.

Each split has 800 training sources and 200 test sources, with 160/40 sources per
class. Label budgets are 1, 2, 5 and 20 unique observations per class. Larger budgets
contain the smaller labelled subsets. Source IDs are shared across methods.

A repetition is not ten initializations of a fixed fold partition: partitions,
label selections and training random streams change between repetitions. There is
one training initialization for each method and labelled-subset condition in each
split. Neural encoder and classifier-head initializations are paired between
methods within a split. Separate seed streams control crops, masks, event corruption,
supervised sample order and dropout. All seeds and source roles are serialized.

Only the last training checkpoint is evaluated. No inner CV, validation-dependent
schedule or test-driven checkpoint selection is used.

## Pretraining history

The primary experiment uses the complete outer training pool with its class labels
hidden from pretraining. CorA-MAPLE-F samples full recordings uniformly with
replacement. CorA-MAPLE-E selects one length from 15, 30, 45 or 60 samples per
minibatch, then samples sources and valid crop starts uniformly. The crop selector
does not inspect class labels, alarm-rate thresholds or whether an excerpt is empty.

Available history is `N_U * T * dt / 60` hours for equal-length recordings. For the
packaged 60-minute observations its numeric value equals `N_U`. Full=800 is an
available-data budget, not a guarantee of exhaustive visitation. `source_coverage.npz`
records source draw counts and a union mask over visited time samples. Repeated
visits contribute to cumulative sampled hours but do not count as new history.
At 800 updates and batch size 32, full episodes process 25,600 cumulative hours;
excerpts process 16,000 hours in expectation, counting repeated observations.

## History-size sensitivity

At fixed k=1, each split has three class-blind random permutations of the outer
training sources. Prefixes of length 1, 2, 5, 10, 20, 50, 100, 200 and 400 produce
nested pools. Both proposals share these pools. Labelled examples and initialization
seeds stay fixed while pool size and pool selection vary. Labelled examples are not
forced into the pretraining pool, so N_U=1 means one recording in total.

The full-pool endpoint references the matching primary task in the same split. It
is not fitted or counted three times. No archived three-fold endpoint is used.

| Study | Distinct classifiers | Distinct pretraining tasks |
|---|---:|---:|
| Primary: 50 splits × 12 methods × 4 label budgets | 2,400 | 250 |
| Reduced history: 50 × 3 pools × 9 sizes × 2 methods | 2,700 | 2,700 |
| Full-pool history endpoint | 100 reused | 100 reused |
| Total unique tasks | 5,100 classifiers | 2,950 pretraining tasks |

## Perturbations

Training leaves a view unchanged with probability 0.2. Otherwise it samples one
of missing, spurious or timing corruption uniformly at severity uniform on
[0.05,0.20]. Masking is independent and still applies to uncorrupted views.

All operators act on reconstructed ACT/RTN notifications, use trace repair and
return to the same binary-state grid. Missing corruption deletes floor(s*n)
notifications. Spurious corruption inserts floor(s*n) notifications with uniformly
sampled tags, types and times. Timing shifts floor(s*n) selected events by a
uniform displacement up to s*10 minutes, clipped to the observation window.

Three structured tests are outside the pretraining operator distribution. Burst
insertion samples times around a random centre with a two-minute half-width. Tag
dropout removes every notification for floor(s*C) sampled channels. Mixed corruption
applies missing events, repairs, then applies spurious events at the same severity
relative to the resulting event count. A leading RTN can imply activity at the
window start under the retained repair convention. Severity is an event-edit or
channel-loss parameter, not the fraction of state cells altered.

The test suite has one clean condition and six operators × three positive severities
(0.1,0.2,0.3) × three random realizations: 55 conditions per fitted classifier.
Conditions are shared by every method within a split; changing worker counts does
not change them. Corruption does not alter ground-truth disturbance classes.

## Aggregation

Macro-F1 is computed with the full five-class label list and zero_division=0.
Reporting first averages perturbation realizations within each classifier and
then applicable pool selections within each fixed repeat/fold/condition. The five
fold scores are averaged for each repetition. The final mean and sample SD use
the ten repetition means. For paired comparisons, method differences are calculated
on matching folds, labels and corruption conditions before the same aggregation.

Per-draw, per-fit, per-split and per-repeat outputs are all retained. SDs describe
repetition variability and are not independent-fold confidence intervals. The
same 1,000 sources are reused across repetitions; this does not create 10,000
independently collected recordings.

## Reference documentation

- scikit-learn RepeatedStratifiedKFold:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html
- PyTorch reproducibility:
  https://docs.pytorch.org/docs/stable/notes/randomness.html
