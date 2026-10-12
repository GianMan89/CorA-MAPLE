# Frozen clean-supervision protocol

Protocol identifier: `cora_maple_clean_ft_cv10x5_v2_1`.
Scientific defaults are in `configs/base.yaml`; study grids are in
`configs/repeated_cv.yaml` and `configs/pool_sizes.yaml`. Full resolved configurations,
code hashes and dataset hashes are written before model construction.

## Source-level roles

Five-fold CV is repeated ten times with stratification by fault class. Each split
contains 800 training and 200 held-out source recordings (160/40 per class).
Splitting precedes excerpt extraction and no held-out source enters either training
stage. Exact duplicate matrices are grouped instead of crossing source roles;
all 1,000 current matrices are unique. Each source appears exactly once as a test
recording within each repetition. Repetitions reuse the same underlying 1,000
recordings.

Nested labelled sets contain k=1,2,5,20 records per class and are shared by all
methods within a split. The same initialization, head, batch and dropout seed
identifiers are paired across pretraining controls where meaningful. These are
one initialization and one label selection per condition per split, not ten
independent initialization replications on a fixed train/test set.

## Full-pool comparison

Six neural conditions (three within E, three within F) and five classical methods
are evaluated at four label budgets:

- 50 splits × 6 neural methods × 4 label budgets = **1,200 neural classifier fits**.
- 50 splits × 5 classical methods × 4 label budgets = **1,000 classical fits**.
- 50 splits × 4 SSL conditions = **200 shared pretraining fits**.

Every classifier is trained on clean labelled episodes only. MAPLE and CorA-MAPLE
within a scope use identical unlabelled sources, architecture, losses and training
schedules, differing only in corruption exposure during SSL. Scratch has no
pretraining, but the same architecture and supervised schedule. Each SSL checkpoint
serves all four label budgets, with newly initialized classifier heads and separate
fine-tuning. No held-out accuracy, F1 or robustness score controls checkpoint
selection or the learning-rate schedule.

E and F use the same architecture and **4,000/2,400 pretraining/supervised updates**.
An E input contains 37.5 samples on average, versus 60 for F, so cumulative temporal
exposure differs by a factor of 1.6 at equal updates. The optional `short_schedule`
diagnostic uses 1,600/900 updates for both scopes in a separate workspace; it is
not included in the default main run.

## Amount of available history

At k=1, CorA-MAPLE-E/F use total pretraining source counts
`[1,2,5,10,20,50,100,200,400,full]`. These are counts **across all classes**, not
per-class budgets. Every TEP source contributes one available hour, so H_U=N_U h
and Full=800 h. Small pools do not have to cover all five classes.

Three nested class-blind random source permutations are sampled per split. The
same pool selections are used by E/F, with no mandatory inclusion of labelled
training sources. At N_U=1, the pool is precisely one recording; supervised labels
remain the independently selected five class-balanced recordings.

There are 50 × 2 × 9 × 3 = **2,700 additional pretraining and classifier fits**.
The full-pool endpoints reuse 100 pretraining tasks and 100 classifier tasks from
the current primary run. Across both studies this yields **2,900 distinct SSL fits**
and **4,900 classifier fits**, or **7,800 unique tasks**. The full primary and
history studies must finish before exporting their manuscript results.

Smaller pools retain the same update schedule and therefore repeat sources more
frequently. Each pretraining task records available hours, sampled input hours,
unique source-time coverage, observed source count and per-operator view counts.
These distinguish available history from cumulative processed data. They do not
claim a minimum sufficient corpus size.

## Test suite and measurement

Six published operators are evaluated at severity 0,0.1,0.2,0.3,0.4,0.5. A single
uncorrupted reference condition and three realizations per operator/positive
severity yield **91 conditions per classifier**. All models and label/pool budgets
reuse the same per-source transformed observations and valid-length metadata
within a split. There is no adaptation using test corruptions.

The primary metric is five-class Macro-F1. Accuracy, per-class predictions,
clean-to-degraded absolute drops and normalized trapezoidal severity areas are
also retained. Within each fitted classifier, average the three perturbation
realizations. For pool studies, average the three pool selections within each
split. Next average five folds within each repetition, then report the mean and
sample standard deviation across ten repetition means.

Group metrics use equally weighted operator means computed within folds before
repetition aggregation. The **trained** group contains false_full, missing_full and
timing, which are sampled only during CorA-MAPLE pretraining. The **test_only** group
contains late_start, missing and spurious, absent from both stages. The redundant
pretraining_only group records the stage explicitly in machine-readable output.
These are operator-balanced diagnostics, not estimates weighted by real-world
failure prevalence. Raw per-pool and per-draw results prevent this aggregation from
concealing selection variability.

## Statistical and external validity

Hyperparameters are fixed for the configured run, and held-out metrics are not
used for checkpoint selection, learning-rate adaptation or in-run model selection.
Repeated CV partitions still reuse the same underlying recordings and do not create
independent external confirmation. General performance claims require independent
recordings, process domains, or an appropriate nested validation design. Any
score-informed change to the configuration must be evaluated as a distinct study.

The full-channel false-alarm mechanism is especially identifiable on full TEP
recordings because every clean first sample is inactive. Channel-forcing, event
loss and delayed history have distinct physical consequences at the same numeric
severity. Initial-state-aware reconstruction assumes a reliable snapshot. These
facts must accompany scientific interpretation of the resulting rankings.

## Optional diagnostics

Each has a separate explicit configuration; none is silently selected for better
scores or substituted for the default protocol.

- `short_schedule`: 1,600/900 updates for both E and F at k=1,20; evaluates training
  duration relative to the common 4,000/2,400 main schedule.
- `variability_k1`: crossed label draws and neural seeds (three each), k=1;
  quantifies the difference between label selection and initialization variability.

The latter can reuse the default context in a separately named study if settings
match, but using a separate workspace keeps its cost and reporting explicit.
