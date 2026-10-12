# Experimental design

## What is compared

The study separates two questions: whether unlabelled representation learning
improves label efficiency (MAPLE versus Scratch), and whether explicit corruption
exposure improves the learned representation (CorA-MAPLE versus MAPLE). All
classifiers receive clean labelled episodes and no supervised augmentation. This
keeps the comparisons focused on pretraining.

E pretrains on excerpts; F uses complete supplied episodes. Both have the same
96-dimensional, three-layer Transformer and identical optimizer settings, loss
weights, and **4,000/2,400 pretraining/supervised update counts**. Four pretrained
models per split (MAPLE/CorA × E/F) each supply four separately fitted classifiers
for k=1,2,5,20. Scratch is a no-pretraining control. Its E/F entries have identical
settings and paired seeds and do not constitute independent model replications.

## Resource comparison

All methods use identical labelled subsets and held-out observations. Only models
with pretraining use the additional unlabelled pool. Equal optimization steps do
not equalize observation exposure: F processes 60 real samples per input and E
37.5 on average, yielding a factor of 1.6. Available historical hours, processed
sample-minutes, visited source counts and source-time coverage are recorded.

The optional `short_schedule` diagnostic applies 1,600/900 updates to **both**
scopes. It must run in a separate workspace and is not selected automatically.
The optional `variability_k1` diagnostic crosses label selections and initialization
seeds to distinguish these two sources of variation.

## Threat model and evaluation

CorA pretraining samples full-channel false activation, full-channel loss, and
timestamp uncertainty at 5–30% severity, with 20% unchanged views. Delayed
observation, missing notifications and additional notifications are excluded from
all training. Tests extend to 50%. Thus, trained-operator severity extrapolation
and transfer to a related but unseen event mechanism are separate claims.

The initial-state-aware event reconstruction and valid-length-aware delay adapter
are shared by all methods. They are part of the observation model, not a neural
advantage or learned class-dependent correction. A reliable initial snapshot must
exist in an intended application.

## Evidence and reporting

All hyperparameters are fixed in the configuration for a run; checkpoints are not
chosen using held-out scores. Full per-method, per-operator, per-pool and per-draw
measurements are retained. Group means are operator-balanced summaries, not estimates
of expected industrial performance. A smaller pool changes source diversity, class
coverage and repeated exposure together, so a history-size curve cannot establish a
universal minimum data requirement.

Software tests validate execution, identities, padding, objectives and arithmetic;
they do not demonstrate the method's accuracy or robustness. Repeated splits of one
synthetic process are not independent external validation. No performance outcome
is assumed or inserted in advance of executing the configured experiment.
