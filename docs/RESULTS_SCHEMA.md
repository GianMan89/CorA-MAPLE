# Result schema and manuscript exports

## Workspace identity

`context.json` records the common resolved settings, source-code SHA-256, dataset
SHA-256, class map and planning environment. `dataset_index.json` records source
IDs and observation hashes. `splits.json` records the 50 source-level splits.
Each study has `studies/<name>/config.json` and `plan.json` containing its exact
pretraining-comparison task graph and seed assignments. Any changed code/data/common settings
requires a new workspace rather than silent continuation.

## Task outputs

`tasks/<kind>_<hash>/` contains `task.json`, `training.json`, and checksummed
`completed.json`. Progress and exceptions are stored as `progress.json` and
`failure.json`. Completion checks verify the task identity and its recorded files.

**Pretraining:** `encoder.pt`, `pretraining.csv`, `source_coverage.npz`. Statistics
include optimizer updates, available sources, visited sources, unique sample
coverage, cumulative processed sample-minutes, per-operator exposures, maximum
observed training severity, and actual scope/schedule.

**Neural classification:** `classifier.pt`, `finetuning.csv`, `model_info.json`,
`predictions.npz`, `metrics.csv`, and `evaluation.json`. Fine-tuning logs include
clean CE, train-only accuracy, and separate encoder/head learning rates. The
corrupted_ce field is identically zero because supervised training is clean-only;
operator_exposures is empty and corrupted_sample_steps is zero. These fields
provide an explicit runtime check that no supervised corruption was applied. No validation/test metric
is used by training. The final checkpoint, not a selected best test checkpoint,
is retained.

**Classical classification:** `baseline.joblib`, `model_info.json`, predictions,
metrics and evaluation metadata. `joblib` is a trusted-local model serialization;
it must not be loaded from untrusted sources. Neural loading uses
`torch.load(..., weights_only=True)`.

**Prediction NPZ:** `predictions[condition,source]`, `conditions`, `ids`, `y`,
`classes`, and `lengths[condition,source]`. `lengths` is the real time-sample count,
not event count. The 91 test conditions include one clean condition plus six
operators × five positive severities × three random realizations.

**Paired cache:** `cache/test/*.npz` stores binary observations, lengths and ordered
source/condition metadata. Its accompanying `.effects.csv` gives source-level
physical input changes. Cache identities include initial-state and delay policies.
All methods share these cache inputs; no per-model perturbation draws are used.

## Numerical reports

`reports/primary/` and `reports/pool_sizes/` contain:

| File | Aggregation / purpose |
|---|---|
| `metrics_by_draw.csv` | Individual fitted-model × test-realization score |
| `per_fit.csv` | Mean across corruption draws for each fitted model |
| `per_split.csv` | Mean across pool/label/model repetitions where configured |
| `per_repeat.csv` | Five fold means averaged within each CV repetition |
| `summary.csv` | Mean and sample SD across ten repetition means |
| `paired_per_fit.csv` | Matched pretraining/proposal/reference differences |
| `paired_summary.csv` | Matched differences averaged to repetition level |
| `absolute_drops.csv` | Clean score minus corrupted score for each fitted model |
| `severity_auc_per_fit.csv` | Trapezoidal area normalized by total severity range |
| `group_per_split.csv` | Equal-weight operator groups within a split |
| `group_per_repeat.csv` | Group scores averaged across folds per repetition |
| `group_summary.csv` | Group mean and sample SD over repetitions |
| `pretraining_exposure.csv` | Available versus processed/covered history per SSL task |
| `test_input_effects_per_source.csv` | Physical corruption effects, not multiplied by model count |
| `test_input_effects_per_draw.csv` | Source-averaged effects per split/operator/draw |
| `pool_to_full_per_split.csv` | Reduced-history minus current full-history paired score |
| `config.json` | Complete resolved scientific configuration |
| `REPORT_STATUS.json` | Completeness audit, protocol and file checksums |

Paired method comparisons are MAPLE minus Scratch, CorA-MAPLE minus MAPLE,
and CorA-MAPLE minus Scratch within each E/F schedule, plus each proposal minus
each classical baseline and the F-minus-E difference. All neural comparisons use
clean-only supervised training. E/F comparisons use equal optimization budgets but different input contexts and
cumulative temporal exposure. Scratch-E/F are identical paired controls under the
default common schedule, not independent model replications.

## Table and figure interfaces

`export-paper` audits both studies and produces:

- `tables/robustness_all_table.tex`: **11 methods**, clean plus six corruptions at
  the predeclared headline severity 0.5; every cell has mean ± repetition SD.
- `tables/robustness_main_table.tex`: compact seven-method version (proposals and
  classical baselines). It is an alternative display, not selective evaluation.
- `tables/ablation_factorial_table.tex`: six neural conditions (scope × pretraining), their SSL
  and classifier states, and clean/trained/test-only group scores at 0.5.
- `tables/unlabelled_pool_table_compact.tex`: clean and full-channel false-alarm
  performance versus available historical hours at k=1.
- `tables/unlabelled_pool_by_group.tex`: clean/trained/test-only history sensitivity.
- `figures/figure1_two_columns.pdf`, `.svg`, `.png`: label-budget clean comparison
  and false-full severity curve; seven methods; one right-side shared legend.
- `figures/figure1_one_column.*`: stacked single-column alternative.
- `figures/all_six_robustness_profiles.*`: all selected perturbation severity curves.
- `numeric_facts.json`, `paper_numeric_summary.csv`, study provenance and
  `RESULTS_STATUS.json`.

Figure axes show the entire F1 range, not only favorable upper segments. Solid lines
identify CorA-MAPLE; classical methods use dashed/dotted curves. The training ceiling
0.30 is marked for the trained conditions. Headers distinguish P (self-supervised pretraining) and U (test-only) **for the proposed methods**.
Ablation exposure follows its own specified stage switches, not the table group
name. Color or bold mean maxima do not imply statistical significance.

No numerical results are supplied before the configured studies are executed.
The exporter accepts only complete, audited reports matching the protocol, source
identities and resolved settings.
