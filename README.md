# CorA-MAPLE

**Corruption-Aware Pretraining for Robust and Label-Efficient Offline Alarm Flood Classification**

CorA-MAPLE learns alarm representations from unlabelled data and adapts them to
fault classification with few labelled alarm floods. **CorA-MAPLE-E** samples
excerpts without requiring historical flood boundaries. **CorA-MAPLE-F** uses
complete, already segmented episodes. A student encoder receives masked and
corrupted observations while a moving-average teacher supplies reference features.

This repository contains the TEP data snapshot, model and baseline implementations,
fixed hyperparameters, training and evaluation scripts, saved three-fold results,
and export tools for the manuscript tables and figures.

## Evaluation protocol

The default experiment is **five-fold cross-validation repeated ten times**. Each
repetition independently reshuffles complete recordings with class stratification.
There are 50 train/test splits, each with 800 training and 200 test recordings.
Label selection, initializations and test corruptions are seeded and paired across
methods. No test recording enters pretraining or classifier fitting.

| Component | Configuration |
|---|---|
| Dataset | 1,000 TEP recordings, 5 classes, 50 binary alarm channels, 60 samples at 1 min |
| Label budgets | 1, 2, 5 and 20 labelled episodes per class |
| Primary comparison | 2 proposed methods, 5 ablations and 5 literature baselines |
| Pretraining | 800 updates, batch size 32 |
| Supervised adaptation | 400 updates, batch size min(32, number of labelled episodes) |
| History-size experiment | 1, 2, 5, 10, 20, 50, 100, 200, 400 and Full=800 available hours |
| Reduced-pool sampling | 3 nested, unstratified source selections per train/test split |
| Test perturbations | Missing events, spurious events, timing shifts, bursts, tag dropout and missing–spurious composition |
| Test severity | 0, 0.1, 0.2 and 0.3; 3 paired realizations at each positive severity |

One recording contributes one hour of **available history**. Repeated visits to a
recording do not increase available history. Full episodes process 60 minutes per
sampled input; excerpts process 37.5 minutes on average. Actual sampled duration
and unique source/time-sample coverage are recorded for each pretraining task.

The checked-in `results/reference_cv3/` directory contains the measured three-fold
results accompanying the manuscript. The repeated-CV result directory is empty
until an evaluation is executed. The two protocols have separate identifiers and
are never combined by the reporting pipeline.

## Installation

Use Python 3.11–3.13 and execute commands from this repository's root.

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
```

For a CUDA environment, install the scientific dependencies with
`python -m pip install -r requirements.txt`, install the appropriate PyTorch 2.10.0
CUDA wheel, and run `python -m pip install -e . --no-deps`.
The package uses deterministic operations and records the execution environment.
CPU and GPU results are kept in separate workspaces.
See [installation details](docs/INSTALLATION.md).

## Run the paper experiments

The following command runs the twelve-method comparison and the history-size
experiment, then writes their summaries:

```bash
python scripts/run_studies.py --workspace runs/tep_cv10x5 --workers 4
```

For a single GPU, use `--device cuda --workers 1`. For CPU execution, `--workers`
controls concurrent tasks; each worker uses one numerical thread by default.

The primary study contains **2,400 classifier fits** and 250 pretraining tasks.
The history-size study adds **2,700 classifier fits** and 2,700 pretraining tasks.
Its 100 full-pool classifier endpoints and associated encoders reuse the matching
repeated-CV primary tasks. The two studies therefore contain 5,100 distinct fitted
classifiers in total.

Run individual studies with:

```bash
python -m cora_maple run --config configs/repeated_cv.yaml --workspace runs/tep_cv10x5 --workers 4
python -m cora_maple run --config configs/pool_sizes.yaml --workspace runs/tep_cv10x5 --workers 4
```

Re-running a command resumes completed, checksum-verified tasks. An interrupted
individual task restarts from its recorded seed. Shard work with `--only-repeat 0`
and/or `--only-fold 0` (both zero-indexed). Use `--max-tasks` to cap the number of
new tasks in an invocation. A non-training plan can be created with:

```bash
python scripts/plan_all.py --workspace runs/tep_cv10x5
python -m cora_maple status --workspace runs/tep_cv10x5 --study primary
```

## Generate tables, figures and numerical summaries

```bash
python -m cora_maple report --workspace runs/tep_cv10x5 --study primary
python -m cora_maple report --workspace runs/tep_cv10x5 --study pool_sizes
python -m cora_maple export-paper --workspace runs/tep_cv10x5 --destination paper/generated
```

Reporting verifies task identities and recomputes metrics from saved class
predictions. Incomplete studies are not exported as publication results.

The exporter writes `main_table.tex`, `transfer_table.tex`,
`unlabelled_pool_table_compact.tex`, `clean_tep.pdf`, `spurious_tep_k1.pdf`, vector
SVGs, PNG previews, numerical facts and provenance. Figure panels contain only the
five literature baselines and the two proposed methods. Ablations appear in the
tables. Table 4 uses available history in hours.

Reports also preserve per-realization, per-fit, per-fold and per-repetition values,
paired method differences, clean-to-degraded drops, normalized severity areas and
pretraining exposure. The principal summaries average realizations and pool
selections first, folds within each repetition second, and then report the mean
and sample standard deviation of the ten repetition means. These are descriptive
standard deviations, not confidence intervals based on 50 independent folds.

## Methods

| Paper name | Configuration key | Pretraining / supervised augmentation |
|---|---|---|
| Scratch | `scratch` | No pretraining; uncorrupted labelled inputs |
| Scratch+Aug | `scratch_aug` | No pretraining; eight fixed corrupted views per labelled episode |
| Recon-E | `recon_excerpt` | Masked binary reconstruction on excerpts, no latent or variance loss |
| MAPLE-E | `maple_excerpt` | Combined objective on excerpts, no event corruption |
| MAPLE-F | `maple_full` | Combined objective on complete episodes, no event corruption |
| CorA-MAPLE-E | `cora_maple_excerpt` | Combined corruption-aware objective on excerpts |
| CorA-MAPLE-F | `cora_maple_full` | Combined corruption-aware objective on complete episodes |
| EAC-1NN, MBW-LR, JAC-1NN, WDI-1NN, ACM-SVM | Same as paper name | Fixed literature-based representations and classifiers |

All pretrained variants use clean, unmasked inputs during supervised adaptation.
The five ablations, the finite-bank Scratch+Aug policy, model dimensions, losses
and training budgets follow the manuscript. Additional controlled experiments are
separate configurations under `configs/diagnostics/`; they are not run by default.

## Repository map

```text
configs/                       Repeated-CV and history-size study definitions
configs/diagnostics/            Optional augmentation, loss and training-budget studies
src/cora_maple/                 Planning, training, corruption, inference and reporting
src/afc_excerpt_ssl/            Shared data and encoder primitives
vendor/afc_robustness/           AFC baseline and event-processing implementations
data/raw/tep/                  1,000 first-hour CSV recordings
data/processed/                Binary data tensor, source index and dataset statistics
results/reference_cv3/         Measured three-fold results, checkpoints and predictions
results/cv10x5/                Repeated-CV result registry
scripts/                       Run, plan, data-rebuild and archive-replay commands
tests/                         Software tests on synthetic inputs and orchestration
paper/                         LaTeX integration and bibliography
docs/                          Method, protocol, data and result-schema documentation
runs/                          Local resumable experiment workspaces
```

## Software checks and archived results

Software tests can be invoked explicitly:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Release integrity can be checked without starting experiments:

```bash
python scripts/verify_release.py
```

The measured three-fold archive has separate commands for metric recomputation and
checkpoint replay. See [archived results](docs/REFERENCE_RESULTS.md). These commands
do not treat the archived models as repeated-CV models.

## Citation and licences

Citation metadata is in `CITATION.cff`. The underlying TEP alarm dataset is cited as
IEEE DataPort, DOI **10.21227/326k-qr90**. Methodological references are retained in
`paper/ifacconf.bib`.

Software is MIT-licensed. Third-party notices and dataset provenance are documented
in `THIRD_PARTY_NOTICES.md` and `data/README.md`.
