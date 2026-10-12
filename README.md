# CorA-MAPLE

**Corruption-aware self-supervised pretraining for robust, label-efficient offline alarm flood classification.**

CorA-MAPLE-E learns from unlabelled historical alarm excerpts; CorA-MAPLE-F learns
from complete, unlabelled flood episodes. A patch Transformer predicts reference
features from masked and corrupted observations using a student–teacher objective.
The student encoder and a new classification head are then jointly fine-tuned on
**clean labelled episodes only**.

This repository contains the exact 1,000-recording TEP dataset snapshot, the model
and five classical AFC algorithms, fixed configurations, complete tests, source-level
cross-validation, paired test-corruption caches, resumable task execution, prediction
audits, and publication table/vector-figure exporters. **No precomputed benchmark
scores or trained model checkpoints are included.** The declared studies produce
those artifacts in the workspace.

## Fixed experiment

| Component | Setting |
|---|---|
| Data | 1,000 TEP records; five classes; 50 binary channels; 60 one-minute samples |
| Source splitting | Stratified five-fold CV repeated ten times; 800 train / 200 test records per split |
| Label budgets | 1, 2, 5, 20 episodes per class; nested and identical across methods |
| Architecture | 96-dimensional, three-layer, four-head Transformer; patch length 3; FFN 192; dropout 0.1 |
| Pretraining steps — **both E and F** | **4,000** |
| Clean supervised steps — **both E and F** | **2,400** |
| E inputs | Uniformly sampled 15, 30, 45 or 60-sample excerpts, one length per minibatch |
| F inputs | Complete 60-sample unlabelled episodes |
| CorA pretraining corruption | Full-channel false activation, full-channel loss, timestamp uncertainty |
| Training severity | Uniform 5–30%; 20% unchanged views; independent 30% patch masking |
| Test-only mechanisms | Delayed observation, missing ACT/RTN notifications, additional ACT/RTN notifications |
| Test severities | 0, 10, 20, 30, 40, 50%; three paired draws at positive severity |
| History study | k=1; 1, 2, 5, 10, 20, 50, 100, 200, 400, and all 800 unlabelled records |
| History selections | Three nested class-blind source selections per split, shared by E/F |

No held-out recording enters either training stage, feature fitting or checkpoint
selection. Equal update counts do **not** equalize total observed sample-minutes:
F processes 60 samples per input versus 37.5 on average for E, giving 1.6 times the
expected cumulative temporal exposure. Source coverage and processed duration are
recorded separately from available history.

## Methods and controls

| Condition, evaluated for both E/F scopes | Self-supervised pretraining | Supervised training |
|---|---|---|
| Scratch-E/F | None | Clean cross-entropy |
| MAPLE-E/F | Uncorrupted, masked SSL | Clean cross-entropy |
| **CorA-MAPLE-E/F** | Corruption-aware, masked SSL | Clean cross-entropy |

All six neural conditions use the same encoder and update counts. Scratch-E/F are
kept as paired references for the E/F comparisons: neither uses excerpt sampling,
and they are numerically identical when settings, data, seeds and device match.
They are not independent scientific replications. An SSL checkpoint is shared
across the four label budgets; each classifier receives a new head and is fine-tuned
separately.

The classical methods are **EAC-1NN, MBW-LR, JAC-1NN, WDI-1NN, and ACM-SVM**.
They use the same labelled subsets and held-out observations as the neural methods.
Only pretrained models also use the unlabelled training pool.

The default experiment contains **11 method entries**, **2,200 primary classifier
fits**, **2,700 additional history-study classifier fits**, and **2,900 distinct SSL
fits**: **7,800 unique tasks** across both studies. Full-pool history endpoints
reuse the corresponding primary models.

## Installation — Windows PowerShell

Run from the extracted repository root. Python **3.13** and the pinned CPU
requirements define the tested environment.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

`requirements-dev.txt` includes the CPU dependencies and the test tools. For an
NVIDIA CUDA setup, install the appropriate PyTorch 2.10.0 CUDA wheel and
`requirements.txt` instead of the CPU requirement file. See
[installation](docs/INSTALLATION.md).

## Check, plan, and run

```powershell
python scripts/verify_release.py
python -m pytest -q
python scripts/plan_all.py --workspace runs/tep_cv10x5_cleanft_v2_1
```

Planning writes the source assignments, hashes and exact workload without training.
Run both main studies on CPU:

```powershell
python scripts/run_studies.py --workspace runs/tep_cv10x5_cleanft_v2_1 --device cpu --workers 4 --threads 1
```

With CUDA-enabled PyTorch:

```powershell
python scripts/run_studies.py --workspace runs/tep_cv10x5_cleanft_v2_1 --device cuda --workers 1
```

Use the same run command to resume an interrupted experiment. Completed tasks are
validated and reused; interrupted optimization tasks restart from their stored
seed. Use a separate workspace whenever code, data, scientific settings, or the
execution environment changes. Do not mix CPU and GPU runs in one workspace.

To evaluate only one split with the full training schedule:

```powershell
python -m cora_maple run --config configs/repeated_cv.yaml --workspace runs/tep_cv10x5_cleanft_v2_1 --only-repeat 0 --only-fold 0 --device cpu --workers 1 --threads 1
```

The complete run later reuses finished tasks. The history study uses
`--config configs/pool_sizes.yaml` with the same flags.

## Monitor progress

From another activated terminal:

```powershell
python -m cora_maple status --workspace runs/tep_cv10x5_cleanft_v2_1 --study primary
python -m cora_maple status --workspace runs/tep_cv10x5_cleanft_v2_1 --study pool_sizes
```

For an automatically refreshing view, use `scripts/monitor.ps1`. Task-local
`progress.json`, `pretraining.csv`, and `finetuning.csv` record optimizer progress.
Task percentage is not a wall-time estimate; progress can be stale after a process
stops. Errors remain available in `failure.json`.

## Audit and export publication results

When both studies finish:

```powershell
python -m cora_maple audit --workspace runs/tep_cv10x5_cleanft_v2_1 --study primary
python -m cora_maple audit --workspace runs/tep_cv10x5_cleanft_v2_1 --study pool_sizes
python -m cora_maple export-paper --workspace runs/tep_cv10x5_cleanft_v2_1 --destination paper/generated
```

The exporter requires complete studies and validates saved-prediction arithmetic.
It generates the full 11-method robustness table, a compact seven-method alternative,
the six-condition pretraining ablation, available-history tables, Figure 1 with a
shared legend, and all six severity-response profiles. No measurements are inserted
before an experiment is run.

The figures distinguish **P** (present in CorA pretraining) from **U** (test-only),
with the 30% training ceiling marked. Statistics average corruption draws and pool
selections, then folds within each repetition, and report mean ± sample SD across
**ten repetition means**. These SDs are not confidence intervals. All methods and
conditions remain in the detailed reports, including unfavorable outcomes.

## Scientific assumptions

Recorded per-channel initial states are independent snapshots, not corruptible
notifications. Event reconstruction cannot infer past activity from an RTN on an
initially inactive channel. Unavailable delay padding is excluded from neural and
classical feature processing. A lost RTN can still prolong reconstructed activity.
These assumptions require a reliable initial-state observation in an application.

A common numeric severity denotes different physical interventions across operator
families. Full-channel false activation is particularly identifiable on these TEP
records because their clean first samples are inactive. Repeated partitions of a
single synthetic process dataset are not independent external validation. Frozen
settings and complete reporting do not guarantee performance, and general claims
require independent industrial recordings or other process domains.

[Method](docs/METHOD.md) · [Protocol](docs/PROTOCOL.md) ·
[Design](docs/DECISIONS.md) · [Perturbations](docs/PERTURBATIONS.md) ·
[Result schema](docs/RESULTS_SCHEMA.md) · [Reproducibility](reproducibility/README.md)

Dataset source: IEEE DataPort **10.21227/326k-qr90**; exact snapshot provenance is
in `data/`. Code attribution and licence notices are in `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and `vendor/`. The software licence does not replace
the original data terms. Citation metadata is provided in `CITATION.cff`.
