# Installation and execution environments

Run from a clone or extracted repository so configurations, data and archived
results are available. An editable install provides the `cora-maple` entry point.
Python 3.13 is the default CI environment; Python 3.11 and 3.12 are also compatible
with the declared dependency versions.

## CPU

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
```

The file pins PyTorch 2.10.0 and the scientific Python dependencies. Linux and
Windows use the official CPU wheel; macOS uses the standard wheel.

## NVIDIA GPU

For example, for the PyTorch CUDA 12.6 wheel:

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e . --no-deps
python scripts/run_studies.py --workspace runs/tep_cv10x5_cuda --device cuda --workers 1
```

PyTorch publishes additional 2.10.0 CUDA builds at
https://pytorch.org/get-started/previous-versions/ . Use a build compatible with
the local driver. MPS execution is not an experiment configuration in this release.

## Parallel work and resumption

`--workers` controls concurrent independent tasks; `--threads` controls numerical
threads per worker. Task IDs include the full experiment context, so different
studies in the same workspace reuse only identical tasks. Process spawning is used
for Windows compatibility. Locks protect cache generation and task output.

Completed files are hashed. Failed or interrupted tasks are rerun from their
recorded seeds. Resumption is at task boundaries, not individual optimizer updates.
Different source/data/configuration hashes or incompatible environments are rejected
within an existing workspace. Use distinct workspace names for distinct scientific
configurations or CPU/GPU environments.

For manual multi-GPU sharding, invoke one worker per GPU and select different
`--only-repeat` values while keeping the workspace and dependency versions identical.
The execution record stores device family and worker thread policy.

## Determinism

Seeds cover source splitting, label and pool sampling, model initialization,
crop selection, masking, perturbations and dropout. Deterministic PyTorch operations
are requested. Exact equality across platforms or PyTorch releases is not assumed.
The package records Python, dependency versions, platform and device in each
workspace. See the PyTorch reproducibility documentation cited in `PROTOCOL.md`.

## Output size

Model checkpoints and paired test caches are stored in the selected workspace.
Caches use compressed uint8 alarm-state tensors. The `runs/` directory is excluded
from Git tracking. Publication asset exports contain the compact summaries,
tables and vector figures without duplicating caches.
