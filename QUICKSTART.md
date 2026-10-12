# Quick start

Extract the ZIP into its own directory. Open PowerShell in the `CorA-MAPLE`
repository root, where `pyproject.toml` and `configs/` are located.

## 1. Install and verify

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python scripts/verify_release.py
python -m pytest -q
```

CPU dependencies are pinned. CUDA instructions are in `docs/INSTALLATION.md`.
Do not install the CPU requirement file over an intended CUDA PyTorch build.

## 2. Plan and run

```powershell
python scripts/plan_all.py --workspace runs/tep_cv10x5_cleanft_v2_1
python scripts/run_studies.py --workspace runs/tep_cv10x5_cleanft_v2_1 --device cpu --workers 4 --threads 1
```

Both E and F use **4,000 pretraining and 2,400 supervised updates**. The command
runs the full primary comparison and the available-history experiment, using
10 repetitions of stratified five-fold CV. Planning itself fits no model.
Adjust CPU workers to the machine's resources. For a configured NVIDIA setup,
replace the run command with:

```powershell
python scripts/run_studies.py --workspace runs/tep_cv10x5_cleanft_v2_1 --device cuda --workers 1
```

Reissue the same command to resume. Use a separate workspace when changing any
configuration, code, dataset, or execution environment.

## 3. Monitor from a second terminal

```powershell
.\.venv\Scripts\Activate.ps1
python -m cora_maple status --workspace runs/tep_cv10x5_cleanft_v2_1 --study primary
python -m cora_maple status --workspace runs/tep_cv10x5_cleanft_v2_1 --study pool_sizes
```

## 4. Audit and generate all paper assets

```powershell
python -m cora_maple audit --workspace runs/tep_cv10x5_cleanft_v2_1 --study primary
python -m cora_maple audit --workspace runs/tep_cv10x5_cleanft_v2_1 --study pool_sizes
python -m cora_maple export-paper --workspace runs/tep_cv10x5_cleanft_v2_1 --destination paper/generated
```

The workspace retains source assignments, full resolved configurations, every
checkpoint, training logs, test inputs, raw predictions, and detailed scores.
`paper/generated/` receives the validated tables, vector plots, numeric facts,
and provenance. No benchmark scores are included in the initial release.
