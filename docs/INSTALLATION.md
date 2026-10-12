# Installation, devices and progress

A fresh **Python 3.13** environment is the documented default. The project code
supports Python >=3.11 and <3.15, subject to availability of the pinned scientific
wheels. Python 3.9 or older is not supported. Do not downgrade pip to work around an
old interpreter. `python --version` and `py -0p` identify Windows interpreter choices.

## Windows / CPU

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
```

`requirements-cpu.txt` installs PyTorch 2.10.0 CPU plus exact scientific dependency
versions. On macOS it selects the standard PyTorch distribution. Run from the
repository root, or set `CORA_MAPLE_ROOT` to that directory when using an installed
package elsewhere. Do not move the data after preparing a workspace.

## NVIDIA CUDA

Install a PyTorch 2.10.0 CUDA wheel compatible with the host driver using the
PyTorch installation instructions, then install `requirements.txt` and the editable
project without replacing PyTorch with the CPU lock:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use `--device cuda --workers 1` or `--device cuda:0 --workers 1`. Multiple GPU
invocations may partition work with `--only-repeat`/`--only-fold` and distinct
CUDA indices. Tasks and test caches are protected by file locks. Do not start
multiple unfiltered invocations against the same entire task set.

For CPU use `--workers 4 --threads 1` as an example, adjusting workers to physical
cores and memory. Each worker loads a dataset and task-local model; shared test
caches avoid regenerating every corruption for every classifier. Spawn is used
on every platform. GPU execution uses one process per invocation; observed runtime
hardware and package versions are saved with the workspace.

The runner requests deterministic PyTorch operations, disables TF32 and CUDA
attention kernels incompatible with deterministic operation, and uses separate RNG
streams. CPU/GPU or library-version changes can still change floating-point results;
these runs must use different workspaces. No GPU performance claim is made from CPU
software tests.

## Monitoring

From a second PowerShell terminal with the same environment activated:

```powershell
while ($true) {
    Clear-Host
    Get-Date
    python -m cora_maple status --workspace runs/tep_cv10x5_cleanft_v2_1 --study primary
    Start-Sleep -Seconds 60
}
```

Use `--study pool_sizes` for the history study. Ctrl+C in the monitoring terminal
does not stop training in another terminal. The task percentage counts completed
jobs, not estimated wall time: E/F and SSL/supervised jobs differ in cost. The last
step/loss is written in task-local `progress.json`; after an interruption that
record can be stale. Runtime exceptions are retained in `failure.json`.

Resume with the same run command. Complete tasks are checked and reused, while an
interrupted optimization task restarts from its recorded seed. This is task-level
resume, not optimizer checkpoint restart.

## Tests

```powershell
python -m pip install pytest==9.0.2
python -m pytest -q
```

Tests include short synthetic training through the six-condition neural grid, validity
masking, reconstruction edge cases, deterministic pairing, full-pool reuse, metadata,
report completeness, and numerical prediction replay. They do not run the 1,000-source
TEP benchmark. The released validation record states the actual tested environment.
