# Reproducibility

The reproducible experiment is defined by the current source, `configs/`, the
bundled data snapshot, and the pinned dependency files. There are no independent
code copies or precomputed performance measurements in this directory.

- `study_plan.json`: verified configured task counts; planning fits no models.
- `environment_cpu.json`: scientific dependency versions used for software tests.
- `data_verification.json`: source/processed-data equality and checksum checks.
- `RELEASE_MANIFEST.json` at the repository root: exact distributed file hashes.

At runtime, every workspace records full resolved configurations, code/data hashes,
source-role assignments, random seed identities, device/runtime versions and a
complete task graph before fitting starts. Task completion is checked against
checksums. Test perturbations are generated once per source/split/condition and
shared among all methods and label budgets.

A task can be resumed only in a compatible workspace. Completed tasks are reused;
an interrupted optimization task restarts with its recorded random seeds.
`python -m cora_maple audit` recomputes metric arithmetic from saved predictions.
`scripts/replay_workspace.py` additionally reloads selected classifiers and
recomputes their predictions from the saved paired test inputs.

The primary comparison and history-size experiment must both be complete before
manuscript export. Repetitions share the underlying dataset; the reported sample
SD across repetition means is descriptive, not a confidence interval or proof of
external generalization. Hyperparameter choices and model claims require appropriate
validation separate from the held-out prediction calculation.
