# Results

This distribution contains no precomputed benchmark scores, trained checkpoints,
or generated performance plots. `STATUS.json` records this state explicitly.

Executing the frozen experiment writes all numerical data to the selected
workspace, not into the source tree:

- `tasks/`: trained checkpoints, training histories, predictions and per-draw scores;
- `cache/test/`: paired corrupted inputs, valid lengths and measured input effects;
- `reports/primary/`: the full classifier comparison;
- `reports/pool_sizes/`: the available-history study;
- `context.json`, `dataset_index.json`, `splits.json`, `studies/`: reproducibility data.

After completion, use `audit` and `export-paper` as described in `QUICKSTART.md`.
The resulting plots and tables are placed in `paper/generated/`. An explicitly
published results snapshot should include both complete report directories,
their resolved configurations, checksum metadata, and raw predictions or a stable
location for those predictions. The software export tools reject incomplete
studies or mismatched settings instead of supplying unmeasured values.
