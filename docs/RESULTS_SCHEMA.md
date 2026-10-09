# Results and provenance schema

## Workspace identity

`context.json` stores the protocol, all shared configuration values, dataset and
numerical source hashes, and planning environment. `execution_environment.json`
stores the runtime dependency versions, device family and numerical thread policy.
`splits.json` identifies training and test source rows for each repeat/fold.
`dataset_index.json` maps rows to source IDs, labels, classes and channel names.

Study-specific plans reside in `studies/<study>/plan.json`, with a resolved
`config.json`. Tasks are addressed by SHA-256-derived identities that include the
shared context and their source selections, method, seeds and dependencies.

## Task outputs

All task outputs are under `tasks/<task_id>/`.

| File | Contents |
|---|---|
| `task.json` | Exact task definition, seeds, source indices and dependencies |
| `encoder.pt` | Pretrained student weights and model configuration |
| `pretraining.csv` | Objective components at logged updates |
| `source_coverage.npz` | Per-source draw counts, visited time-sample union, and source indices |
| `classifier.pt` | Neural classifier weights and dimensions |
| `model_info.json` | Class names, channel identities and observation metadata |
| `baseline.joblib` | Fitted classical model; trusted locally produced files only |
| `finetuning.csv` | Supervised minibatch loss and clean training metrics |
| `training.json` | Update count, duration and training/source-exposure statistics |
| `predictions.npz` | Predicted classes, condition keys, source IDs and ground-truth labels |
| `metrics.csv` | One Macro-F1/accuracy row per perturbation condition |
| `evaluation.json` | Test cache identity and condition/source counts |
| `completed.json` | Output hashes, device, thread policy and elapsed time |
| `failure.json` | Exception information for an interrupted or failed task |

Pretraining tasks and classifier tasks are separate. Their successful completion
records are only written after all outputs have been created. A partial or altered
task cannot be silently reused.

## Condition keys

Each key encodes `operator|severity|realization`, for example `spurious|0.3|2`.
One clean condition has severity zero. Every positive severity has three shared
realizations per operator, for 55 conditions. The perturbation cache stores uint8
states and is immutable under a given workspace identity.

## Reports

- `metrics_by_draw.csv`: all per-condition observations plus task metadata.
- `per_fit.csv`: averages the three corruption realizations within each model.
- `per_split.csv`: additionally averages applicable pool/label/initialization replicates.
- `per_repeat.csv`: averages the five folds within each repetition.
- `summary.csv`: means and sample SDs over the ten repetition means.
- `paired_per_fit.csv`, `paired_summary.csv`: contrasts on paired sources/conditions.
- `absolute_drops.csv`: clean minus degraded performance.
- `severity_auc_per_fit.csv`: normalized trapezoidal area across the tested severity range.
- `pretraining_exposure.csv`: available history, cumulative sampled duration and source coverage.
- `pool_to_full_per_split.csv`: matched differences between each reduced pool and its full endpoint.

Optional variability reports separate labelled-subset variation and initialization
variation. These experiments are not merged with the default twelve-method summary.

## Manuscript assets

`export-paper` creates the three numerical tables with the names expected by the
LaTeX manuscript, both result-figure panels, numeric facts and provenance. The
available-history column in Table 4 is measured in hours. The Full endpoint for the
default protocol is 800 hours. Table means/SDs come from repeated-CV aggregation;
archived three-fold tables remain in a different directory.

## Prediction and checkpoint replay

A fitted neural model can classify a compatible NPZ through:

```bash
python scripts/predict.py --checkpoint runs/tep_cv10x5/tasks/TASK_ID/classifier.pt --inputs data/processed/tep_60min.npz --output runs/predictions.csv
```

The command requires the checkpoint's `model_info.json` and checks channel order
and observation dimensions. `TASK_ID` is a completed neural task ID from the
workspace. Using the entire bundled dataset is an inference example, not a test
performance estimate, because it also contains training recordings.

Completed neural models can be replayed on their paired held-out observations:

```bash
python scripts/replay_workspace.py --workspace runs/tep_cv10x5 --study primary
```

Replay compares saved and regenerated class predictions without optimizer updates.
The separate `audit` command recomputes metrics for both neural and classical tasks.
