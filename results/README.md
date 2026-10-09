# Result collections

| Collection | Cross-validation | Measurement status |
|---|---|---|
| `reference_cv3/` | Three stratified folds | Recorded predictions, checkpoints and metrics included |
| `cv10x5/` | Five folds repeated ten times | No measurements stored; status recorded in `status.json` |

The command-line runner writes resumable workspaces to `runs/`. Tables and figures
exported from a workspace use only its completed tasks. The default 10×5 study does
not import three-fold models or data-budget endpoints.
