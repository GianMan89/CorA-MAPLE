# Measured three-fold reference results

`results/reference_cv3/` contains the saved experiments corresponding to the
manuscript's existing numerical values. It is a separate protocol from the default
10×5 evaluation and retains its own configurations, source selections and seeds.
There are 666–667 training recordings per outer fold, not 800.

## Contents

`tep_primary/` contains the original primary fits, metrics and predictions.
`tep_additional_controls/` contains MAPLE-F and exposure-matched excerpt runs.
Together, the primary and MAPLE-F entries provide the twelve-method comparison.
`tep_absolute_scaling/` contains the nine reduced-history sizes with three nested
pool selections per fold, plus references to full-pool results. `tep_long_supervised/`
contains the separately recorded 1,200-update supervised diagnostic.

Each classifier directory contains `classifier.pt` where applicable,
`predictions.npz`, `metrics.csv`, training traces and source-role provenance.
Pretraining checkpoint directories contain `encoder.pt` and pretraining traces.
Aggregates and paired comparisons are retained in `revision_summary/`. Publication
assets matching the manuscript are in `assets/tables/` and `assets/figures/`.

## Recorded identifiers

Archived mode names predate the paper abbreviations:

| Archived key | Paper name |
|---|---|
| `mae_excerpt` | Recon-E |
| `latent_excerpt` | MAPLE-E |
| `latent_full` | MAPLE-F |
| `latent_robust_excerpt` | CorA-MAPLE-E |
| `latent_robust_full` | CorA-MAPLE-F |
| `latent_robust_excerpt_matched` | Exposure-matched CorA-MAPLE-E |

The names in the archived paths are preserved so checkpoint references stay
resolvable. `config.lock.yaml` records the exact historical settings. References
to paths in earlier directory structures are provenance strings; the data are
consolidated under the current `data/` directory. The absolute-history execution source is retained in
`tep_absolute_scaling/execution_source/`.

Descriptive metadata terminology was standardized during repository assembly.
`metadata_normalization.json` lists those text files and their before/after hashes.
CSV measurements, saved prediction arrays and checkpoint bytes are unchanged.
Original config/source hashes remain identifiers of the original executions.

## Verify archived measurements

```bash
python scripts/reference_results.py
```

This recomputes each per-fit metric from stored predictions and checks observation
identities. Reused full-pool endpoints remain in their historical summary files;
they are not counted as independently fitted pool repetitions.

Clean neural predictions can additionally be replayed with:

```bash
python scripts/reference_results.py --replay-clean
```

This loads the archived weights with `weights_only=True` and evaluates the original
held-out data. It does not fit models or regenerate the repeated-CV experiments.
The archived CPU environment is documented in its run manifests. Platform or
library differences can affect numerical replay.

## Separation from repeated CV

The main CLI reads only tasks whose identities match the active workspace. It does
not search this reference directory for missing models, endpoints or scores.
Repeated-CV exports come exclusively from completed tasks under `runs/tep_cv10x5`.
