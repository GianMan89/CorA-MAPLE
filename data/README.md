# TEP alarm data

The dataset contains **1,000 source recordings**, 200 for each of five classes:
`idv1`, `idv2`, `idv6`, `idv14`, and `idv1_and_idv5`. Each source contributes the first
60 one-minute samples of 50 binary alarm-state channels. A one means active and a
zero inactive. There are no process measurements or extra horizon samples in the
model inputs.

The source is the author-supplied extension of the Tennessee-Eastman-Process Alarm
Management Dataset, IEEE DataPort DOI **10.21227/326k-qr90**, following the selected
fault design of the authors' ETFA robustness study. The bundled matrices identify
the exact experimental snapshot. The historical DOI is not asserted to contain a
byte-identical export of this study-specific extension.

## Files

- `raw/tep/<class>/alarm_timeseries_<run>.csv`: the 1,000 first-hour CSVs, with local
  row labels corresponding to original samples 1–60. Channel order is fixed.
- `processed/tep_60min.npz`: arrays `X` (1000×50×60, uint8), `y` (class strings),
  `ids`, `run_ids`, `tags`, and exact-input `hashes`.
- `processed/tep_manifest.csv`: source IDs, labels, hashes and event statistics.
- `processed/audit.json`: stored dataset dimensions and summary statistics.
- `source_manifest.json`: original and retained CSV hashes plus archive provenance.

The average first-hour ACT count is 30.019. Raw-event conversion uses local sample
times 0–59 minutes, with initially active channels represented by ACT at zero.
The distinction between event conversion and original sample labels is documented
in the perturbation module.

## Rebuild the processed tensor

```bash
python scripts/prepare_data.py --source data/raw --output runs/rebuilt_data --observation-samples 60
```

This command preserves original source ordering, verifies binary states and channel
mapping, and derives hashes without fitting statistics. It does not extrapolate
missing samples. A different output directory avoids overwriting the bundled data.
The NPZ is loaded without pickle.

Source classes are used to establish stratified train/test partitions and select
labelled examples. Class labels and source identifiers are not inputs to pretraining.
The source sampling policy does not filter excerpts by alarm activity.

## Terms

Dataset rights and source terms remain with the original authors and data provider.
The software MIT licence does not assign a new licence to the dataset. Attribution
and source identifiers are retained in this directory and `THIRD_PARTY_NOTICES.md`.
