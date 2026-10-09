# Manuscript assets

The exporter creates `generated/tables/` and `generated/figures/` with the filenames
used in the manuscript:

```bash
python -m cora_maple export-paper --workspace runs/tep_cv10x5 --destination paper/generated
```

`figure_and_table_includes.tex` contains standalone insertion blocks. The captions
state that deviations are across ten repetition means. All table method names are
italicized. The two figure panels contain the five literature baselines and the
two proposed methods, while ablations are included in the numerical tables.

`evaluation_protocol.tex` describes the ten-times repeated five-fold design without
inserting any performance claims. The exporter also writes `numeric_facts.json`
for the completed protocol. Existing numerical values remain available with their
three-fold provenance in `results/reference_cv3/assets/`.

The bibliography uses the compact IFAC entry names retained by the paper. IFAC
class and bibliography-style files are not included or modified. Method/architecture
diagram alternatives and manuscript review comments are not part of this repository.
