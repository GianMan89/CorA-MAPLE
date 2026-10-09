# Attribution and licensing

The project software is distributed under the MIT licence in `LICENSE`.
The original notices remain applicable to reused code.

`vendor/afc_robustness` contains the baseline classifiers, event representations,
perturbations and repair functions from AFC-RobustBench. Its MIT notice is retained
in `vendor/AFCRobustBench_LICENSE`; `vendor/PROVENANCE.json` identifies the source
archive. `vendor/SUBSET_MANIFEST.json` records the retained source hashes. Package
initialization exports only the five included classifiers.

The TEP data snapshot derives from the Tennessee-Eastman-Process Alarm Management
Dataset (IEEE DataPort, DOI 10.21227/326k-qr90) and the supplied study-specific
extension. Source information and sample hashes are retained in `data/`. The
software licence applies to code and does not replace the data source's terms.
No additional dataset licence is assigned by this distribution.

Third-party Python dependencies retain their respective upstream licences.
Dependency binaries, external research papers, and IFAC template files are not
bundled. The bibliography contains citations, not copies of those publications.
