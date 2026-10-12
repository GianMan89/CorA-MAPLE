# Software validation

`RELEASE_VALIDATION.json` records the actual software-test, installation, dataset,
planning and release checks. `software_tests.txt` contains the successful pytest
summary. The test suite includes small synthetic optimizations and prediction
replay, not scientific TEP benchmark fitting. No test-generated model, prediction
array, or performance graphic is included as an experimental result.

Run `python -m pytest -q` to execute the tests and
`python scripts/verify_release.py` to verify the distributed source and data hashes.
The primary and history-size studies must be run separately to obtain scientific
performance measurements.
