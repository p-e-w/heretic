Run the tests with

```sh
uv run run_tests.py
```

To update the hashes after a logic change, run the tests, then execute

```sh
cd TEST_DIR/model
sha256sum * > ../SHA256SUMS.LABEL
```

where `LABEL` describes the type of system you are running the tests on.
Since PyTorch does not guarantee exact cross-system reproducibility regardless of configuration,
multiple valid hashes can be provided for each output file. The above update must be performed
for each `TEST_DIR` and on each type of system.
