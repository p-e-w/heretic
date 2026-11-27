# Test Guide

This directory contains two kinds of tests:

1. **Reproducibility tests**: end-to-end tests that verify whether your new changes affect a model's reproducibility. See [Reproducibility tests](#1-reproducibility-tests).
2. **Unit tests**: tests for individual units of heretic. See [Unit tests](#2-unit-tests).

## 1. Reproducibility tests

Whenever we change any code-logic related to `src/heretic/model.py`, `src/heretic/main.py`, `src/heretic/modifier.py`, `src/heretic/evaluator.py` *(operations, modifier, evaluation, batch size, etc.)*, or `src/heretic/config.py`, `src/heretic/plugin.py`, `src/heretic/utils.py`, `src/heretic/scorer.py` *(settings, plugin interface, response prefix, chat template, etc.)*, or **anything** which can affect a model's *reproducibility*, use these tests which are designed to verify that those changes do *not* affect reproducibility, unless they *are* meant to.

For example, when fixing bugs *(e.g. fixing the `response_prefix` logic in `src/heretic/main.py`, which **changes** the previous behavior)*. Such changes will affect the reproducibility of the models, and in such cases, you may update the hash for the files that differ from their existing hash as mentioned in the existing model's `SHA256SUMS.LABEL` file.

### How to test

```bash
uv run run_tests.py
```

### How to add a new test

You may add a new test case with a new model when implementing new plugins *(e.g. a new modifier, a new scorer, etc.)*, or when testing different optimization choices / non-default values *(e.g. different value of `row_normalization` in `src/heretic/modifiers/abliteration.py`, different value of `lora_rank` in `src/heretic/modifiers/ara.py`, etc.)*.

1. Choose any model from [tiny-random](https://huggingface.co/tiny-random) org which provides tiny models useful for debugging.

* **Example**: [tiny-random/minicpm5](https://huggingface.co/tiny-random/minicpm5).

2. Clone that model repository using **Git** and generate the `SHA256SUMS.LABEL` file using the `sha256sum` tool:

> [!IMPORTANT]
> Delete the unnecessary files such as `README.md` and `.gitattributes` before generating the hash file.

* **On Linux**:

```bash
sha256sum -b * > ../SHA256SUMS.LABEL
```

* **On Windows**:

```bash
sha256sum * | Out-File -Encoding utf8NoBOM ../SHA256SUMS.LABEL
```

Where `LABEL` describes the type of system you are running the tests on.

**Example**: 

- `SHA256SUMS.windows` (For windows)
- `SHA256SUMS.ci` (For GitHub CI)
- `SHA256SUMS.linux` (For linux)

> [!TIP]
> On Windows, `sha256sum` is generally pre-installed by **Git for Windows**.

* **Verify with**:

```bash
Get-Command sha256sum
```

* **Expected**:

```bash
CommandType     Name                                               Version    Source
-----------     ----                                               -------    ------
Application     sha256sum.exe                                      0.0.0.0    C:\Program Files\Git\usr\bin\sha256sum...
```

> [!NOTE]
> You must use Windows PowerShell `v7.x` not the core which is `v5.1`. This is required for `-Encoding utf8NoBOM` to work.
>
> See [Differences between Windows PowerShell 5.1 and PowerShell 7.x](https://learn.microsoft.com/en-us/powershell/scripting/whats-new/differences-from-windows-powershell?view=powershell-7.6) documentation.

3. Run the tests with:

```bash
uv run run_tests.py
```

The output hashes should **FAIL** against the `Valid hashes` in `SHA256SUMS.LABEL` file of the test model you added. This is **expected** since Heretic modifies the model. Without **Step 2**, the test model's folder will simply be ignored because it will not have a checksums file to compare against.

4. After that, go to the output folder `TEST_MODEL_DIR/model` and re-generate the actual hashes based on the system you are using.

```bash
cd TEST_MODEL_DIR/model
sha256sum -b * > ../SHA256SUMS.LABEL  # or use windows command.
```

5. Re-run the tests with:

```bash
uv run run_tests.py
```

This time, the tests should **PASS** because we added the new hashes which are *expected* to be reproduced on the same system.

6. After that, push the `SHA256SUMS.LABEL` files and wait for the **GitHub CI** actions to run those tests.

> [!NOTE]
> The GitHub CI actions will run the tests only if your PR is targeting the `master` branch of Heretic.

Since **PyTorch** does *not* guarantee exact cross-system reproducibility regardless of configuration, multiple valid hashes can be provided for each output file. The above update must be performed for each `TEST_MODEL_DIR` and on each type of system *(or at least your system plus the GitHub CI, e.g. Linux or Windows + GitHub CI)*.

For this, copy the `Actual hash` value for *each* **unidentical** file into a new `SHA256SUMS.ci` file.

7. After that, push the `SHA256SUMS.ci` files and wait for GitHub CI actions to re-run those tests.

This time the tests *should* **PASS** because we added the new hashes which are *expected* to be reproduced on the CI.

> [!NOTE]
> The GitHub CI is usually non-deterministic across different Python versions on which we run the tests, probably due to their internal infrastructure.
>
> Due to this, if the tests constantly fail due to hash mismatch for one or more Python versions *(e.g. Python 3.12 and Python 3.14)*, then you may add **alternate** hashes for the same system, i.e. for the GitHub CI.
>
> For example, `SHA256SUMS.ci2` and `SHA256SUMS.ci3` files.

## 2. Unit tests

The unit tests are written with Python's built-in `unittest` framework.

```bash
cd tests
python -m unittest
```
