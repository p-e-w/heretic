# Test Suite Guide

Whenever we change any code-logic related to `src/heretic/model.py` or `config.toml` *(e.g. `row_normalization`, `full_normalization_lora_rank`, `winsorization_quantile`, etc)* which can affect a model's reproduciblity; Use these tests which are designed to verify that those changes does not affect reproducibility, unless they are meant to (like when we'll integrate ARA branch in future).

## How to test

1. Choose any model from [tiny-random](https://huggingface.co/tiny-random) org which provides tiny models useful for debugging. 

**Example**: [tiny-random/minicpm5](https://huggingface.co/tiny-random/minicpm5).

2. Clone that model repository using Git and generate the SHA256 hashes using `sha256sum`:

**On Linux**:

```bash
sha256sum -b * > ../SHA256SUMS.LABEL
```

**On Windows**:

```bash
sha256sum * | Out-File -Encoding utf8NoBOM ../SHA256SUMS.LABEL
```

> [!TIP]
> On windows, `sha256sum` is generally pre-installed by *Git for windows*.

**Verify with**:

```bash
Get-Command sha256sum`
```

**Expected**:

```bash
CommandType     Name                                               Version    Source
-----------     ----                                               -------    ------
Application     sha256sum.exe                                      0.0.0.0    C:\Program Files\Git\usr\bin\sha256sum...
```

> [!NOTE]
> You must use Windows Powershell `v7.X` not the core which is `v5.1`. This is required for `-Encoding utf8NoBOM` to work.
>
> See [Differences between Windows PowerShell 5.1 and PowerShell 7.x](https://learn.microsoft.com/en-us/powershell/scripting/whats-new/differences-from-windows-powershell?view=powershell-7.6) documentation.

Where `LABEL` describes the type of system you are running the tests on.

**Example**: 

- `SHA256SUMS.windows` (For windows)
- `SHA256SUMS.ci` (For GitHub CI)
- `SHA256SUMS.linux` (For linux)

3. Run the tests with:

```bash
uv run run_tests.py
```

The output hashes *should FAIL* against the `Valid hashes` in `SHA256SUMS` file of the test model you added. This is expected since Heretic changes the model. Without **Step 2**, the test model's folder will simply be ignored because it will not have a hash SUMS file to compare against.

4. After that go to the output `TEST_MODEL_DIR/model` folder and re-generate the Actual hashes based on the system you are using.

```bash
cd TEST_MODEL_DIR/model
sha256sum -b * > ../SHA256SUMS.LABEL # or use windows command.
```

5. Re-run the tests with:

```bash
uv run run_tests.py
```

This time the tests *should PASS* because we added the new hashes which are expected to be reproduced on the same system.

6. After that push the `SHA256SUMS.LABEL` files and wait for GitHub CI actions to run those tests.

Since PyTorch does not guarantee exact cross-system reproducibility regardless of configuration, multiple valid hashes can be provided for each output file. The above update must be performed for each `TEST_MODEL_DIR` and on each type of system.

For this, copy the `Actual hash` value for *each mismatched unidentical* file into a `SHA256SUMS.ci` file.

7. After that push the `SHA256SUMS.ci` files and wait for GitHub CI actions to re-run those tests.

This time the tests *should* PASS because we added the new hashes which are expected to be reproduced on CI.
