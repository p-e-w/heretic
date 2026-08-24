# Reproducible Local Dataset Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make verified `datasets.save_to_disk()` inputs reconstructable from an exact public Hugging Face source through Heretic's native reproduction system.

**Architecture:** Add a typed provenance object to `DatasetSpecification`, isolate canonical hashing/materialization/verification in a focused module, route provenance-aware inputs through that module in `load_prompts`, and sanitize local paths when generating reproduction metadata. Emit schema v4 only when the new semantics are present while continuing to read existing v3 manifests.

**Tech Stack:** Python 3.10+, Pydantic 2, Hugging Face `datasets`, `huggingface_hub`, `unittest`, TOML/JSON reproduction metadata.

**Spec:** `docs/superpowers/specs/2026-08-24-local-dataset-provenance-design.md`

## Global Constraints

- Do not upload the local dataset or embed raw prompts in reproduction files.
- Do not treat an unverified local path as reproducible.
- Do not expose a local filesystem path in generated reproduction files.
- Preserve the one-prompt-per-line text-file contract.
- Continue accepting existing reproduce.json schema version `3`.
- Abort or clearly disable reproduction on any content-hash mismatch.

---

### Task 1: Provenance schema and canonical content hash

**Files:**
- Modify: `src/heretic/config.py`
- Create: `src/heretic/dataset_provenance.py`
- Modify: `tests/test_config.py`
- Create: `tests/test_dataset_provenance.py`

**Interfaces:**
- Produces: `HuggingFaceDatasetProvenance` with `dataset`, `revision`, `split`, `indices`, `column`, and `content_sha256`.
- Produces: `get_prompt_content_sha256(prompts: Iterable[str]) -> str`.
- Produces: `get_dataset_content_sha256(dataset: Dataset, column: str) -> str`.

- [ ] **Step 1: Add failing schema tests**

Add tests that construct a valid nested provenance object and that reject a branch name as `revision`, malformed SHA-256, negative indices, and an explicit empty index list. Use literal 40/64-character hashes.

```python
provenance = HuggingFaceDatasetProvenance(
    dataset="fka/awesome-chatgpt-prompts",
    revision="a" * 40,
    split="train",
    indices=[3, 104],
    column="prompt",
    content_sha256="b" * 64,
)
self.assertEqual(provenance.indices, [3, 104])
```

- [ ] **Step 2: Add a failing multiline hash test**

Assert that `get_prompt_content_sha256(["alpha\nbeta", "", "gamma"])` equals a hand-generated literal digest and that changing order or record boundaries changes the digest.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_config tests.test_dataset_provenance -v
```

Expected: import errors because the provenance model/module do not exist.

- [ ] **Step 4: Implement the minimal schema and hash functions**

Add `HuggingFaceDatasetProvenance` before `DatasetSpecification`, validate exact lowercase-normalized hex digests, validate the dataset ID with `validate_repo_id`, and add `provenance: HuggingFaceDatasetProvenance | None = None` to `DatasetSpecification`.

Implement the v1 domain-separated, uint64-length-prefixed UTF-8 hash. Raise `TypeError` when a prompt value is not a string and `ValueError` when the requested column is missing.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run the Task 1 command and expect all tests to pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add -- src/heretic/config.py src/heretic/dataset_provenance.py tests/test_config.py tests/test_dataset_provenance.py
git commit -m "feat: define local dataset provenance"
```

---

### Task 2: Verify local materializations and rebuild public sources

**Files:**
- Modify: `src/heretic/dataset_provenance.py`
- Modify: `src/heretic/utils.py`
- Modify: `tests/test_dataset_provenance.py`

**Interfaces:**
- Consumes: Task 1 provenance and hash APIs.
- Produces: `materialize_source_dataset(provenance: HuggingFaceDatasetProvenance) -> Dataset`.
- Produces: `load_verified_dataset(specification: DatasetSpecification) -> Dataset`.

- [ ] **Step 1: Add failing local verification tests**

Use a real temporary `Dataset.from_dict({"prompt": ["alpha\nbeta", "gamma"]}).save_to_disk()` directory. Assert `load_prompts()` returns the two intact values when the hash matches, and raises a `ValueError` containing `content hash` when one prompt changes without updating provenance.

- [ ] **Step 2: Add failing source rematerialization tests**

Patch only the external `load_dataset` call to return a real in-memory source dataset. Give provenance ordered indices `[2, 0]`, serialize the public source ID as `specification.dataset`, and assert `load_prompts()` returns source rows 2 then 0 with newlines intact. Assert the loader was called with the exact revision and source split.

- [ ] **Step 3: Run tests and confirm RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_dataset_provenance -v
```

Expected: failures because provenance-aware loading is absent.

- [ ] **Step 4: Implement materialization and verification**

In `materialize_source_dataset`, call `load_dataset(provenance.dataset, revision=provenance.revision, split=provenance.split)`, reject `DatasetDict`, and apply `Dataset.select(provenance.indices)` when present.

In `load_verified_dataset`, require source/local column equality. Load an existing `state.json` directory with `load_from_disk`; otherwise materialize the public source. Compute the entire materialized column hash and raise a message containing expected and actual hashes on mismatch.

Route `load_prompts` through `load_verified_dataset` before the ordinary HF/local branches whenever provenance is present, then apply the existing Heretic split slice and prompt transformations.

- [ ] **Step 5: Run tests and confirm GREEN**

Run the Task 2 command and the original config tests.

- [ ] **Step 6: Commit Task 2**

```bash
git add -- src/heretic/dataset_provenance.py src/heretic/utils.py tests/test_dataset_provenance.py
git commit -m "feat: verify and rebuild materialized datasets"
```

---

### Task 3: Reproduction eligibility, sanitization, and schema compatibility

**Files:**
- Modify: `src/heretic/dataset_provenance.py`
- Modify: `src/heretic/utils.py`
- Modify: `src/heretic/reproduce.py`
- Modify: `src/heretic/main.py`
- Create: `tests/test_reproduction_metadata.py`

**Interfaces:**
- Consumes: Task 2 verified loaders.
- Produces: `get_dataset_reproducibility_error(specification: DatasetSpecification) -> str | None`.
- Produces: `sanitize_dataset_provenance_paths(value: Any) -> tuple[Any, bool]`.
- Produces: `is_supported_reproduction_version(version: Any) -> bool` accepting strings `"3"` and `"4"`.

- [ ] **Step 1: Add failing eligibility tests**

Test these real behaviors:

- a local dataset without provenance returns an explanatory error;
- a local dataset with matching local and rematerialized source hashes returns `None`;
- a source mismatch returns an error containing `does not match its public source`;
- a direct HF dataset with a pinned commit remains eligible.

- [ ] **Step 2: Add failing serialization tests**

Build settings with a temporary local path and valid provenance. Assert generated JSON:

- uses version `"4"`;
- contains the public source ID instead of the temporary path;
- contains provenance and no raw prompt values.

Assert ordinary HF-only settings still produce version `"3"`. Assert generated reproduction TOML also omits the local path.

- [ ] **Step 3: Add failing reader compatibility tests**

Assert `is_supported_reproduction_version("3")` and `("4")` are true while `"2"`, `5`, and `None` are false.

- [ ] **Step 4: Run tests and confirm RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_reproduction_metadata -v
```

- [ ] **Step 5: Implement eligibility and sanitization**

Eligibility must verify both the local saved dataset and a fresh exact-source materialization. Return errors instead of raising so `main.py` can disable the offer and print labeled reasons without private paths.

Recursively copy dumped settings. Whenever a dict has non-null provenance, validate it and replace its `dataset` value with `provenance.dataset`; return `True` from the recursive aggregate to select schema v4.

Use sanitized settings in both `generate_reproduce_json` and a new `generate_reproduction_config_toml`. Update `create_reproduce_folder` to call the latter.

- [ ] **Step 6: Wire eligibility and version support into main**

Replace the dataset portion of the inline gate with calls to `get_dataset_reproducibility_error`. Print one warning per dataset role only when it blocks reproduction. Accept versions 3 and 4 in `--reproduce` and retain the existing message for pre-plugin versions.

- [ ] **Step 7: Run tests and confirm GREEN**

Run all unit tests:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

- [ ] **Step 8: Commit Task 3**

```bash
git add -- src/heretic/dataset_provenance.py src/heretic/utils.py src/heretic/reproduce.py src/heretic/main.py tests/test_reproduction_metadata.py
git commit -m "feat: reproduce verified local datasets"
```

---

### Task 4: User documentation and public-source integration proof

**Files:**
- Modify: `config.default.toml`
- Modify: `src/heretic/utils.py`
- Create: `tests/test_public_dataset_provenance.py`

**Interfaces:**
- Consumes all prior tasks.
- Verifies the end-to-end public-source/local-materialization contract without loading a language model.

- [ ] **Step 1: Add the pinned integration test**

Gate the network test behind `HERETIC_RUN_NETWORK_TESTS=1` so normal unit tests remain offline. The test must load:

```python
repo_id = "fka/awesome-chatgpt-prompts"
revision = "ca0bf873b687e093f27beaddce8421f92d8ea7b4"
indices = [3, 104]
column = "prompt"
```

It must save the selected rows to a temporary directory, load them through the local provenance path, generate and parse reproduction JSON, assert the local path is absent, reconstruct `Settings` from the JSON, load through the public source path, and assert identical hashes and prompt lists with embedded newlines.

- [ ] **Step 2: Run the integration test and confirm GREEN**

Run:

```bash
HERETIC_RUN_NETWORK_TESTS=1 .venv/bin/python -m unittest tests.test_public_dataset_provenance -v
```

- [ ] **Step 3: Document the optional provenance table**

Add a commented example to `config.default.toml` explaining the exact revision, source split/indices, column, hash helper, and that arbitrary transformations are unsupported. Do not enable provenance in defaults.

Update the reproduction README dataset links to display the public provenance source and revision rather than any local path.

- [ ] **Step 4: Commit Task 4**

```bash
git add -- config.default.toml src/heretic/utils.py tests/test_public_dataset_provenance.py
git commit -m "docs: explain reproducible local datasets"
```

---

### Task 5: Full verification and PR preparation

**Files:**
- Review all changed files.

**Interfaces:**
- Produces a clean branch ready for a draft pull request.

- [ ] **Step 1: Run format and static checks**

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/ty check
```

- [ ] **Step 2: Run all offline unit tests**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

- [ ] **Step 3: Re-run the pinned network integration test**

```bash
HERETIC_RUN_NETWORK_TESTS=1 .venv/bin/python -m unittest tests.test_public_dataset_provenance -v
```

- [ ] **Step 4: Inspect privacy and compatibility invariants**

Generate reproduction JSON/TOML from a temporary local path and search both outputs for that path and for the literal prompt values; both searches must return no matches. Parse a version 3 fixture through the supported-version check.

- [ ] **Step 5: Inspect the final diff and status**

```bash
git diff origin/master...HEAD --check
git status --short --branch
git log --oneline origin/master..HEAD
```

- [ ] **Step 6: Push and create a draft PR**

Push `codex/local-dataset-provenance` to the authenticated fork, resolve the exact cross-repository head, and create a draft PR against `p-e-w/heretic:master` with the investigation evidence, design, tests, privacy guarantees, and schema compatibility notes.
