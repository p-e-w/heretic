# Reproducible Local Dataset Provenance Design

## Goal

Allow a `datasets.save_to_disk()` input to participate in Heretic's native
reproduction system when it is a verified materialization of an exact,
public, commit-pinned Hugging Face dataset selection.

## Current failure

Heretic can load a saved `Dataset` with `load_from_disk()`, but the native
reproducibility gate requires every dataset path to be a Hugging Face repo ID
with a pinned `commit`. A local saved dataset therefore disables reproduction.
If the existing `reproduce.json` generator is called directly, it serializes
the local filesystem path and no dataset content hash, so another machine
cannot rebuild or verify the input.

This is distinct from plain text handling. Heretic's text format intentionally
uses one prompt per line; this feature does not change that contract.

## User-facing schema

`DatasetSpecification` gains an optional nested `provenance` object:

```toml
[good_prompts]
dataset = "/data/materialized-prompts"
split = "train[:]"
column = "prompt"

[good_prompts.provenance]
dataset = "fka/awesome-chatgpt-prompts"
revision = "ca0bf873b687e093f27beaddce8421f92d8ea7b4"
split = "train"
indices = [3, 104]
column = "prompt"
content_sha256 = "383dc9e12acf9ea26fb2f85837bb88eae9465d4da740cab9082b9093a3950dd6"
```

The fields mean:

- `dataset`: public Hugging Face dataset ID.
- `revision`: exact 40-character commit SHA; branches and tags are rejected.
- `split`: original split or Hugging Face split slice.
- `indices`: optional ordered row indices applied after loading `split`.
  Omitting it materializes the complete selected split.
- `column`: source prompt column, which must match the local specification's
  `column` in this first version. Column renames and arbitrary maps are not
  silently inferred.
- `content_sha256`: hash of the complete ordered materialized prompt column.

`split` plus optional `indices` covers exact slices and arbitrary row
selections without embedding prompts. Future deterministic transformations can
be added as typed, versioned operations; arbitrary code transformations are out
of scope because they cannot be safely reconstructed from parameters alone.

## Canonical prompt-content hash

The hash is SHA-256 over this byte stream:

1. The ASCII domain separator `heretic-prompt-content-v1\0`.
2. For every prompt in order: its UTF-8 byte length as an unsigned 64-bit
   big-endian integer, followed by the exact UTF-8 bytes.

No Unicode normalization or whitespace normalization is performed. Length
prefixes make record boundaries unambiguous, including multiline prompts,
empty strings, and embedded NUL characters. Only the prompt column is hashed
because it is the dataset content that Heretic consumes; unrelated metadata
columns cannot affect the run.

The public `get_prompt_content_sha256()` helper lets materialization scripts
compute the value using the same documented algorithm.

## Loading and verification

The existing behavior remains unchanged when `provenance` is absent.

When provenance is present:

- If `dataset` is an existing `save_to_disk()` directory, Heretic loads it,
  verifies the source/local column agreement, computes the local content hash,
  and fails clearly if it differs from `content_sha256`.
- If the path is unavailable or has been sanitized to the public source ID,
  Heretic loads the exact `dataset@revision`, applies `split` and `indices`,
  verifies the content hash, and uses that materialized in-memory dataset.
- The normal `DatasetSpecification.split`, prefix, suffix, and system-prompt
  processing is applied after materialization, exactly as it is for the local
  saved dataset.

Thus a copied reproduction config works without the original local directory.

## Reproduction eligibility

A direct Hugging Face dataset remains eligible when it has a pinned commit.

A local saved dataset is eligible only when all of these checks pass:

1. It has a valid provenance object.
2. Its declared local prompt-content hash matches its actual content.
3. The exact public source revision can be fetched.
4. Rematerializing the declared split and indices yields the same hash.

Failures disable the reproducibility offer and emit a clear reason. A local
path with no provenance never becomes reproducible merely because it exists.

## Reproduction files and privacy

For verified local provenance, generated `config.toml` and `reproduce.json`
replace the local path with the public source dataset ID while retaining the
provenance object. Raw prompts and local filesystem paths are not emitted, and
nothing is uploaded as a dataset.

Manifests without materialized provenance remain schema version `3`. A manifest
containing the new semantics uses version `4`. `heretic --reproduce` accepts
both versions, preserving support for existing version 3 manifests. Older
Heretic releases reject version 4 rather than ignoring provenance and silently
using the wrong rows.

## Error handling

- Invalid repo IDs, non-commit revisions, malformed hashes, empty explicit
  index lists, negative indices, and source/local column disagreement are
  rejected with actionable validation errors.
- A local content mismatch fails prompt loading when provenance was explicitly
  claimed.
- An inaccessible source or source hash mismatch prevents native
  reproducibility and identifies the affected dataset role without printing a
  private local path.
- During reproduction, a rematerialization hash mismatch aborts before model
  optimization begins.

## Test strategy

Unit tests cover schema validation, the canonical hash including multiline
records, local `save_to_disk()` verification, source rematerialization and
ordered indices, mismatch failures, eligibility, path sanitization, schema
version selection, and version 3/4 reader compatibility.

The final integration test uses
`fka/awesome-chatgpt-prompts@ca0bf873b687e093f27beaddce8421f92d8ea7b4`,
rows `[3, 104]`, and column `prompt`: save locally, load through Heretic,
serialize reproduction settings without the local path, rematerialize from the
public source, verify the identical hash, and load the same two multiline
prompts.

## Non-goals

- Changing the one-prompt-per-line text format.
- Uploading local data to the Hub.
- Embedding raw prompts in reproduction metadata.
- Trusting paths, filenames, or unverified user-supplied hashes.
- Reproducing arbitrary Python `map()` functions or private source datasets.
