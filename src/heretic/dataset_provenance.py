# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import hashlib
from collections.abc import Iterable
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME

from .config import DatasetSpecification, HuggingFaceDatasetProvenance


_PROMPT_CONTENT_HASH_DOMAIN = b"heretic-prompt-content-v1\0"


def get_prompt_content_sha256(prompts: Iterable[str]) -> str:
    """Hash an ordered prompt sequence without losing record boundaries."""

    digest = hashlib.sha256(_PROMPT_CONTENT_HASH_DOMAIN)

    for prompt in prompts:
        if not isinstance(prompt, str):
            raise TypeError("Prompt content values must be strings")

        encoded = prompt.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)

    return digest.hexdigest()


def get_dataset_content_sha256(dataset: Dataset, column: str) -> str:
    """Hash the complete ordered prompt column of a materialized dataset."""

    if column not in dataset.column_names:
        raise ValueError(f'Dataset does not contain provenance column "{column}"')

    return get_prompt_content_sha256(dataset[column])


def materialize_source_dataset(
    provenance: HuggingFaceDatasetProvenance,
) -> Dataset:
    """Rebuild a materialized dataset from its exact public source selection."""

    dataset = load_dataset(
        provenance.dataset,
        revision=provenance.revision,
        split=provenance.split,
    )
    if isinstance(dataset, DatasetDict):
        raise ValueError("Provenance source split resolved to a DatasetDict")

    if provenance.indices is not None:
        dataset = dataset.select(provenance.indices)

    return dataset


def load_verified_dataset(specification: DatasetSpecification) -> Dataset:
    """Load a local materialization or rebuild it, then verify its prompt hash."""

    provenance = specification.provenance
    if provenance is None:
        raise ValueError("Dataset provenance is required for verified loading")

    if specification.column != provenance.column:
        raise ValueError(
            "Materialized dataset column does not match its provenance column "
            f'("{specification.column}" != "{provenance.column}")'
        )

    path = Path(specification.dataset)
    state_path = path / DATASET_STATE_JSON_FILENAME

    if state_path.exists():
        dataset = load_from_disk(path)
        if isinstance(dataset, DatasetDict):
            raise ValueError("Loading materialized DatasetDict inputs is not supported")
    elif path.exists():
        raise ValueError(
            "Dataset provenance is only supported for directories created with "
            "datasets.save_to_disk()"
        )
    else:
        dataset = materialize_source_dataset(provenance)

    actual_sha256 = get_dataset_content_sha256(dataset, provenance.column)
    if actual_sha256 != provenance.content_sha256:
        raise ValueError(
            "Materialized dataset content hash does not match provenance "
            f"(expected {provenance.content_sha256}, got {actual_sha256})"
        )

    return dataset
