# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import huggingface_hub
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from huggingface_hub.utils import validate_repo_id

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
        name=provenance.configuration,
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


def get_dataset_reproducibility_error(
    specification: DatasetSpecification,
) -> str | None:
    """Return why a dataset cannot be reproduced, or None when it can be."""

    path = Path(specification.dataset)
    provenance = specification.provenance

    if provenance is None:
        if path.exists():
            return "local dataset has no verified public provenance"

        try:
            validate_repo_id(specification.dataset)
        except ValueError:
            return "local dataset has no verified public provenance"

        if specification.commit is None:
            return "Hugging Face dataset is not pinned to a commit"
        return None

    if path.exists():
        try:
            load_verified_dataset(specification)
        except Exception as error:
            message = str(error)
            if "content hash" in message or "column" in message:
                return message
            return (
                "local materialized dataset could not be verified "
                f"({type(error).__name__})"
            )

    try:
        source_info = huggingface_hub.dataset_info(
            provenance.dataset,
            revision=provenance.revision,
        )
    except Exception as error:
        return (
            f"public provenance source {provenance.dataset}@{provenance.revision} "
            f"could not be inspected ({type(error).__name__})"
        )

    if source_info.sha != provenance.revision:
        return (
            "public provenance revision did not resolve to the declared commit "
            f"(expected {provenance.revision}, got {source_info.sha})"
        )

    if source_info.private is not False or source_info.gated is not False:
        return "provenance source is not public and ungated"

    try:
        source_dataset = materialize_source_dataset(provenance)
        source_sha256 = get_dataset_content_sha256(
            source_dataset,
            provenance.column,
        )
    except Exception as error:
        return (
            f"public provenance source {provenance.dataset}@{provenance.revision} "
            f"could not be materialized ({type(error).__name__})"
        )

    if source_sha256 != provenance.content_sha256:
        return (
            "materialized dataset does not match its public source "
            f"(expected {provenance.content_sha256}, got {source_sha256})"
        )

    return None


def sanitize_dataset_provenance_paths(value: Any) -> tuple[Any, bool]:
    """Copy settings data while replacing provenance-backed local paths."""

    if isinstance(value, list):
        sanitized_items = []
        has_provenance = False
        for item in value:
            sanitized, item_has_provenance = sanitize_dataset_provenance_paths(item)
            sanitized_items.append(sanitized)
            has_provenance = has_provenance or item_has_provenance
        return sanitized_items, has_provenance

    if not isinstance(value, dict):
        return value, False

    sanitized_dict = {}
    has_provenance = False
    for key, item in value.items():
        sanitized, item_has_provenance = sanitize_dataset_provenance_paths(item)
        sanitized_dict[key] = sanitized
        has_provenance = has_provenance or item_has_provenance

    raw_provenance = sanitized_dict.get("provenance")
    if "dataset" in sanitized_dict and raw_provenance is not None:
        provenance = HuggingFaceDatasetProvenance.model_validate(raw_provenance)
        sanitized_dict["dataset"] = provenance.dataset
        has_provenance = True

    return sanitized_dict, has_provenance
