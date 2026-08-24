# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import hashlib
from collections.abc import Iterable

from datasets import Dataset


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
