# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from datasets import Dataset, NamedSplit

from heretic.config import (
    DatasetSpecification,
    HuggingFaceDatasetProvenance,
    Settings,
)
from heretic.dataset_provenance import get_prompt_content_sha256
from heretic.utils import load_prompts


class PromptContentHashTests(unittest.TestCase):
    def test_hashes_multiline_and_empty_prompts_canonically(self) -> None:
        sha256 = get_prompt_content_sha256(["alpha\nbeta", "", "gamma"])

        self.assertEqual(
            sha256,
            "4a2a61a12a4d0af07c3aea6ea2514db765db945b43c6e04d3ec3b8d15aaad77a",
        )

    def test_hash_preserves_record_boundaries(self) -> None:
        self.assertNotEqual(
            get_prompt_content_sha256(["ab", "c"]),
            get_prompt_content_sha256(["a", "bc"]),
        )

    def test_hash_preserves_record_order(self) -> None:
        self.assertNotEqual(
            get_prompt_content_sha256(["first", "second"]),
            get_prompt_content_sha256(["second", "first"]),
        )


class VerifiedDatasetLoadingTests(unittest.TestCase):
    def test_rejects_plain_text_file_claiming_dataset_provenance(self) -> None:
        with TemporaryDirectory() as temp_directory:
            text_path = Path(temp_directory) / "prompts.txt"
            text_path.write_text("alpha\nbeta\n", encoding="utf-8")
            specification = DatasetSpecification(
                dataset=str(text_path),
                split="train[:]",
                column="prompt",
                provenance=HuggingFaceDatasetProvenance(
                    dataset="source/public",
                    revision="a" * 40,
                    split="train",
                    column="prompt",
                    content_sha256="b" * 64,
                ),
            )

            with self.assertRaisesRegex(ValueError, "save_to_disk"):
                load_prompts(
                    Settings.model_construct(system_prompt="System"),
                    specification,
                )

    def test_rejects_source_and_local_column_disagreement(self) -> None:
        specification = DatasetSpecification(
            dataset="source/public",
            split="train[:]",
            column="local_prompt",
            provenance=HuggingFaceDatasetProvenance(
                dataset="source/public",
                revision="a" * 40,
                configuration="default",
                split="train",
                column="prompt",
                content_sha256="b" * 64,
            ),
        )

        with self.assertRaisesRegex(ValueError, "does not match.*column"):
            load_prompts(
                Settings.model_construct(system_prompt="System"),
                specification,
            )

    def test_loads_verified_save_to_disk_dataset_with_multiline_prompts(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_directory:
            dataset_path = Path(temp_directory) / "dataset"
            Dataset.from_dict(
                {"prompt": ["alpha\nbeta", "gamma"]},
                split=NamedSplit("train"),
            ).save_to_disk(dataset_path)
            specification = DatasetSpecification(
                dataset=str(dataset_path),
                split="train[:]",
                column="prompt",
                provenance=HuggingFaceDatasetProvenance(
                    dataset="source/public",
                    revision="a" * 40,
                    split="train",
                    column="prompt",
                    content_sha256="c271e718182a2ca383440ee2105f65c15546a4d03f8f906fed0e38223144c31b",
                ),
            )
            settings = Settings.model_construct(
                model="unused/model",
                system_prompt="You are a helpful assistant.",
                good_prompts=specification,
            )

            prompts = load_prompts(settings, specification)

        self.assertEqual([prompt.user for prompt in prompts], ["alpha\nbeta", "gamma"])

    def test_rejects_changed_local_content(self) -> None:
        with TemporaryDirectory() as temp_directory:
            dataset_path = Path(temp_directory) / "dataset"
            Dataset.from_dict(
                {"prompt": ["alpha\nCHANGED", "gamma"]},
                split=NamedSplit("train"),
            ).save_to_disk(dataset_path)
            specification = DatasetSpecification(
                dataset=str(dataset_path),
                split="train[:]",
                column="prompt",
                provenance=HuggingFaceDatasetProvenance(
                    dataset="source/public",
                    revision="a" * 40,
                    split="train",
                    column="prompt",
                    content_sha256="c271e718182a2ca383440ee2105f65c15546a4d03f8f906fed0e38223144c31b",
                ),
            )
            settings = Settings.model_construct(
                model="unused/model",
                system_prompt="You are a helpful assistant.",
                good_prompts=specification,
            )

            with self.assertRaisesRegex(ValueError, "content hash"):
                load_prompts(settings, specification)

    def test_rematerializes_ordered_source_indices(self) -> None:
        source = Dataset.from_dict(
            {"prompt": ["row0\nline2", "row1", "row2\nline2"]},
            split=NamedSplit("train"),
        )
        specification = DatasetSpecification(
            dataset="source/public",
            split="train[:]",
            column="prompt",
            provenance=HuggingFaceDatasetProvenance(
                dataset="source/public",
                revision="a" * 40,
                configuration="default",
                split="train",
                indices=[2, 0],
                column="prompt",
                content_sha256="d79e92c036c2bbd14afa53ef9dc9745f679d6997c358fcd254d3fef711450b71",
            ),
        )
        settings = Settings.model_construct(
            model="unused/model",
            system_prompt="You are a helpful assistant.",
            good_prompts=specification,
        )

        with patch(
            "heretic.dataset_provenance.load_dataset", return_value=source
        ) as load_dataset_mock:
            prompts = load_prompts(settings, specification)

        self.assertEqual(
            [prompt.user for prompt in prompts], ["row2\nline2", "row0\nline2"]
        )
        load_dataset_mock.assert_called_once_with(
            "source/public",
            name="default",
            revision="a" * 40,
            split="train",
        )


if __name__ == "__main__":
    unittest.main()
