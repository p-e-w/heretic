# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from datasets import Dataset, NamedSplit
from optuna.trial import FrozenTrial, create_trial

from heretic.config import (
    DatasetSpecification,
    HuggingFaceDatasetProvenance,
    Settings,
)
from heretic.dataset_provenance import (
    get_dataset_reproducibility_error,
    sanitize_dataset_provenance_paths,
)
from heretic.reproduce import is_supported_reproduction_version
from heretic.utils import (
    generate_reproduce_json,
    generate_reproduction_config_toml,
)


CONTENT_SHA256 = "c271e718182a2ca383440ee2105f65c15546a4d03f8f906fed0e38223144c31b"


def make_provenance() -> HuggingFaceDatasetProvenance:
    return HuggingFaceDatasetProvenance(
        dataset="source/public",
        revision="a" * 40,
        split="train",
        column="prompt",
        content_sha256=CONTENT_SHA256,
    )


def make_settings(specification: DatasetSpecification) -> Settings:
    return Settings.model_construct(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        system_prompt="You are a helpful assistant.",
        good_prompts=specification,
    )


def make_trial() -> FrozenTrial:
    return create_trial(
        values=[],
        user_attrs={
            "direction_index": None,
            "parameters": {},
            "scores": [],
        },
    )


class DatasetReproducibilityEligibilityTests(unittest.TestCase):
    def test_rejects_local_dataset_without_provenance(self) -> None:
        with TemporaryDirectory() as temp_directory:
            specification = DatasetSpecification(
                dataset=temp_directory,
                split="train[:]",
                column="prompt",
            )

            error = get_dataset_reproducibility_error(specification)

        self.assertIsNotNone(error)
        self.assertIn("no verified public provenance", error or "")

    def test_accepts_pinned_hugging_face_dataset(self) -> None:
        specification = DatasetSpecification(
            dataset="source/public",
            commit="a" * 40,
            split="train[:2]",
            column="prompt",
        )

        self.assertIsNone(get_dataset_reproducibility_error(specification))

    def test_accepts_local_dataset_matching_public_source(self) -> None:
        source = Dataset.from_dict(
            {"prompt": ["alpha\nbeta", "gamma"]}, split=NamedSplit("train")
        )
        with TemporaryDirectory() as temp_directory:
            dataset_path = Path(temp_directory) / "dataset"
            source.save_to_disk(dataset_path)
            specification = DatasetSpecification(
                dataset=str(dataset_path),
                split="train[:]",
                column="prompt",
                provenance=make_provenance(),
            )

            with patch("heretic.dataset_provenance.load_dataset", return_value=source):
                error = get_dataset_reproducibility_error(specification)

        self.assertIsNone(error)

    def test_rejects_local_dataset_not_matching_public_source(self) -> None:
        local = Dataset.from_dict(
            {"prompt": ["alpha\nbeta", "gamma"]}, split=NamedSplit("train")
        )
        changed_source = Dataset.from_dict(
            {"prompt": ["different", "source"]}, split=NamedSplit("train")
        )
        with TemporaryDirectory() as temp_directory:
            dataset_path = Path(temp_directory) / "dataset"
            local.save_to_disk(dataset_path)
            specification = DatasetSpecification(
                dataset=str(dataset_path),
                split="train[:]",
                column="prompt",
                provenance=make_provenance(),
            )

            with patch(
                "heretic.dataset_provenance.load_dataset",
                return_value=changed_source,
            ):
                error = get_dataset_reproducibility_error(specification)

        self.assertIsNotNone(error)
        self.assertIn("does not match its public source", error or "")


class ReproductionSerializationTests(unittest.TestCase):
    def test_schema_v4_sanitizes_local_dataset_path(self) -> None:
        local_path = "/private/users/alice/materialized-prompts"
        specification = DatasetSpecification(
            dataset=local_path,
            split="train[:]",
            column="prompt",
            provenance=make_provenance(),
        )
        settings = make_settings(specification)

        contents = generate_reproduce_json(
            settings,
            make_trial(),
            timestamp="2026-08-24T00:00:00",
            uploaded_model_hashes={},
            include_system_information=False,
        )
        manifest = json.loads(contents)

        self.assertEqual(manifest["version"], "4")
        self.assertEqual(
            manifest["settings"]["good_prompts"]["dataset"], "source/public"
        )
        self.assertNotIn(local_path, contents)
        self.assertNotIn("alpha\\nbeta", contents)
        self.assertEqual(
            manifest["settings"]["good_prompts"]["provenance"]["revision"],
            "a" * 40,
        )

    def test_hugging_face_only_manifest_remains_schema_v3(self) -> None:
        specification = DatasetSpecification(
            dataset="source/public",
            commit="a" * 40,
            split="train[:2]",
            column="prompt",
        )

        contents = generate_reproduce_json(
            make_settings(specification),
            make_trial(),
            timestamp="2026-08-24T00:00:00",
            uploaded_model_hashes={},
            include_system_information=False,
        )

        self.assertEqual(json.loads(contents)["version"], "3")

    def test_reproduction_toml_sanitizes_local_dataset_path(self) -> None:
        local_path = "/private/users/alice/materialized-prompts"
        specification = DatasetSpecification(
            dataset=local_path,
            split="train[:]",
            column="prompt",
            provenance=make_provenance(),
        )

        contents = generate_reproduction_config_toml(make_settings(specification))

        self.assertNotIn(local_path, contents)
        self.assertIn('dataset = "source/public"', contents)
        self.assertIn("[good_prompts.provenance]", contents)

    def test_sanitizes_nested_plugin_dataset_path(self) -> None:
        raw_settings = {
            "scorer": {
                "KeywordRate": {
                    "prompts": {
                        "dataset": "/private/plugin-prompts",
                        "provenance": make_provenance().model_dump(),
                    }
                }
            }
        }

        sanitized, has_provenance = sanitize_dataset_provenance_paths(raw_settings)

        self.assertTrue(has_provenance)
        self.assertEqual(
            sanitized["scorer"]["KeywordRate"]["prompts"]["dataset"],
            "source/public",
        )
        self.assertEqual(
            raw_settings["scorer"]["KeywordRate"]["prompts"]["dataset"],
            "/private/plugin-prompts",
        )


class ReproductionVersionTests(unittest.TestCase):
    def test_accepts_current_and_materialized_dataset_schemas(self) -> None:
        self.assertTrue(is_supported_reproduction_version("3"))
        self.assertTrue(is_supported_reproduction_version("4"))

    def test_rejects_unsupported_schema_versions(self) -> None:
        for version in ["2", 5, None]:
            with self.subTest(version=version):
                self.assertFalse(is_supported_reproduction_version(version))


if __name__ == "__main__":
    unittest.main()
