# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from tempfile import TemporaryDirectory

from optuna.trial import create_trial
from pydantic import ValidationError

from heretic.config import DatasetSpecification, ScorerConfig, Settings
from heretic.utils import (
    generate_config_toml,
    generate_reproduce_json,
    generate_reproduce_readme,
    get_reproduction_dataset,
    is_dataset_reproducible,
)


class ScorerConfigTests(unittest.TestCase):
    def test_accepts_slug_like_instance_name(self) -> None:
        config = ScorerConfig(
            plugin="heretic.scorers.keyword_rate.KeywordRate",
            optimization="minimize",
            instance_name="small-1",
        )

        self.assertEqual(config.instance_name, "small-1")

    def test_rejects_empty_instance_name(self) -> None:
        with self.assertRaises(ValidationError):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name=" \t",
            )

    def test_rejects_whitespace_in_instance_name(self) -> None:
        for instance_name in ["small name", "small\tname", "small\nname"]:
            with self.subTest(instance_name=instance_name):
                with self.assertRaisesRegex(
                    ValidationError, "whitespace is not allowed"
                ):
                    ScorerConfig(
                        plugin="heretic.scorers.keyword_rate.KeywordRate",
                        optimization="minimize",
                        instance_name=instance_name,
                    )

    def test_rejects_dot_in_instance_name(self) -> None:
        with self.assertRaisesRegex(ValidationError, "'\\.' is not allowed"):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name="small.name",
            )


class LocalDatasetSourceTests(unittest.TestCase):
    def test_public_source_makes_local_dataset_reproducible(self) -> None:
        with TemporaryDirectory() as local_path:
            specification = DatasetSpecification(
                dataset=local_path,
                source_dataset="source/public",
                commit="a" * 40,
            )

            self.assertTrue(is_dataset_reproducible(specification))
            for update in (
                {"source_dataset": "not a valid repo!"},
                {"source_dataset": None},
                {"commit": None},
            ):
                self.assertFalse(
                    is_dataset_reproducible(specification.model_copy(update=update))
                )

    def test_reproduction_settings_replace_only_local_datasets(self) -> None:
        local_path = "C:/private/materialized-prompts"
        specification = DatasetSpecification(
            dataset=local_path,
            source_dataset="source/public",
            commit="a" * 40,
        )

        self.assertEqual(specification.model_dump()["dataset"], local_path)
        self.assertEqual(specification.model_dump()["source_dataset"], "source/public")

        settings = Settings.model_construct(
            model="source/model",
            good_prompts=specification,
            scorer={
                "KeywordRate": {"prompts": specification.model_dump(exclude_none=True)}
            },
        )
        trial = create_trial(
            values=[],
            user_attrs=dict(direction_index=None, index=0, parameters={}, scores=[]),
        )
        artifacts = (
            generate_config_toml(settings),
            generate_reproduce_json(settings, trial, "2026-09-12", {}, False),
            generate_reproduce_readme(settings, "study.log", trial, False),
        )

        for artifact in artifacts:
            self.assertNotIn(local_path, artifact)
            self.assertNotIn("source_dataset", artifact)
            self.assertIn("source/public", artifact)
        self.assertIn('"version": "3"', artifacts[1])
        self.assertEqual(
            get_reproduction_dataset("source/actual", "source/other"),
            "source/actual",
        )


if __name__ == "__main__":
    unittest.main()
