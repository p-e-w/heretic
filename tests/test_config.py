# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from pydantic import ValidationError

from heretic.config import (
    DatasetSpecification,
    HuggingFaceDatasetProvenance,
    ScorerConfig,
)


class HuggingFaceDatasetProvenanceTests(unittest.TestCase):
    def test_accepts_exact_public_source_selection(self) -> None:
        provenance = HuggingFaceDatasetProvenance(
            dataset="fka/awesome-chatgpt-prompts",
            revision="a" * 40,
            split="train",
            indices=[3, 104],
            column="prompt",
            content_sha256="b" * 64,
        )

        specification = DatasetSpecification(
            dataset="/data/materialized-prompts",
            split="train[:]",
            column="prompt",
            provenance=provenance,
        )

        self.assertEqual(specification.provenance, provenance)

    def test_rejects_non_commit_revision(self) -> None:
        with self.assertRaisesRegex(ValidationError, "40-character commit SHA"):
            HuggingFaceDatasetProvenance(
                dataset="fka/awesome-chatgpt-prompts",
                revision="main",
                split="train",
                column="prompt",
                content_sha256="b" * 64,
            )

    def test_rejects_malformed_content_hash(self) -> None:
        with self.assertRaisesRegex(ValidationError, "64-character SHA-256"):
            HuggingFaceDatasetProvenance(
                dataset="fka/awesome-chatgpt-prompts",
                revision="a" * 40,
                split="train",
                column="prompt",
                content_sha256="not-a-hash",
            )

    def test_rejects_negative_indices(self) -> None:
        with self.assertRaises(ValidationError):
            HuggingFaceDatasetProvenance(
                dataset="fka/awesome-chatgpt-prompts",
                revision="a" * 40,
                split="train",
                indices=[-1],
                column="prompt",
                content_sha256="b" * 64,
            )

    def test_rejects_explicit_empty_indices(self) -> None:
        with self.assertRaisesRegex(ValidationError, "at least one row index"):
            HuggingFaceDatasetProvenance(
                dataset="fka/awesome-chatgpt-prompts",
                revision="a" * 40,
                split="train",
                indices=[],
                column="prompt",
                content_sha256="b" * 64,
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


if __name__ == "__main__":
    unittest.main()
