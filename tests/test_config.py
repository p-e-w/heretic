# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from heretic.config import ScorerConfig, Settings


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


class StoredSettingsTests(unittest.TestCase):
    def test_keeps_stored_plugin_tables(self) -> None:
        stored = {
            "model": "model",
            "modifier": {"Abliteration": {"row_normalization": "pre"}},
        }

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(
                'save_directory = "out"\n'
                "[modifier.Abliteration]\nwinsorization_quantile = 0.5\n"
            )
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                with patch.object(sys, "argv", ["heretic"]):
                    settings = Settings.from_stored(stored)
            finally:
                os.chdir(previous_directory)

        self.assertEqual(
            settings.model_extra,
            {"modifier": {"Abliteration": {"row_normalization": "pre"}}},
        )
        # Values that weren't stored still come from config.toml.
        self.assertEqual(settings.save_directory, "out")
