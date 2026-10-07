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


class StoredSettingsTests(unittest.TestCase):
    def from_stored(
        self, stored: dict, config_toml: str, argv: tuple[str, ...] = ("heretic",)
    ) -> Settings:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(config_toml)
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                with patch.object(sys, "argv", list(argv)):
                    return Settings.from_stored(stored, Settings(model="current"))
            finally:
                os.chdir(previous_directory)

    def test_keeps_stored_plugin_tables(self) -> None:
        settings = self.from_stored(
            {
                "model": "model",
                "modifier": {"Abliteration": {"row_normalization": "pre"}},
            },
            'save_directory = "out"\n'
            "[modifier.Abliteration]\nwinsorization_quantile = 0.5\n",
        )

        self.assertEqual(
            settings.model_extra,
            {"modifier": {"Abliteration": {"row_normalization": "pre"}}},
        )
        # Values that weren't stored still come from config.toml.
        self.assertEqual(settings.save_directory, "out")

    def test_drops_plugin_tables_that_were_not_stored(self) -> None:
        settings = self.from_stored(
            {"model": "model"},
            "[modifier.Abliteration]\nwinsorization_quantile = 0.5\n",
        )

        self.assertEqual(settings.model_extra, {})

    def test_does_not_read_other_sources(self) -> None:
        with patch.dict(os.environ, {"HERETIC_MAX_RESPONSE_LENGTH": "9"}):
            settings = self.from_stored(
                {"model": "model"},
                "n_trials = 7\n",
                ("heretic", "--n-startup-trials", "3"),
            )

        self.assertEqual(settings.model, "model")
        for name in ["n_trials", "max_response_length", "n_startup_trials"]:
            self.assertEqual(
                getattr(settings, name), Settings.model_fields[name].get_default()
            )

    def test_keeps_stored_top_level_tables(self) -> None:
        settings = self.from_stored(
            {"model": "model", "max_memory": {"0": "20GiB"}},
            'max_memory = { "1" = "8GiB" }\n',
        )

        self.assertEqual(settings.max_memory, {"0": "20GiB"})


if __name__ == "__main__":
    unittest.main()
