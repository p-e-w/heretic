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


class ModelValidateInitOnlyTests(unittest.TestCase):
    def validate(
        self, values: dict, config_toml: str, argv: tuple[str, ...] = ("heretic",)
    ) -> Settings:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(config_toml)
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                with patch.object(sys, "argv", list(argv)):
                    return Settings.model_validate_init_only(
                        values, Settings(model="current")
                    )
            finally:
                os.chdir(previous_directory)

    def test_keeps_plugin_tables_from_values(self) -> None:
        settings = self.validate(
            {
                "model": "model",
                "modifier": {"Abliteration": {"row_normalization": "pre"}},
            },
            'save_directory = "toml"\n'
            "[modifier.Abliteration]\nwinsorization_quantile = 0.5\n",
            ("heretic", "--save-directory", "current"),
        )

        self.assertEqual(
            settings.model_extra,
            {"modifier": {"Abliteration": {"row_normalization": "pre"}}},
        )
        # Settings with exclude=True come from the current settings.
        self.assertEqual(settings.save_directory, "current")

    def test_drops_plugin_tables_missing_from_values(self) -> None:
        settings = self.validate(
            {"model": "model"},
            "[modifier.Abliteration]\nwinsorization_quantile = 0.5\n",
        )

        self.assertEqual(settings.model_extra, {})

    def test_other_sources_do_not_override_values(self) -> None:
        values = {
            "model": "model",
            "n_trials": 50,
            "max_response_length": 60,
            "n_startup_trials": 20,
        }

        with patch.dict(os.environ, {"HERETIC_MAX_RESPONSE_LENGTH": "9"}):
            settings = self.validate(
                values,
                "n_trials = 7\n",
                ("heretic", "--n-startup-trials", "3"),
            )

        for name, value in values.items():
            self.assertEqual(getattr(settings, name), value)

    def test_settings_missing_from_values_get_defaults(self) -> None:
        with patch.dict(os.environ, {"HERETIC_MAX_RESPONSE_LENGTH": "9"}):
            settings = self.validate(
                {"model": "model"},
                "n_trials = 7\n",
                ("heretic", "--n-startup-trials", "3"),
            )

        for name in ["n_trials", "max_response_length", "n_startup_trials"]:
            self.assertEqual(
                getattr(settings, name), Settings.model_fields[name].default
            )

    def test_keeps_top_level_tables_from_values(self) -> None:
        settings = self.validate(
            {"model": "model", "max_memory": {"0": "20GiB"}},
            'max_memory = { "1" = "8GiB" }\n',
        )

        self.assertEqual(settings.max_memory, {"0": "20GiB"})


class SettingsUploadTagsTests(unittest.TestCase):
    def test_default_upload_tags(self) -> None:
        with patch("sys.argv", ["heretic"]):
            settings = Settings(model="dummy")
            self.assertEqual(
                settings.upload_tags,
                ["uncensored", "decensored", "abliterated"],
            )

    def test_custom_upload_tags_override(self) -> None:
        with patch("sys.argv", ["heretic"]):
            settings = Settings(model="dummy", upload_tags=["custom-tag", "experiment"])
            self.assertEqual(settings.upload_tags, ["custom-tag", "experiment"])

    def test_empty_upload_tags(self) -> None:
        with patch("sys.argv", ["heretic"]):
            settings = Settings(model="dummy", upload_tags=[])
            self.assertEqual(settings.upload_tags, [])

    def test_upload_tags_from_cli(self) -> None:
        with patch("sys.argv", ["heretic", "--upload-tags", '["cli-tag", "test"]']):
            settings = Settings(model="dummy")
            self.assertEqual(settings.upload_tags, ["cli-tag", "test"])


if __name__ == "__main__":
    unittest.main()
