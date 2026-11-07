# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from heretic.config import Settings
from heretic.plugin import Plugin
from heretic.utils import generate_config_toml


class DummyNestedSettings(BaseModel):
    commit: str | None = None
    name: str = "default"


class DummySettings(BaseModel):
    value: int = 0
    optional: str | None = None
    nested: DummyNestedSettings = DummyNestedSettings()


FULL_SETTINGS = {
    "value": 42,
    "optional": None,
    "nested": {"commit": None, "name": "default"},
}


class DummyPlugin(Plugin):
    settings: DummySettings


class WriteSettingsToConfigTests(unittest.TestCase):
    def make_settings(self) -> Settings:
        # Ensure no config.toml or HERETIC_* environment variables
        # from the surrounding environment leak into the test.
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("HERETIC_")
            }
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                with (
                    patch.dict(os.environ, environment, clear=True),
                    patch.object(sys, "argv", ["heretic"]),
                ):
                    return Settings(model="dummy")
            finally:
                os.chdir(previous_directory)

    def make_plugin(self, heretic_settings: Settings) -> DummyPlugin:
        return DummyPlugin(
            heretic_settings=heretic_settings,
            settings=DummySettings(value=42),
        )

    def test_records_effective_settings_without_config_file(self) -> None:
        # Without a configuration file, model_extra starts out empty,
        # and the plugin's settings must still be recorded.
        heretic_settings = self.make_settings()
        self.assertEqual(heretic_settings.model_extra, {})

        self.make_plugin(heretic_settings).write_settings_to_config("scorer", None)

        self.assertEqual(
            heretic_settings.model_extra,
            {"scorer": {"DummyPlugin": FULL_SETTINGS}},
        )

    def test_records_instance_settings_under_instance_key(self) -> None:
        heretic_settings = self.make_settings()

        self.make_plugin(heretic_settings).write_settings_to_config("scorer", "small")

        self.assertEqual(
            heretic_settings.model_extra,
            {"scorer": {"DummyPlugin_small": FULL_SETTINGS}},
        )

    def test_captures_runtime_mutations(self) -> None:
        # Plugins resolve settings values at runtime (e.g. pinned dataset
        # commits); the recorded settings must reflect those mutations.
        heretic_settings = self.make_settings()
        plugin = self.make_plugin(heretic_settings)

        plugin.settings.value = 7
        plugin.settings.optional = "resolved"
        plugin.settings.nested.commit = "abc"

        plugin.write_settings_to_config("modifier", None)

        self.assertEqual(
            heretic_settings.model_extra,
            {
                "modifier": {
                    "DummyPlugin": {
                        "value": 7,
                        "optional": "resolved",
                        "nested": {"commit": "abc", "name": "default"},
                    }
                }
            },
        )

    def test_recorded_settings_survive_serialization_roundtrip(self) -> None:
        heretic_settings = self.make_settings()
        self.make_plugin(heretic_settings).write_settings_to_config("scorer", None)

        # This is how settings are stored in study journal files and
        # reproduce.json, and how they are restored when resuming
        # a study or reproducing a model.
        restored = Settings.model_validate_init_only(
            json.loads(heretic_settings.model_dump_json()), heretic_settings
        )

        self.assertEqual(
            restored.model_extra,
            {"scorer": {"DummyPlugin": FULL_SETTINGS}},
        )

    def test_recorded_settings_appear_in_generated_config_toml(self) -> None:
        heretic_settings = self.make_settings()
        self.make_plugin(heretic_settings).write_settings_to_config("scorer", None)

        exported = generate_config_toml(heretic_settings)

        # None values are omitted (TOML has no null value); they fall back
        # to their defaults when the settings are validated.
        self.assertIn("[scorer.DummyPlugin]", exported)
        self.assertIn("value = 42", exported)
        self.assertNotIn("optional", exported)
        self.assertIn("[scorer.DummyPlugin.nested]", exported)
        self.assertIn('name = "default"', exported)
        self.assertNotIn("commit", exported)

    def test_none_values_are_omitted_from_config_toml(self) -> None:
        heretic_settings = self.make_settings()
        plugin = self.make_plugin(heretic_settings)
        plugin.settings.nested.commit = "abc"
        plugin.write_settings_to_config("scorer", None)

        exported = generate_config_toml(heretic_settings)

        self.assertNotIn("optional", exported)
        self.assertIn("[scorer.DummyPlugin.nested]", exported)
        self.assertIn('commit = "abc"', exported)
        self.assertIn('name = "default"', exported)

    def test_recorded_settings_feed_back_into_plugin_validation(self) -> None:
        # The recorded settings must be readable by the same mechanism
        # that reads raw settings from the configuration file, so that
        # resuming a study or reproducing a model restores them exactly.
        heretic_settings = self.make_settings()
        self.make_plugin(heretic_settings).write_settings_to_config("scorer", None)

        raw_settings = DummyPlugin.get_settings_raw(
            heretic_settings.model_extra, "scorer", None
        )
        validated = DummyPlugin.validate_settings(raw_settings)

        self.assertEqual(validated, DummySettings(value=42))


if __name__ == "__main__":
    unittest.main()
