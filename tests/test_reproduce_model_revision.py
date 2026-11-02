# SPDX-License-Identifier: AGPL-3.0-or-later
"""The reproduction manifest must identify the model revision actually loaded."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from heretic import utils
from heretic.config import InitOnlySettings


class ReproduceModelRevisionTests(unittest.TestCase):
    def create_folder(self, directory, revision, lookup_error=None):
        settings = InitOnlySettings(model="org/base", model_commit=revision)
        trial = SimpleNamespace(user_attrs={"parameters": {}, "scores": []})
        checkpoint = directory / "study.jsonl"
        checkpoint.write_text("study evidence\n")
        with ExitStack() as patches:
            lookup = patches.enter_context(patch.object(
                utils.huggingface_hub, "model_info",
                return_value=SimpleNamespace(sha="b" * 40), side_effect=lookup_error,
            ))
            patches.enter_context(patch.object(utils, "generate_requirements_txt", return_value=""))
            patches.enter_context(patch.object(utils, "generate_config_toml", return_value=""))
            patches.enter_context(patch.object(utils, "generate_reproduce_readme", return_value=""))
            patches.enter_context(patch.object(utils, "get_requirements_dict", return_value={}))
            patches.enter_context(patch.object(utils, "get_heretic_version_info", return_value=SimpleNamespace(
                version="test", is_standard_pypi=True, metadata={},
            )))
            utils.create_reproduce_folder(
                directory / "export", settings, [], checkpoint, trial, {}, False,
            )
        manifest = json.loads((directory / "export/reproduce/reproduce.json").read_text())
        self.assertEqual((directory / "export/reproduce/study.jsonl").read_text(), "study evidence\n")
        return settings, manifest, lookup

    def test_pinned_revision_is_kept_in_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, manifest, lookup = self.create_folder(Path(directory), "a" * 40)
        self.assertEqual(settings.model_commit, "a" * 40)
        self.assertEqual(manifest["settings"]["model_commit"], "a" * 40)
        lookup.assert_not_called()

    def test_symbolic_revision_selection_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, manifest, lookup = self.create_folder(Path(directory), "release-2026")
        self.assertEqual(settings.model_commit, "release-2026")
        self.assertEqual(manifest["settings"]["model_commit"], "release-2026")
        lookup.assert_not_called()

    def test_pinned_revision_needs_no_hub_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest, lookup = self.create_folder(
                Path(directory), "a" * 40, RuntimeError("Hub unavailable"),
            )
        self.assertEqual(manifest["settings"]["model_commit"], "a" * 40)
        lookup.assert_not_called()

    def test_unpinned_revision_is_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, manifest, lookup = self.create_folder(Path(directory), None)
        self.assertEqual(settings.model_commit, "b" * 40)
        self.assertEqual(manifest["settings"]["model_commit"], "b" * 40)
        lookup.assert_called_once_with("org/base")


if __name__ == "__main__":
    unittest.main()
