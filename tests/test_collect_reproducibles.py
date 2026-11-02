# SPDX-License-Identifier: AGPL-3.0-or-later
"""Collection must archive the file revision identified by its filename."""

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from heretic import reproduce


class CollectReproduciblesTests(unittest.TestCase):
    def collect(self, directory, metadata_revision, head_revision):
        payloads = {}
        for revision in {metadata_revision, head_revision}:
            payload = directory / f"cache-{revision}.json"
            payload.write_text(f'{{"revision": "{revision}"}}', encoding="utf-8")
            payloads[revision] = payload

        def download(repo_id, filename, *, revision=None):
            self.assertEqual(repo_id, "owner/model")
            self.assertEqual(filename, "reproduce/reproduce.json")
            return str(payloads[revision or head_revision])

        with ExitStack() as patches:
            api = patches.enter_context(patch.object(reproduce, "HfApi")).return_value
            api.list_models.return_value = [
                SimpleNamespace(id="owner/model", tags=["heretic"], gated=False)
            ]
            api.get_paths_info.return_value = [
                SimpleNamespace(last_commit=SimpleNamespace(oid=metadata_revision))
            ]
            downloaded = patches.enter_context(
                patch.object(reproduce, "hf_hub_download", side_effect=download)
            )
            patches.enter_context(patch.object(reproduce, "print"))
            patches.enter_context(patch.object(reproduce, "disable_progress_bars"))
            patches.enter_context(patch.object(reproduce, "enable_progress_bars"))
            reproduce.collect_reproducibles(str(directory / "archive"))
        return downloaded

    def archived_file(self, directory, revision):
        return directory / "archive/huggingface.co/owner" / f"model-{revision[:7]}.json"

    def test_frozen_head_downloads_the_identified_revision(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            downloaded = self.collect(directory, revision, revision)
            self.assertEqual(
                self.archived_file(directory, revision).read_text(encoding="utf-8"),
                f'{{"revision": "{revision}"}}',
            )
        downloaded.assert_called_once_with(
            "owner/model", "reproduce/reproduce.json", revision=revision
        )

    def test_moving_head_cannot_change_the_archived_revision(self):
        observed, moved = "a" * 40, "b" * 40
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.collect(directory, observed, moved)
            self.assertEqual(
                self.archived_file(directory, observed).read_text(encoding="utf-8"),
                f'{{"revision": "{observed}"}}',
            )

    def test_new_revision_is_archived_without_rewriting_previous_version(self):
        first, second, moved = "a" * 40, "b" * 40, "c" * 40
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.collect(directory, first, first)
            self.collect(directory, second, moved)
            for revision in (first, second):
                self.assertEqual(
                    self.archived_file(directory, revision).read_text(encoding="utf-8"),
                    f'{{"revision": "{revision}"}}',
                )

    def test_existing_revision_is_not_downloaded_again_when_head_moves(self):
        revision, moved = "a" * 40, "b" * 40
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.collect(directory, revision, revision)
            downloaded = self.collect(directory, revision, moved)
            self.assertEqual(
                self.archived_file(directory, revision).read_text(encoding="utf-8"),
                f'{{"revision": "{revision}"}}',
            )
        downloaded.assert_not_called()


if __name__ == "__main__":
    unittest.main()
