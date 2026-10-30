# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from optuna.trial import FrozenTrial

from heretic.config import Settings
from heretic.utils import create_reproduce_folder, generate_reproduce_readme


def make_trial() -> FrozenTrial:
    return cast(
        FrozenTrial,
        SimpleNamespace(user_attrs={"index": 0, "scores": []}, params={}),
    )


class ReproduceChecksumFileTests(unittest.TestCase):
    def test_readme_links_checksum_file_relatively(self) -> None:
        readme = generate_reproduce_readme(
            Settings.model_construct(model="org/base", model_commit=None),
            [],
            "study.jsonl",
            make_trial(),
            include_system_information=False,
        )

        # The link must stay relative and carry an extension, so that
        # Hugging Face renders it as a file (/blob/) rather than a
        # directory (/tree/).
        self.assertIn("[`SHA256SUMS.txt`](SHA256SUMS.txt)", readme)
        self.assertIn("`sha256sum -c SHA256SUMS.txt`", readme)
        self.assertNotIn("(SHA256SUMS)", readme)
        self.assertNotIn("sha256sum -c SHA256SUMS`", readme)
        self.assertNotIn("/blob/", readme)

    def test_folder_writes_checksum_file_with_extension(self) -> None:
        hashes = {"model.safetensors": "0" * 64}
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "heretic.utils.huggingface_hub.model_info",
                return_value=SimpleNamespace(sha="abc123"),
            ),
            patch("heretic.utils.generate_reproduce_json", return_value="{}"),
        ):
            path = Path(tmpdir)
            create_reproduce_folder(
                path,
                Settings.model_construct(model="org/base", model_commit=None),
                [],
                checkpoint_path=path / "missing.jsonl",
                trial=make_trial(),
                uploaded_model_hashes=hashes,
                include_system_information=False,
            )

            reproduce_dir = path / "reproduce"
            self.assertFalse((reproduce_dir / "SHA256SUMS").exists())
            self.assertEqual(
                (reproduce_dir / "SHA256SUMS.txt").read_text(encoding="utf-8"),
                f"{'0' * 64} *model.safetensors\n",
            )


if __name__ == "__main__":
    unittest.main()
