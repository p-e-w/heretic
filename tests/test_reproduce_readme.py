# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import inspect
import unittest
from types import SimpleNamespace
from typing import cast

from optuna.trial import FrozenTrial

from heretic.config import Settings
from heretic.utils import create_reproduce_folder, generate_reproduce_readme


class GenerateReproduceReadmeTests(unittest.TestCase):
    def test_sha256sums_link_uses_huggingface_blob(self) -> None:
        trial = cast(
            FrozenTrial,
            SimpleNamespace(user_attrs={"index": 0, "scores": []}),
        )
        cases = {
            "heretic-org/example-heretic": (
                "https://huggingface.co/heretic-org/example-heretic"
                "/blob/main/reproduce/SHA256SUMS"
            ),
            "heretic-org/LFM-2.5-2.6B-heretic": (
                "https://huggingface.co/heretic-org/LFM-2.5-2.6B-heretic"
                "/blob/main/reproduce/SHA256SUMS"
            ),
        }
        for repo_id, href in cases.items():
            with self.subTest(repo_id=repo_id):
                readme = generate_reproduce_readme(
                    Settings.model_construct(model="org/base", model_commit=None),
                    [],
                    "study.jsonl",
                    trial,
                    include_system_information=False,
                    repo_id=repo_id,
                )

                self.assertIn(f"[`SHA256SUMS`]({href})", readme)
                self.assertIn("(requirements.txt)", readme)
                self.assertIn("(config.toml)", readme)
                self.assertIn("(reproduce.json)", readme)
                self.assertIn("(study.jsonl)", readme)
                self.assertIn("sha256sum -c SHA256SUMS", readme)
                self.assertNotIn("(SHA256SUMS)", readme)

    def test_repo_id_has_no_default(self) -> None:
        for function in (generate_reproduce_readme, create_reproduce_folder):
            with self.subTest(function=function.__name__):
                parameter = inspect.signature(function).parameters["repo_id"]
                self.assertIs(parameter.default, inspect.Parameter.empty)


if __name__ == "__main__":
    unittest.main()
