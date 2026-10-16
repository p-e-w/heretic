# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from typing import Any, cast
from unittest.mock import patch

from huggingface_hub.hf_api import RepoSibling

from heretic.config import Settings
from heretic.utils import upload_reproduce_folder


class FakeModelInfo:
    def __init__(self, siblings: list[RepoSibling]) -> None:
        self.siblings = siblings


class UploadReproduceFolderTests(unittest.TestCase):
    def test_raises_when_weight_file_has_no_lfs_info(self) -> None:
        info = FakeModelInfo([RepoSibling(rfilename="model.safetensors", lfs=None)])

        with patch("heretic.utils.huggingface_hub.HfApi") as api:
            api.return_value.model_info.return_value = info

            with self.assertRaisesRegex(RuntimeError, "uploaded model hashes"):
                upload_reproduce_folder(
                    "user/model",
                    Settings.model_construct(),
                    [],
                    "token",
                    checkpoint_path="checkpoint.jsonl",
                    trial=cast(Any, None),
                    include_system_information=False,
                )


if __name__ == "__main__":
    unittest.main()
