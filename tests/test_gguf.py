# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.utils import OfflineModeIsEnabled

from heretic.config import Settings
from heretic.model import Model, _is_gguf_checkpoint

_GEMMA_GGUF_FILES = [
    ".gitattributes",
    "README.md",
    "gemma-4-26B-it-mmproj.gguf",
    "gemma-4-26B_q4_0-it.gguf",
]


class GGUFRejectionTests(unittest.TestCase):
    def _settings(
        self,
        model: str,
        *,
        evaluate_model: str | None = None,
        model_commit: str | None = None,
    ) -> Settings:
        return Settings.model_construct(
            model=model,
            evaluate_model=evaluate_model,
            model_commit=model_commit,
        )

    def _assert_rejected(
        self,
        model: str,
        *,
        evaluate_model: str | None = None,
        model_commit: str | None = None,
        files: list[str] | None = None,
    ) -> str:
        settings = self._settings(
            model,
            evaluate_model=evaluate_model,
            model_commit=model_commit,
        )

        with (
            patch(
                "heretic.model.list_repo_files",
                return_value=files,
            ) as listing,
            patch(
                "heretic.model.AutoTokenizer.from_pretrained",
                side_effect=AssertionError("tokenizer should not be loaded"),
            ) as tokenizer,
            self.assertRaises(SystemExit) as caught,
        ):
            Model(settings)

        tokenizer.assert_not_called()
        message = str(caught.exception)
        self.assertIn("GGUF checkpoint", message)
        self.assertIn("safetensors", message)
        self.assertIn("gguf_file", message)
        self.assertIn("sentencepiece", message)
        if files is None:
            listing.assert_not_called()
        else:
            listing.assert_called()
        return message

    def test_rejects_hub_gguf_repo_from_file_listing(self) -> None:
        # Repro from https://github.com/p-e-w/heretic/issues/533
        # The id is not what is matched. The repo listing is: two .gguf files,
        # no config.json and no safetensors.
        model = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
        message = self._assert_rejected(model, files=_GEMMA_GGUF_FILES)
        self.assertIn(model, message)
        self.assertIn("model '", message)

    def test_rejects_hub_gguf_repo_without_gguf_in_the_id(self) -> None:
        model = "someone/example-model-q4-k-m"
        message = self._assert_rejected(
            model,
            files=["README.md", "example-model-q4_k_m.gguf"],
        )
        self.assertIn(model, message)

    def test_passes_model_commit_when_listing_the_hub_repo(self) -> None:
        model = "org/example-gguf"
        commit = "abc123def"
        with (
            patch(
                "heretic.model.list_repo_files",
                return_value=["model.gguf"],
            ) as listing,
            patch(
                "heretic.model.AutoTokenizer.from_pretrained",
                side_effect=AssertionError("tokenizer should not be loaded"),
            ),
            self.assertRaises(SystemExit),
        ):
            Model(self._settings(model, model_commit=commit))

        listing.assert_called_once_with(model, revision=commit, repo_type="model")

    def test_rejects_local_file_with_gguf_magic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # The name is not a .gguf suffix. The header is.
            path = Path(tmp) / "weights.bin"
            path.write_bytes(b"GGUF" + b"\x00" * 8)
            message = self._assert_rejected(str(path))
            self.assertIn(str(path), message)

    def test_rejects_local_directory_of_gguf_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("gguf only\n", encoding="utf-8")
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "model.gguf").write_bytes(b"GGUF-not-a-real-file")
            message = self._assert_rejected(str(root))
            self.assertIn(str(root), message)

    def test_rejects_gguf_evaluate_model_before_tokenizer_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluate_model = Path(tmp) / "gemma-q4_0.gguf"
            evaluate_model.write_bytes(b"GGUF")
            message = self._assert_rejected(
                "google/gemma-2-2b-it",
                evaluate_model=str(evaluate_model),
                files=["config.json", "model.safetensors", "tokenizer.json"],
            )
            self.assertIn("evaluation model", message)
            self.assertIn(str(evaluate_model), message)
            self.assertNotIn("google/gemma-2-2b-it", message)

    def test_safetensors_repo_that_also_ships_gguf_is_not_rejected(self) -> None:
        model = "org/Model_GGUF"
        files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "model-q4_k_m.gguf",
        ]
        with (
            patch("heretic.model.list_repo_files", return_value=files),
            patch(
                "heretic.model.AutoTokenizer.from_pretrained",
                side_effect=AssertionError("tokenizer-was-called"),
            ),
            self.assertRaisesRegex(AssertionError, "tokenizer-was-called"),
        ):
            Model(self._settings(model))

    def test_listing_failure_does_not_guess_from_the_repo_id(self) -> None:
        model = "org/model-q4_0-gguf"
        with (
            patch(
                "heretic.model.list_repo_files",
                side_effect=OSError("hub unavailable"),
            ),
            patch(
                "heretic.model.AutoTokenizer.from_pretrained",
                side_effect=AssertionError("tokenizer-was-called"),
            ),
            self.assertRaisesRegex(AssertionError, "tokenizer-was-called"),
        ):
            Model(self._settings(model))

    def test_offline_cache_listing_detects_gguf(self) -> None:
        repo = "org/cached-quant"
        commit = "abc123"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models--org--cached-quant"
            snapshot = root / "snapshots" / commit
            snapshot.mkdir(parents=True)
            (snapshot / "weights.gguf").write_bytes(b"placeholder")
            (root / "refs").mkdir()
            (root / "refs" / "main").write_text(commit, encoding="utf-8")

            with (
                patch.dict(os.environ, {"HF_HUB_CACHE": tmp}),
                patch(
                    "heretic.model.list_repo_files",
                    side_effect=OfflineModeIsEnabled("offline"),
                ),
            ):
                self.assertTrue(_is_gguf_checkpoint(repo))

    def test_offline_cache_with_safetensors_is_not_gguf_only(self) -> None:
        repo = "org/cached-full"
        commit = "abc123"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models--org--cached-full"
            snapshot = root / "snapshots" / commit
            snapshot.mkdir(parents=True)
            (snapshot / "model.safetensors").write_bytes(b"not-gguf")
            (snapshot / "extra.gguf").write_bytes(b"GGUF")
            (root / "refs").mkdir()
            (root / "refs" / "main").write_text(commit, encoding="utf-8")

            with (
                patch.dict(os.environ, {"HF_HUB_CACHE": tmp}),
                patch(
                    "heretic.model.list_repo_files",
                    side_effect=OfflineModeIsEnabled("offline"),
                ),
            ):
                self.assertFalse(_is_gguf_checkpoint(repo))

    def test_local_non_gguf_files_are_not_treated_as_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "model.gguf"
            fake.write_bytes(b"this is not a gguf file")
            safetensors = root / "model.safetensors"
            safetensors.write_bytes(b"not-a-real-safetensors")
            mixed = root / "mixed"
            mixed.mkdir()
            (mixed / "model.safetensors").write_bytes(b"st")
            (mixed / "also.gguf").write_bytes(b"GGUF-real-enough")

            self.assertFalse(_is_gguf_checkpoint(str(fake)))
            self.assertFalse(_is_gguf_checkpoint(str(safetensors)))
            self.assertFalse(_is_gguf_checkpoint(str(mixed)))

    def test_unreadable_id_is_not_treated_as_gguf(self) -> None:
        # Not a local file, and not something the Hub listing can resolve.
        # Do not guess from the name.
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HF_HUB_CACHE": tmp}):
                self.assertFalse(_is_gguf_checkpoint(r"C:\models\bar.gguf"))
                with patch(
                    "heretic.model.list_repo_files",
                    side_effect=OSError("hub unavailable"),
                ) as listing:
                    self.assertFalse(_is_gguf_checkpoint("org/model.gguf.txt"))
                listing.assert_called_once()


if __name__ == "__main__":
    unittest.main()
