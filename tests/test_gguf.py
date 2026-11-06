# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from huggingface_hub.utils import OfflineModeIsEnabled
from transformers import GgufConfig

from heretic.config import QuantizationMethod, Settings
from heretic.model import Model, _is_gguf_checkpoint, _resolve_gguf_checkpoint

_GEMMA_GGUF_FILES = [
    ".gitattributes",
    "README.md",
    "gemma-4-26B-it-mmproj.gguf",
    "gemma-4-26B_q4_0-it.gguf",
]


class GGUFLoadTests(unittest.TestCase):
    def _settings(
        self,
        model: str,
        *,
        evaluate_model: str | None = None,
        model_commit: str | None = None,
        quantization: QuantizationMethod = QuantizationMethod.NONE,
    ) -> Settings:
        return Settings.model_construct(
            model=model,
            evaluate_model=evaluate_model,
            model_commit=model_commit,
            dtypes=["float16"],
            max_memory=None,
            device_map="cpu",
            quantization=quantization,
        )

    def _load_until_model(
        self,
        settings: Settings,
        *,
        files: list[str] | None = None,
        list_side_effect: BaseException | None = None,
    ):
        tokenizer_calls: list[tuple[tuple, dict]] = []
        model_calls: list[tuple[tuple, dict]] = []
        messages: list[str] = []

        def tokenizer_from_pretrained(*args, **kwargs):
            tokenizer_calls.append((args, kwargs))
            tokenizer = MagicMock()
            tokenizer.pad_token = 0
            tokenizer.apply_chat_template.return_value = "HERETIC_SYSTEM_ROLE_PROBE"
            return tokenizer

        class _Loader:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                model_calls.append((args, kwargs))
                raise SystemExit("stop-after-model-load")

        list_kwargs: dict = {}
        if list_side_effect is not None:
            list_kwargs["side_effect"] = list_side_effect
        else:
            list_kwargs["return_value"] = files

        with (
            patch("heretic.model.list_repo_files", **list_kwargs) as listing,
            patch(
                "heretic.model.AutoTokenizer.from_pretrained",
                side_effect=tokenizer_from_pretrained,
            ),
            patch("heretic.model.get_model_class", return_value=_Loader),
            patch(
                "heretic.model.print",
                side_effect=lambda *args, **kwargs: messages.append(
                    " ".join(str(arg) for arg in args)
                ),
            ),
            self.assertRaises(SystemExit),
        ):
            Model(settings)

        return tokenizer_calls, model_calls, messages, listing

    def _assert_dequantized_load(
        self,
        settings: Settings,
        *,
        model_id: str,
        gguf_file: str,
        files: list[str] | None = None,
        list_side_effect: BaseException | None = None,
    ) -> str:
        tokenizer_calls, model_calls, messages, listing = self._load_until_model(
            settings,
            files=files,
            list_side_effect=list_side_effect,
        )
        self.assertEqual(tokenizer_calls[0][0][0], model_id)
        self.assertEqual(tokenizer_calls[0][1]["gguf_file"], gguf_file)
        self.assertNotIn("quantization_config", tokenizer_calls[0][1])

        self.assertEqual(model_calls[0][0][0], model_id)
        self.assertEqual(model_calls[0][1]["gguf_file"], gguf_file)
        config = model_calls[0][1]["quantization_config"]
        self.assertIsInstance(config, GgufConfig)
        self.assertTrue(config.dequantize)

        if settings.model_commit is not None:
            self.assertEqual(tokenizer_calls[0][1]["revision"], settings.model_commit)
            self.assertEqual(model_calls[0][1]["revision"], settings.model_commit)
            if files is not None:
                listing.assert_called_once_with(
                    settings.model,
                    revision=settings.model_commit,
                    repo_type="model",
                )

        message = "\n".join(messages)
        self.assertIn(settings.model, message)
        self.assertIn(gguf_file, message)
        self.assertIn("dense reconstruction", message)
        self.assertIn("not the original full-precision", message)
        self.assertIn("safetensors", message)
        self.assertNotIn("not supported", message)
        if files is None and list_side_effect is None:
            listing.assert_not_called()
        return message

    def test_hub_gguf_repo_passes_model_file_not_mmproj(self) -> None:
        # Repro from https://github.com/p-e-w/heretic/issues/533
        # The id is not what is matched. The repo listing is: two .gguf files,
        # no safetensors. The projector sidecar is not the model.
        model = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
        self._assert_dequantized_load(
            self._settings(model),
            model_id=model,
            gguf_file="gemma-4-26B_q4_0-it.gguf",
            files=_GEMMA_GGUF_FILES,
        )

    def test_hub_gguf_repo_without_gguf_in_the_id(self) -> None:
        model = "someone/example-model-q4-k-m"
        filename = "example-model-q4_k_m.gguf"
        self._assert_dequantized_load(
            self._settings(model),
            model_id=model,
            gguf_file=filename,
            files=["README.md", filename],
        )

    def test_passes_model_commit_when_listing_and_loading(self) -> None:
        model = "org/example"
        commit = "abc123def"
        self._assert_dequantized_load(
            self._settings(model, model_commit=commit),
            model_id=model,
            gguf_file="model.gguf",
            files=["model.gguf"],
        )

    def test_prefers_the_quant_closest_to_full_precision(self) -> None:
        model = "org/several-quants"
        files = [
            "README.md",
            "weights-Q4_K_M.gguf",
            "weights-Q2_K.gguf",
            "weights-Q8_0.gguf",
            "weights-mmproj-f16.gguf",
        ]
        source = None
        with patch("heretic.model.list_repo_files", return_value=files):
            source = _resolve_gguf_checkpoint(model)
        assert source is not None
        self.assertEqual(source.gguf_file, "weights-Q8_0.gguf")
        self.assertEqual(
            source.alternatives,
            ("weights-Q2_K.gguf", "weights-Q4_K_M.gguf"),
        )

        message = self._assert_dequantized_load(
            self._settings(model),
            model_id=model,
            gguf_file="weights-Q8_0.gguf",
            files=files,
        )
        self.assertIn("weights-Q4_K_M.gguf", message)
        self.assertNotIn("mmproj", message)

    def test_local_file_with_gguf_magic_passes_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # The name is not a .gguf suffix. The header is.
            path = Path(tmp) / "weights.bin"
            path.write_bytes(b"GGUF" + b"\x00" * 8)
            resolved = path.resolve()
            self._assert_dequantized_load(
                self._settings(str(path)),
                model_id=str(resolved.parent),
                gguf_file=str(resolved),
            )

    def test_local_directory_of_gguf_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("gguf only\n", encoding="utf-8")
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "model.gguf").write_bytes(b"GGUF-not-a-real-file")
            resolved_root = root.resolve()
            self._assert_dequantized_load(
                self._settings(str(root)),
                model_id=str(resolved_root),
                gguf_file=str((root / "model.gguf").resolve()),
            )

    def test_bitsandbytes_is_not_combined_with_gguf(self) -> None:
        model = "org/quant"
        message = self._assert_dequantized_load(
            self._settings(model, quantization=QuantizationMethod.BNB_4BIT),
            model_id=model,
            gguf_file="model-Q4_0.gguf",
            files=["model-Q4_0.gguf"],
        )
        self.assertIn("bitsandbytes", message)

    def test_safetensors_repo_that_also_ships_gguf_is_left_alone(self) -> None:
        model = "org/Model_GGUF"
        files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "model-q4_k_m.gguf",
        ]
        tokenizer_calls, model_calls, messages, _listing = self._load_until_model(
            self._settings(model),
            files=files,
        )
        self.assertNotIn("gguf_file", tokenizer_calls[0][1])
        self.assertNotIn("gguf_file", model_calls[0][1])
        self.assertNotIn("quantization_config", model_calls[0][1])
        self.assertNotIn("dense reconstruction", "\n".join(messages))
        with patch("heretic.model.list_repo_files", return_value=files):
            self.assertIsNone(_resolve_gguf_checkpoint(model))

    def test_listing_failure_does_not_guess_from_the_repo_id(self) -> None:
        model = "org/model-q4_0-gguf"
        tokenizer_calls, model_calls, messages, listing = self._load_until_model(
            self._settings(model),
            list_side_effect=OSError("hub unavailable"),
        )
        listing.assert_called()
        self.assertNotIn("gguf_file", tokenizer_calls[0][1])
        self.assertNotIn("gguf_file", model_calls[0][1])
        self.assertNotIn("dense reconstruction", "\n".join(messages))

    def test_evaluate_model_gguf_does_not_change_the_initial_load(self) -> None:
        # evaluate_model is swapped in later, by reset_model. The model loaded
        # at startup is still the base checkpoint.
        with tempfile.TemporaryDirectory() as tmp:
            evaluate_model = Path(tmp) / "gemma-q4_0.gguf"
            evaluate_model.write_bytes(b"GGUF")
            base = "google/gemma-2-2b-it"
            tokenizer_calls, model_calls, _messages, _listing = self._load_until_model(
                self._settings(base, evaluate_model=str(evaluate_model)),
                files=["config.json", "model.safetensors", "tokenizer.json"],
            )
            self.assertEqual(tokenizer_calls[0][0][0], base)
            self.assertNotIn("gguf_file", tokenizer_calls[0][1])
            self.assertNotIn("gguf_file", model_calls[0][1])

    def test_reset_model_dequantizes_a_swapped_in_gguf_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluate_model = Path(tmp) / "gemma-q4_0.gguf"
            evaluate_model.write_bytes(b"GGUF")
            resolved = evaluate_model.resolve()
            base = "google/gemma-2-2b-it"
            settings = self._settings(base, evaluate_model=str(evaluate_model))

            model = Model.__new__(Model)
            model.settings = settings
            model.revision_kwargs = {}
            model.needs_reload = False
            model.gguf_file = None
            model.pretrained_model_id = base
            model.loaded_from = base
            model._announced_gguf = set()
            model._gguf_alternatives = ()
            model.dtype = torch.float16
            model.max_memory = None
            model.trusted_models = set()
            model.model = MagicMock()
            model.model.config.name_or_path = base

            # Same swap main() performs before evaluating another checkpoint.
            settings.model = str(evaluate_model)
            calls: list[tuple[tuple, dict]] = []

            class _Loader:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    calls.append((args, kwargs))
                    loaded = MagicMock()
                    loaded.config.name_or_path = args[0]
                    return loaded

            with (
                patch("heretic.model.empty_cache"),
                patch("heretic.model.get_model_class", return_value=_Loader),
                patch("heretic.model.print"),
            ):
                fast = model.reset_model()

            self.assertFalse(fast)
            self.assertEqual(calls[0][0][0], str(resolved.parent))
            self.assertEqual(calls[0][1]["gguf_file"], str(resolved))
            config = calls[0][1]["quantization_config"]
            self.assertIsInstance(config, GgufConfig)
            self.assertTrue(config.dequantize)

    def test_reset_model_fast_path_does_not_reload(self) -> None:
        settings = self._settings("google/gemma-2-2b-it")
        model = Model.__new__(Model)
        model.settings = settings
        model.needs_reload = False
        model.gguf_file = None
        model.pretrained_model_id = settings.model
        model.loaded_from = settings.model
        model.model = MagicMock()
        model.model.config.name_or_path = settings.model
        model.model.named_modules.return_value = []

        with patch("heretic.model.get_model_class") as get_model_class:
            self.assertTrue(model.reset_model())
        get_model_class.assert_not_called()

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
                source = _resolve_gguf_checkpoint(repo)
                assert source is not None
                self.assertEqual(source.gguf_file, "weights.gguf")

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
