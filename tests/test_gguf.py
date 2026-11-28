# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from transformers import GgufConfig

from heretic.config import QuantizationMethod, Settings
from heretic.model import Model, _only_gguf_file


class OnlyGgufFileTests(unittest.TestCase):
    def test_picks_the_single_model_file_and_skips_mmproj(self) -> None:
        # google/gemma-4-26B-A4B-it-qat-q4_0-gguf (https://github.com/p-e-w/heretic/issues/533)
        self.assertEqual(
            _only_gguf_file(
                [
                    ".gitattributes",
                    "README.md",
                    "gemma-4-26B-it-mmproj.gguf",
                    "gemma-4-26B_q4_0-it.gguf",
                ]
            ),
            "gemma-4-26B_q4_0-it.gguf",
        )

    def test_uses_the_only_gguf_file(self) -> None:
        self.assertEqual(
            _only_gguf_file(["README.md", "mmproj.gguf"]),
            "mmproj.gguf",
        )

    def test_leaves_safetensors_and_pytorch_checkpoints_alone(self) -> None:
        self.assertIsNone(_only_gguf_file(["model.safetensors", "model-q4_k_m.gguf"]))
        self.assertIsNone(
            _only_gguf_file(
                [
                    "model.safetensors.index.json",
                    "model-00001-of-00002.safetensors",
                    "extra.gguf",
                ]
            )
        )
        self.assertIsNone(
            _only_gguf_file(
                ["pytorch_model.bin", "pytorch_model.bin.index.json", "model.gguf"]
            )
        )

    def test_does_not_rank_several_model_files(self) -> None:
        self.assertIsNone(
            _only_gguf_file(
                [
                    "weights-Q8_0.gguf",
                    "weights-Q4_K_M.gguf",
                    "weights-mmproj-f16.gguf",
                ]
            )
        )


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

        message = "\n".join(messages)
        self.assertIn(gguf_file, message)
        self.assertIn("dense reconstruction", message)
        self.assertIn("not the original full-precision", message)
        self.assertIn("safetensors", message)
        if files is None and list_side_effect is None:
            listing.assert_not_called()
        return message

    def test_hub_repo_passes_the_single_model_file(self) -> None:
        model = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
        filename = "gemma-4-26B_q4_0-it.gguf"
        self._assert_dequantized_load(
            self._settings(model),
            model_id=model,
            gguf_file=filename,
            files=[
                ".gitattributes",
                "README.md",
                "gemma-4-26B-it-mmproj.gguf",
                filename,
            ],
        )

    def test_passes_model_commit_when_listing_and_loading(self) -> None:
        model = "org/example"
        commit = "abc123def"
        tokenizer_calls, model_calls, _messages, listing = self._load_until_model(
            self._settings(model, model_commit=commit),
            files=["model.gguf"],
        )
        self.assertEqual(tokenizer_calls[0][1]["gguf_file"], "model.gguf")
        self.assertEqual(tokenizer_calls[0][1]["revision"], commit)
        self.assertEqual(model_calls[0][1]["revision"], commit)
        config = model_calls[0][1]["quantization_config"]
        self.assertIsInstance(config, GgufConfig)
        self.assertTrue(config.dequantize)
        listing.assert_called_once_with(model, revision=commit, repo_type="model")

    def test_local_gguf_file_is_passed_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "weights.gguf"
            path.write_bytes(b"not-inspected")
            # An explicit file wins even when safetensors sit beside it.
            (root / "model.safetensors").write_bytes(b"st")
            resolved = path.resolve()
            self._assert_dequantized_load(
                self._settings(str(path)),
                model_id=str(resolved.parent),
                gguf_file=str(resolved),
            )

    def test_local_directory_with_one_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("gguf only\n", encoding="utf-8")
            (root / "mmproj.gguf").write_bytes(b"projector")
            model_file = root / "model.gguf"
            model_file.write_bytes(b"weights")
            resolved_root = root.resolve()
            self._assert_dequantized_load(
                self._settings(str(root)),
                model_id=str(resolved_root),
                gguf_file=str(model_file.resolve()),
            )

    def test_bitsandbytes_is_not_combined_with_gguf(self) -> None:
        message = self._assert_dequantized_load(
            self._settings("org/quant", quantization=QuantizationMethod.BNB_4BIT),
            model_id="org/quant",
            gguf_file="model.gguf",
            files=["model.gguf"],
        )
        self.assertIn("bitsandbytes", message)

    def test_safetensors_repo_that_also_ships_gguf_is_left_alone(self) -> None:
        tokenizer_calls, model_calls, messages, _listing = self._load_until_model(
            self._settings("org/both"),
            files=["config.json", "model.safetensors", "model-q4_k_m.gguf"],
        )
        self.assertNotIn("gguf_file", tokenizer_calls[0][1])
        self.assertNotIn("gguf_file", model_calls[0][1])
        self.assertNotIn("quantization_config", model_calls[0][1])
        self.assertNotIn("dense reconstruction", "\n".join(messages))

    def test_several_gguf_files_are_not_guessed(self) -> None:
        tokenizer_calls, model_calls, messages, listing = self._load_until_model(
            self._settings("org/several-quants"),
            files=["weights-Q8_0.gguf", "weights-Q4_K_M.gguf"],
        )
        listing.assert_called_once()
        self.assertNotIn("gguf_file", tokenizer_calls[0][1])
        self.assertNotIn("gguf_file", model_calls[0][1])
        self.assertNotIn("dense reconstruction", "\n".join(messages))

    def test_listing_failure_does_not_guess_from_the_repo_id(self) -> None:
        tokenizer_calls, model_calls, messages, listing = self._load_until_model(
            self._settings("org/model-q4_0-gguf"),
            list_side_effect=OSError("hub unavailable"),
        )
        listing.assert_called()
        self.assertNotIn("gguf_file", tokenizer_calls[0][1])
        self.assertNotIn("gguf_file", model_calls[0][1])
        self.assertNotIn("dense reconstruction", "\n".join(messages))

    def test_evaluate_model_does_not_change_the_initial_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluate_model = Path(tmp) / "gemma-q4_0.gguf"
            evaluate_model.write_bytes(b"GGUF")
            base = "google/gemma-2-2b-it"
            tokenizer_calls, model_calls, _messages, _listing = self._load_until_model(
                self._settings(base, evaluate_model=str(evaluate_model)),
                files=["config.json", "model.safetensors"],
            )
            self.assertEqual(tokenizer_calls[0][0][0], base)
            self.assertNotIn("gguf_file", tokenizer_calls[0][1])
            self.assertNotIn("gguf_file", model_calls[0][1])

    def test_reset_model_dequantizes_a_swapped_in_gguf_file(self) -> None:
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
            model.dtype = torch.float16
            model.max_memory = None
            model.trusted_models = set()
            model.model = MagicMock()
            model.model.config.name_or_path = base

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


if __name__ == "__main__":
    unittest.main()
