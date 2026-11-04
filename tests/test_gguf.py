# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from unittest.mock import patch

from heretic.config import Settings
from heretic.model import Model, _looks_like_gguf


class GGUFRejectionTests(unittest.TestCase):
    def _assert_rejected(self, model: str, *, evaluate_model: str | None = None) -> str:
        settings = Settings.model_construct(
            model=model,
            evaluate_model=evaluate_model,
        )

        with (
            patch(
                "heretic.model.AutoTokenizer.from_pretrained",
                side_effect=AssertionError("tokenizer should not be loaded"),
            ) as tokenizer,
            self.assertRaises(SystemExit) as caught,
        ):
            Model(settings)

        tokenizer.assert_not_called()
        message = str(caught.exception)
        self.assertIn("GGUF models are not supported yet", message)
        self.assertIn("Safetensors", message)
        self.assertIn("Hugging Face", message)
        return message

    def test_rejects_gguf_model_id_before_tokenizer_load(self) -> None:
        # Repro from https://github.com/p-e-w/heretic/issues/533
        model = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
        message = self._assert_rejected(model)
        self.assertIn(model, message)
        self.assertIn("model '", message)

    def test_rejects_gguf_paths_and_markers(self) -> None:
        models = [
            "org/Model_GGUF",
            "weights/foo.gguf",
            "weights/FOO.GGUF",
            r"C:\models\bar.gguf",
            "dir/model.gguf/",
        ]
        for model in models:
            with self.subTest(model=model):
                message = self._assert_rejected(model)
                self.assertIn(model, message)

    def test_rejects_gguf_evaluate_model_before_tokenizer_load(self) -> None:
        evaluate_model = "/models/gemma-q4_0.gguf"
        message = self._assert_rejected(
            "google/gemma-2-2b-it",
            evaluate_model=evaluate_model,
        )
        self.assertIn("evaluation model", message)
        self.assertIn(evaluate_model, message)
        self.assertNotIn("google/gemma-2-2b-it", message)

    def test_normal_model_ids_are_not_treated_as_gguf(self) -> None:
        models = [
            "google/gemma-2-2b-it",
            "meta-llama/Llama-3.1-8B-Instruct",
            "/tmp/model.safetensors",
            "org/gguf",
            "org/model.gguf.txt",
        ]
        for model in models:
            with self.subTest(model=model):
                self.assertFalse(_looks_like_gguf(model))


if __name__ == "__main__":
    unittest.main()
