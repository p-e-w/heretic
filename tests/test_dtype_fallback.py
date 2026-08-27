# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import torch

from heretic.config import QuantizationMethod, Settings
from heretic.model import Model, classify_dtype_load_error


def make_settings(dtypes: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        dtypes=dtypes,
        model="test-model",
        model_commit=None,
        max_memory=None,
        device_map=None,
        system_prompt="",
        quantization=QuantizationMethod.NONE,
    )


class DtypeFallbackTests(unittest.TestCase):
    def test_classifies_non_retryable_and_retryable_errors(self) -> None:
        self.assertEqual(
            classify_dtype_load_error(ModuleNotFoundError("mamba_ssm")),
            "non-retryable",
        )
        self.assertEqual(
            classify_dtype_load_error(
                ValueError("Unrecognized configuration class LlavaConfig")
            ),
            "non-retryable",
        )
        self.assertEqual(
            classify_dtype_load_error(
                AttributeError("module 'torch' has no attribute 'fp8'")
            ),
            "non-retryable",
        )
        self.assertEqual(
            classify_dtype_load_error(RuntimeError("CUDA out of memory")),
            "retryable",
        )

    def test_stops_after_non_retryable_error(self) -> None:
        tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos")
        model_class = Mock()
        model_class.from_pretrained.side_effect = [
            ModuleNotFoundError("No module named 'mamba_ssm'"),
            SimpleNamespace(dtype=torch.float16),
        ]

        with (
            patch(
                "heretic.model.AutoTokenizer.from_pretrained", return_value=tokenizer
            ),
            patch("heretic.model.get_model_class", return_value=model_class),
            patch("heretic.model.empty_cache") as empty_cache,
        ):
            with self.assertRaisesRegex(Exception, r"auto \[non-retryable\]"):
                Model(cast(Settings, make_settings(["auto", "float16"])))

        model_class.from_pretrained.assert_called_once()
        empty_cache.assert_called_once()

    def test_retries_after_retryable_error(self) -> None:
        tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos")
        model_class = Mock()
        loaded_model = SimpleNamespace(dtype=torch.float16)
        model_class.from_pretrained.side_effect = [
            RuntimeError("CUDA out of memory"),
            loaded_model,
        ]

        with (
            patch(
                "heretic.model.AutoTokenizer.from_pretrained", return_value=tokenizer
            ),
            patch("heretic.model.get_model_class", return_value=model_class),
            patch("heretic.model.empty_cache") as empty_cache,
            patch.object(Model, "generate"),
            patch.object(Model, "_apply_lora"),
            patch.object(Model, "get_layers", return_value=[]),
        ):
            model = Model(cast(Settings, make_settings(["auto", "float16"])))

        self.assertIs(model.model, loaded_model)
        self.assertEqual(empty_cache.call_count, 1)
        self.assertEqual(
            [
                call.kwargs["dtype"]
                for call in model_class.from_pretrained.call_args_list
            ],
            ["auto", "float16"],
        )


if __name__ == "__main__":
    unittest.main()
