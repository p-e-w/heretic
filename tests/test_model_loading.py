# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from heretic.config import QuantizationMethod
from heretic.model import Model


def make_model_for_loading_test(dtypes: list[str]) -> Model:
    model = Model.__new__(Model)
    model.settings = SimpleNamespace(
        dtypes=dtypes,
        model="test-model",
        quantization=QuantizationMethod.NONE,
        device_map=None,
        system_prompt="",
    )
    model.max_memory = None
    model.revision_kwargs = {}
    model.trusted_models = set()
    model.model = None
    model._get_quantization_config = Mock(return_value=None)
    model.generate = Mock()
    return model


class ModelLoadingTests(unittest.TestCase):
    def test_reports_each_dtype_failure_and_clears_cache(self) -> None:
        model = make_model_for_loading_test(["auto", "float16"])
        model_class = Mock()
        model_class.from_pretrained.side_effect = [
            ValueError("auto is unsupported"),
            RuntimeError("float16 ran out of memory"),
        ]

        with (
            patch("heretic.model.get_model_class", return_value=model_class),
            patch("heretic.model.empty_cache") as empty_cache,
        ):
            with self.assertRaises(Exception) as raised:
                model._load_model_with_dtype_fallback()

        message = str(raised.exception)
        self.assertIn("auto: auto is unsupported", message)
        self.assertIn("float16: float16 ran out of memory", message)
        self.assertEqual(empty_cache.call_count, 2)
        self.assertIsNone(model.model)

    def test_continues_to_next_dtype_after_failure(self) -> None:
        model = make_model_for_loading_test(["auto", "float16"])
        model_class = Mock()
        loaded_model = SimpleNamespace(dtype=torch.float16)
        model_class.from_pretrained.side_effect = [
            ValueError("auto is unsupported"),
            loaded_model,
        ]

        with (
            patch("heretic.model.get_model_class", return_value=model_class),
            patch("heretic.model.empty_cache") as empty_cache,
        ):
            model._load_model_with_dtype_fallback()

        self.assertIs(model.model, loaded_model)
        self.assertEqual(model.dtype, torch.float16)
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
