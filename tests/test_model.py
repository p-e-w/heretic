# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from unittest.mock import Mock, patch

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear
from transformers import GPT2Config, GPT2LMHeadModel

from heretic.config import QuantizationMethod, Settings
from heretic.model import Model


class ModelMergeTests(unittest.TestCase):
    def test_interrupted_merge_requires_full_reload(self):
        base = GPT2LMHeadModel(
            GPT2Config(
                n_layer=1,
                n_head=1,
                n_embd=8,
                vocab_size=16,
                bos_token_id=0,
                eos_token_id=0,
            )
        )
        adapter = get_peft_model(
            base,
            LoraConfig(
                r=1, lora_alpha=1, target_modules=["c_attn"], fan_in_fan_out=True
            ),
        )
        model = object.__new__(Model)
        model.model = adapter
        model.settings = Mock(spec=Settings, quantization=QuantizationMethod.NONE)
        model.needs_reload = False
        layer = adapter.get_submodule("base_model.model.transformer.h.0.attn.c_attn")
        assert isinstance(layer, Linear)
        with torch.no_grad():
            layer.get_parameter("lora_A.default.weight").fill_(1)
            layer.get_parameter("lora_B.default.weight").fill_(1)
        weight = layer.get_parameter("base_layer.weight")
        before = weight.detach().clone()
        original_merge = layer.merge

        def interrupted_merge(*args, **kwargs):
            original_merge(*args, **kwargs)
            raise KeyboardInterrupt

        with patch.object(layer, "merge", side_effect=interrupted_merge):
            with self.assertRaises(KeyboardInterrupt):
                model.get_merged_model()
        self.assertFalse(torch.equal(before, weight))
        self.assertTrue(model.needs_reload)


if __name__ == "__main__":
    unittest.main()
