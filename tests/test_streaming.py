# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import copy
import unittest
from typing import cast
from unittest.mock import patch

import torch
from model_fixtures import PROMPTS, TinyModel
from peft.tuners.lora.layer import Linear

from heretic.modifier import Context
from heretic.modifiers.ara import ARA, Parameters, Settings
from heretic.streaming import LayerStream


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_replay_matches_native_forward_after_each_edit(self):
        for hybrid, multimodal in [(False, False), (True, False), (True, True)]:
            with self.subTest(hybrid=hybrid, multimodal=multimodal):
                model = TinyModel(hybrid=hybrid, multimodal=multimodal)
                model.apply_lora(2)
                stream = LayerStream(model, PROMPTS)
                for layer_index in range(len(model.get_layers())):
                    for component, modules in model.get_layer_modules(
                        layer_index
                    ).items():
                        module = cast(Linear, modules[0])
                        expected = model.get_module_io_batched(PROMPTS)[layer_index][
                            component
                        ][0]
                        actual = stream.collect(module)
                        for a, e in zip(actual, expected):
                            torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)
                        # Change all token positions before collecting the next projection.
                        with torch.no_grad():
                            cast(torch.Tensor, module.lora_A["default"].weight).fill_(
                                0.1
                            )
                            cast(torch.Tensor, module.lora_B["default"].weight).fill_(
                                0.3
                            )
                    stream.advance()
                self.assertTrue(
                    all(batch.hidden.device.type == "cpu" for batch in stream.batches)
                )

    def test_hooks_are_removed_when_capture_or_collection_fails(self):
        model = TinyModel()
        layers = model.get_layers()
        with patch.object(
            model, "generate", side_effect=RuntimeError("capture failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                LayerStream(model, PROMPTS)
        self.assertTrue(all(not layer._forward_pre_hooks for layer in layers))

        stream = LayerStream(model, PROMPTS)
        module = model.get_layer_modules(0)["attn.o_proj"][0]
        with patch.object(stream, "_run", side_effect=RuntimeError("replay failed")):
            with self.assertRaisesRegex(RuntimeError, "replay failed"):
                stream.collect(module)
        self.assertFalse(module._forward_hooks)

        with patch.object(model, "generate", side_effect=RuntimeError("native failed")):
            with self.assertRaisesRegex(RuntimeError, "native failed"):
                model.get_module_io(PROMPTS)
        self.assertTrue(
            all(not module._forward_hooks for module in model.model.modules())
        )

    def test_empty_prompts_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one prompt"):
            LayerStream(TinyModel(), [])

    def test_ara_streaming_refreshes_inputs_and_repeats_after_reset(self):
        model = TinyModel()
        ctx = Context(model.settings, model)
        modifier = ARA(
            heretic_settings=model.settings,
            settings=Settings(
                streaming=True, lora_rank=2, n_optimization_steps=1, max_iter=3
            ),
        )
        with patch.object(ctx, "load_prompts", side_effect=[PROMPTS[:2], PROMPTS[2:]]):
            modifier.init(ctx)
        parameters = Parameters(0, 3, 0.5, 0.1, 0.2, 1)
        baseline = copy.deepcopy(model.model.state_dict())
        original_modify = modifier._modify_module
        seen = []

        def modify(module, good_io, bad_io, parameters, generator, label):
            layer, component, index = label.split("/")
            expected = model.get_module_io_batched(PROMPTS)[int(layer)][component][
                int(index)
            ]
            torch.testing.assert_close(torch.cat([good_io[0], bad_io[0]]), expected[0])
            seen.append(label)
            original_modify(module, good_io, bad_io, parameters, generator, label)

        with patch.object(modifier, "_modify_module", side_effect=modify):
            modifier.modify_model(ctx, parameters)
        self.assertEqual(len(seen), 6)
        first = copy.deepcopy(model.model.state_dict())
        self.assertTrue(any(not torch.equal(first[k], baseline[k]) for k in first))
        modifier.reset_model(ctx)
        for key, value in baseline.items():
            torch.testing.assert_close(
                model.model.state_dict()[key], value, atol=0, rtol=0
            )
        torch.manual_seed(999)
        modifier.modify_model(ctx, parameters)
        for key, value in first.items():
            torch.testing.assert_close(
                model.model.state_dict()[key], value, atol=0, rtol=0
            )
        self.assertTrue(all(p.grad is None for p in model.model.parameters()))


if __name__ == "__main__":
    unittest.main()
