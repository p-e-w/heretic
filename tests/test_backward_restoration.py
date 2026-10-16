# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import copy
import unittest
from typing import cast
from unittest.mock import patch

import torch
from model_fixtures import PROMPTS, TinyModel
from peft import LoraConfig
from peft.tuners.lora.layer import Linear
from pydantic import ValidationError

from heretic.modifier import Context
from heretic.modifiers.backward_restoration import (
    BackwardRestoration,
    Parameters,
    Settings,
    collect_frozen_projection_io,
    fit_projection,
    flatten_io,
    restoration_loss,
)


class BackwardRestorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_population_weight_is_independent_of_dataset_size(self):
        targets = torch.zeros(3, 2)
        prediction = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        scale = torch.tensor(2.0)
        loss = restoration_loss(prediction, targets, 1, scale, 4.0)
        repeated = torch.cat([prediction[:1], prediction[1:].repeat(5, 1)])
        actual = restoration_loss(repeated, torch.zeros_like(repeated), 1, scale, 4.0)
        self.assertEqual(loss.item(), 13.5)
        torch.testing.assert_close(actual, loss)

    def test_fit_restores_good_and_preserves_retention_with_bias(self):
        base = torch.nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            base.weight.zero_()
            base.bias.fill_(0.5)
        module = Linear(base, "default", LoraConfig(), r=2, lora_alpha=2)
        inputs = torch.eye(3)
        targets = torch.tensor([[2.0, -1.0], [0.5, 0.5], [0.5, 0.5]])
        fit_projection(
            module,
            inputs,
            targets,
            1,
            torch.tensor(1.0),
            4.0,
            Settings(original_model="original", n_optimization_steps=3, max_iter=25),
            torch.Generator().manual_seed(42),
            "test",
        )
        torch.testing.assert_close(module(inputs), targets, atol=1e-3, rtol=0)
        self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_incompatible_teachers_find_weighted_compromise(self):
        base = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(base.weight)
        module = Linear(base, "default", LoraConfig(), r=1, lora_alpha=1)
        fit_projection(
            module,
            torch.ones(2, 1),
            torch.tensor([[1.0], [0.0]]),
            1,
            torch.tensor(1.0),
            4.0,
            Settings(
                original_model="original",
                update_penalty=0,
                n_optimization_steps=3,
                max_iter=25,
            ),
            torch.Generator().manual_seed(42),
            "test",
        )
        torch.testing.assert_close(
            module(torch.ones(1, 1)), torch.tensor([[0.2]]), atol=1e-5, rtol=0
        )

    def test_nonfinite_fit_clears_gradients(self):
        module = Linear(
            torch.nn.Linear(2, 2), "default", LoraConfig(), r=1, lora_alpha=1
        )
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite restoration loss"):
            fit_projection(
                module,
                torch.ones(2, 2),
                torch.full((2, 2), float("nan")),
                1,
                torch.tensor(1.0),
                4.0,
                Settings(original_model="original"),
                torch.Generator().manual_seed(42),
                "test",
            )
        self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_identical_bfloat16_teachers_need_no_correction(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                model = TinyModel()
                cast(torch.nn.Module, model.model).to(torch.bfloat16)
                model.dtype = torch.bfloat16
                original = TinyModel()
                cast(torch.nn.Module, original.model).to(torch.bfloat16)
                original.dtype = torch.bfloat16
                before = model.get_logits_batched(PROMPTS)
                native = flatten_io(original.get_module_io_batched(PROMPTS[:2]))
                teacher = collect_frozen_projection_io(original, PROMPTS[:2])
                self.assertTrue(
                    any(
                        not torch.equal(outputs, native[key][1].float())
                        for key, (_, outputs) in teacher.items()
                    )
                )
                self.assertTrue(
                    all(
                        outputs.dtype == torch.float32
                        for _, outputs in teacher.values()
                    )
                )
                ctx = Context(model.settings, model)
                modifier = BackwardRestoration(
                    heretic_settings=model.settings,
                    settings=Settings(
                        original_model="same-original",
                        streaming=streaming,
                        lora_rank=4,
                        n_optimization_steps=2,
                        max_iter=6,
                    ),
                )
                with (
                    patch.object(
                        ctx, "load_prompts", side_effect=[PROMPTS[:2], PROMPTS[2:]]
                    ),
                    patch.object(model, "reference_model") as reference,
                ):
                    reference.return_value.__enter__.return_value = original
                    modifier.init(ctx)
                modifier.modify_model(ctx, Parameters(0, 3, 4.0))
                for module in model.model.modules():
                    if isinstance(module, Linear):
                        update = module.get_delta_weight("default")
                        torch.testing.assert_close(
                            update, torch.zeros_like(update), atol=1e-7, rtol=0
                        )
                torch.testing.assert_close(
                    model.get_logits_batched(PROMPTS), before, atol=0, rtol=0
                )
                model.get_merged_model()
                torch.testing.assert_close(
                    model.get_logits_batched(PROMPTS), before, atol=0, rtol=0
                )

    def test_frozen_targets_reject_unmerged_teachers(self):
        model = TinyModel()
        model.apply_lora(2)
        with self.assertRaisesRegex(ValueError, "merged teacher checkpoints"):
            collect_frozen_projection_io(model, PROMPTS)

    def test_teachers_stay_frozen_and_trials_repeat_in_both_modes(self):
        original = TinyModel()
        original_outputs = {
            key: pair[1]
            for key, pair in flatten_io(
                original.get_module_io_batched(PROMPTS[:2])
            ).items()
        }
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                model = TinyModel()
                with torch.no_grad():
                    model.model.get_parameter(
                        "model.layers.0.self_attn.o_proj.weight"
                    ).add_(0.05)
                frozen = flatten_io(model.get_module_io_batched(PROMPTS))
                ctx = Context(model.settings, model)
                modifier = BackwardRestoration(
                    heretic_settings=model.settings,
                    settings=Settings(
                        original_model="original",
                        streaming=streaming,
                        lora_rank=4,
                        n_optimization_steps=2,
                        max_iter=6,
                    ),
                )
                with patch.object(
                    ctx, "load_prompts", side_effect=[PROMPTS[:2], PROMPTS[2:]]
                ):
                    with patch.object(
                        modifier,
                        "_collect_original_outputs",
                        return_value=original_outputs,
                    ):
                        modifier.init(ctx)
                targets = copy.deepcopy(modifier.targets)
                scales = copy.deepcopy(modifier.scales)
                for key, target in targets.items():
                    torch.testing.assert_close(target[:2], original_outputs[key])
                    torch.testing.assert_close(target[2:], frozen[key][1][2:])
                parameters = Parameters(0, 3, 4.0)
                calls = []

                def fit(*args):
                    label = args[-1]
                    layer, component, index = label.split("/")
                    key = (int(layer), component, int(index))
                    expected = (
                        flatten_io(model.get_module_io_batched(PROMPTS))[key][0]
                        if streaming
                        else frozen[key][0]
                    )
                    torch.testing.assert_close(args[1], expected)
                    calls.append(key)
                    fit_projection(*args)

                with patch(
                    "heretic.modifiers.backward_restoration.fit_projection",
                    side_effect=fit,
                ):
                    modifier.modify_model(ctx, parameters)
                self.assertEqual(len(calls), 6)
                trained = copy.deepcopy(model.model.state_dict())
                for key in targets:
                    torch.testing.assert_close(
                        modifier.targets[key], targets[key], atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        modifier.scales[key], scales[key], atol=0, rtol=0
                    )
                modifier.reset_model(ctx)
                torch.manual_seed(999)
                modifier.modify_model(ctx, Parameters.from_dict(parameters.to_dict()))
                for key, value in trained.items():
                    torch.testing.assert_close(
                        model.model.state_dict()[key], value, atol=0, rtol=0
                    )

    def test_settings_and_reproduction_requirements(self):
        for kwargs in [
            {"lora_rank": 0},
            {"learning_rate": 0},
            {"update_penalty": -1},
            {"learning_rate": float("inf")},
        ]:
            with self.assertRaises(ValidationError):
                Settings.model_validate({"original_model": "original", **kwargs})
        model = TinyModel()
        modifier = BackwardRestoration(
            heretic_settings=model.settings,
            settings=Settings(original_model="org/original"),
        )
        self.assertFalse(modifier.reproducible)
        modifier.settings.original_model_commit = "a" * 40
        self.assertTrue(modifier.reproducible)
        self.assertIs(modifier.parameters_class, Parameters)
        modifier.validate_contract()
        for params in [
            Parameters(0, 0, 4),
            Parameters(0, 4, 4),
            Parameters(0, 1, float("nan")),
        ]:
            with self.assertRaises(ValueError):
                modifier.modify_model(Context(model.settings, model), params)

    def test_mismatched_teacher_layout_is_rejected_before_fitting(self):
        model = TinyModel()
        ctx = Context(model.settings, model)
        modifier = BackwardRestoration(
            heretic_settings=model.settings,
            settings=Settings(original_model="original"),
        )
        with patch.object(ctx, "load_prompts", side_effect=[PROMPTS[:2], PROMPTS[2:]]):
            with patch.object(modifier, "_collect_original_outputs", return_value={}):
                with self.assertRaisesRegex(ValueError, "matching projection layouts"):
                    modifier.init(ctx)

    def test_existing_adapter_is_not_overwritten(self):
        model = TinyModel()
        model.apply_lora(2)
        modifier = BackwardRestoration(
            heretic_settings=model.settings,
            settings=Settings(original_model="original"),
        )
        before = copy.deepcopy(model.model.state_dict())
        with self.assertRaisesRegex(ValueError, "merged edited checkpoint"):
            modifier.init(Context(model.settings, model))
        for key, value in before.items():
            torch.testing.assert_close(
                model.model.state_dict()[key], value, atol=0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
