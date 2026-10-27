# SPDX-License-Identifier: AGPL-3.0-or-later
"""Additional output projection configuration for custom decoder layers."""

import unittest
from unittest.mock import patch

from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from pydantic import ValidationError
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from heretic.config import Settings
from heretic.model import Model


class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.o_proj = nn.Linear(8, 4)
        self.feedforward = nn.Module()
        self.feedforward.down_proj = nn.Linear(16, 4)


class StandardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(8, 4)


class TinyConfig(PretrainedConfig):
    model_type = "additional-layer-modules-test"
    hidden_size = 4


class TinyModel(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([StandardLayer()])


def make_model(layers, targets=None):
    model = Model.__new__(Model)
    if targets is None:
        model.settings = Settings.model_construct(model="test/model")
    else:
        model.settings = Settings.model_construct(
            model="test/model", layer_module_targets=targets
        )
    model.model = TinyModel(TinyConfig())
    model.model.model.layers = nn.ModuleList(layers)
    return model


def make_peft_model(targets):
    base_model = TinyModel(TinyConfig())
    peft_model = get_peft_model(
        base_model,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["self_attn.o_proj"],
        ),
    )
    model = Model.__new__(Model)
    model.settings = Settings.model_construct(
        model="test/model", layer_module_targets=targets
    )
    model.model = peft_model
    return model


class LayerModuleTargetsTest(unittest.TestCase):
    def setUp(self) -> None:
        # Settings parses CLI arguments by default; isolate it from unittest flags.
        argv_patch = patch("sys.argv", ["heretic"])
        argv_patch.start()
        self.addCleanup(argv_patch.stop)

    def test_settings_round_trip_targets(self):
        targets = {
            "attn.o_proj": ["attention.o_proj"],
            "mlp.down_proj": ["feedforward.down_proj"],
        }
        settings = Settings.model_validate(
            {
                "model": "test/model",
                "layer_module_targets": targets,
            }
        )

        restored = Settings.model_validate_json(settings.model_dump_json())
        self.assertEqual(
            restored.layer_module_targets,
            targets,
        )

    def test_settings_accept_custom_component(self):
        settings = Settings.model_validate(
            {
                "model": "test/model",
                "layer_module_targets": {
                    "shared_expert": ["mlp.shared_expert.down_proj"]
                },
            }
        )
        self.assertIn("shared_expert", settings.layer_module_targets)

    def test_settings_reject_blank_patterns(self):
        with self.assertRaisesRegex(ValidationError, "non-empty patterns"):
            Settings.model_validate(
                {"model": "test/model", "layer_module_targets": {"custom": [" "]}}
            )

    def test_defaults_preserve_builtin_discovery(self):
        model = make_model([StandardLayer()])

        found = model.get_layer_modules(0)

        self.assertEqual(list(found), ["attn.o_proj"])
        self.assertIs(found["attn.o_proj"][0], model.get_layers()[0].self_attn.o_proj)

    def test_configured_targets_replace_defaults(self):
        layer = StandardLayer()
        layer.self_attn.extra_o_proj = nn.Linear(8, 4)
        model = make_model([layer], {"custom": ["self_attn.extra_o_proj"]})
        found = model.get_layer_modules(0)
        self.assertEqual(list(found), ["custom"])
        self.assertIs(found["custom"][0], layer.self_attn.extra_o_proj)

    def test_empty_component_list_disables_selection(self):
        layer = CustomLayer()
        model = make_model(
            [layer],
            {"attn.o_proj": [], "mlp.down_proj": ["feedforward.down_proj"]},
        )
        self.assertEqual(list(model.get_layer_modules(0)), ["mlp.down_proj"])

    def test_default_moe_paths(self):
        class MoeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.experts = nn.ModuleList(
                    [nn.ModuleDict({"down_proj": nn.Linear(8, 4)}) for _ in range(2)]
                )

        model = make_model([MoeLayer()])
        self.assertEqual(len(model.get_layer_modules(0)["mlp.down_proj"]), 2)

    def test_custom_architecture_paths(self):
        model = make_model(
            [CustomLayer(), CustomLayer()],
            {
                "attn.o_proj": ["attention.o_proj"],
                "mlp.down_proj": ["feedforward.down_proj"],
            },
        )
        for i in range(2):
            found = model.get_layer_modules(i)
            self.assertEqual(set(found), {"attn.o_proj", "mlp.down_proj"})
            self.assertIs(
                found["attn.o_proj"][0], model.get_layers()[i].attention.o_proj
            )
            self.assertIs(
                found["mlp.down_proj"][0], model.get_layers()[i].feedforward.down_proj
            )

    def test_empty_configuration_selects_nothing(self):
        model = make_model([StandardLayer()], {})
        with self.assertRaisesRegex(AssertionError, "No abliterable modules"):
            model.get_layer_modules(0)

    def test_missing_modules_include_diagnostics(self):
        model = make_model([CustomLayer()])
        with self.assertRaisesRegex(AssertionError, r"attention.o_proj \(8->4\)"):
            model.get_layer_modules(0)

    def test_glob_matches_multiple_experts(self):
        class Experts(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.experts = nn.ModuleList(
                    [nn.ModuleDict({"down_proj": nn.Linear(8, 4)}) for _ in range(3)]
                )

        model = make_model([Experts()], {"mlp.down_proj": ["mlp.experts.*.down_proj"]})
        self.assertEqual(len(model.get_layer_modules(0)["mlp.down_proj"]), 3)

    def test_pattern_can_match_only_some_layers(self):
        model = make_model(
            [CustomLayer(), StandardLayer()],
            {
                "attn.o_proj": ["self_attn.o_proj"],
                "mlp.down_proj": ["feedforward.down_proj"],
            },
        )
        self.assertEqual(list(model.get_layer_modules(1)), ["attn.o_proj"])

    def test_unmatched_pattern_fails_on_layer_discovery(self):
        model = make_model([CustomLayer()], {"mlp.down_proj": ["missing.down_proj"]})
        with self.assertRaisesRegex(AssertionError, "No abliterable modules"):
            model.get_layer_modules(0)

    def test_overlapping_patterns_are_deduplicated(self):
        model = make_model(
            [StandardLayer()], {"attn.o_proj": ["self_attn.o_proj", "self_attn.*"]}
        )
        found = model.get_layer_modules(0)
        self.assertEqual(len(found["attn.o_proj"]), 1)

    def test_duplicate_across_components_is_rejected(self):
        model = make_model(
            [StandardLayer()],
            {
                "attn.o_proj": ["self_attn.o_proj"],
                "mlp.down_proj": ["self_attn.o_proj"],
            },
        )
        with self.assertRaisesRegex(ValueError, "already assigned"):
            model.get_layer_modules(0)

    def test_wrapped_projection_resolves(self):
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_layer = nn.Linear(8, 4)

        layer = CustomLayer()
        layer.attention.o_proj = Wrapper()
        model = make_model([layer], {"attn.o_proj": ["attention.o_proj"]})
        self.assertIs(
            model.get_layer_modules(0)["attn.o_proj"][0], layer.attention.o_proj
        )

    def test_peft_wrapper_is_atomic_for_glob_selection(self):
        model = make_peft_model({"attn.o_proj": ["self_attn.*"]})

        found = model.get_layer_modules(0)

        self.assertEqual(len(found["attn.o_proj"]), 1)
        self.assertIsInstance(found["attn.o_proj"][0], BaseTunerLayer)

    def test_peft_internals_cannot_be_selected_directly(self):
        internal_patterns = [
            ("self_attn.o_proj.base_layer", "self_attn.o_proj.base_layer"),
            ("self_attn.o_proj.lora_A.*", "self_attn.o_proj.lora_A.default"),
            ("self_attn.o_proj.lora_B.*", "self_attn.o_proj.lora_B.default"),
        ]

        for pattern, internal_name in internal_patterns:
            with self.subTest(pattern=pattern):
                model = make_peft_model({"attn.o_proj": [pattern]})
                self.assertIn(
                    internal_name,
                    dict(model.get_layers()[0].named_modules()),
                )
                with self.assertRaisesRegex(AssertionError, "No abliterable modules"):
                    model.get_layer_modules(0)

    def test_incompatible_output_size_is_rejected(self):
        layer = CustomLayer()
        layer.feedforward.down_proj = nn.Linear(8, 8)
        model = make_model([layer], {"mlp.down_proj": ["feedforward.down_proj"]})
        with self.assertRaisesRegex(ValueError, "out_features=8.*expected 4"):
            model.get_layer_modules(0)

    def test_broad_glob_rejects_incompatible_projections(self):
        layer = StandardLayer()
        layer.self_attn.q_proj = nn.Linear(4, 8)
        layer.self_attn.k_proj = nn.Linear(4, 2)
        layer.self_attn.v_proj = nn.Linear(4, 2)
        model = make_model([layer], {"attn.o_proj": ["self_attn.*"]})
        with self.assertRaisesRegex(ValueError, "self_attn.q_proj.*out_features=8"):
            model.get_layer_modules(0)

    def test_container_is_rejected(self):
        model = make_model([CustomLayer()], {"attn.o_proj": ["attention"]})
        with self.assertRaisesRegex(ValueError, "2D weight matrix"):
            model.get_layer_modules(0)


if __name__ == "__main__":
    unittest.main()
