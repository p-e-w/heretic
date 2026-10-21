# SPDX-License-Identifier: AGPL-3.0-or-later
"""Additional output projection configuration for custom decoder layers."""

import unittest
from types import SimpleNamespace

from torch import nn

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


def make_model(layers, extra):
    model = Model.__new__(Model)
    model.settings = SimpleNamespace(additional_layer_modules=extra)
    model.model = nn.Module()
    model.model.model = nn.Module()
    model.model.model.layers = nn.ModuleList(layers)
    return model


class AdditionalLayerModulesTest(unittest.TestCase):
    def test_custom_architecture_paths(self):
        model = make_model(
            [CustomLayer(), CustomLayer()],
            {
                "attn.o_proj": ["attention.o_proj"],
                "mlp.down_proj": ["feedforward.down_proj"],
            },
        )
        model._validate_additional_layer_modules()
        for i in range(2):
            found = model.get_layer_modules(i)
            self.assertEqual(set(found), {"attn.o_proj", "mlp.down_proj"})
            self.assertIs(
                found["attn.o_proj"][0], model.get_layers()[i].attention.o_proj
            )
            self.assertIs(
                found["mlp.down_proj"][0], model.get_layers()[i].feedforward.down_proj
            )

    def test_missing_modules_include_diagnostics(self):
        model = make_model([CustomLayer()], {})
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
        model._validate_additional_layer_modules()
        self.assertEqual(len(model.get_layer_modules(0)["mlp.down_proj"]), 3)

    def test_pattern_can_match_only_some_layers(self):
        model = make_model(
            [CustomLayer(), StandardLayer()],
            {"mlp.down_proj": ["feedforward.down_proj"]},
        )
        model._validate_additional_layer_modules()
        self.assertEqual(list(model.get_layer_modules(1)), ["attn.o_proj"])

    def test_unmatched_pattern_fails_before_optimization(self):
        model = make_model([CustomLayer()], {"mlp.down_proj": ["missing.down_proj"]})
        with self.assertRaisesRegex(ValueError, "matches no module"):
            model._validate_additional_layer_modules()

    def test_duplicate_builtin_or_overlapping_patterns(self):
        model = make_model(
            [StandardLayer()], {"attn.o_proj": ["self_attn.o_proj", "self_attn.*"]}
        )
        found = model.get_layer_modules(0)
        self.assertEqual(len(found["attn.o_proj"]), 1)

    def test_duplicate_across_components_is_rejected(self):
        model = make_model([StandardLayer()], {"mlp.down_proj": ["self_attn.o_proj"]})
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

    def test_container_is_rejected(self):
        model = make_model([CustomLayer()], {"attn.o_proj": ["attention"]})
        with self.assertRaisesRegex(ValueError, "2D weight matrix"):
            model.get_layer_modules(0)


if __name__ == "__main__":
    unittest.main()
