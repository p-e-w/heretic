# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch

try:
    from auto_round_extension.torch.qlinear_torch_zp import QuantLinear
except ImportError:
    QuantLinear = None
from torch import nn
from torch.nn import ModuleList

from heretic.config import Settings as HereticSettings
from heretic.modifiers.abliteration import (
    Abliteration,
    Parameters,
    WeightDistribution,
)
from heretic.modifiers.abliteration import (
    Settings as AbliterationSettings,
)


class FakeExpert(nn.Module):
    def __init__(self, down_proj: nn.Module) -> None:
        super().__init__()
        self.down_proj = down_proj


class FakeLayer(nn.Module):
    def __init__(self, down_proj: nn.Module) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = ModuleList([FakeExpert(down_proj)])


class FakeModel:
    def __init__(self, layers: ModuleList) -> None:
        self.layers = layers

    def get_layers(self) -> ModuleList:
        return self.layers


class FakeContext:
    def __init__(self, model: FakeModel) -> None:
        self.model = model

    def get_model(self) -> FakeModel:
        return self.model


class Qwen4BakeTests(unittest.TestCase):
    def test_bakes_every_quantized_expert(self) -> None:
        if QuantLinear is None:
            self.skipTest("auto-round is not installed")
        torch.manual_seed(2)
        out_features, in_features = 256, 128
        weight = 0.05 * torch.randn(out_features, in_features)
        direction = torch.randn(out_features)
        direction /= direction.norm()

        groups = in_features // 128
        qlinear = QuantLinear(4, 128, in_features, out_features, False)
        scales = torch.maximum(
            weight.reshape(out_features, groups, 128).clamp_min(0).amax(dim=2) / 7.0,
            (-weight.reshape(out_features, groups, 128).clamp_max(0)).amax(dim=2) / 8.0,
        ).clamp_min(1e-8)
        zeros = torch.full((out_features, groups), 8.0)
        dense = nn.Linear(in_features, out_features, bias=False)
        dense.weight.data = weight
        qlinear.pack(dense, scales, zeros)
        original = qlinear(torch.eye(in_features)).t()

        layer = FakeLayer(qlinear)
        model = FakeModel(ModuleList([layer]))
        modifier = Abliteration(
            heretic_settings=HereticSettings.model_construct(),
            settings=AbliterationSettings(),
        )
        modifier.residual_directions = torch.stack(
            [torch.zeros(out_features), direction]
        )
        parameters = Parameters(
            direction_index=None,
            weight_distributions={
                "mlp.down_proj": WeightDistribution(
                    max_weight=0.8,
                    max_weight_position=0.0,
                    min_weight=0.8,
                    min_weight_distance=1.0,
                )
            },
        )

        modifier.bake(FakeContext(model), None, parameters)

        expected = original - 0.8 * torch.outer(direction, direction @ original)
        baked = qlinear(torch.eye(in_features)).t()
        self.assertTrue(
            torch.allclose(baked, expected, atol=5e-2),
            f"max bake error: {float((baked - expected).abs().max())}",
        )


if __name__ == "__main__":
    unittest.main()
