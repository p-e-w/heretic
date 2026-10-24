# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Bake routed-expert ablation into AutoRound W4A16 checkpoints.

Heretic cannot attach LoRA adapters to the packed quantized expert weights in
Qwen4-Exp checkpoints. The routed experts therefore stay untouched during the
normal optimization, even though they account for most of the MoE's weights.
This module applies the selected refusal direction to each expert's
``down_proj`` at export time and repacks the result.

The operation is a rank-1 projection on the expert output::

    W' = W - lambda * v * (v^T W)

which is exactly what the MoE block's forward projection does, but baked into
the checkpoint.
"""

import torch
from torch import Tensor, nn
from torch.nn import ModuleList

from .utils import print


def bake_quantized_experts(
    layers: ModuleList,
    directions: list[Tensor | None],
    weights: list[float],
) -> int:
    """Bake the projection into all packed routed experts.

    Returns the number of expert ``down_proj`` matrices that were modified.
    """
    total = 0
    for layer_index, layer in enumerate(layers):
        weight = weights[layer_index]
        if weight == 0:
            continue

        direction = directions[layer_index]
        if direction is None:
            continue

        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is None:
            continue

        baked = 0
        for _, expert in experts.named_children():
            down_proj = getattr(expert, "down_proj", None)
            if down_proj is None:
                continue
            if not all(
                hasattr(down_proj, attribute)
                for attribute in ("qweight", "qzeros", "scales")
            ):
                continue
            bake_quantized_linear(down_proj, direction, weight)
            baked += 1

        if baked:
            print(f"  * Layer {layer_index}: baked {baked} routed experts")
            total += baked

    return total


def bake_quantized_linear(qlinear: nn.Module, direction: Tensor, weight: float) -> None:
    """Apply the projection to one packed AutoRound linear and repack it."""
    # Decode and repack on the host. This also works for the Triton-backed
    # QuantLinear class, whose pack() helper is intentionally absent.
    qlinear.to("cpu")

    bits = qlinear.bits  # ty: ignore[unresolved-attribute]
    group_size = qlinear.group_size  # ty: ignore[unresolved-attribute]
    infeatures = qlinear.infeatures  # ty: ignore[unresolved-attribute]
    outfeatures = qlinear.outfeatures  # ty: ignore[unresolved-attribute]
    if infeatures % group_size != 0:
        raise ValueError(
            f"Expert down_proj input size {infeatures} is not divisible by "
            f"group size {group_size}."
        )
    groups = infeatures // group_size
    pack = 32 // bits
    maxq = (1 << bits) - 1

    wf = torch.arange(0, 32, bits, dtype=torch.int32)
    qweight = qlinear.qweight.to("cpu")  # ty: ignore[unresolved-attribute]
    qzeros = qlinear.qzeros.to("cpu")  # ty: ignore[unresolved-attribute]
    scales = qlinear.scales.to("cpu")  # ty: ignore[unresolved-attribute]

    packed_weight = (
        (qweight.unsqueeze(1).expand(-1, pack, -1) >> wf.unsqueeze(-1)) & maxq
    ).reshape(infeatures, outfeatures)
    packed_zeros = (
        (qzeros.unsqueeze(2).expand(-1, -1, pack) >> wf.unsqueeze(0)) & maxq
    ).reshape(groups, outfeatures)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = packed_zeros.repeat_interleave(group_size, dim=0)
    W = (scales * (packed_weight - zeros)).t().contiguous().float()

    v = direction.to(device=W.device, dtype=torch.float32)
    W = W - float(weight) * torch.outer(v, v @ W)

    Wg = W.reshape(outfeatures, groups, group_size)
    new_scales = torch.maximum(
        Wg.clamp_min(0).amax(dim=2) / 7.0,
        (-Wg.clamp_max(0)).amax(dim=2) / 8.0,
    ).clamp_min(1e-8)
    new_zeros = torch.full((outfeatures, groups), 8.0, dtype=torch.float32)

    # Use the reference packing implementation. It writes the same GPTQ layout
    # that both the torch and Triton QuantLinear classes consume.
    from auto_round_extension.torch.qlinear_torch_zp import QuantLinear

    dense = nn.Linear(infeatures, outfeatures, bias=False)
    dense.weight.data = W
    packed = QuantLinear(bits, group_size, infeatures, outfeatures, False)
    packed.pack(dense, new_scales, new_zeros)

    qlinear.qweight.data.copy_(packed.qweight)  # ty: ignore[unresolved-attribute]
    qlinear.qzeros.data.copy_(packed.qzeros)  # ty: ignore[unresolved-attribute]
    qlinear.scales.data.copy_(packed.scales)  # ty: ignore[unresolved-attribute]


__all__ = [
    "bake_quantized_experts",
    "bake_quantized_linear",
]
