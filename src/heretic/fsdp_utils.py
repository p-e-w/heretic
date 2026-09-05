# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
FSDP (Fully Sharded Data Parallel) utilities for TPU model parallelism.

This module provides helpers for wrapping models with FSDP on PyTorch/XLA
for multi-core TPU training and inference (torch_xla 2.8+ API:
XlaFullyShardedDataParallel).
"""

import json
import os
from functools import partial
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel

from .system import _is_torch_xla_available, detect_tpu, get_xla_device_count


def get_fsdp_layer_classes(model: PreTrainedModel) -> set[type[nn.Module]]:
    """Resolve transformer layer classes present in the model for FSDP wrapping."""
    return {
        type(module)
        for module in model.modules()
        if "DecoderLayer" in type(module).__name__
        or "TransformerBlock" in type(module).__name__
    }


def get_fsdp_layer_class_names(model: PreTrainedModel) -> list[str]:
    """Get the transformer layer class names for FSDP wrapping based on model architecture."""
    model_class = model.__class__.__name__
    return TRANSFORMER_LAYER_CLASSES.get(model_class, [])


def get_default_fsdp_config(model: PreTrainedModel) -> dict[str, Any]:
    """Generate a default FSDP configuration for the given model."""
    return {
        "fsdp_transformer_layer_cls_to_wrap": get_fsdp_layer_class_names(model),
    }


def load_fsdp_config(path: str) -> dict[str, Any]:
    """Load FSDP configuration from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_fsdp_config(config: dict[str, Any], path: str) -> None:
    """Save FSDP configuration to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def fsdp_uses_spmd() -> bool:
    """True when the single-process SPMD FSDP path applies (Kaggle/Colab TPU VMs)."""
    if not (detect_tpu() and _is_torch_xla_available()):
        return False
    import torch_xla.runtime as xr

    return xr.process_count() == 1 and get_xla_device_count() > 1


def _get_spmd_mesh() -> Any:
    """Create the SPMD device mesh over all physical TPU cores."""
    from torch_xla.distributed.spmd import Mesh

    n = get_xla_device_count()
    return Mesh(list(range(n)), (n,), ("fsdp",))


def _spmd_shard_output(output: Any, mesh: Any) -> None:
    """Shard the logits output of a wrapped model across the SPMD mesh."""
    from torch_xla.distributed import spmd

    logits = output if isinstance(output, torch.Tensor) else getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Cannot shard output of type {type(output)}")
    spmd.mark_sharding(logits, mesh, (None, None, "fsdp"))


def wrap_model_fsdp(
    model: PreTrainedModel,
    config: dict[str, Any] | str | None = None,
) -> PreTrainedModel:
    """
    Wrap a model with FSDP for TPU model parallelism (torch_xla 2.8+).

    Uses the SPMD variant (SpmdFullyShardedDataParallel) in single-process
    multi-core setups (e.g. one Kaggle/Colab TPU VM process driving all chips),
    and the classic XlaFullyShardedDataParallel for multi-process jobs.

    Args:
        model: The model to wrap (on CPU).
        config: FSDP configuration dict or path to JSON file.

    Returns:
        FSDP-wrapped model.
    """
    if not detect_tpu():
        raise RuntimeError("FSDP wrapping only supported on TPU")

    if not _is_torch_xla_available():
        raise RuntimeError("torch_xla not available")

    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

    if isinstance(config, str):
        config = load_fsdp_config(config)
    elif config is None:
        config = get_default_fsdp_config(model)

    layer_names = config.get("fsdp_transformer_layer_cls_to_wrap", [])
    layer_classes = get_fsdp_layer_classes(model)
    if layer_names:
        by_name = {type(m).__name__: type(m) for m in model.modules()}
        named = {by_name[name] for name in layer_names if name in by_name}
        if named:
            layer_classes = named

    if not layer_classes:
        raise ValueError(
            f"No transformer layer classes found for model {model.__class__.__name__}"
        )

    wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls=layer_classes
    )

    import torch_xla.runtime as xr

    if fsdp_uses_spmd():
        # Single process driving multiple chips: SPMD FSDP.
        from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
            SpmdFullyShardedDataParallel,
        )

        xr.use_spmd()
        model = SpmdFullyShardedDataParallel(
            model,
            mesh=_get_spmd_mesh(),
            shard_output=_spmd_shard_output,
            auto_wrap_policy=wrap_policy,
        )
    else:
        # Multi-process job: classic FSDP (one process per core).
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel

        model = XlaFullyShardedDataParallel(
            model,
            auto_wrap_policy=wrap_policy,
            flatten_parameters=False,
            compute_dtype=torch.bfloat16,
        )

    return model


def unwrap_fsdp_model(model: PreTrainedModel) -> PreTrainedModel:
    """Unwrap a FSDP-wrapped model."""
    if not _is_torch_xla_available():
        return model

    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel

    if isinstance(model, XlaFullyShardedDataParallel):
        return model._fsdp_wrapped_module  # ty:ignore[attr-defined]

    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel,
    )

    if isinstance(model, SpmdFullyShardedDataParallel):
        return model.module

    return model


def is_fsdp_model(model: PreTrainedModel) -> bool:
    """Check if model is wrapped with FSDP."""
    if not _is_torch_xla_available():
        return False

    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel

    if isinstance(model, XlaFullyShardedDataParallel):
        return True

    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel,
    )

    return isinstance(model, SpmdFullyShardedDataParallel)


# Architecture-specific transformer layer class names for FSDP wrapping
TRANSFORMER_LAYER_CLASSES = {
    "LlamaForCausalLM": ["LlamaDecoderLayer"],
    "LlamaForSequenceClassification": ["LlamaDecoderLayer"],
    "CodeLlamaForCausalLM": ["LlamaDecoderLayer"],
    "Qwen2ForCausalLM": ["Qwen2DecoderLayer"],
    "Qwen2MoeForCausalLM": ["Qwen2MoeDecoderLayer"],
    "Qwen3ForCausalLM": ["Qwen3DecoderLayer"],
    "Qwen3MoeForCausalLM": ["Qwen3MoeDecoderLayer"],
    "GemmaForCausalLM": ["GemmaDecoderLayer"],
    "Gemma2ForCausalLM": ["Gemma2DecoderLayer"],
    "Gemma3ForCausalLM": ["Gemma3DecoderLayer"],
    "Gemma3ForConditionalGeneration": ["Gemma3DecoderLayer"],
    "MistralForCausalLM": ["MistralDecoderLayer"],
    "MixtralForCausalLM": ["MixtralDecoderLayer"],
    "Phi3ForCausalLM": ["Phi3DecoderLayer"],
    "Phi4ForCausalLM": ["Phi4DecoderLayer"],
    "YiForCausalLM": ["YiDecoderLayer"],
    "InternLM2ForCausalLM": ["InternLM2DecoderLayer"],
    "InternLM2ForSequenceClassification": ["InternLM2DecoderLayer"],
    "GraniteForCausalLM": ["GraniteDecoderLayer"],
    "GraniteMoeForCausalLM": ["GraniteMoeDecoderLayer"],
    "CohereForCausalLM": ["CohereDecoderLayer"],
    "SmolLM3ForCausalLM": ["SmolLM3DecoderLayer"],
    "DeepseekV2ForCausalLM": ["DeepseekV2DecoderLayer"],
    "DeepseekV3ForCausalLM": ["DeepseekV3DecoderLayer"],
}
