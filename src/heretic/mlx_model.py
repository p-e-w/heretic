# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
MLX backend for heretic – runs abliteration natively on Apple Silicon via
the MLX framework and mlx-lm library.
"""

import json
import math
import os
from contextlib import suppress
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .config import Settings
from .model import AbliterationParameters
from .utils import Prompt, batchify, print


# ---------------------------------------------------------------------------
# Lazy MLX imports (so the module can be imported on non-Apple platforms
# without crashing – the caller should guard with is_mlx_available()).
# ---------------------------------------------------------------------------

def _import_mlx():
    import mlx.core as mx
    import mlx.nn as nn
    return mx, nn


def _import_mlx_lm():
    import mlx_lm
    return mlx_lm


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def is_mlx_available() -> bool:
    """Return True if both mlx and mlx-lm can be imported."""
    try:
        _import_mlx()
        _import_mlx_lm()
        return True
    except ImportError:
        return False


def is_mlx_model(path: str) -> bool:
    """
    Heuristic: a directory is an MLX model if it contains a config.json
    whose ``model_type`` is in the set of architectures shipped with mlx-lm
    **and** the weights are in safetensors format (not bin/pytorch format).
    """
    if not os.path.isdir(path):
        return False

    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    # Check for quantization config that indicates MLX format
    # (MLX models have quantize_config with group_size/bits at top level)
    if "quantization" in config:
        return True

    # Also detect by checking for mlx-specific weight files
    # MLX models use safetensors, but so do some HF models.
    # A stronger signal is the absence of pytorch_model.bin and presence of
    # model*.safetensors without a model.safetensors.index.json that references
    # pytorch format.
    has_safetensors = any(
        f.endswith(".safetensors") for f in os.listdir(path) if f.startswith("model")
    )
    has_pytorch = any(
        f.endswith(".bin") for f in os.listdir(path) if f.startswith("pytorch_model")
    )

    if has_safetensors and not has_pytorch:
        # Check if the config has a model_type that mlx-lm supports
        model_type = config.get("model_type", "")
        try:
            mlx_lm = _import_mlx_lm()
            # mlx_lm.models has submodules named after model types
            import importlib

            importlib.import_module(f"mlx_lm.models.{model_type}")
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    return False


# ---------------------------------------------------------------------------
# MLX Model class
# ---------------------------------------------------------------------------


class MLXModel:
    """
    Model backend that uses Apple MLX for inference and weight manipulation.

    Provides the same public interface as ``heretic.model.Model`` so that
    ``evaluator.py`` and ``main.py`` can use either backend transparently.
    """

    def __init__(self, settings: Settings):
        mx, nn = _import_mlx()
        mlx_lm = _import_mlx_lm()

        self.settings = settings
        self.response_prefix = ""
        self.needs_reload = False
        self._mx = mx
        self._nn = nn
        self._mlx_lm = mlx_lm

        print()
        print(f"Loading model [bold]{settings.model}[/] (MLX backend)...")

        self.mlx_model, self.tokenizer = mlx_lm.load(
            settings.model,
            tokenizer_config={"trust_remote_code": settings.trust_remote_code or False},
        )

        # Fallback pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Left-padding for decoder-only generation.
        self.tokenizer.padding_side = "left"

        # Read hidden_size from config.
        config_path = os.path.join(settings.model, "config.json")
        with open(config_path) as f:
            self._model_config = json.load(f)
        self._hidden_size = self._model_config["hidden_size"]
        self._vocab_size = self._model_config["vocab_size"]

        # Storage for original weights (used by reset_model).
        self._original_weights: dict[int, dict[str, Any]] = {}

        # Test run.
        print("* Testing inference... ", end="")
        try:
            self.get_responses(
                [Prompt(system=settings.system_prompt, user="What is 1+1?")],
                skip_special_tokens=True,
            )
            print("[green]Ok[/]")
        except Exception as error:
            print(f"[red]Failed[/] ({error})")
            raise

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        all_components: dict[str, int] = {}
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                if component not in all_components:
                    all_components[component] = 0
                all_components[component] += len(modules)
        for component, count in all_components.items():
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

    # -------------------------------------------------------------------
    # Layer / module introspection
    # -------------------------------------------------------------------

    def get_layers(self) -> list:
        """Return the list of transformer decoder layers."""
        return self.mlx_model.model.layers

    def get_layer_modules(self, layer_index: int) -> dict[str, list]:
        """
        Return abliterable modules for a given layer, matching the same
        component keys as the PyTorch backend.
        """
        nn = self._nn
        layer = self.get_layers()[layer_index]
        modules: dict[str, list] = {}

        def try_add(component: str, module: Any):
            if isinstance(module, nn.Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)

        # Standard self-attention out-projection.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_proj)

        # Qwen3.5 MoE hybrid linear attention.
        with suppress(Exception):
            try_add("attn.o_proj", layer.linear_attn.out_proj)

        # Dense MLP down_proj.
        with suppress(Exception):
            if hasattr(layer.mlp, "down_proj") and isinstance(
                layer.mlp.down_proj, nn.Module
            ):
                try_add("mlp.down_proj", layer.mlp.down_proj)

        # MoE SwitchGLU down_proj (stacked experts in one module).
        with suppress(Exception):
            if hasattr(layer.mlp, "switch_mlp"):
                try_add("mlp.down_proj", layer.mlp.switch_mlp.down_proj)

        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No abliterable modules found in layer"

        return modules

    def get_abliterable_components(self) -> list[str]:
        components: set[str] = set()
        for layer_index in range(len(self.get_layers())):
            components.update(self.get_layer_modules(layer_index).keys())
        return sorted(components)

    # -------------------------------------------------------------------
    # Tokenization helpers
    # -------------------------------------------------------------------

    def _tokenize_prompts(self, prompts: list[Prompt]) -> list[str]:
        """Apply chat template to prompts, returning formatted strings."""
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]

        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        if self.response_prefix:
            chat_prompts = [p + self.response_prefix for p in chat_prompts]

        return chat_prompts

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        mx = self._mx
        mlx_lm = self._mlx_lm

        chat_prompts = self._tokenize_prompts(prompts)
        responses = []

        from mlx_lm.sample_utils import make_sampler

        # Greedy decoding (temp=0) for deterministic outputs.
        sampler = make_sampler(temp=0.0)

        for prompt_str in chat_prompts:
            response = mlx_lm.generate(
                self.mlx_model,
                self.tokenizer,
                prompt=prompt_str,
                max_tokens=self.settings.max_response_length,
                verbose=False,
                sampler=sampler,
            )
            # mlx_lm.generate returns the full text including prompt.
            # Extract only the generated part.
            if response.startswith(prompt_str):
                response = response[len(prompt_str):]

            if skip_special_tokens:
                # Re-decode through tokenizer to strip special tokens.
                token_ids = self.tokenizer.encode(response, add_special_tokens=False)
                response = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            responses.append(response)

        return responses

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        responses = []
        for batch in batchify(prompts, max(self.settings.batch_size, 1)):
            responses.extend(
                self.get_responses(batch, skip_special_tokens=skip_special_tokens)
            )
        return responses

    # -------------------------------------------------------------------
    # Hidden-state extraction (residuals)
    # -------------------------------------------------------------------

    def _forward_with_hidden_states(
        self, input_ids_list: list[list[int]]
    ) -> tuple[Any, list[list[Any]]]:
        """
        Run a manual forward pass through the model layers, capturing
        the hidden state after each layer.

        Returns (logits, all_hidden_states) where all_hidden_states is
        a list (per prompt) of lists (per layer+embeddings) of mx.arrays.
        """
        mx = self._mx

        all_hidden_states = []
        all_logits = []

        for input_ids in input_ids_list:
            tokens = mx.array([input_ids])

            # Embedding layer
            h = self.mlx_model.model.embed_tokens(tokens)
            hidden_states = [h[:, -1, :]]  # Last token position

            # Create attention mask
            from mlx_lm.models.base import create_attention_mask

            mask = create_attention_mask(h, cache=None)

            # Forward through each layer
            for layer in self.mlx_model.model.layers:
                h = layer(h, mask=mask, cache=None)
                hidden_states.append(h[:, -1, :])  # Last token position

            # Final norm
            h = self.mlx_model.model.norm(h)

            # LM head for logits.
            # When tie_word_embeddings is True the model has no separate
            # lm_head; instead it reuses the embedding matrix.
            if hasattr(self.mlx_model, "lm_head"):
                logits = self.mlx_model.lm_head(h)
            else:
                logits = self.mlx_model.model.embed_tokens.as_linear(h)

            all_hidden_states.append(hidden_states)
            all_logits.append(logits[:, -1, :])  # Last position logits

            mx.eval(*hidden_states, logits)

        return all_logits, all_hidden_states

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        mx = self._mx

        chat_prompts = self._tokenize_prompts(prompts)
        input_ids_list = [
            self.tokenizer.encode(p, add_special_tokens=False) for p in chat_prompts
        ]

        _, all_hidden_states = self._forward_with_hidden_states(input_ids_list)

        # Stack into (batch, layers+1, hidden_size) tensor
        batch_residuals = []
        for prompt_hidden_states in all_hidden_states:
            # Each element is (1, hidden_size), squeeze the batch dim
            layer_residuals = mx.stack(
                [hs.squeeze(0) for hs in prompt_hidden_states], axis=0
            )
            batch_residuals.append(layer_residuals)

        residuals_mx = mx.stack(batch_residuals, axis=0)
        mx.eval(residuals_mx)

        # Convert to torch float32
        residuals = torch.from_numpy(np.array(residuals_mx.astype(mx.float32)))

        if 0 <= self.settings.winsorization_quantile < 1:
            abs_residuals = torch.abs(residuals)
            thresholds = torch.quantile(
                abs_residuals,
                self.settings.winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            return torch.clamp(residuals, -thresholds, thresholds)

        return residuals

    def get_residuals_batched(self, prompts: list[Prompt]) -> Tensor:
        residuals = []
        for batch in batchify(prompts, max(self.settings.batch_size, 1)):
            residuals.append(self.get_residuals(batch))
        return torch.cat(residuals, dim=0)

    # -------------------------------------------------------------------
    # Log-probability extraction
    # -------------------------------------------------------------------

    def get_logprobs(self, prompts: list[Prompt]) -> Tensor:
        mx = self._mx

        chat_prompts = self._tokenize_prompts(prompts)
        input_ids_list = [
            self.tokenizer.encode(p, add_special_tokens=False) for p in chat_prompts
        ]

        all_logits, _ = self._forward_with_hidden_states(input_ids_list)

        # Stack logits: (batch, vocab_size)
        logits_mx = mx.concatenate(all_logits, axis=0)
        mx.eval(logits_mx)

        logits_torch = torch.from_numpy(np.array(logits_mx.astype(mx.float32)))
        return F.log_softmax(logits_torch, dim=-1)

    def get_logprobs_batched(self, prompts: list[Prompt]) -> Tensor:
        logprobs = []
        for batch in batchify(prompts, max(self.settings.batch_size, 1)):
            logprobs.append(self.get_logprobs(batch))
        return torch.cat(logprobs, dim=0)

    # -------------------------------------------------------------------
    # Weight manipulation helpers
    # -------------------------------------------------------------------

    def _get_weight_f32(self, module: Any):
        """Get a module's weight as a float32 mx.array, dequantizing if needed."""
        mx = self._mx

        if hasattr(module, "scales"):
            # QuantizedLinear or QuantizedSwitchLinear
            biases = getattr(module, "biases", None)
            return mx.dequantize(
                module.weight,
                module.scales,
                biases,
                module.group_size,
                module.bits,
            ).astype(mx.float32)
        else:
            return module.weight.astype(mx.float32)

    def _save_original_weights(self, module: Any):
        """Save a copy of the module's weight state for later restoration."""
        mx = self._mx
        mid = id(module)
        if mid in self._original_weights:
            return  # Already saved

        state: dict[str, Any] = {"weight": mx.array(module.weight)}
        if hasattr(module, "scales"):
            state["scales"] = mx.array(module.scales)
            biases = getattr(module, "biases", None)
            if biases is not None:
                state["biases"] = mx.array(biases)
            state["group_size"] = module.group_size
            state["bits"] = module.bits
            state["is_quantized"] = True
        else:
            state["is_quantized"] = False

        self._original_weights[mid] = state

    def _restore_original_weights(self, module: Any):
        """Restore a module's weights from the saved originals."""
        mid = id(module)
        if mid not in self._original_weights:
            return

        state = self._original_weights[mid]
        module.weight = state["weight"]
        if state["is_quantized"]:
            module.scales = state["scales"]
            if "biases" in state:
                module.biases = state["biases"]

    def _set_weight_requantize(self, module: Any, W_new):
        """
        Write a modified float32 weight back into a module.
        If the module was quantized, requantize to the same bit width.
        """
        mx = self._mx

        if hasattr(module, "scales"):
            group_size = module.group_size
            bits = module.bits
            mode = getattr(module, "mode", "affine")
            result = mx.quantize(W_new, group_size=group_size, bits=bits)
            module.weight = result[0]
            module.scales = result[1]
            if len(result) > 2:
                module.biases = result[2]
        else:
            module.weight = W_new.astype(module.weight.dtype)

    # -------------------------------------------------------------------
    # Abliteration
    # -------------------------------------------------------------------

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        mx = self._mx

        # Convert refusal directions from torch to MLX
        refusal_dirs_mx = mx.array(refusal_directions.numpy())

        if direction_index is not None:
            weight_frac, index = math.modf(direction_index + 1)
            rd = refusal_dirs_mx[int(index)] * (1.0 - weight_frac) + \
                 refusal_dirs_mx[int(index) + 1] * weight_frac
            # Normalize
            refusal_direction = rd / mx.linalg.norm(rd)
        else:
            refusal_direction = None

        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                params = parameters[component]

                distance = abs(layer_index - params.max_weight_position)
                if distance > params.min_weight_distance:
                    continue

                ablation_weight = params.max_weight + (
                    distance / params.min_weight_distance
                ) * (params.min_weight - params.max_weight)

                if refusal_direction is None:
                    layer_refusal_direction = refusal_dirs_mx[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                for module in modules:
                    self._save_original_weights(module)

                    # Get weight as float32
                    W = self._get_weight_f32(module)

                    v = layer_refusal_direction

                    if W.ndim == 3:
                        # SwitchLinear: (num_experts, out_dims, in_dims)
                        # v is (out_dims,)
                        # Per-expert: lora_A_i = v @ W_i → (in_dims,)
                        # delta_W_i = -weight * outer(v, v @ W_i)
                        vT_W = mx.einsum("j,ejk->ek", v, W)  # (experts, in_dims)
                        delta_W = -ablation_weight * v[None, :, None] * vT_W[:, None, :]
                        W_new = W + delta_W
                    else:
                        # Regular Linear: (out_dims, in_dims)
                        W_flat = W.reshape(W.shape[0], -1)
                        lora_A = (v @ W_flat).reshape(1, -1)
                        lora_B = (-ablation_weight * v).reshape(-1, 1)
                        W_new = W + (lora_B @ lora_A).reshape(W.shape)

                    mx.eval(W_new)
                    self._set_weight_requantize(module, W_new)

    # -------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------

    def reset_model(self):
        """Restore all abliterated modules to their original weights."""
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                for module in modules:
                    self._restore_original_weights(module)

        self._original_weights.clear()
        self.needs_reload = False

    # -------------------------------------------------------------------
    # Merged model (for saving)
    # -------------------------------------------------------------------

    def get_merged_model(self):
        """
        Return the MLX model with current abliteration applied.
        For MLX, the weights are already modified in-place, so we just
        return the model itself.
        """
        return self.mlx_model

    def save_pretrained(self, save_directory: str):
        """Save the MLX model weights, config, and tokenizer to a directory."""
        import shutil

        mx = self._mx
        os.makedirs(save_directory, exist_ok=True)

        # Save weights using mlx utilities
        weights = dict(self.mlx_model.parameters())
        # Flatten nested dicts to dot-separated keys
        flat_weights = {}

        def _flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, key)
                else:
                    flat_weights[key] = v

        _flatten(weights)
        mx.save_safetensors(os.path.join(save_directory, "model.safetensors"), flat_weights)

        # Copy config.json
        src_config = os.path.join(self.settings.model, "config.json")
        if os.path.exists(src_config):
            shutil.copy2(src_config, os.path.join(save_directory, "config.json"))

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

    # -------------------------------------------------------------------
    # Chat streaming
    # -------------------------------------------------------------------

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        mlx_lm = self._mlx_lm

        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        import builtins

        full_response = ""
        for response in mlx_lm.stream_generate(
            self.mlx_model,
            self.tokenizer,
            prompt=chat_prompt,
            max_tokens=4096,
        ):
            token_text = response.text
            builtins.print(token_text, end="", flush=True)
            full_response += token_text

        builtins.print()  # Newline after streaming
        return full_response
