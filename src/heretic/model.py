# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import math
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Generator, Type, cast

import bitsandbytes as bnb
import torch
import torch.linalg as LA
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,  # ty:ignore[possibly-missing-import]
)

with suppress(ImportError):
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion  # type: ignore[assignment]

from .config import QuantizationMethod, RowNormalization, Settings
from .system import empty_cache
from .utils import Prompt, batchify, format_exception, print


def get_model_class(
    model: str,
) -> Type[AutoModelForImageTextToText] | Type[AutoModelForCausalLM]:
    configs = PretrainedConfig.get_config_dict(model)

    for config in configs:
        if isinstance(config, dict) and config.get("model_type") == "diffusion_gemma":
            if "DiffusionGemmaForBlockDiffusion" not in globals():
                raise ImportError(
                    "DiffusionGemma support requires a newer version of the transformers library."
                )
            return DiffusionGemmaForBlockDiffusion  # type: ignore[return-value]

    if any(
        [("vision_config" in config) for config in configs if isinstance(config, dict)]
    ):
        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    # Set for multimodal models, None for text-only ones.
    processor: ProcessorMixin | None
    peft_config: LoraConfig

    def __init__(self, settings: Settings):
        self.settings = settings
        self.needs_reload = False

        self.revision_kwargs = {}
        if settings.model_commit is not None:
            self.revision_kwargs["revision"] = settings.model_commit

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model,
            trust_remote_code=settings.trust_remote_code,
            **self.revision_kwargs,
        )

        # Multimodal models have a processor we'll want to save.
        self.processor = None
        if get_model_class(settings.model) == AutoModelForImageTextToText:
            self.processor = AutoProcessor.from_pretrained(
                settings.model,
                trust_remote_code=settings.trust_remote_code,
                **self.revision_kwargs,
            )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        self.model = None  # ty:ignore[invalid-assignment]
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )
        self.trusted_models = {settings.model: settings.trust_remote_code}

        if self.settings.evaluate_model is not None:
            self.trusted_models[settings.evaluate_model] = settings.trust_remote_code

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]...")

            try:
                quantization_config = self._get_quantization_config(dtype)

                extra_kwargs = {}
                # Only include quantization_config if it's not None
                # (some models like gpt-oss have issues with explicit None).
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config

                self.model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=self.trusted_models.get(settings.model),
                    **self.revision_kwargs,
                    **extra_kwargs,
                )

                # If we reach this point and the model requires trust_remote_code,
                # either the user accepted, or settings.trust_remote_code is True.
                if self.trusted_models.get(settings.model) is None:
                    self.trusted_models[settings.model] = True

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(
                    [
                        Prompt(
                            system=settings.system_prompt,
                            user="What is 1+1?",
                        )
                    ],
                    max_new_tokens=1,
                )
            except Exception as error:
                self.model = None  # ty:ignore[invalid-assignment]
                empty_cache()
                formatted = format_exception(error)
                if "\n" in formatted:
                    print(f"* [red]Failed[/]:\n{formatted}")
                else:
                    print(f"* [red]Failed[/] ({formatted})")
                continue

            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("* Quantized to 4-bit precision")

            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        self._apply_lora()

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.

        if self._is_diffusion_gemma():
            self._save_dg_expert_weights()

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")

        all_components = {}
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                if component not in all_components:
                    all_components[component] = 0
                all_components[component] += len(modules)

        print("* Abliterable components:")
        for component, count in all_components.items():
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

    def _apply_lora(self):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for abliteration (faster reload, no weight modification).
        # Collect actual leaf module names from the model for LoRA targeting.
        # This is more robust than splitting component keys (e.g. "attn.o_proj" -> "o_proj")
        # because hybrid models like Qwen3.5 MoE have modules with different names
        # across layers (e.g. "o_proj" on attention layers, "out_proj" on linear attention layers).
        target_modules_set: set[str] = set()

        module_id_to_full_name = {
            id(module): module_name
            for module_name, module in self.model.named_modules()
        }

        for layer_index in range(len(self.get_layers())):
            for modules in self.get_layer_modules(layer_index).values():
                for module in modules:
                    full_name = module_id_to_full_name.get(id(module))
                    if full_name is not None:
                        target_modules_set.add(full_name)

        target_modules = sorted(target_modules_set)

        if self.settings.row_normalization != RowNormalization.FULL:
            # Rank 1 is sufficient for directional ablation without renormalization.
            lora_rank = 1
        else:
            # Row magnitude preservation introduces nonlinear effects.
            lora_rank = self.settings.full_normalization_lora_rank

        # DiffusionGemmaForBlockDiffusion uses its own generation mixin and does not
        # implement prepare_inputs_for_generation, so CAUSAL_LM task type makes PEFT
        # fail. Use FEATURE_EXTRACTION instead (no generation hooks required).
        task_type = "FEATURE_EXTRACTION" if self._is_diffusion_gemma() else "CAUSAL_LM"

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type=task_type,
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        display_targets = sorted({name.rsplit(".", 1)[-1] for name in target_modules})
        print(
            f"* LoRA adapters initialized (target types: {', '.join(display_targets)})"
        )

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig or None
        """
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        # Check if we need special handling for quantized models
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # Quantized models need special handling - we must reload the base model
            # in full precision to merge the LoRA adapters

            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=self.trusted_models.get(self.settings.model),
                **self.revision_kwargs,
            )

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base_model, self.peft_config)

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            # Non-quantized model - can merge directly
            print("* Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self):
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.
        """
        current_model = getattr(self.model.config, "name_or_path", None)
        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation).
            for name, module in self.model.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            # Restore expert weights that were modified in-place by EGA
            if self._is_diffusion_gemma():
                self._restore_dg_expert_weights()
            return

        dtype = self.model.dtype

        # Purge existing model object from memory to make space.
        self.model = None  # ty:ignore[invalid-assignment]
        empty_cache()

        quantization_config = self._get_quantization_config(str(dtype).split(".")[-1])

        # Build kwargs, only include quantization_config if it's not None.
        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config

        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=self.trusted_models.get(self.settings.model),
            **self.revision_kwargs,
            **extra_kwargs,
        )

        self._apply_lora()

        # On full reload the expert weights are restored from disk; just re-save
        # them so future _restore_dg_expert_weights() calls have fresh copies.
        if self._is_diffusion_gemma():
            self._save_dg_expert_weights()

        self.needs_reload = False

    @contextmanager
    def _dg_lora_merged(self) -> Generator[None, None, None]:
        """Temporarily merge LoRA adapters into the shared encoder/decoder weight tensors.

        DiffusionGemma ties encoder and decoder weights (same data_ptr). PEFT wraps
        only the encoder layers, so the decoder's forward passes won't see the LoRA
        abliteration. We temporarily fold the delta (lora_B @ lora_A) into the shared
        base weight tensor so that the decoder-driven diffusion generation also reflects
        the current abliteration state, then restore the originals when done.
        """
        saved: dict[int, tuple[Tensor, Tensor]] = {}

        for module in self.model.modules():
            if (
                isinstance(module, Linear)
                and hasattr(module, "lora_A")
                and hasattr(module, "lora_B")
            ):
                base_w = module.base_layer.weight.data
                ptr = base_w.data_ptr()
                if ptr in saved:
                    continue  # Already handled this shared tensor.
                if base_w.dtype == torch.uint8 or hasattr(base_w, "quant_state"):
                    raise RuntimeError(
                        "DiffusionGemma LoRA merge is not supported with 4-bit quantization."
                    )
                lora_A_w = cast(Tensor, module.lora_A["default"].weight.data)
                lora_B_w = cast(Tensor, module.lora_B["default"].weight.data)
                delta = (lora_B_w @ lora_A_w).to(base_w.dtype)
                saved[ptr] = (base_w, delta)
                base_w.add_(delta)

        try:
            yield
        finally:
            for base_w, delta in saved.values():
                base_w.sub_(delta)

    def _save_dg_expert_weights(self) -> None:
        """Save CPU copies of expert down_proj tensors for fast trial reset."""
        self._dg_expert_saved: dict[int, Tensor] = {}
        layers = self.get_layers()
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, "experts") and hasattr(layer.experts, "down_proj"):
                self._dg_expert_saved[layer_idx] = (
                    layer.experts.down_proj.data.cpu().clone()
                )

    def _restore_dg_expert_weights(self) -> None:
        """Restore expert weights modified in-place by EGA, ready for the next trial."""
        if not hasattr(self, "_dg_expert_saved"):
            return
        layers = self.get_layers()
        for layer_idx, layer in enumerate(layers):
            if layer_idx in self._dg_expert_saved:
                layer.experts.down_proj.data.copy_(
                    self._dg_expert_saved[layer_idx].to(layer.experts.down_proj.device)
                )

    def _abliterate_dg_experts(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None:
        """EGA: abliterate each expert's down_proj slice in-place.

        Expert weights [n_experts, hidden, moe_inter] are nn.Parameters, not nn.Linear,
        so LoRA can't wrap them. We apply norm-preserving biprojected ablation to every
        expert slice using the same refusal direction and kernel as mlp.down_proj.
        Because encoder and decoder share these tensors (confirmed same data_ptr),
        abliterating here automatically fixes the decoder too.
        """
        layers = self.get_layers()
        for layer_idx, layer in enumerate(layers):
            if not (hasattr(layer, "experts") and hasattr(layer.experts, "down_proj")):
                continue

            if "mlp.down_proj" in parameters:
                params = parameters["mlp.down_proj"]
                distance = cast(float, abs(layer_idx - params.max_weight_position))
                if distance > params.min_weight_distance:
                    continue
                weight_scale = params.max_weight + (
                    distance / params.min_weight_distance
                ) * (params.min_weight - params.max_weight)
            else:
                weight_scale = 1.0

            # Select refusal direction for this layer (same logic as abliterate())
            if direction_index is None:
                layer_refusal_direction = refusal_directions[layer_idx + 1]
            else:
                w_frac, idx = math.modf(direction_index + 1)
                layer_refusal_direction = F.normalize(
                    refusal_directions[int(idx)].lerp(
                        refusal_directions[int(idx) + 1], w_frac
                    ),
                    p=2,
                    dim=0,
                )

            expert_down = layer.experts.down_proj  # [n_experts, hidden, moe_inter]
            v = F.normalize(layer_refusal_direction.float(), dim=0).to(
                expert_down.device
            )

            for expert_idx in range(expert_down.shape[0]):
                # W: [hidden, moe_inter] — out_features=hidden, in_features=moe_inter
                W = expert_down.data[expert_idx].float()
                W_norms = W.norm(dim=1, keepdim=True)  # [hidden, 1]
                W_dirs = F.normalize(W, dim=1)  # Row-normalised.

                # Projection 1: remove refusal component from each column of W
                refusal_comp = v @ W_dirs  # [moe_inter]
                W_dirs = F.normalize(
                    W_dirs - weight_scale * v.unsqueeze(1) * refusal_comp.unsqueeze(0),
                    dim=1,
                )

                # Projection 2: biprojection to catch residual leakage
                refusal_comp2 = v @ W_dirs
                W_dirs = F.normalize(
                    W_dirs - v.unsqueeze(1) * refusal_comp2.unsqueeze(0), dim=1
                )

                expert_down.data[expert_idx] = (W_norms * W_dirs).to(expert_down.dtype)

    def _is_diffusion_gemma(self) -> bool:
        model = self.model
        if isinstance(model, PeftModel):
            model = model.base_model.model
        return "DiffusionGemmaForBlockDiffusion" in type(model).__name__

    def _get_dg_encoder(self) -> Module:
        """Return the DiffusionGemmaEncoderModel (not the text sub-model)."""
        model = self.model
        if isinstance(model, PeftModel):
            model = model.base_model.model
        return model.model.encoder

    def _get_dg_lm_head(self) -> Module:
        model = self.model
        if isinstance(model, PeftModel):
            model = model.base_model.model
        return model.lm_head

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # DiffusionGemma encoder-decoder (encoder layers are abliterated;
        # decoder shares weights via tied_weights_keys so it gets them for free).
        with suppress(Exception):
            return model.model.encoder.language_model.layers

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers

        # Text-only models.
        return model.model.layers

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA)
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            else:
                # Assert for unexpected types (catches architecture changes)
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} - expected nn.Module"
                )

        # Standard self-attention out-projection (most models).
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_proj)  # ty:ignore[possibly-missing-attribute]

        # Qwen3.5 MoE hybrid layers use GatedDeltaNet (linear attention) instead of
        # standard self-attention, so self_attn.o_proj doesn't exist on those layers.
        with suppress(Exception):
            try_add("attn.o_proj", layer.linear_attn.out_proj)  # ty:ignore[possibly-missing-attribute]

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.w2)  # ty:ignore[possibly-missing-attribute]

        # LFM dense operator blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.conv.out_proj)  # ty:ignore[possibly-missing-attribute]

        with suppress(Exception):
            try_add("mlp.down_proj", layer.feed_forward.w2)  # ty:ignore[possibly-missing-attribute]

        # LFM transformer blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.out_proj)  # ty:ignore[possibly-missing-attribute]

        with suppress(Exception):
            for expert in layer.feed_forward.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.w2)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.output_linear)  # ty:ignore[possibly-missing-attribute]

        # We need at least one module across all components for abliteration to work.
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No abliterable modules found in layer"

        return modules

    def get_abliterable_components(self) -> list[str]:
        components: set[str] = set()

        # Scan all layers because hybrid models (e.g. Qwen3.5 MoE) have different
        # components on different layers (some have self_attn, others linear_attn).
        for layer_index in range(len(self.get_layers())):
            components.update(self.get_layer_modules(layer_index).keys())

        return sorted(components)

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        if direction_index is None:
            refusal_direction = None
        else:
            # The index must be shifted by 1 because the first element
            # of refusal_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                params = parameters[component]

                # Type inference fails here for some reason.
                distance = cast(float, abs(layer_index - params.max_weight_position))

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    # The index must be shifted by 1 because the first element
                    # of refusal_directions is the direction for the embeddings.
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                for module in modules:
                    # FIXME: This cast is potentially invalid, because the program logic
                    #        does not guarantee that the module is of type Linear, and in fact
                    #        the retrieved modules might not conform to the interface assumed
                    #        below (though they do in practice). However, this is difficult
                    #        to fix cleanly, because get_layer_modules is called twice on
                    #        different model configurations, and PEFT employs different
                    #        module types depending on the chosen quantization.
                    module = cast(Linear, module)

                    # LoRA abliteration: delta W = -lambda * v * (v^T W)
                    # lora_B = -lambda * v
                    # lora_A = v^T W

                    # Use the FP32 refusal direction directly (no downcast/upcast)
                    # and move to the correct device.
                    v = layer_refusal_direction.to(module.weight.device)

                    # Get W (dequantize if necessary).
                    #
                    # FIXME: This cast is valid only under the assumption that the original
                    #        module wrapped by the LoRA adapter has a weight attribute.
                    #        See the comment above for why this is currently not guaranteed.
                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W = base_weight.to(torch.float32)
                    else:
                        # 4-bit quantization.
                        # This cast is always valid. Type inference fails here because the
                        # bnb.functional module is not found by ty for some reason.
                        W = cast(
                            Tensor,
                            bnb.functional.dequantize_4bit(  # ty:ignore[possibly-missing-attribute]
                                base_weight.data,
                                quant_state,
                            ).to(torch.float32),
                        )

                    # Flatten weight matrix to (out_features, in_features).
                    W = W.view(W.shape[0], -1)

                    if self.settings.row_normalization != RowNormalization.NONE:
                        # Keep a reference to the original weight matrix so we can subtract it later.
                        W_org = W
                        # Get the row norms.
                        W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                        # Normalize the weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)

                    # Calculate lora_A = v^T W
                    # v is (d_out,), W is (d_out, d_in)
                    # v @ W -> (d_in,)
                    lora_A = (v @ W).view(1, -1)

                    # Calculate lora_B = -weight * v
                    # v is (d_out,)
                    lora_B = (-weight * v).view(-1, 1)

                    if self.settings.row_normalization == RowNormalization.PRE:
                        # Make the LoRA adapter apply to the original weight matrix.
                        lora_B = W_row_norms * lora_B
                    elif self.settings.row_normalization == RowNormalization.FULL:
                        # Approximates https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                        W = W + lora_B @ lora_A
                        # Normalize the adjusted weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)
                        # Restore the original row norms of the weight matrix.
                        W = W * W_row_norms
                        # Subtract the original matrix to turn W into a delta.
                        W = W - W_org
                        # Use a low-rank SVD to get an approximation of the matrix.
                        r = self.peft_config.r
                        if self.settings.seed is not None:
                            torch.manual_seed(self.settings.seed)
                        U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
                        # Truncate it to the part we want to store in the LoRA adapter.
                        # Note: svd_lowrank actually returns V, so transpose it to get Vh.
                        U = U[:, :r]
                        S = S[:r]
                        Vh = Vh[:, :r].T
                        # Transfer it into the LoRA adapter components. Split the singular values
                        # evenly between the two components to keep their norms balanced and avoid
                        # potential issues with numerical stability.
                        sqrt_S = torch.sqrt(S)
                        lora_B = U @ torch.diag(sqrt_S)
                        lora_A = torch.diag(sqrt_S) @ Vh

                    # Assign to adapters. The adapter name is "default", because that's
                    # what PEFT uses when no name is explicitly specified, as above.
                    # These casts are therefore valid.
                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

        # For DiffusionGemma, also abliterate the batched MoE expert parameters in-place.
        # LoRA can't wrap nn.Parameter tensors, so EGA modifies them directly. The
        # decoder shares these tensors (same data_ptr), so it's abliterated for free.
        if self._is_diffusion_gemma():
            self._abliterate_dg_experts(refusal_directions, direction_index, parameters)

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor | Any]:
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]

        # This cast is valid because list[str] is the return type
        # for batched operation with tokenize=False.
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        if self.settings.response_prefix:
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [
                prompt + self.settings.response_prefix for prompt in chat_prompts
            ]

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        if self._is_diffusion_gemma():
            # DiffusionGemmaGenerationMixin.generate() does not support do_sample,
            # pad_token_id, output_hidden_states, output_logits, return_dict_in_generate,
            # or use_cache — these are handled via the diffusion sampler config instead.
            dg_allowed = {
                "max_new_tokens",
                "streamer",
                "generation_config",
                "logits_processor",
                "stopping_criteria",
                "max_length",
            }
            dg_kwargs = {k: v for k, v in kwargs.items() if k in dg_allowed}
            # Merge LoRA into shared encoder/decoder tensors so the decoder-driven
            # diffusion generation sees the current abliteration state.
            with self._dg_lora_merged():
                outputs = self.model.generate(  # ty:ignore[call-non-callable]
                    input_ids=cast(Tensor, inputs["input_ids"]),
                    **dg_kwargs,
                )
            return inputs, outputs

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        # DiffusionGemmaGenerationOutput stores the full sequence in .sequences.
        sequences: LongTensor = (
            outputs.sequences if hasattr(outputs, "sequences") else outputs
        )  # ty:ignore[assignment]

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            sequences[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(
                batch,
                skip_special_tokens=skip_special_tokens,
            ):
                responses.append(response)

        return responses

    def _get_residuals_dg(self, prompts: list[Prompt]) -> Tensor:
        """Get per-layer residuals for DiffusionGemma by hooking the encoder layers.

        DiffusionGemma's block-diffusion generate() doesn't expose per-layer hidden
        states, so we run a causal encoder forward pass instead and capture each
        layer's output via forward hooks.
        """
        chats = [
            [
                {"role": "system", "content": p.system},
                {"role": "user", "content": p.user},
            ]
            for p in prompts
        ]
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats, add_generation_prompt=True, tokenize=False
            ),
        )
        if self.settings.response_prefix:
            chat_prompts = [cp + self.settings.response_prefix for cp in chat_prompts]

        inputs = self.tokenizer(
            chat_prompts, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.model.device)

        encoder = self._get_dg_encoder()
        text_model = encoder.language_model

        # layer_idx -> (batch, hidden) tensor captured at the last prompt token
        captured: dict[int, Tensor] = {}
        hooks = []

        # Capture the embed_tokens output as "layer 0" (embedding layer).
        def make_embed_hook():
            def hook(module: Module, input: Any, output: Tensor) -> None:
                captured[-1] = output.detach()

            return hook

        hooks.append(text_model.embed_tokens.register_forward_hook(make_embed_hook()))

        for idx, layer in enumerate(text_model.layers):

            def make_layer_hook(i: int):
                def hook(module: Module, input: Any, output: Tensor) -> None:
                    # Encoder layers return a plain Tensor, not a tuple.
                    captured[i] = (
                        output.detach()
                        if isinstance(output, Tensor)
                        else output[0].detach()
                    )

                return hook

            hooks.append(layer.register_forward_hook(make_layer_hook(idx)))

        try:
            with torch.no_grad():
                encoder(
                    input_ids=cast(Tensor, inputs["input_ids"]),
                    attention_mask=inputs.get("attention_mask"),  # type: ignore[arg-type]
                )
        finally:
            for h in hooks:
                h.remove()

        # Build (prompt, layer, component) tensor; heretic convention: index 0 = embeddings.
        layer_outputs = []
        if -1 in captured:
            layer_outputs.append(captured[-1][:, -1, :])  # Last token position.
        for idx in range(len(text_model.layers)):
            if idx in captured:
                layer_outputs.append(captured[idx][:, -1, :])

        residuals = torch.stack(layer_outputs, dim=1).to(torch.float32)

        if 0 <= self.settings.winsorization_quantile < 1:
            abs_residuals = torch.abs(residuals)
            thresholds = torch.quantile(
                abs_residuals, self.settings.winsorization_quantile, dim=2, keepdim=True
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)

        if self.settings.offload_outputs_to_cpu:
            residuals = residuals.cpu()
            empty_cache()

        return residuals

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        if self._is_diffusion_gemma():
            return self._get_residuals_dg(prompts)

        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # KV cache is unnecessary here because we only need the hidden states
            # for the first generated token.
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Hidden states for the first (only) generated token.
        # This cast is valid because we passed output_hidden_states=True above.
        hidden_states = cast(tuple[tuple[FloatTensor]], outputs.hidden_states)[0]

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        residuals = residuals.to(torch.float32)

        if 0 <= self.settings.winsorization_quantile < 1:
            # Apply symmetric winsorization to each layer of the per-prompt residuals.
            abs_residuals = torch.abs(residuals)
            # Get the (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
            thresholds = torch.quantile(
                abs_residuals,
                self.settings.winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)

        if self.settings.offload_outputs_to_cpu:
            residuals = residuals.cpu()
            empty_cache()

        return residuals

    def get_residuals_batched(self, prompts: list[Prompt]) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))

        return torch.cat(residuals, dim=0)

    def get_residuals_mean(self, prompts: list[Prompt]) -> Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty")

        running_sum = None
        total_count = 0

        for batch in batchify(prompts, self.settings.batch_size):
            batch_residuals = self.get_residuals(batch)

            # Accumulate in high precision on CPU to reduce peak VRAM usage.
            batch_sum = batch_residuals.sum(dim=0, dtype=torch.float64).cpu()

            if running_sum is None:
                running_sum = batch_sum
            else:
                running_sum += batch_sum

            total_count += batch_residuals.shape[0]

        assert running_sum is not None

        return (running_sum / total_count).to(torch.float32)

    def _get_logprobs_dg(self, prompts: list[Prompt]) -> Tensor:
        """Get next-token logprobs for DiffusionGemma.

        Runs the encoder in causal mode and applies lm_head to the last-position
        hidden state, giving the autoregressive prediction for the first new token.
        This is the appropriate signal for KL divergence: it captures how the
        abliteration has shifted the encoder's output distribution.
        """
        chats = [
            [
                {"role": "system", "content": p.system},
                {"role": "user", "content": p.user},
            ]
            for p in prompts
        ]
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats, add_generation_prompt=True, tokenize=False
            ),
        )
        if self.settings.response_prefix:
            chat_prompts = [cp + self.settings.response_prefix for cp in chat_prompts]

        inputs = self.tokenizer(
            chat_prompts, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.model.device)

        with torch.no_grad():
            encoder = self._get_dg_encoder()
            enc_out = encoder(
                input_ids=cast(Tensor, inputs["input_ids"]),
                attention_mask=inputs.get("attention_mask"),  # type: ignore[arg-type]
            )
            last_hidden = enc_out.last_hidden_state[:, -1, :]  # (batch, hidden)
            lm_head = self._get_dg_lm_head()
            logits = lm_head(last_hidden.to(lm_head.weight.dtype))  # (batch, vocab)

        logprobs = F.log_softmax(logits.float(), dim=-1)

        if self.settings.offload_outputs_to_cpu:
            logprobs = logprobs.cpu()
            empty_cache()

        return logprobs

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[Prompt]) -> Tensor:
        if self._is_diffusion_gemma():
            return self._get_logprobs_dg(prompts)

        # We only generate one token, and we return the (log) probability distributions
        # over the vocabulary at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Use raw logits, not processed generation scores; processors can insert
        # -inf for suppressed tokens, which can make KL divergence evaluate to NaN.
        logits = cast(tuple[FloatTensor], outputs.logits)[0]

        # The returned tensor has shape (prompt, token).
        logprobs = F.log_softmax(logits, dim=-1)

        if self.settings.offload_outputs_to_cpu:
            del outputs, logits
            logprobs = logprobs.cpu()
            empty_cache()

        return logprobs

    def get_logprobs_batched(self, prompts: list[Prompt]) -> Tensor:
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch))

        return torch.cat(logprobs, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,  # ty:ignore[invalid-argument-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        if self._is_diffusion_gemma():
            with self._dg_lora_merged():
                outputs = self.model.generate(  # ty:ignore[call-non-callable]
                    input_ids=cast(Tensor, inputs["input_ids"]),
                    streamer=streamer,
                    max_new_tokens=4096,
                )
            sequences: LongTensor = (
                outputs.sequences if hasattr(outputs, "sequences") else outputs
            )  # ty:ignore[assignment]
            return cast(
                str,
                self.tokenizer.decode(
                    sequences[0, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ),
            )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        # This cast is valid because str is the return type
        # when passing a sequence of token IDs.
        return cast(
            str,
            self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        )
