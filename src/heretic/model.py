# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable, Type, TypeAlias, cast

import bitsandbytes as bnb
import torch
import torch.linalg as LA
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear
from psutil import virtual_memory
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from torch.optim import LBFGS
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,  # ty:ignore[possibly-missing-import]
)

from .config import QuantizationMethod, RowNormalization, Settings
from .utils import Prompt, batchify, empty_cache, mean_distances_to_knn, print


def get_model_class(
    model: str,
) -> Type[AutoModelForImageTextToText] | Type[AutoModelForCausalLM]:
    configs = PretrainedConfig.get_config_dict(model)

    if any([("vision_config" in config) for config in configs]):
        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


@dataclass
class ARAParameters:
    start_layer_index: int
    end_layer_index: int
    preserve_good_behavior_weight: float
    steer_bad_behavior_weight: float
    overcorrect_relative_weight: float
    neighbor_count: int


# The list contains one element per layer.
# Each element maps from the component name to a (possibly sparse) mapping
# from the module index to an (input, output) tuple containing the I/O
# tensors of shape (prompt, component).
ModuleIO: TypeAlias = list[dict[str, dict[int, tuple[Tensor, Tensor]]]]


# Headroom (in bytes) we keep free on any device when planning a snapshot,
# so that we don't starve the LBFGS optimizer or the inference path.
_SNAPSHOT_SAFETY_MARGIN_BYTES = 2 * 1024**3  # 2 GB


class ARAWeightCache:
    """Caches original weights of ARA-touched modules between trials.

    Holds the invariant: every entry stores the pristine, original weight
    of its module. To preserve this, the caller must:
      1. Restore from cache before any new optimization (model is clean).
      2. Drop entries for modules not used on the next trial (free RAM).
      3. Snapshot newly-targeted modules BEFORE mutating them.

    Storage tier is per source device: snapshots stay on the source GPU
    when there is room (intra-device copy on restore, fastest), otherwise
    spill to system RAM. If neither tier can host the snapshot,
    try_snapshot returns False and the caller is expected to set
    needs_reload so the next reset_model() falls back to a disk reload.
    """

    def __init__(self):
        self._snapshots: dict[int, Tensor] = {}
        self._modules: dict[int, Module] = {}
        self.needs_reload = False

    def __len__(self) -> int:
        return len(self._snapshots)

    def __contains__(self, module: Module) -> bool:
        return id(module) in self._snapshots

    def cached_modules(self) -> list[Module]:
        return list(self._modules.values())

    def restore_all(self) -> None:
        with torch.no_grad():
            for mid, snapshot in self._snapshots.items():
                target = cast(Tensor, self._modules[mid].weight)
                if snapshot.device == target.device:
                    target.data.copy_(snapshot)
                else:
                    target.data.copy_(
                        snapshot.to(target.device, non_blocking=True)
                    )

    def free(self, modules: list[Module]) -> None:
        for module in modules:
            mid = id(module)
            self._snapshots.pop(mid, None)
            self._modules.pop(mid, None)

    def clear(self) -> None:
        # Required after any full reload: cached module ids are stale.
        self._snapshots.clear()
        self._modules.clear()
        self.needs_reload = False

    @staticmethod
    def _module_size_bytes(module: Module) -> int:
        weight = cast(Tensor, module.weight)
        return weight.numel() * weight.element_size()

    @classmethod
    def estimate_total_size(cls, modules: list[Module]) -> int:
        return sum(cls._module_size_bytes(m) for m in modules)

    @staticmethod
    def _gpu_has_room(device: torch.device, size_bytes: int) -> bool:
        if device.type != "cuda":
            return False
        try:
            free, _ = torch.cuda.mem_get_info(device)
        except Exception:
            return False
        return free >= size_bytes + _SNAPSHOT_SAFETY_MARGIN_BYTES

    def try_snapshot(self, modules: list[Module]) -> bool:
        """Either every module is snapshotted (returns True) or none is
        (returns False). Partial snapshots are never taken, to keep the
        cache invariant clean."""
        if not modules:
            return True

        # Tally size per source device so we make a single accept/reject
        # decision per device.
        size_per_device: dict[torch.device, int] = {}
        for module in modules:
            weight = cast(Tensor, module.weight)
            size_per_device[weight.device] = (
                size_per_device.get(weight.device, 0)
                + self._module_size_bytes(module)
            )

        keep_on_source_device: set[torch.device] = set()
        cpu_required_bytes = 0
        for device, size in size_per_device.items():
            if device.type == "cuda" and self._gpu_has_room(device, size):
                keep_on_source_device.add(device)
            else:
                cpu_required_bytes += size

        if cpu_required_bytes > 0:
            available = virtual_memory().available
            if (
                available
                < cpu_required_bytes + _SNAPSHOT_SAFETY_MARGIN_BYTES
            ):
                return False

        with torch.no_grad():
            for module in modules:
                weight = cast(Tensor, module.weight)
                target_device = (
                    weight.device
                    if weight.device in keep_on_source_device
                    else torch.device("cpu")
                )
                snapshot = weight.detach().to(target_device, copy=True)
                self._snapshots[id(module)] = snapshot
                self._modules[id(module)] = module

        return True


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    peft_config: LoraConfig

    def __init__(self, settings: Settings):
        self.settings = settings
        self.response_prefix = ""
        self.needs_reload = False
        # Caches original weights of ARA-touched modules so reset_model()
        # can restore them in O(memcpy) instead of reloading from disk.
        # Only populated when settings.use_ara is True.
        self._ara_cache = ARAWeightCache()
        # Tracks the dtype of the currently loaded model so reset_model()
        # can recover even if self.model has been nulled out by an
        # interrupted previous reset (e.g. Ctrl+C during disk reload).
        self._dtype: torch.dtype | None = None

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model,
            trust_remote_code=settings.trust_remote_code,
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
                print(f"* [red]Failed[/] ({error})")
                continue

            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("* Quantized to 4-bit precision")

            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        # Remember the dtype that worked, so reset_model() can recover
        # the disk-reload path even if self.model has been nulled out.
        self._dtype = self.model.dtype

        if not settings.use_ara:
            self._apply_lora()

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        all_components = {}
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                if component not in all_components:
                    all_components[component] = 0
                all_components[component] += len(modules)
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

        for layer_index, layer in enumerate(self.get_layers()):
            module_id_to_leaf_name = {
                id(module): module_name.split(".")[-1]
                for module_name, module in layer.named_modules()
            }

            for modules in self.get_layer_modules(layer_index).values():
                for module in modules:
                    if id(module) in module_id_to_leaf_name:
                        target_modules_set.add(module_id_to_leaf_name[id(module)])

        target_modules = list(target_modules_set)

        if self.settings.row_normalization != RowNormalization.FULL:
            # Rank 1 is sufficient for directional ablation without renormalization.
            lora_rank = 1
        else:
            # Row magnitude preservation introduces nonlinear effects.
            lora_rank = self.settings.full_normalization_lora_rank

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type="CAUSAL_LM",
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        print(f"* LoRA adapters initialized (targets: {', '.join(target_modules)})")

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
        - Recovery: If self.model is None (e.g. a previous reset was
          interrupted between purging self.model and from_pretrained), do
          a full disk reload using the last known good dtype.
        - Fast path (non-ARA): If the same model is loaded and doesn't
          need full reload, resets LoRA adapter weights to zero.
        - Fast path (ARA): If the same model is loaded and the snapshot
          cache is healthy, restores cached original weights via memcpy.
          No disk I/O.
        - Slow path: If switching models, after merge_and_unload(), or
          when the ARA cache reports it can't fully back the next trial,
          performs a full model reload with quantization config.
        """
        # Recovery path: a previous reset_model() may have been
        # interrupted (Ctrl+C, OOM) between ``self.model = None`` and
        # the from_pretrained call below, leaving us with no live model.
        # Reload using the dtype we recorded at the last successful load.
        if self.model is None:
            assert self._dtype is not None, (
                "self.model is None and no dtype recorded — this should "
                "be unreachable because __init__ always records a dtype "
                "after a successful load."
            )
            self._reload_from_disk(self._dtype)
            return

        current_model = getattr(self.model.config, "name_or_path", None)
        same_model = current_model == self.settings.model

        # Fast path (non-ARA): zero out LoRA adapters.
        if (
            same_model
            and not self.needs_reload
            and not self.settings.use_ara
        ):
            for name, module in self.model.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            return

        # Fast path (ARA): restore from in-memory snapshot cache.
        if (
            same_model
            and not self.needs_reload
            and self.settings.use_ara
            and not self._ara_cache.needs_reload
        ):
            if len(self._ara_cache) > 0:
                self._ara_cache.restore_all()
            return

        # Slow path.
        self._reload_from_disk(self.model.dtype)

    def _reload_from_disk(self, dtype: torch.dtype) -> None:
        """Drop self.model and re-instantiate it from disk with the given
        dtype. Used by the slow path of reset_model() and by the recovery
        path when self.model has been nulled out."""
        print("* Reloading weights from disk...")

        # Purge existing model object from memory to make space.
        self.model = None  # ty:ignore[invalid-assignment]
        # Module IDs captured in the ARA cache reference the model object
        # we just dropped, so the cache must be cleared before any reload.
        self._ara_cache.clear()
        empty_cache()

        quantization_config = self._get_quantization_config(str(dtype).split(".")[-1])

        # Build kwargs, only include quantization_config if it's not None
        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config

        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=self.trusted_models.get(self.settings.model),
            **extra_kwargs,
        )

        if not self.settings.use_ara:
            self._apply_lora()

        self._dtype = self.model.dtype
        self.needs_reload = False

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers

        # Text-only models.
        return model.model.layers

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            if component not in self.settings.target_components:
                return

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

        # Qwen3.5 MoE hybrid layers use GatedDeltaNet (linear attention) instead
        # of standard self-attention, so self_attn.o_proj doesn't exist on those layers.
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
        # Scan all layers because hybrid models (e.g. Qwen3.5 MoE) have different
        # components on different layers (some have self_attn, others linear_attn).
        components: set[str] = set()
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

    def _collect_ara_target_modules(
        self, parameters: ARAParameters
    ) -> list[Module]:
        """Enumerate every module that ara_abliterate will mutate for
        these parameters. Used by the snapshot cache planner.
        """
        target_modules: list[Module] = []
        for layer_index in range(
            parameters.start_layer_index,
            parameters.end_layer_index,
        ):
            for modules in self.get_layer_modules(layer_index).values():
                target_modules.extend(modules)
        return target_modules

    def _prepare_ara_cache(self, target_modules: list[Module]) -> None:
        """Diff the snapshot cache against this trial's targets:

        1. Drop snapshots for modules we won't touch (free memory).
        2. Snapshot newly-targeted modules (which currently hold
           originals, because reset_model() either restored them from
           the cache or freshly loaded them from disk).

        On insufficient memory, prints a warning and sets needs_reload
        so the next reset_model() falls back to a disk reload.
        """
        cache = self._ara_cache
        target_ids = {id(m) for m in target_modules}

        # Step 1: drop entries no longer needed.
        to_drop = [
            m for m in cache.cached_modules() if id(m) not in target_ids
        ]
        if to_drop:
            cache.free(to_drop)
            empty_cache()

        # Step 2: snapshot newly-targeted modules.
        to_add = [m for m in target_modules if m not in cache]
        if not to_add:
            return

        size_gb = cache.estimate_total_size(to_add) / (1024**3)

        if cache.try_snapshot(to_add):
            # Only print when the cache does meaningful work, to avoid
            # noise on small models where the snapshot is essentially free.
            if size_gb >= 0.1:
                print(
                    f"* Snapshotted [bold]{len(to_add)}[/] module(s) "
                    f"([bold]{size_gb:.2f} GB[/]) for fast reset"
                )
            return

        # Could not fit the planned snapshot on any tier. Run the trial
        # anyway, but flag the cache so the next reset_model() reloads
        # from disk to recover the originals we're about to overwrite.
        print(
            f"* [yellow]Not enough RAM to snapshot all targeted layers "
            f"(would need ~[bold]{size_gb:.2f} GB[/]). "
            f"Weights will be reloaded from disk after this iteration...[/]"
        )
        cache.needs_reload = True

    def ara_abliterate(
        self,
        good_module_io: ModuleIO,
        bad_module_io: ModuleIO,
        parameters: ARAParameters,
    ):
        # Plan/refresh the snapshot cache BEFORE any mutation, so the
        # invariant 'cache holds originals' is preserved. reset_model()
        # must have been called just before this, leaving every module
        # pristine.
        self._prepare_ara_cache(
            self._collect_ara_target_modules(parameters)
        )

        for layer_index in range(
            parameters.start_layer_index,
            parameters.end_layer_index,
        ):
            for component, modules in self.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    # See above for a (partial) justification of this cast.
                    module = cast(Linear, module)
                    matrix = module.weight

                    row_norms = LA.vector_norm(matrix, dim=1, keepdim=True).detach()

                    # Helper function for reparameterization (row-norm preservation constraint).
                    def get_matrix() -> Tensor:
                        if self.settings.row_normalization == RowNormalization.FULL:
                            # See https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                            return row_norms * F.normalize(matrix, p=2, dim=1)
                        else:
                            return matrix

                    good_input, good_output = good_module_io[layer_index][component][
                        module_index
                    ]
                    bad_input, bad_output = bad_module_io[layer_index][component][
                        module_index
                    ]

                    good_input = good_input.to(matrix.device)
                    good_output = good_output.to(matrix.device)
                    bad_input = bad_input.to(matrix.device)
                    bad_output = bad_output.to(matrix.device)

                    def objective(matrix: Tensor) -> Tensor:
                        new_good_output = good_input @ matrix.T
                        new_bad_output = bad_input @ matrix.T

                        # The outputs for "good" prompts should change as little as possible.
                        preserve_good_behavior = (
                            (new_good_output - good_output) ** 2
                        ).mean()

                        steer_bad_behavior = (
                            # Pull the outputs for "bad" prompts towards
                            # the original outputs for "good" prompts.
                            mean_distances_to_knn(
                                new_bad_output,
                                good_output,
                                parameters.neighbor_count,
                            ).mean()
                            # Push the outputs for "bad" prompts away from
                            # the original outputs for "bad" prompts.
                            # In combination with the above, this overcorrects
                            # away from the original residuals, which results
                            # in stronger steering that can overcome more complex
                            # refusal mechanisms.
                            + parameters.overcorrect_relative_weight
                            * -mean_distances_to_knn(
                                new_bad_output,
                                bad_output,
                                parameters.neighbor_count,
                            ).mean()
                        )

                        return (
                            parameters.preserve_good_behavior_weight
                            * preserve_good_behavior
                            + parameters.steer_bad_behavior_weight * steer_bad_behavior
                        )

                    optimizer = LBFGS(
                        [matrix],
                        lr=1.0,
                        max_iter=20,  # Number of internal iterations per step, *not* the number of steps.
                        history_size=10,
                        line_search_fn="strong_wolfe",
                    )

                    def closure() -> Tensor:
                        optimizer.zero_grad()
                        loss = objective(get_matrix())
                        loss.backward()
                        return loss

                    # Convergence usually happens within 2-3 steps, so this is more than enough.
                    for step in range(5):
                        loss = optimizer.step(closure)
                        # print(
                        #    f"\\[{layer_index}/{component}/{module_index}] Step: {step}, Loss: {loss.item():.6f}"
                        # )

                    with torch.no_grad():
                        matrix.copy_(get_matrix())

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
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

        if self.response_prefix:
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [prompt + self.response_prefix for prompt in chat_prompts]

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

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

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
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

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
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
            return torch.clamp(residuals, -thresholds, thresholds)

        return residuals

    def get_residuals_batched(self, prompts: list[Prompt]) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))

        return torch.cat(residuals, dim=0)

    def get_module_io(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        # The list contains one element per layer.
        # Each element maps from the component name to a (possibly sparse) mapping
        # from the module index to an (input, output) tuple containing the I/O
        # tensors of shape (prompt, component).
        module_io: ModuleIO = []

        def get_hook(
            layer_index: int,
            component: str,
            module_index: int,
        ) -> Callable[[Module, tuple[Tensor, ...], Tensor], None]:
            def hook(
                module: Module,
                inputs: tuple[Tensor, ...],
                outputs: Tensor,
            ) -> None:
                if len(module_io) == layer_index:
                    # First invocation of the hook for this layer.
                    module_io.append({})

                # Layers are invoked in order during inference,
                # so this should always hold.
                assert len(module_io) == layer_index + 1

                if component not in module_io[layer_index]:
                    module_io[layer_index][component] = {}

                # Each module should be invoked at most once per inference step.
                assert module_index not in module_io[layer_index][component]

                # inputs[0] and outputs have shape (prompt, position, component),
                # so this extracts the input/output at the end of each prompt.
                # Move to CPU to decouple from device assignments, which can
                # change between model reloads in multi-GPU configurations.
                input = inputs[0][:, -1, :].detach().clone().cpu()
                output = outputs[:, -1, :].detach().clone().cpu()

                # The modules associated with a component (e.g. expert MLPs)
                # are not necessarily invoked in order, nor are all of them
                # necessarily invoked in each inference step, so we cannot
                # use a list here.
                module_io[layer_index][component][module_index] = (input, output)

            return hook

        hook_handles: list[RemovableHandle] = []

        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    hook_handles.append(
                        module.register_forward_hook(
                            get_hook(layer_index, component, module_index)
                        )
                    )

        self.generate(prompts, max_new_tokens=1)

        for hook_handle in hook_handles:
            hook_handle.remove()

        return module_io

    def get_module_io_batched(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        # Aggregating batch results is more complicated for module I/O
        # than for other get_*_batched methods, because the structure of the results
        # might differ between batches, as whether individual modules activate
        # can depend on the prompt (in particular for MoE models).
        # In practice, inhomogeneous results should be very rare, but to be fully
        # generic, this logic is required.
        module_io_batches: list[ModuleIO] = [
            self.get_module_io(batch)
            for batch in batchify(prompts, self.settings.batch_size)
        ]

        module_io: ModuleIO = []

        for layer_index in range(len(self.get_layers())):
            module_io.append({})

            for module_io_batch in module_io_batches:
                for component, io_map in module_io_batch[layer_index].items():
                    if component not in module_io[layer_index]:
                        module_io[layer_index][component] = {}

                    for module_index in io_map:
                        if module_index not in module_io[layer_index][component]:
                            # This is a placeholder; the actual aggregation happens below.
                            # We need to iterate over the batches twice because we don't
                            # know in advance which components and module indices are present.
                            module_io[layer_index][component][module_index] = (
                                torch.empty(0),
                                torch.empty(0),
                            )

            for component, io_map in module_io[layer_index].items():
                for module_index in io_map:
                    inputs_outputs = [
                        module_io_batch[layer_index][component][module_index]
                        for module_io_batch in module_io_batches
                        if component in module_io_batch[layer_index]
                        and module_index in module_io_batch[layer_index][component]
                    ]
                    input = torch.cat(
                        [input_output[0] for input_output in inputs_outputs],
                        dim=0,
                    )
                    output = torch.cat(
                        [input_output[1] for input_output in inputs_outputs],
                        dim=0,
                    )

                    # The key already exists, and replacing existing values
                    # in a dictionary while iterating over the same dictionary
                    # is safe in Python.
                    module_io[layer_index][component][module_index] = (input, output)

        return module_io

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the (log) probability distributions
        # over the vocabulary at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # This cast is valid because we passed output_scores=True above.
        logits = cast(tuple[FloatTensor], outputs.scores)[0]

        # The returned tensor has shape (prompt, token).
        return F.log_softmax(logits, dim=-1)

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
