# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import copy
import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Type, cast

import bitsandbytes as bnb
import torch
import torch.linalg as LA
import torch.nn as nn
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

from .config import QuantizationMethod, RowNormalization, Settings
from .system import (
    _get_tpu_core_count_from_env,
    detect_tpu,
    empty_cache,
    get_xla_device,
    mark_step,
    setup_tpu_environment,
)
from .utils import Prompt, batchify, format_exception, print


def _is_torch_xla_available() -> bool:
    try:
        import torch_xla  # noqa: F401
        return True
    except ImportError:
        return False


def _get_torch_xla():
    """Lazy import of torch_xla."""
    import torch_xla.core.xla_model as xm
    return xm


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


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    # Set for multimodal models, None for text-only ones.
    processor: ProcessorMixin | None
    peft_config: LoraConfig
    dtype: torch.dtype
    _is_tpu: bool
    _xla_device: torch.device | None

    def __init__(self, settings: Settings):
        self.settings = settings
        self.needs_reload = False
        self._is_tpu = detect_tpu()
        self._xla_device = None

        if self._is_tpu:
            # Resolve auto-detected TPU parallelism before computing use_fsdp.
            # Client init must stay after setup_tpu_environment so the SPMD
            # flag is in place (see _ensure_spmd_if_multichip).
            if settings.tpu_cores is None:
                # Auto-detection: use the fewest cores that fit the model.
                # Fewer cores mean less FSDP all-gather overhead per XLA
                # execution, which dominates latency for small models (a
                # 3B model generates ~9x faster on 4 cores than on 8).
                settings.tpu_cores = Model.auto_tpu_cores(settings)
            if settings.tpu_use_fsdp is None:
                settings.tpu_use_fsdp = settings.tpu_cores > 1
            use_fsdp = (
                settings.tpu_use_fsdp
                and settings.tpu_cores > 1
            )
            setup_tpu_environment(enable_spmd=use_fsdp)
            settings = settings.adjust_for_tpu()
            # SPMD must be enabled before ANY device access; it is only needed
            # for the single-process multi-core FSDP path. Single-core runs must
            # NOT use SPMD: the deviceless virtual device breaks memory probing
            # and tensor fetches accumulate (null data crashes after ~6 steps).
            self._xla_device = get_xla_device(0, enable_spmd=use_fsdp)
        else:
            use_fsdp = False

        self.revision_kwargs = {}
        if settings.model_commit is not None:
            self.revision_kwargs["revision"] = settings.model_commit

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model,
            **self.revision_kwargs,
        )

        # Multimodal models have a processor we'll want to save.
        self.processor = None
        if get_model_class(settings.model) == AutoModelForImageTextToText:
            self.processor = AutoProcessor.from_pretrained(
                settings.model,
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

        self.trusted_models = set()

        # FSDP mode: shard a CPU-loaded model across all TPU cores.
        # torch_xla 2.8: single-process multi-core setups (Kaggle/Colab TPU VMs)
        # use the SPMD variant, which accepts bf16 parameters; multi-process jobs
        # use classic FSDP, which requires fp32 parameters (compute stays bf16).
        use_fsdp = (
            self._is_tpu
            and settings.tpu_use_fsdp
            and settings.tpu_cores > 1
        )

        # On TPU, force bfloat16 and disable quantization
        effective_dtypes = settings.dtypes
        effective_quantization = settings.quantization
        if self._is_tpu:
            if use_fsdp:
                from .fsdp_utils import fsdp_uses_spmd

                effective_dtypes = ["bfloat16" if fsdp_uses_spmd() else "float32"]
            else:
                effective_dtypes = ["bfloat16"]
            effective_quantization = QuantizationMethod.NONE
            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("* [yellow]bitsandbytes quantization not supported on TPU. Using bfloat16 instead.[/]")

        for dtype in effective_dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]...")

            try:
                quantization_config = self._get_quantization_config(dtype)

                extra_kwargs = {}
                # Only include quantization_config if it's not None
                # (some models like gpt-oss have issues with explicit None).
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config

                # For TPU, device_map="auto" works with Accelerate's Big Model Inference.
                # With FSDP, the model must load on CPU so the wrapper can shard it.
                device_map = None if use_fsdp else settings.device_map

                self.model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=True
                    if settings.model in self.trusted_models
                    else None,
                    **self.revision_kwargs,
                    **extra_kwargs,
                )

                self.dtype = self.model.dtype

                # If we reach this point and the model requires trust_remote_code,
                # the user must have agreed when prompted to execute remote code,
                # because from_pretrained raises an exception otherwise.
                self.trusted_models.add(settings.model)

                # Shard the model with FSDP BEFORE attaching LoRA adapters,
                # so the adapter parameters stay unsharded (replicated on every
                # core) and can be updated directly by apply_abliteration().
                if use_fsdp:
                    from .fsdp_utils import wrap_model_fsdp

                    print("* Sharding model with FSDP across TPU cores...")
                    self.model = wrap_model_fsdp(
                        self.model, config=settings.tpu_fsdp_config
                    )

                # On TPU, move model to XLA device if device_map didn't handle it
                # (skipped in FSDP mode - the wrapper handles device placement).
                if (
                    self._is_tpu
                    and self._xla_device is not None
                    and not use_fsdp
                    and not str(next(self.model.parameters()).device).startswith("xla")
                ):
                    self.model = self.model.to(self._xla_device)

                # CRITICAL for TPU: the model must be in eval mode. Without it,
                # dropout/RNG ops stay in the graph, so every forward produces a
                # slightly different HLO and XLA recompiles per step. Each fresh
                # executable keeps its own TPU memory plan (~2.4GB for a 0.5B
                # model), so after ~6-7 steps the 16.9GB HBM is exhausted,
                # executions fail, and tensor fetches crash with null data.
                self.model.eval()

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                #
                # On TPU, the test run triggers XLA compilation which can produce
                # harmless PJRT cleanup errors during process exit (version mismatch
                # between torch_xla and libtpu). Skip the test on TPU since XLA
                # compilation happens naturally on first real use anyway.
                if not self._is_tpu:
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
                    print(f"* [red]Failed:\n{formatted}[/]")
                else:
                    print(f"* [red]Failed ({formatted})[/]")

                continue

            if effective_quantization == QuantizationMethod.BNB_4BIT:
                print("* Quantized to 4-bit precision")

            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        self._apply_lora()

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.

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

    @staticmethod
    def _estimate_model_footprint_gb(model: str, revision_kwargs: dict[str, Any]) -> float:
        """Estimate the bf16 footprint of a model on the meta device (no weights)."""
        with suppress(Exception):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_model = get_model_class(model).from_pretrained(
                    model,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    **revision_kwargs,
                )
                return meta_model.get_memory_footprint() / (1024**3)
        return 0.0

    @staticmethod
    def auto_tpu_cores(settings: Settings) -> int:
        """Fewest TPU cores that fit the model, power-of-two, 2-core floor.

        On v5e-8 each core has 16GB HBM; ~12GB is usable after XLA overhead.
        Single-core plain-mode XLA is unreliable on v5e (null-data tensor
        fetches after a few steps), so we never auto-select one core. This
        matters for speed: FSDP all-gather overhead per XLA execution grows
        with the number of cores, so small models run much faster on fewer
        cores (measured ~9x on a 3B model: 4 cores vs 8).
        """
        available = _get_tpu_core_count_from_env() or 1
        if available <= 1:
            return 1
        revision_kwargs: dict[str, Any] = {}
        if settings.model_commit is not None:
            revision_kwargs["revision"] = settings.model_commit
        footprint_gb = Model._estimate_model_footprint_gb(settings.model, revision_kwargs)
        if footprint_gb <= 0:
            # Estimation failed - fall back to all cores (safe for big models).
            return available
        budget_per_core = 12.0
        needed = max(2, math.ceil(footprint_gb / budget_per_core))
        cores = 1
        while cores < needed:
            cores *= 2
        return min(cores, available)

    def _model_device(self) -> torch.device:
        """Resolve the device of the current model, tolerating FSDP wrappers."""
        device = getattr(self.model, "device", None)
        if isinstance(device, torch.device):
            return device
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def _model_dtype(self) -> torch.dtype:
        """Resolve the dtype of the current model, tolerating FSDP wrappers."""
        dtype = getattr(self.model, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        for param in self.model.parameters():
            return param.dtype
        return torch.float32

    def _collect_target_modules(self, model: nn.Module | None = None) -> list[str]:
        """Collect leaf module full names for LoRA targeting (FSDP-tolerant).

        Uses the same module traversal as get_layer_modules() against the given
        model (default: self.model). Names are read back from named_modules() of
        that exact model, so FSDP's in-place "_orig_module" wrappers are included
        on the sharded model and absent on a plain CPU reload - each model gets
        targets that match its own naming.
        """
        if model is None:
            model = self.model
        if model is self.model:
            saved_model = None
        else:
            saved_model = self.model
            self.model = model
        try:
            # get_layers()/get_layer_modules() read self.model, so the swap
            # above must stay in place for the whole traversal. Restoring it
            # earlier would walk the sharded model's layers against this
            # model's module ids, yielding an empty target list.
            module_id_to_full_name = {
                id(module): module_name
                for module_name, module in model.named_modules()
            }

            target_modules_set: set[str] = set()
            for layer_index in range(len(self.get_layers())):
                for modules in self.get_layer_modules(layer_index).values():
                    for module in modules:
                        full_name = module_id_to_full_name.get(id(module))
                        if full_name is not None:
                            target_modules_set.add(full_name)

            return sorted(target_modules_set)
        finally:
            if saved_model is not None:
                self.model = saved_model

    def _apply_lora(self):
        # Guard against calling this method at the wrong time.
        # (After FSDP wrapping, self.model is an FSDP wrapper, not a
        # PreTrainedModel, so check for any nn.Module here.)
        assert isinstance(self.model, nn.Module) and not isinstance(
            self.model, PeftModel
        )

        # Always use LoRA adapters for abliteration (faster reload, no weight modification).
        # Collect actual leaf module names from the model for LoRA targeting.
        # This is more robust than splitting component keys (e.g. "attn.o_proj" -> "o_proj")
        # because hybrid models like Qwen3.5 MoE have modules with different names
        # across layers (e.g. "o_proj" on attention layers, "out_proj" on linear attention layers).
        target_modules = self._collect_target_modules()

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
        # bitsandbytes quantization not supported on TPU
        if self._is_tpu:
            return None

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

        # On TPU (especially FSDP-sharded), merging on-device is unreliable.
        # The CPU reload path below works everywhere and is the safe choice:
        # quantized models need it anyway, and TPU models merge cleanly on CPU.
        if self.settings.quantization == QuantizationMethod.BNB_4BIT or self._is_tpu:
            # Quantized models need special handling - we must reload the base model
            # in full precision to merge the LoRA adapters

            # On TPU, materialize the model to CPU first. Cloning/copying XLA lazy
            # tensors after abliteration fails with OSError EPERM (errno 1) when HBM
            # is near capacity, and CPU merging avoids TPU device issues entirely.
            if self._is_tpu:
                print("* Moving model to CPU...")
                self.model = self.model.to("cpu")

            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    # FSDP wraps modules in-place and exposes the wrapped module
                    # as "_orig_module", which pollutes parameter names (e.g.
                    # "..._orig_module.mlp.down_proj..."). Strip it so the keys
                    # match the canonical names of the freshly built PEFT model
                    # below; otherwise every adapter copy is silently skipped and
                    # the "merged" model is byte-identical to the base.
                    adapter_state[name.replace("_orig_module.", "")] = (
                        param.data.clone()
                    )

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self._model_dtype(),
                device_map="cpu",
                trust_remote_code=True
                if self.settings.model in self.trusted_models
                else None,
                **self.revision_kwargs,
            )

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            # Re-derive the LoRA target list from the CPU base model itself.
            # self.peft_config.target_modules was collected from the sharded
            # (FSDP) model and contains "_orig_module" segments that do not
            # exist on the plain reload; reusing it makes PEFT match only a
            # subset of modules (e.g. o_proj but not mlp.down_proj) and the
            # merged model silently misses entire components.
            peft_config = copy.deepcopy(self.peft_config)
            peft_config.target_modules = self._collect_target_modules(base_model)
            peft_model = get_peft_model(base_model, peft_config)

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

        # If a prior model load was interrupted/cancelled mid-process, self.model will be None.
        current_model = None
        if self.model is not None:
            current_model = getattr(self.model.config, "name_or_path", None)

        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation).
            for name, module in self.model.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            return

        # Purge existing model object from memory to make space.
        self.model = None  # ty:ignore[invalid-assignment]
        empty_cache()

        quantization_config = self._get_quantization_config(
            str(self.dtype).split(".")[-1]
        )

        # Build kwargs, only include quantization_config if it's not None.
        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config

        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=self.dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=True
            if self.settings.model in self.trusted_models
            else None,
            **self.revision_kwargs,
            **extra_kwargs,
        )

        self._apply_lora()

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
        residual_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        if direction_index is None:
            residual_direction = None
        else:
            # The index must be shifted by 1 because the first element
            # of residual_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)
            residual_direction = F.normalize(
                residual_directions[int(index)].lerp(
                    residual_directions[int(index) + 1],
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

                # A weight of 0 disables this component's ablation. reset_model() has
                # already left the adapter at identity, so abort before the otherwise
                # wasteful decomposition (which would also be operating on a zero matrix).
                if weight == 0:
                    continue

                if residual_direction is None:
                    # The index must be shifted by 1 because the first element
                    # of residual_directions is the direction for the embeddings.
                    layer_residual_direction = residual_directions[layer_index + 1]
                else:
                    layer_residual_direction = residual_direction

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

                    # Use the FP32 residual direction directly (no downcast/upcast)
                    # and move to the correct device.
                    v = layer_residual_direction.to(module.weight.device)

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
                        # 4-bit quantization (bitsandbytes).
                        # On TPU, quantization is disabled so this shouldn't happen,
                        # but handle gracefully just in case.
                        if self._is_tpu:
                            raise RuntimeError(
                                "4-bit quantized model detected on TPU. "
                                "This should not happen as quantization is disabled on TPU."
                            )
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

                    if self.settings.row_normalization == RowNormalization.FULL:
                        # Keep a reference to the original weight matrix so we can subtract it later.
                        W_org = W

                    if self.settings.row_normalization != RowNormalization.NONE:
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

                        # svd_lowrank is randomized:
                        # https://github.com/pytorch/pytorch/blob/20919052303c0b5ba87f8bf7e19237dc33ab09d3/torch/_lowrank.py#L108-L109
                        # Reseed immediately before the call so restoring a trial is independent of RNG history.
                        torch.manual_seed(self.settings.seed)
                        # "It's safe to call this function if CUDA is not available;
                        # in that case, it is silently ignored."
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(self.settings.seed)  # ty:ignore[invalid-argument-type]
                        elif self._is_tpu and _is_torch_xla_available():
                            import torch_xla.core.xla_model as xm
                            xm.set_rng_state(self.settings.seed, device=self._xla_device)
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

    def _tokenize_prompts(self, prompts: list[Prompt]) -> BatchEncoding:
        """Tokenize prompts with chat template, returning inputs on model device."""
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

        # On TPU, force fixed max_length so every batch has the same shape.
        # Without this, XLA recompiles a new graph for each unique (batch, seq_len),
        # causing unbounded memory growth. 128 covers prompt (~60 tokens) plus
        # max_response_length (20) with margin; larger values only slow down
        # XLA compilation and steady-state throughput without benefit.
        # padding="max_length" (not just max_length+truncation) is critical:
        # plain padding pads each batch to its own longest sequence, so batches
        # of prompts with different lengths produce different shapes and XLA
        # recompiles per batch.
        tokenizer_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": "max_length" if self._is_tpu else True,
            "return_token_type_ids": False,
        }
        if self._is_tpu:
            tokenizer_kwargs["max_length"] = 128
            tokenizer_kwargs["truncation"] = True

        inputs = self.tokenizer(
            chat_prompts,
            **tokenizer_kwargs,
        ).to(self._model_device())

        return inputs

    def forward(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> Any:
        """Direct forward pass — XLA-compatible, no generation loop.

        Unlike generate(), this calls model(**inputs) directly, which is
        pure matrix multiplication and fully traceable by XLA. Use this
        when you only need logits or hidden states for a single position.
        """
        inputs = self._tokenize_prompts(prompts)
        outputs = self.model(**inputs, **kwargs)
        mark_step()
        return inputs, outputs

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        inputs = self._tokenize_prompts(prompts)

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        # Mark step for XLA lazy execution
        mark_step()

        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> list[str]:
        if self._is_tpu:
            return self._get_responses_xla(
                prompts,
                skip_special_tokens=skip_special_tokens,
            )

        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
            **kwargs,
        )

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def _get_responses_xla(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """XLA-compatible greedy generation using a manual loop.

        Avoids model.generate() which uses Python control flow that forces
        XLA to fall back to CPU. Uses a fixed-size pre-allocated buffer
        so the tensor shape never changes — this lets XLA compile the
        entire loop into a single traced graph.
        """
        inputs = self._tokenize_prompts(prompts)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        max_new = self.settings.max_response_length
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device
        total_len = prompt_len + max_new

        # Pre-allocate fixed-size buffers. Shape NEVER changes.
        full_ids = torch.zeros(batch_size, total_len, device=device, dtype=input_ids.dtype)
        full_ids[:, :prompt_len] = input_ids
        full_mask = torch.zeros(batch_size, total_len, device=device, dtype=attention_mask.dtype)
        full_mask[:, :prompt_len] = attention_mask

        # Track which sequences have finished.
        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
        eos_tensor = torch.tensor(eos_id, device=device, dtype=torch.long) if eos_id is not None else None

        for _ in range(max_new):
            # Always pass the full fixed-size tensor — XLA sees the same shape.
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_mask,
            )
            # Read logits at the LAST position. This index must stay a Python
            # constant across steps: XLA bakes integer constants into the
            # compiled HLO, so a varying index (cur_pos) produces a different
            # graph per decode step (~25s compile each = ~8 min for 20 tokens).
            # A constant index yields ONE compiled graph reused by all steps.
            next_logits = outputs.logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1)

            # If already finished, emit pad token.
            if eos_tensor is not None:
                next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)

            # Shift the window one token left (constant-shape op) and write the
            # new token at the fixed last position. Keeps every step's graph
            # identical, so the whole decode loop compiles once.
            full_ids[:, :-1] = full_ids[:, 1:]
            full_ids[:, -1] = next_token
            full_mask[:, :-1] = full_mask[:, 1:]
            full_mask[:, -1] = 1

            # Update finished state.
            if eos_tensor is not None:
                finished = finished | (next_token == eos_tensor)

            # mark_step per step: keeps each execution small (bounded HBM;
            # a single unrolled graph for all steps instead exhausts HBM via
            # per-step logits buffers and thrashes evictions).
            mark_step()

        mark_step()

        new_tokens = full_ids[:, -max_new:]
        return self.tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        # On TPU, execute prompts in fixed chunks. Every XLA execution costs
        # ~2.4GB of device memory that is never reclaimed, so only ~5-6
        # executions fit in the 16.9GB HBM; a single 100-prompt x 100-token
        # generation graph exhausts it (bench: chunk=20 OK at 91s, chunk=40
        # fails with RESOURCE_EXHAUSTED). Chunking is a BATCHING change, not a
        # computation change: the same responses are produced, just in
        # independent graphs. empty_cache between chunks reclaims the SVD
        # residue that otherwise forces the next generation to recompile.
        if self._is_tpu:
            responses = []
            for i in range(0, len(prompts), 20):
                responses.extend(
                    self.get_responses(
                        prompts[i : i + 20],
                        skip_special_tokens=skip_special_tokens,
                    )
                )
                empty_cache()
            return responses

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
        if self._is_tpu:
            # XLA-compatible path: direct forward pass avoids generate()'s
            # Python control flow that forces XLA to fall back to CPU.
            _, outputs = self.forward(
                prompts,
                output_hidden_states=True,
                use_cache=False,
            )
        else:
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

        if self._is_tpu:
            # hidden_states is a tuple of (batch, seq_len, hidden_size) per layer.
            # We want the last position of the input sequence (the generation point).
            hidden_states = outputs.hidden_states
        else:
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

        # Mark step for XLA lazy execution
        mark_step()

        if self.settings.offload_outputs_to_cpu or self._is_tpu:
            # On TPU, offloading is mandatory: same device-memory accumulation
            # issue as get_logits (retained hidden states exhaust HBM after a
            # few batches and later executions fail with null tensor data).
            del outputs
            residuals = residuals.cpu()
            empty_cache()

        return residuals

    def get_residuals_batched(self, prompts: list[Prompt]) -> Tensor:
        # On TPU, single execution for all prompts: see get_responses_batched.
        if self._is_tpu:
            return self.get_residuals(prompts)

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

    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the raw logits over the vocabulary
        # at that token position, for each prompt.
        if self._is_tpu:
            # XLA-compatible path: direct forward pass avoids generate()'s
            # Python control flow that forces XLA to fall back to CPU.
            _, outputs = self.forward(
                prompts,
                use_cache=False,
            )
            logits = outputs.logits[:, -1, :]
        else:
            _, outputs = self.generate(
                prompts,
                max_new_tokens=1,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=False,
            )
            outputs = cast(GenerateDecoderOnlyOutput, outputs)
            # Logits for the first (only) generated token.
            # Use raw logits, not processed generation scores; processors can insert
            # -inf for suppressed tokens, which can make KL divergence evaluate to NaN.
            # This cast is valid because we passed output_logits=True above.
            logits = cast(tuple[FloatTensor], outputs.logits)[0]

        # Mark step for XLA lazy execution
        mark_step()

        # The returned tensor has shape (prompt, token).
        if self.settings.offload_outputs_to_cpu or self._is_tpu:
            # On TPU, offloading is mandatory, not optional: retaining each
            # batch's full logits tensor on the device accumulates ~155MB per
            # batch, and once roughly six batches are held, the TPU HBM is
            # exhausted and subsequent executions fail silently, leaving
            # tensors with null data. The next shape access then crashes with
            # "Check failed: data->tensor_data". Fetching to CPU immediately
            # frees device memory each iteration; the host has plenty of RAM.
            del outputs
            logits = logits.cpu()
            empty_cache()

        return logits

    def get_logits_batched(self, prompts: list[Prompt]) -> Tensor:
        # On TPU, single execution for all prompts: see get_responses_batched.
        if self._is_tpu:
            return self.get_logits(prompts)

        logits = []

        for batch in batchify(prompts, self.settings.batch_size):
            logits.append(self.get_logits(batch))

        return torch.cat(logits, dim=0)

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
