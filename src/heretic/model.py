# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# ROCm PyTorch compatibility: torch.distributed.tensor doesn't exist in ROCm builds
# PEFT checks for this in tuners_utils.py, causing AttributeError
# Add a dummy module at the very top to prevent the crash
import torch
if hasattr(torch, 'distributed') and not hasattr(torch.distributed, 'tensor'):
    class _DummyDTensor:
        pass
    torch.distributed.tensor = type('tensor', (), {'DTensor': _DummyDTensor})()

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Type, cast

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
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
)

from .config import QuantizationMethod, RowNormalization, Settings
from .utils import Prompt, batchify, empty_cache, print

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
    peft_config: LoraConfig

    def __init__(self, settings: Settings):
        self.settings = settings
        self.response_prefix = settings.response_prefix
        self.needs_reload = False

        print(f"Loading model [bold]{settings.model}[/]... [grey50](Engine v1.2.1-hybrid-fix)[/]")

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model,
            trust_remote_code=settings.trust_remote_code,
            token=settings.hf_token,
        )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        self.tokenizer.padding_side = "left"

        self.model = None
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

            try:
                quantization_config = self._get_quantization_config(dtype)

                extra_kwargs = {}
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config

                self.model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=settings.trust_remote_code,
                    token=settings.hf_token,
                    **extra_kwargs,
                )

                # Test run to detect dtype issues
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
                self.model = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue

            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("[green]Ok[/] (quantized to 4-bit precision)")
            else:
                print("[green]Ok[/]")

            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        self._apply_lora()

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        self.all_components = {}
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                if component not in self.all_components:
                    self.all_components[component] = 0
                self.all_components[component] += len(modules)
        for component, count in self.all_components.items():
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

        if "ssm.out_proj" not in self.all_components:
            print("[yellow]WARNING: No SSM components found! This model may not be fully abliterated.[/] If this is a hybrid model, check the architecture definition.")

    def _apply_lora(self):
        assert isinstance(self.model, PreTrainedModel)

        # Collect actual leaf module names for LoRA targeting
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

        # PEFT Workaround: Bypass the Mamba-based model check that forbids 'out_proj'
        # This allows LoRA on hybrid architectures like Falcon-H1.
        import peft.tuners.tuners_utils
        peft.tuners.tuners_utils._check_lora_target_modules_mamba = lambda *args, **kwargs: None

        if self.settings.row_normalization != RowNormalization.FULL:
            lora_rank = 1
        else:
            lora_rank = self.settings.full_normalization_lora_rank

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))
        self.model.eval()
        self.model.requires_grad_(False)

        print(f"* LoRA adapters initialized (targets: {', '.join(target_modules)})")

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
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
        assert isinstance(self.model, PeftModel)

        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            print("* Loading base model on CPU (this may take a while)...")
            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=self.settings.trust_remote_code,
            )

            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base_model, self.peft_config)

            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            print("* Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            self.needs_reload = True
            return merged_model

    def reset_model(self):
        current_model = getattr(self.model.config, "name_or_path", None)
        if current_model == self.settings.model and not self.needs_reload:
            for name, module in self.model.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            return

        dtype = self.model.dtype
        self.model = None
        empty_cache()

        quantization_config = self._get_quantization_config(str(dtype).split(".")[-1])

        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config

        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=self.settings.trust_remote_code,
            **extra_kwargs,
        )

        self._apply_lora()
        self.needs_reload = False

    def get_layers(self) -> ModuleList:
        model = self.model

        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Check for common hybrid model structures (e.g., Falcon-H1R, Mamba-2)
        # Search for any attribute that is a ModuleList and likely contains the layers.
        # We start with known common paths for efficiency.
        with suppress(Exception):
            return model.model.language_model.layers
        
        with suppress(Exception):
            return model.model.layers

        with suppress(Exception):
            return model.transformer.h

        # More general search for the layer list
        for name, module in model.named_modules():
            if isinstance(module, ModuleList) and name.endswith(".layers"):
                return module

        # This should cover most Transformers-compatible models
        if hasattr(model, "layers") and isinstance(model.layers, ModuleList):
            return model.layers

        raise Exception("Could not find model layers")

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        """
        Returns a mapping from logical component names (e.g. "attn.o_proj",
        "mlp.down_proj", "ssm.out_proj") to the corresponding nn.Modules
        in a given transformer layer.

        Enhanced with:
        - Deduplication by module id
        - Shape validation for SSM components (if hidden_size available)
        - Recursive search for out_proj within known SSM block names
        - Debug prints to aid architecture discovery
        """
        layer = self.get_layers()[layer_index]

        modules: dict[str, list[Module]] = {}
        seen_ids = set()

        hidden_size = getattr(self.model.config, "hidden_size", None)

        def try_add(component: str, module: Any) -> None:
            if not isinstance(module, Module):
                assert not isinstance(module, Tensor), \
                    f"Unexpected Tensor in {component} - expected nn.Module"
                return

            mod_id = id(module)
            if mod_id in seen_ids:
                return
            seen_ids.add(mod_id)

            # Shape validation for SSM components (ensure it's the residual projection)
            if component == "ssm.out_proj" and hidden_size is not None:
                out_features = getattr(module, "out_features", None)
                if out_features is not None and out_features != hidden_size:
                    # Not the right projection (likely input/gate)
                    return

            modules.setdefault(component, []).append(module)

        # 1. Standard attention out-projections
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_proj)
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.out_proj)
        with suppress(Exception):
            try_add("attn.o_proj", layer.linear_attn.out_proj)

        # 2. MLP down-projections (using recursive search for all feed-forward components)
        # We look for all Linear modules within 'mlp' or 'feed_forward' blocks
        # that have the correct output dimension (hidden_size).
        for block_name in ["mlp", "feed_forward", "block_sparse_moe", "moe"]:
            if hasattr(layer, block_name):
                block = getattr(layer, block_name)
                for name, module in block.named_modules():
                    # Skip LoRA internals if LoRA is already applied
                    if "base_layer" in name or "lora_" in name:
                        continue
                    if isinstance(module, torch.nn.Linear):
                        # The final projection in an MLP (down-proj) restores the hidden size.
                        if hidden_size is not None and module.out_features == hidden_size:
                            try_add("mlp.down_proj", module)
                        # Fallback: support common names if hidden_size is unknown or mismatched
                        elif any(x in name.lower() for x in ["down_proj", "w2", "dense_4h_to_h", "output_linear", "out_proj"]):
                            try_add("mlp.down_proj", module)

        # 3. SSM/Mamba/Recurrent blocks (recursive search)
        for name, module in layer.named_modules():
            # Skip LoRA internals if LoRA is already applied
            if "base_layer" in name or "lora_" in name:
                continue
            name_lower = name.lower()
            if any(x in name_lower for x in ["ssm", "mamba", "mixer", "recurrent", "hydra", "scan"]):
                for subname, submod in module.named_modules():
                    # Skip internals of internals
                    if "base_layer" in subname or "lora_" in subname:
                        continue
                    # Look for the final projection restoring hidden size
                    if isinstance(submod, torch.nn.Linear) or "Linear" in submod.__class__.__name__:
                        out_f = getattr(submod, "out_features", None)
                        if hidden_size is not None and out_f == hidden_size:
                            try_add("ssm.out_proj", submod)
                        elif subname.endswith(("out_proj", "output", "proj", "dense")):
                            try_add("ssm.out_proj", submod)

        # ---- Final check ----
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, f"No abliterable modules found in layer {layer_index}"

        return modules

    def get_abliterable_components(self) -> list[str]:
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
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                params = parameters[component]

                distance = cast(float, abs(layer_index - params.max_weight_position))

                if distance > params.min_weight_distance:
                    continue

                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                for module in modules:
                    # Ensure module is a Linear-like with base_layer
                    # (PEFT wraps original module)
                    module = cast(Linear, module)

                    v = layer_refusal_direction.to(module.weight.device)

                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W = base_weight.to(torch.float32)
                    else:
                        W = cast(
                            Tensor,
                            bnb.functional.dequantize_4bit(
                                base_weight.data,
                                quant_state,
                            ).to(torch.float32),
                        )

                    W = W.view(W.shape[0], -1)

                    if self.settings.row_normalization != RowNormalization.NONE:
                        W_org = W
                        W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                        W = F.normalize(W, p=2, dim=1)

                    lora_A = (v @ W).view(1, -1)
                    lora_B = (-weight * v).view(-1, 1)

                    if self.settings.row_normalization == RowNormalization.PRE:
                        lora_B = W_row_norms * lora_B
                    elif self.settings.row_normalization == RowNormalization.FULL:
                        W = W + lora_B @ lora_A
                        W = F.normalize(W, p=2, dim=1)
                        W = W * W_row_norms
                        W = W - W_org
                        r = self.peft_config.r
                        U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
                        U = U[:, :r]
                        S = S[:r]
                        Vh = Vh[:, :r].T
                        sqrt_S = torch.sqrt(S)
                        lora_B = U @ torch.diag(sqrt_S)
                        lora_A = torch.diag(sqrt_S) @ Vh

                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

        # Confirm abliteration occurred
        # print(f"  * [grey50]Abliteration applied to {sum(len(m) for m in self.all_components.values() if m)} modules[/]")

        # NEW: Global diagnostic for KL issue
        with torch.no_grad():
            avg_a_norm = 0.0
            avg_b_norm = 0.0
            count = 0
            for name, param in self.model.named_parameters():
                if "lora_A" in name:
                    avg_a_norm += param.norm().item()
                    count += 1
                elif "lora_B" in name:
                    avg_b_norm += param.norm().item()
            if count > 0:
                print(f"  * [grey50]Active LoRA norms: A={avg_a_norm/count:.4f}, B={avg_b_norm/count:.4f}[/]")

    def generate(
        self,
        prompts: list[Prompt],
        use_prefix: bool = True,
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

        if use_prefix and self.response_prefix:
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
        use_prefix: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            use_prefix=use_prefix,
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

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(
                batch,
                skip_special_tokens=skip_special_tokens,
                **kwargs,
            ):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[Prompt], use_prefix: bool = True) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            use_prefix=use_prefix,
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

    def get_residuals_batched(self, prompts: list[Prompt], use_prefix: bool = True) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch, use_prefix=use_prefix))

        return torch.cat(residuals, dim=0)

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[Prompt], use_prefix: bool = True, **kwargs: Any) -> Tensor:
        # We only generate one token, and we return the (log) probability distributions
        # for that token.
        _, outputs = self.generate(
            prompts,
            use_prefix=use_prefix,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # This cast is valid because we passed output_scores=True above.
        logits = cast(tuple[FloatTensor], outputs.scores)[0]

        # The returned tensor has shape (prompt, token).
        logprobs = F.log_softmax(logits, dim=-1)

        return logprobs

    def get_logprobs_batched(self, prompts: list[Prompt], use_prefix: bool = True, **kwargs: Any) -> Tensor:
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch, use_prefix=use_prefix, **kwargs))

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

        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
