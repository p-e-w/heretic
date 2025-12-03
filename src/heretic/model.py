# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import LongTensor, Tensor
from torch.nn import ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation.utils import GenerateOutput

from .config import QuantizationMethod, Settings
from .utils import batchify, empty_cache, print


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Model:
    def __init__(self, settings: Settings):
        self.settings = settings

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            settings.model,
            trust_remote_code=settings.trust_remote_code,
        )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        self.model = None
        self.trusted_models = {settings.model: settings.trust_remote_code}

        if self.settings.evaluate_model is not None:
            self.trusted_models[settings.evaluate_model] = settings.trust_remote_code

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

            try:
                quantization_config = None
                if settings.quantization == QuantizationMethod.BNB_4BIT:
                    # BitsAndBytesConfig expects a torch.dtype, not a string.
                    if dtype == "auto":
                        compute_dtype = torch.bfloat16
                    else:
                        compute_dtype = getattr(torch, dtype)

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )

                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    dtype=dtype,
                    quantization_config=quantization_config,
                    device_map=settings.device_map,
                    trust_remote_code=self.trusted_models.get(settings.model),
                )

                # If we reach this point and the model requires trust_remote_code,
                # the user must have confirmed it.
                if self.trusted_models.get(settings.model) is None:
                    self.trusted_models[settings.model] = True

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(["Test"], max_new_tokens=1)
            except Exception as error:
                self.model = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue

            print("[green]Ok[/]")
            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("[bold green]Model loaded in 4-bit precision.[/]")
            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        # Always use LoRA adapters for abliteration
        print("* Initializing LoRA adapters...")
        target_modules = self.get_abliterable_components()
        peft_config = LoraConfig(
            r=1,  # Rank 1 is sufficient for directional ablation
            target_modules=target_modules,
            lora_alpha=1,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.

        self.loaded_model_name = settings.model

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component, modules in self.get_layer_modules(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(modules)}[/] modules per layer"
            )

    def reload_model(self):
        if self.loaded_model_name == self.settings.model:
            # Reset LoRA adapters to zero
            for name, module in self.model.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            return

        dtype = self.model.dtype

        # Purge existing model object from memory to make space.
        self.model = None
        empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model,
            dtype=dtype,
            device_map=self.settings.device_map,
            trust_remote_code=self.trusted_models.get(self.settings.model),
        )

        if self.trusted_models.get(self.settings.model) is None:
            self.trusted_models[self.settings.model] = True

        self.loaded_model_name = self.settings.model

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers

        # Text-only models.
        return model.model.layers

    def get_layer_modules(
        self, layer_index: int
    ) -> dict[str, list[torch.nn.Module | Tensor]]:
        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            if component not in modules:
                modules[component] = []

            # Validate that the module is either a torch.nn.Module or a Tensor
            if not isinstance(module, (torch.nn.Module, torch.Tensor)):
                # If it's not a standard module/tensor, check if it's a wrapper (like in GPT-OSS)
                # that we can extract a tensor from, otherwise skip or raise error.
                # For now, we strictly enforce Module or Tensor as requested.
                return

            modules[component].append(module)

        # Exceptions aren't suppressed here, because there is currently
        # no alternative location for the attention out-projection.
        try_add("attn.o_proj", layer.self_attn.o_proj)

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:
                try_add("mlp.down_proj", expert.down_proj)

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:
                try_add("mlp.down_proj", expert.w2)

        # gpt-oss MoE.
        with suppress(Exception):
            # The implementation of gpt-oss in Transformers differs from many other MoE models
            # in that it stores the down-projections for all experts in a single 3D tensor,
            # but thanks to PyTorch's broadcasting magic, it all just works anyway.
            try_add("mlp.down_proj", layer.mlp.experts.down_proj)

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:
                try_add("mlp.down_proj", expert.output_linear)

        # We need at least one MLP down-projection.
        assert modules["mlp.down_proj"]

        return modules

    def get_abliterable_components(self) -> list[str]:
        return list(self.get_layer_modules(0).keys())

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

                distance = abs(layer_index - params.max_weight_position)

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
                    if hasattr(module, "lora_A"):
                        # LoRA abliteration: delta W = -lambda * v * (v^T W)
                        # lora_B = -lambda * v
                        # lora_A = v^T W

                        # Use the FP32 refusal direction directly (no downcast/upcast)
                        # and move to the correct device
                        v = layer_refusal_direction.to(module.weight.device)

                        # Get W (dequantize if necessary)
                        if hasattr(module.weight, "quant_state"):
                            W = bnb.functional.dequantize_4bit(
                                module.weight.data, module.weight.quant_state
                            ).to(torch.float32)
                        else:
                            W = module.weight.to(torch.float32)

                        # Calculate lora_A = v^T W
                        # v is (d_out,), W is (d_out, d_in)
                        # v @ W -> (d_in,)
                        lora_A = (v @ W).view(1, -1)

                        # Calculate lora_B = -weight * v
                        # v is (d_out,)
                        lora_B = (-weight * v).view(-1, 1)

                        # Assign to adapters
                        # We assume the default adapter name "default"
                        module.lora_A["default"].weight.data = lora_A.to(
                            module.lora_A["default"].weight.dtype
                        )
                        module.lora_B["default"].weight.data = lora_B.to(
                            module.lora_B["default"].weight.dtype
                        )
                    else:
                        # Direct weight modification (for non-LoRA mode or modules without LoRA adapters)
                        # This handles cases like GPT-OSS where down_proj is an nn.Parameter

                        # Projects any right-multiplied vector(s) onto the subspace
                        # spanned by the refusal direction.
                        # We use the property (r r^T) W = r (r^T W) to avoid computing
                        # the O(d^2) projector matrix and the O(d^2 k) matrix multiplication.
                        # (α is the weight)
                        # W_new = W - α(r (r^T W))
                        r = layer_refusal_direction.to(self.model.dtype)

                        matrix = module.weight if hasattr(module, "weight") else module

                        # Handle Triton tensors (e.g., from MXFP4 quantization) by extracting
                        # the underlying PyTorch tensor via the .data attribute.
                        if hasattr(matrix, "data") and torch.is_tensor(matrix.data):
                            matrix = matrix.data

                        assert torch.is_tensor(matrix)

                        # Ensure r is on the same device as the matrix
                        r_device = r.to(matrix.device)

                        # Calculate the projection scalars: (r^T W)
                        # r is (d,), matrix is (d, k) -> result is (k,)
                        r_transpose_W = torch.matmul(r_device, matrix)

                        # Compute the rank-1 update r (r^T W) using the outer product form
                        # r_device: (d,)          — projection direction
                        # r_transpose_W: (k,)     — r^T W result for this matrix
                        # torch.outer(r_device, r_times_W) constructs the (d, k) matrix with
                        # entries r[i] * (r^T W)[j], equivalent to the outer product of two
                        # vectors, avoiding materializing the full (d x d) projector.
                        matrix.sub_(weight * torch.outer(r_device, r_transpose_W))

    def get_chat(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateOutput | LongTensor]:
        chats = [self.get_chat(prompt) for prompt in prompts]

        chat_prompts: list[str] = self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        return inputs, self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )

    def get_responses(self, prompts: list[str]) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        # Return only the newly generated part.
        return self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

    def get_responses_batched(self, prompts: list[str]) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(batch):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[str]) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Hidden states for the first (only) generated token.
        hidden_states = outputs.hidden_states[0]

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
        return residuals.to(torch.float32)

    def get_residuals_batched(self, prompts: list[str]) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))

        return torch.cat(residuals, dim=0)

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[str]) -> Tensor:
        # We only generate one token, and we return the (log) probability distributions
        # over the vocabulary at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Logits for the first (only) generated token.
        logits = outputs.scores[0]

        # The returned tensor has shape (prompt, token).
        return F.log_softmax(logits, dim=-1)

    def get_logprobs_batched(self, prompts: list[str]) -> Tensor:
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch))

        return torch.cat(logprobs, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        chat_prompt: str = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )

        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
