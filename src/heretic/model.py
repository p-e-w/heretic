# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch.nn import ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation.utils import GenerateOutput

from .config import Settings
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
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    dtype=dtype,
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
            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component, matrices in self.get_layer_matrices(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(matrices)}[/] matrices per layer"
            )

    def reload_model(self):
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

    def get_layers(self) -> ModuleList:
        # Most multimodal models.
        with suppress(Exception):
            return self.model.model.language_model.layers

        # Text-only models.
        return self.model.model.layers

    def get_layer_matrices(self, layer_index: int) -> dict[str, list[Tensor]]:
        layer = self.get_layers()[layer_index]

        matrices = {}

        def try_add(component: str, matrix: Any):
            # Handle Triton tensors (e.g., from MXFP4 quantization) by extracting
            # the underlying PyTorch tensor via the .data attribute.
            if hasattr(matrix, "data") and torch.is_tensor(matrix.data):
                matrix = matrix.data

            assert torch.is_tensor(matrix)

            if component not in matrices:
                matrices[component] = []

            matrices[component].append(matrix)

        # Exceptions aren't suppressed here, because there is currently
        # no alternative location for the attention out-projection.
        try_add("attn.o_proj", layer.self_attn.o_proj.weight)

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj.weight)

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:
                try_add("mlp.down_proj", expert.down_proj.weight)

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:
                try_add("mlp.down_proj", expert.w2.weight)

        # gpt-oss MoE.
        with suppress(Exception):
            # The implementation of gpt-oss in Transformers differs from many other MoE models
            # in that it stores the down-projections for all experts in a single 3D tensor,
            # but thanks to PyTorch's broadcasting magic, it all just works anyway.
            try_add("mlp.down_proj", layer.mlp.experts.down_proj)

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear.weight)

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:
                try_add("mlp.down_proj", expert.output_linear.weight)

        # We need at least one MLP down-projection.
        assert matrices["mlp.down_proj"]

        return matrices

    def get_abliterable_components(self) -> list[str]:
        return list(self.get_layer_matrices(0).keys())

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
            for component, matrices in self.get_layer_matrices(layer_index).items():
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

                # Projects any right-multiplied vector(s) onto the subspace
                # spanned by the refusal direction.
                # We use the property (r r^T) W = r (r^T W) to avoid computing
                # the O(d^2) projector matrix and the O(d^2 k) matrix multiplication.
                # (α is the weight)
                # W_new = W - α(r (r^T W))
                #r = layer_refusal_direction.to(self.model.dtype)
                r = layer_refusal_direction
                
                printed_debug = False
                for matrix in matrices:
                    # Ensure r is on the same device AND dtype as the matrix
                    target_device = matrix.device
                    target_dtype = matrix.dtype
                    r_device = r.to(device=target_device, dtype=target_dtype)

                    try:
                        # r^T W
                        r_transpose_W = torch.matmul(r_device, matrix)

                        # Handle both 2D (d,k) and 3D (E,d,k) matrices
                        if r_transpose_W.ndim == 1:
                            update = torch.outer(r_device, r_transpose_W)
                        elif r_transpose_W.ndim == 2:
                            update = torch.einsum("d,ek->edk", r_device, r_transpose_W)
                        else:
                            leading = r_transpose_W.shape[:-1]
                            expanded_r = r_device.view((1,) * len(leading) + r_device.shape)
                            expanded_r = expanded_r.expand(*leading, *r_device.shape)
                            expanded_r_tW = r_transpose_W.unsqueeze(-2)
                            update = expanded_r.unsqueeze(-1) * expanded_r_tW

                        scalar = matrix.new_tensor(weight)
                        matrix.sub_(scalar * update)

                        if not printed_debug:
                            print(
                                f"[ABLITERATE DEBUG]\n"
                                f"  layer_index   = {layer_index}\n"
                                f"  component     = {component}\n"
                                f"  matrix.shape  = {tuple(matrix.shape)}\n"
                                f"  matrix.device = {matrix.device}\n"
                                f"  matrix.dtype  = {matrix.dtype}\n"
                                f"  r.shape       = {tuple(r_device.shape)}\n"
                                f"  r.device      = {r_device.device}\n"
                                f"  r.dtype       = {r_device.dtype}\n"
                                f"  rTW.shape     = {tuple(r_transpose_W.shape)}\n"
                                f"  weight        = {weight}\n"
                            )
                            printed_debug = True

                    except Exception as e:
                        print("\n[ABLITERATE ERROR]")
                        print(f"  layer_index   = {layer_index}")
                        print(f"  component     = {component}")
                        print(f"  matrix.shape  = {tuple(matrix.shape)}")
                        print(f"  matrix.device = {matrix.device}")
                        print(f"  matrix.dtype  = {matrix.dtype}")
                        print(f"  r.shape       = {tuple(r.shape)}")
                        print(f"  r.device      = {r.device}")
                        print(f"  r.dtype       = {r.dtype}")
                        print(f"  weight        = {weight}")
                        raise e

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
