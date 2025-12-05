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
from .schemas import (
    ContextMetadata,
    GenerationStep,
    GenerationTrace,
    ResponseMetadata,
)


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Model:
    SUPPORTED_METADATA_FIELDS = {
        "prompt_text",
        "finish_reason",
        "input_ids",
        "response_ids",
        "response_tokens",
        "token_logprobs",
        "token_logits",
        "generation_steps",
        "response_embedding",
        "prompt_embedding",
        "last_hidden_states",
        "residuals_last_token_per_layer",
    }
    SUPPORTED_CONTEXT_METADATA_FIELDS = {
        "system_prompt",
        "model_name",
        "generation_params",
    }

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
        self.requested_metadata_fields = set()
        self.requested_context_metadata_fields = set()

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

    def set_requested_metadata_fields(self, requested_fields: set[str]):
        self.requested_metadata_fields = requested_fields
        unsupported_metadata_fields = (
            self.requested_metadata_fields - self.SUPPORTED_METADATA_FIELDS
        )

        if unsupported_metadata_fields:
            print(
                "[yellow]"
                + "Warning: unsupported metadata fields requested; they will be ignored: "
                + ", ".join(sorted(unsupported_metadata_fields))
                + "[/]"
            )
            self.requested_metadata_fields -= unsupported_metadata_fields

    def get_context_metadata(self) -> ContextMetadata:
        """
        Build context-level metadata to pass to taggers at initialization time.
        """
        requested = self.requested_context_metadata_fields

        return ContextMetadata(
            system_prompt=self.settings.system_prompt
            if "system_prompt" in requested
            else None,
            model_name=self.settings.model if "model_name" in requested else None,
            generation_params={
                "max_new_tokens": self.settings.max_response_length,
                "do_sample": False,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if "generation_params" in requested
            else None,
        )

    def set_requested_context_metadata_fields(self, requested_fields: set[str]):
        self.requested_context_metadata_fields = requested_fields

        unsupported_context_fields = (
            self.requested_context_metadata_fields
            - self.SUPPORTED_CONTEXT_METADATA_FIELDS
        )

        if unsupported_context_fields:
            print(
                "[yellow]"
                + "Warning: unsupported context metadata fields requested; they will be ignored: "
                + ", ".join(sorted(unsupported_context_fields))
                + "[/]"
            )
            self.requested_context_metadata_fields -= unsupported_context_fields

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
                projector = torch.outer(
                    layer_refusal_direction,
                    layer_refusal_direction,
                ).to(self.model.dtype)

                for matrix in matrices:
                    # Ensure projector is on the same device as the matrix for multi-GPU support.
                    device_projector = projector.to(matrix.device)
                    # In-place subtraction is safe as we're not using Autograd.
                    matrix.sub_(weight * (device_projector @ matrix))

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

    def get_responses_batched(
        self, prompts: list[str]
    ) -> tuple[list[str], list[ResponseMetadata]]:
        responses: list[str] = []
        metadata: list[ResponseMetadata] = []

        for batch in batchify(prompts, self.settings.batch_size):
            if self.requested_metadata_fields:
                batch_responses, batch_metadata = self._get_responses_with_metadata(
                    batch
                )
            else:
                batch_responses = self.get_responses(batch)
                batch_metadata = [self._build_metadata_stub(prompt) for prompt in batch]

            responses.extend(batch_responses)
            metadata.extend(batch_metadata)

        return responses, metadata

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

    def _build_metadata_stub(self, prompt: str) -> ResponseMetadata:
        if "prompt_text" in self.requested_metadata_fields:
            return ResponseMetadata(prompt_text=prompt)

        return ResponseMetadata()

    def _get_responses_with_metadata(
        self, prompts: list[str]
    ) -> tuple[list[str], list[ResponseMetadata]]:
        # Determine which optional outputs we need from `generate`.
        needs_token_scores = bool(
            self.requested_metadata_fields
            & {"token_logprobs", "token_logits", "generation_steps"}
        )
        needs_hidden_states = bool(
            self.requested_metadata_fields
            & {
                "response_embedding",
                "prompt_embedding",
                "last_hidden_states",
                "residuals_last_token_per_layer",
            }
        )

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.settings.max_response_length,
            "output_scores": needs_token_scores,
            "return_dict_in_generate": True,
            "output_hidden_states": needs_hidden_states,
        }

        inputs, outputs = self.generate(prompts, **generate_kwargs)

        responses = self.tokenizer.batch_decode(
            outputs.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        metadata = self._collect_response_metadata(
            prompts=prompts,
            inputs=inputs,
            outputs=outputs,
            needs_scores=needs_token_scores,
            needs_hidden_states=needs_hidden_states,
            generate_kwargs=generate_kwargs,
        )

        return responses, metadata

    def _collect_response_metadata(
        self,
        prompts: list[str],
        inputs: BatchEncoding,
        outputs: GenerateOutput | LongTensor,
        needs_scores: bool,
        needs_hidden_states: bool,
        generate_kwargs: dict[str, Any],
    ) -> list[ResponseMetadata]:
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        input_length = inputs["input_ids"].shape[1]

        hidden_state_summaries = None
        if needs_hidden_states:
            hidden_state_summaries = self._compute_hidden_state_metadata(
                sequences=sequences,
                input_length=input_length,
            )

        metadata: list[ResponseMetadata] = []

        for prompt_index, prompt in enumerate(prompts):
            response_ids = sequences[prompt_index, input_length:].tolist()
            response_tokens = self.tokenizer.convert_ids_to_tokens(response_ids)

            entry = ResponseMetadata()

            if "prompt_text" in self.requested_metadata_fields:
                entry.prompt_text = prompt

            if "model_name" in self.requested_metadata_fields:
                entry.model_name = self.settings.model

            if "generation_params" in self.requested_metadata_fields:
                entry.generation_params = generate_kwargs

            if "finish_reason" in self.requested_metadata_fields:
                entry.finish_reason = self._infer_finish_reason(
                    response_ids, generate_kwargs
                )

            if "input_ids" in self.requested_metadata_fields:
                entry.input_ids = inputs["input_ids"][prompt_index].tolist()

            if "response_ids" in self.requested_metadata_fields:
                entry.response_ids = response_ids

            if "response_tokens" in self.requested_metadata_fields:
                entry.response_tokens = response_tokens

            if needs_scores and hasattr(outputs, "scores"):
                token_logprobs: list[float] = []
                token_logits: list[float] = []
                generation_steps: list[GenerationStep] = []

                for step_index, token_id in enumerate(response_ids):
                    score_row = outputs.scores[step_index][prompt_index]
                    logprobs_row = F.log_softmax(score_row, dim=-1)
                    token_logprob = logprobs_row[token_id].item()

                    if "token_logprobs" in self.requested_metadata_fields:
                        token_logprobs.append(token_logprob)

                    if "token_logits" in self.requested_metadata_fields:
                        token_logits.append(score_row[token_id].item())

                    if "generation_steps" in self.requested_metadata_fields:
                        topk_values, topk_indices = logprobs_row.topk(5)
                        topk = [
                            {
                                self.tokenizer.decode(
                                    [topk_index.item()], skip_special_tokens=True
                                ): topk_value.item()
                            }
                            for topk_index, topk_value in zip(topk_indices, topk_values)
                        ]

                        probs_row = logprobs_row.exp()
                        entropy = -torch.sum(probs_row * logprobs_row).item()

                        generation_steps.append(
                            GenerationStep(
                                step_index=step_index,
                                token_id=token_id,
                                token=self.tokenizer.decode(
                                    [token_id], skip_special_tokens=True
                                ),
                                logprob=token_logprob,
                                topk=topk,
                                entropy=entropy,
                            )
                        )

                if "token_logprobs" in self.requested_metadata_fields:
                    entry.token_logprobs = token_logprobs

                if "token_logits" in self.requested_metadata_fields:
                    entry.token_logits = token_logits

                if "generation_steps" in self.requested_metadata_fields:
                    entry.generation_steps = [
                        GenerationTrace(
                            steps=generation_steps,
                            finish_reason=entry.finish_reason,
                        )
                    ]

            if needs_hidden_states and hidden_state_summaries is not None:
                summary = hidden_state_summaries[prompt_index]
                if "prompt_embedding" in self.requested_metadata_fields:
                    entry.prompt_embedding = summary["prompt_embedding"]
                if "response_embedding" in self.requested_metadata_fields:
                    entry.response_embedding = summary["response_embedding"]
                if "last_hidden_states" in self.requested_metadata_fields:
                    entry.last_hidden_states = summary["last_hidden_states"]
                if "residuals_last_token_per_layer" in self.requested_metadata_fields:
                    entry.residuals_last_token_per_layer = summary[
                        "residuals_last_token_per_layer"
                    ]

            metadata.append(entry)

        return metadata

    def _compute_hidden_state_metadata(
        self, sequences: LongTensor, input_length: int
    ) -> list[dict[str, Any]]:
        with torch.no_grad():
            # Build attention mask where padding is tokenizer pad token
            pad_token_id = self.tokenizer.pad_token_id
            attention_mask = (
                sequences.ne(pad_token_id) if pad_token_id is not None else None
            )

            forward_outputs = self.model(
                input_ids=sequences,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = forward_outputs.hidden_states
        last_layer = hidden_states[-1]

        summaries: list[dict[str, Any]] = []

        for batch_index in range(sequences.shape[0]):
            prompt_slice = last_layer[batch_index, :input_length, :]
            response_slice = last_layer[batch_index, input_length:, :]

            prompt_embedding = (
                prompt_slice.mean(dim=0).detach().cpu().tolist()
                if prompt_slice.numel() > 0
                else None
            )
            response_embedding = (
                response_slice.mean(dim=0).detach().cpu().tolist()
                if response_slice.numel() > 0
                else None
            )

            last_hidden_states = (
                response_slice.detach().cpu().tolist()
                if response_slice.numel() > 0
                else None
            )

            residuals_last_token_per_layer = [
                layer[batch_index, -1, :].detach().cpu().tolist()
                for layer in hidden_states
            ]

            summaries.append(
                {
                    "prompt_embedding": prompt_embedding,
                    "response_embedding": response_embedding,
                    "last_hidden_states": last_hidden_states,
                    "residuals_last_token_per_layer": residuals_last_token_per_layer,
                }
            )

        return summaries

    def _infer_finish_reason(
        self, response_ids: list[int], generate_kwargs: dict[str, Any]
    ) -> str | None:
        max_new_tokens = generate_kwargs.get("max_new_tokens")
        if max_new_tokens is None:
            return None

        # TODO: this is pretty primitive, we need to finetune this heuristic in the future
        if len(response_ids) >= max_new_tokens:
            return "length"

        return "stop"
