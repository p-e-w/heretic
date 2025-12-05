# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import LongTensor
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput

from .schemas import (
    ContextMetadata,
    GenerationStep,
    GenerationTrace,
    ResponseMetadata,
)
from .utils import print


class MetadataBuilder:
    SUPPORTED_RESPONSE_FIELDS = {
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

    def __init__(
        self,
        settings: Any,
        tokenizer: PreTrainedTokenizerBase,
        model_getter: Callable[[], Any],
    ):
        self.settings = settings
        self.tokenizer = tokenizer
        self._model_getter = model_getter
        self.requested_response_fields: set[str] = set()
        self.requested_context_fields: set[str] = set()

    def set_requested_response_fields(self, requested_fields: set[str]) -> set[str]:
        self.requested_response_fields = set(requested_fields or [])
        unsupported = self.requested_response_fields - self.SUPPORTED_RESPONSE_FIELDS

        if unsupported:
            print(
                "[yellow]"
                + "Warning: unsupported metadata fields requested; they will be ignored: "
                + ", ".join(sorted(unsupported))
                + "[/]"
            )
            self.requested_response_fields -= unsupported

        return self.requested_response_fields

    def set_requested_context_metadata_fields(
        self, requested_fields: set[str]
    ) -> set[str]:
        self.requested_context_fields = set(requested_fields or [])
        unsupported = (
            self.requested_context_fields - self.SUPPORTED_CONTEXT_METADATA_FIELDS
        )

        if unsupported:
            print(
                "[yellow]"
                + "Warning: unsupported context metadata fields requested; they will be ignored: "
                + ", ".join(sorted(unsupported))
                + "[/]"
            )
            self.requested_context_fields -= unsupported

        return self.requested_context_fields

    def has_response_fields(self) -> bool:
        return bool(self.requested_response_fields)

    def needs_token_scores(self) -> bool:
        return bool(
            self.requested_response_fields
            & {"token_logprobs", "token_logits", "generation_steps"}
        )

    def needs_hidden_states(self) -> bool:
        return bool(
            self.requested_response_fields
            & {
                "response_embedding",
                "prompt_embedding",
                "last_hidden_states",
                "residuals_last_token_per_layer",
            }
        )

    def build_generate_kwargs(
        self, needs_token_scores: bool, needs_hidden_states: bool, max_new_tokens: int
    ) -> dict[str, Any]:
        return {
            "max_new_tokens": max_new_tokens,
            "output_scores": needs_token_scores,
            "return_dict_in_generate": True,
            "output_hidden_states": needs_hidden_states,
        }

    def build_context_metadata(self) -> ContextMetadata:
        requested = self.requested_context_fields
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

    def build_metadata_stub(self, prompt: str) -> ResponseMetadata:
        if "prompt_text" in self.requested_response_fields:
            return ResponseMetadata(prompt_text=prompt)

        return ResponseMetadata()

    def collect_response_metadata(
        self,
        prompts: list[str],
        inputs: BatchEncoding,
        outputs: GenerateOutput | LongTensor,
        generate_kwargs: dict[str, Any],
    ) -> list[ResponseMetadata]:
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        input_length = inputs["input_ids"].shape[1]

        hidden_state_summaries = None
        if self.needs_hidden_states():
            hidden_state_summaries = self._compute_hidden_state_metadata(
                sequences=sequences,
                input_length=input_length,
            )

        metadata: list[ResponseMetadata] = []

        for prompt_index, prompt in enumerate(prompts):
            response_ids = sequences[prompt_index, input_length:].tolist()
            response_tokens = self.tokenizer.convert_ids_to_tokens(response_ids)

            entry = ResponseMetadata()

            if "prompt_text" in self.requested_response_fields:
                entry.prompt_text = prompt

            if "finish_reason" in self.requested_response_fields:
                entry.finish_reason = self._infer_finish_reason(
                    response_ids, generate_kwargs
                )

            if "input_ids" in self.requested_response_fields:
                entry.input_ids = inputs["input_ids"][prompt_index].tolist()

            if "response_ids" in self.requested_response_fields:
                entry.response_ids = response_ids

            if "response_tokens" in self.requested_response_fields:
                entry.response_tokens = response_tokens

            if self.needs_token_scores() and hasattr(outputs, "scores"):
                token_logprobs: list[float] = []
                token_logits: list[float] = []
                generation_steps: list[GenerationStep] = []

                for step_index, token_id in enumerate(response_ids):
                    score_row = outputs.scores[step_index][prompt_index]
                    logprobs_row = F.log_softmax(score_row, dim=-1)
                    token_logprob = logprobs_row[token_id].item()

                    if "token_logprobs" in self.requested_response_fields:
                        token_logprobs.append(token_logprob)

                    if "token_logits" in self.requested_response_fields:
                        token_logits.append(score_row[token_id].item())

                    if "generation_steps" in self.requested_response_fields:
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

                if "token_logprobs" in self.requested_response_fields:
                    entry.token_logprobs = token_logprobs

                if "token_logits" in self.requested_response_fields:
                    entry.token_logits = token_logits

                if "generation_steps" in self.requested_response_fields:
                    entry.generation_steps = [
                        GenerationTrace(
                            steps=generation_steps,
                            finish_reason=entry.finish_reason,
                        )
                    ]

            if self.needs_hidden_states() and hidden_state_summaries is not None:
                summary = hidden_state_summaries[prompt_index]
                if "prompt_embedding" in self.requested_response_fields:
                    entry.prompt_embedding = summary["prompt_embedding"]
                if "response_embedding" in self.requested_response_fields:
                    entry.response_embedding = summary["response_embedding"]
                if "last_hidden_states" in self.requested_response_fields:
                    entry.last_hidden_states = summary["last_hidden_states"]
                if "residuals_last_token_per_layer" in self.requested_response_fields:
                    entry.residuals_last_token_per_layer = summary[
                        "residuals_last_token_per_layer"
                    ]

            metadata.append(entry)

        return metadata

    def _compute_hidden_state_metadata(
        self, sequences: LongTensor, input_length: int
    ) -> list[dict[str, Any]]:
        model = self._model_getter()

        # there's probably a more optimal way to do this than just doing
        # a second forward pass, we should be able to reuse from .generate()
        # we should also warn the user if this is enabled as there will be
        # a significant performance impact
        with torch.no_grad():
            pad_token_id = self.tokenizer.pad_token_id
            attention_mask = (
                sequences.ne(pad_token_id) if pad_token_id is not None else None
            )

            forward_outputs = model(
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

            # simple mean pooling; maybe this should be configurable later?

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

        if len(response_ids) >= max_new_tokens:
            return "len"

        elif response_ids[-1] == self.tokenizer.eos_token_id:
            return "stop"

        else:
            return "unk"
