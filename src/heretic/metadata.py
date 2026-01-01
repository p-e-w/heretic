# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from typing import Any, Callable, Literal, cast

import torch
import torch.nn.functional as F
from torch import LongTensor
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput

from .scorer import Response
from .utils import print


class MetadataBuilder:
    SUPPORTED_RESPONSE_FIELDS = {
        "response_text",
        "prompt_text",
        "finish_reason",
        "input_ids",
        "response_ids",
        "response_tokens",
        "token_logprobs",
        "token_logits",
        "hidden_states",
        "residuals_last_token_per_layer",
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

    def has_response_fields(self) -> bool:
        return bool(self.requested_response_fields)

    def needs_token_scores(self) -> bool:
        return bool(self.requested_response_fields & {"token_logprobs", "token_logits"})

    def needs_hidden_states(self) -> bool:
        return bool(
            self.requested_response_fields
            & {
                "hidden_states",
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

    def collect_response_metadata(
        self,
        prompts: list[str],
        inputs: BatchEncoding,
        outputs: GenerateOutput | LongTensor,
        generate_kwargs: dict[str, Any],
        responses: list[str],
    ) -> list[Response]:
        if isinstance(outputs, LongTensor):
            sequences: LongTensor = outputs
        else:
            sequences = cast(LongTensor, cast(Any, outputs).sequences)

        input_ids = cast(LongTensor, inputs["input_ids"])
        input_length = input_ids.shape[1]

        hidden_state_summaries = None
        if self.needs_hidden_states():
            hidden_state_summaries = self._compute_hidden_state_metadata(
                sequences=sequences,
                input_length=input_length,
            )

        metadata: list[Response] = []

        for prompt_index, prompt in enumerate(prompts):
            response_ids = sequences[prompt_index, input_length:].tolist()
            has_response = bool(response_ids)
            response_tokens = (
                self.tokenizer.convert_ids_to_tokens(response_ids)
                if has_response
                else []
            )

            entry = Response(response_text=responses[prompt_index])

            if "prompt_text" in self.requested_response_fields:
                entry.prompt_text = prompt

            if "finish_reason" in self.requested_response_fields:
                entry.finish_reason = self._infer_finish_reason(
                    response_ids=response_ids,
                    generate_kwargs=generate_kwargs,
                )

            if "input_ids" in self.requested_response_fields:
                entry.input_ids = input_ids[prompt_index].tolist()

            if "response_ids" in self.requested_response_fields:
                entry.response_ids = response_ids

            if "response_tokens" in self.requested_response_fields:
                entry.response_tokens = response_tokens

            if (
                has_response
                and self.needs_token_scores()
                and hasattr(outputs, "scores")
            ):
                token_logprobs: list[float] = []
                token_logits: list[float] = []

                for step_index, token_id in enumerate(response_ids):
                    score_row = cast(Any, outputs).scores[step_index][prompt_index]
                    logprobs_row = F.log_softmax(score_row, dim=-1)
                    token_logprob = logprobs_row[token_id].item()

                    if "token_logprobs" in self.requested_response_fields:
                        token_logprobs.append(token_logprob)

                    if "token_logits" in self.requested_response_fields:
                        token_logits.append(score_row[token_id].item())

                if "token_logprobs" in self.requested_response_fields:
                    entry.token_logprobs = token_logprobs

                if "token_logits" in self.requested_response_fields:
                    entry.token_logits = token_logits

            if self.needs_hidden_states() and hidden_state_summaries is not None:
                summary = hidden_state_summaries[prompt_index]
                if "hidden_states" in self.requested_response_fields:
                    entry.hidden_states = summary.get("hidden_states")
                if "residuals_last_token_per_layer" in self.requested_response_fields:
                    entry.residuals_last_token_per_layer = summary.get(
                        "residuals_last_token_per_layer"
                    )

            metadata.append(entry)

        return metadata

    def _compute_hidden_state_metadata(
        self, sequences: LongTensor, input_length: int
    ) -> list[dict[str, Any]]:
        requested = self.requested_response_fields
        needs_hidden_states = "hidden_states" in requested
        needs_residuals = "residuals_last_token_per_layer" in requested

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
        summaries: list[dict[str, Any]] = []

        for batch_index in range(sequences.shape[0]):
            summary: dict[str, Any] = {}

            if needs_hidden_states:
                summary["hidden_states"] = [
                    layer[batch_index].detach().cpu().tolist()
                    for layer in hidden_states
                ]

            if needs_residuals:
                summary["residuals_last_token_per_layer"] = [
                    layer[batch_index, -1, :].detach().cpu().tolist()
                    for layer in hidden_states
                ]

            summaries.append(summary)

        return summaries

    def _infer_finish_reason(
        self, response_ids: list[int], generate_kwargs: dict[str, Any]
    ) -> Literal["len", "eos", "unk", "empty"] | None:
        max_new_tokens = generate_kwargs.get("max_new_tokens")
        if max_new_tokens is None:
            return None

        if not response_ids:
            return "empty"

        if len(response_ids) >= max_new_tokens:
            return "len"

        if response_ids[-1] == self.tokenizer.eos_token_id:
            return "eos"

        return "unk"
