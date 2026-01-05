# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from typing import Any, Callable, cast

import torch.nn.functional as F
from torch import LongTensor
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput

from .scorer import (
    FinishReason,
    Response,
    ResponseText,
    ResponseTokenization,
    ResponseTokenScores,
)
from .utils import print


class MetadataBuilder:
    # Allowlist for plugin requests. Most of these are "free" (already computed)
    # and exist mainly to keep plugin contracts self-documenting
    # The only request that changes generation behavior is `token_scores`.
    SUPPORTED_RESPONSE_FIELDS = {
        # text group
        "response_text",
        "prompt_text",
        "finish_reason",
        # tokenization group
        "input_ids",
        "response_ids",
        "response_tokens",
        # token scores group (aliases)
        "token_scores",
        "token_logprobs",
        "token_logits",
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
        self._needs_token_scores = False

    def set_requested_response_fields(self, requested_fields: set[str]) -> set[str]:
        requested = set(requested_fields or [])
        unsupported = requested - self.SUPPORTED_RESPONSE_FIELDS

        if unsupported:
            print(
                "[yellow]"
                + "Warning: unsupported metadata fields requested; they will be ignored: "
                + ", ".join(sorted(unsupported))
                + "[/]"
            )
            requested -= unsupported

        # Token scores are opt-in because they require `output_scores=True`.
        self._needs_token_scores = bool(
            requested & {"token_scores", "token_logprobs", "token_logits"}
        )
        return requested

    def needs_token_scores(self) -> bool:
        return self._needs_token_scores

    def build_generate_kwargs(
        self, needs_token_scores: bool, max_new_tokens: int
    ) -> dict[str, Any]:
        return {
            "max_new_tokens": max_new_tokens,
            "output_scores": needs_token_scores,
            "return_dict_in_generate": True,
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

        metadata: list[Response] = []

        for prompt_index, prompt in enumerate(prompts):
            response_ids = sequences[prompt_index, input_length:].tolist()
            has_response = bool(response_ids)
            response_tokens = (
                self.tokenizer.convert_ids_to_tokens(response_ids)
                if has_response
                else []
            )

            finish_reason = self._infer_finish_reason(
                response_ids=response_ids,
                max_new_tokens=cast(int | None, generate_kwargs.get("max_new_tokens")),
            )

            token_logprobs: list[float] = []
            token_logits: list[float] = []

            if (
                has_response
                and self.needs_token_scores()
                and hasattr(outputs, "scores")
            ):
                for step_index, token_id in enumerate(response_ids):
                    score_row = cast(Any, outputs).scores[step_index][prompt_index]
                    logprobs_row = F.log_softmax(score_row, dim=-1)
                    token_logprob = logprobs_row[token_id].item()
                    token_logprobs.append(token_logprob)
                    token_logits.append(score_row[token_id].item())

            metadata.append(
                Response(
                    text=ResponseText(
                        prompt_text=prompt,
                        response_text=responses[prompt_index],
                        finish_reason=finish_reason,
                    ),
                    tokenization=ResponseTokenization(
                        input_ids=input_ids[prompt_index].tolist(),
                        response_ids=response_ids,
                        response_tokens=response_tokens,
                    ),
                    token_scores=ResponseTokenScores(
                        token_logprobs=token_logprobs,
                        token_logits=token_logits,
                    ),
                )
            )

        return metadata
    def _infer_finish_reason(
        self, *, response_ids: list[int], max_new_tokens: int | None
    ) -> FinishReason:
        if max_new_tokens is None:
            return "unk"

        if not response_ids:
            return "empty"

        if len(response_ids) >= max_new_tokens:
            return "len"

        if response_ids[-1] == self.tokenizer.eos_token_id:
            return "eos"

        return "unk"
