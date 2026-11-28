# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace
from typing import Any, cast

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import Tensor, nn
from transformers import PreTrainedTokenizerFast

from heretic.model import Model
from heretic.utils import Prompt

VOCAB = {"<pad>": 0, "<unk>": 1, "alpha": 2, "beta": 3, "gamma": 4, "delta": 5}
CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }} {% endfor %}"


def _make_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel(vocab=VOCAB, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        chat_template=CHAT_TEMPLATE,
    )
    tokenizer.padding_side = "left"
    return tokenizer


class _SharedExpertMLP(nn.Module):
    def __init__(self, projection: nn.Linear, flatten: bool) -> None:
        super().__init__()
        self.shared_expert = nn.Module()
        self.shared_expert.down_proj = projection
        self.flatten = flatten

    def forward(self, hidden_states: Tensor) -> Tensor:
        down_proj = cast(nn.Linear, self.shared_expert.down_proj)
        if self.flatten:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            out = down_proj(hidden_states.view(-1, hidden_dim))
            return out.reshape(batch_size, sequence_length, -1)
        return down_proj(hidden_states)


class _TinyModel(nn.Module):
    def __init__(
        self,
        embeddings: Tensor,
        weight: Tensor,
        flatten: bool,
    ) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.embeddings = embeddings
        projection = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            projection.weight.copy_(weight)
        layer = nn.Module()
        layer.mlp = _SharedExpertMLP(projection, flatten=flatten)
        inner = nn.Module()
        inner.layers = nn.ModuleList([layer])
        self.model = inner

    def generate(self, input_ids: Tensor, **kwargs: Any) -> Tensor:
        hidden_states = self.embeddings[input_ids]
        for layer in cast(nn.ModuleList, self.model.layers):
            hidden_states = cast(nn.Module, layer.mlp)(hidden_states)
        return input_ids


class ModuleIOTests(unittest.TestCase):
    def _build_model(self, flatten: bool) -> tuple[Model, Tensor, Tensor]:
        embeddings = torch.arange(len(VOCAB) * 4, dtype=torch.float32).reshape(
            len(VOCAB), 4
        )
        weight = torch.tensor(
            [
                [1.0, 0.0, 2.0, -1.0],
                [0.5, -1.0, 0.0, 3.0],
                [-2.0, 1.0, 0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        model = Model.__new__(Model)
        model.model = cast(Any, _TinyModel(embeddings, weight, flatten=flatten))
        model.tokenizer = _make_tokenizer()
        model.supports_system_role = True
        model.settings = cast(Any, SimpleNamespace(response_prefix="", batch_size=2))
        return model, embeddings, weight

    def _prompts(self) -> list[Prompt]:
        # Different lengths so left-padding pads the shorter prompt while both
        # end at position -1 with distinct final tokens ("gamma" and "delta").
        return [
            Prompt(system="", user="alpha beta gamma"),
            Prompt(system="", user="delta"),
        ]

    def test_captures_flattened_two_dimensional_projection(self) -> None:
        model, embeddings, weight = self._build_model(flatten=True)

        module_io = model.get_module_io(self._prompts())

        captured_input, captured_output = module_io[0]["mlp.down_proj"][0]
        expected_input = embeddings[[VOCAB["gamma"], VOCAB["delta"]]]
        expected_output = expected_input @ weight.T

        self.assertEqual(captured_input.device.type, "cpu")
        self.assertEqual(captured_output.device.type, "cpu")
        self.assertTrue(torch.equal(captured_input, expected_input))
        self.assertTrue(torch.allclose(captured_output, expected_output))

    def test_captures_three_dimensional_projection(self) -> None:
        model, embeddings, weight = self._build_model(flatten=False)

        module_io = model.get_module_io(self._prompts())

        captured_input, captured_output = module_io[0]["mlp.down_proj"][0]
        expected_input = embeddings[[VOCAB["gamma"], VOCAB["delta"]]]
        expected_output = expected_input @ weight.T

        self.assertEqual(captured_input.device.type, "cpu")
        self.assertEqual(captured_output.device.type, "cpu")
        self.assertTrue(torch.equal(captured_input, expected_input))
        self.assertTrue(torch.allclose(captured_output, expected_output))


if __name__ == "__main__":
    unittest.main()
