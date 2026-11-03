# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace
from typing import cast

import torch
from torch import FloatTensor

from heretic.model import _collapse_flattened_residual_streams


class FlattenedResidualStreamTests(unittest.TestCase):
    def test_collapses_flattened_streams(self) -> None:
        torch.manual_seed(0)
        hidden_states = cast(
            tuple[FloatTensor, ...],
            (
                torch.randn(2, 5, 12),
                torch.randn(2, 5, 12),
                torch.randn(2, 5, 4),
            ),
        )
        config = SimpleNamespace(hidden_size=4, hc_count=3)

        collapsed = _collapse_flattened_residual_streams(hidden_states, config)

        self.assertEqual([h.shape[-1] for h in collapsed], [4, 4, 4])
        self.assertTrue(
            torch.allclose(
                collapsed[0],
                hidden_states[0].unflatten(-1, (3, 4)).mean(-2),
            )
        )
        self.assertIs(collapsed[2], hidden_states[2])

    def test_uses_text_config(self) -> None:
        hidden_states = cast(tuple[FloatTensor, ...], (torch.randn(1, 2, 16),))
        text_config = SimpleNamespace(hidden_size=4, hc_count=4)

        class Config:
            def get_text_config(self) -> SimpleNamespace:
                return text_config

        collapsed = _collapse_flattened_residual_streams(hidden_states, Config())

        self.assertEqual(collapsed[0].shape, (1, 2, 4))

    def test_leaves_single_stream_models_unchanged(self) -> None:
        hidden_states = cast(tuple[FloatTensor, ...], (torch.randn(1, 2, 4),))
        config = SimpleNamespace(hidden_size=4, hc_count=1)

        collapsed = _collapse_flattened_residual_streams(hidden_states, config)

        self.assertIs(collapsed, hidden_states)


if __name__ == "__main__":
    unittest.main()
