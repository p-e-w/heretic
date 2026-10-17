# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch
import torch.linalg as LA


def _dense_output(good_input, W_base, A, B, W_row_norms, preserve):
    # The original formulation: build the dense effective weight, then multiply.
    W_eff = W_base + (B @ A)
    if preserve:
        W_eff = torch.nn.functional.normalize(W_eff, p=2, dim=1) * W_row_norms
    return good_input @ W_eff.T


def _lowrank_output(good_input, good_output_base, W_base, A, B, W_row_norms, preserve):
    # The reformulation used by ARA.modify_model: constant base output plus the
    # rank-r update, with row normalization applied as a per-output-row rescale.
    out = good_output_base + (good_input @ A.T) @ B.T
    if preserve:
        eff_row_norms = LA.vector_norm(W_base + (B @ A), dim=1)
        out = out * (W_row_norms.squeeze(1) / eff_row_norms)
    return out


class LowRankEquivalenceTests(unittest.TestCase):
    def _check(self, preserve: bool) -> None:
        torch.manual_seed(0)
        d_out, d_in, r, n = 128, 320, 8, 40
        W_base = torch.randn(d_out, d_in, dtype=torch.float64)
        A = torch.randn(r, d_in, dtype=torch.float64) * 0.01
        B = torch.randn(d_out, r, dtype=torch.float64) * 0.01
        good_input = torch.randn(n, d_in, dtype=torch.float64)
        # "captured" module output (no bias here): good_input @ W_base.T
        good_output_base = good_input @ W_base.T
        W_row_norms = LA.vector_norm(W_base, dim=1, keepdim=True)

        dense = _dense_output(good_input, W_base, A, B, W_row_norms, preserve)
        lowrank = _lowrank_output(
            good_input, good_output_base, W_base, A, B, W_row_norms, preserve
        )
        self.assertTrue(
            torch.allclose(dense, lowrank, rtol=1e-9, atol=1e-9),
            msg=f"max abs diff = {(dense - lowrank).abs().max().item():.2e}",
        )

    def test_equivalent_without_row_normalization(self) -> None:
        self._check(preserve=False)

    def test_equivalent_with_row_normalization(self) -> None:
        self._check(preserve=True)


if __name__ == "__main__":
    unittest.main()
