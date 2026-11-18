# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch
import torch.linalg as LA

from heretic.modifiers.ara import mean_distances_to_knn


def _dense_output(good_input, W_base, bias, A, B, W_row_norms, preserve):
    # Reference: build the dense effective weight, multiply, then add the bias.
    W_eff = W_base + (B @ A)
    if preserve:
        W_eff = torch.nn.functional.normalize(W_eff, p=2, dim=1) * W_row_norms
    out = good_input @ W_eff.T
    return out if bias is None else out + bias


def _lowrank_output(good_input, W_base, bias, A, B, W_row_norms, preserve):
    # The reformulation used by ARA.modify_model: the base product (computed once,
    # independent of the adapter) plus the rank-r update, rescaled per output row,
    # then the bias.
    product = good_input @ W_base.T
    out = product + (good_input @ A.T) @ B.T
    if preserve:
        eff_row_norms = LA.vector_norm(W_base + (B @ A), dim=1)
        out = out * (W_row_norms.squeeze(1) / eff_row_norms)
    return out if bias is None else out + bias


class LowRankEquivalenceTests(unittest.TestCase):
    def _check(self, preserve: bool, with_bias: bool = False) -> None:
        torch.manual_seed(0)
        d_out, d_in, r, n = 128, 320, 8, 40
        W_base = torch.randn(d_out, d_in, dtype=torch.float64)
        A = torch.randn(r, d_in, dtype=torch.float64) * 0.01
        B = torch.randn(d_out, r, dtype=torch.float64) * 0.01
        good_input = torch.randn(n, d_in, dtype=torch.float64)
        bias = torch.randn(d_out, dtype=torch.float64) if with_bias else None
        W_row_norms = LA.vector_norm(W_base, dim=1, keepdim=True)

        dense = _dense_output(good_input, W_base, bias, A, B, W_row_norms, preserve)
        lowrank = _lowrank_output(good_input, W_base, bias, A, B, W_row_norms, preserve)
        self.assertTrue(
            torch.allclose(dense, lowrank, rtol=1e-9, atol=1e-9),
            msg=f"max abs diff = {(dense - lowrank).abs().max().item():.2e}",
        )

    def test_knn_mean_matches_sorted_topk_and_gradients(self) -> None:
        torch.manual_seed(0)
        a = torch.randn(32, 24, requires_grad=True)
        b = torch.randn(40, 24, requires_grad=True)

        actual = mean_distances_to_knn(a, b, k=5)
        expected = torch.cdist(a, b).topk(5, dim=1, largest=False).values.mean(1)

        self.assertTrue(torch.allclose(actual, expected))
        actual_gradients = torch.autograd.grad(actual.sum(), (a, b), retain_graph=True)
        expected_gradients = torch.autograd.grad(expected.sum(), (a, b))
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            self.assertTrue(torch.allclose(actual_gradient, expected_gradient))

    def test_equivalent_without_row_normalization(self) -> None:
        self._check(preserve=False)

    def test_equivalent_with_row_normalization(self) -> None:
        self._check(preserve=True)

    def test_equivalent_with_bias(self) -> None:
        self._check(preserve=False, with_bias=True)

    def test_equivalent_with_bias_and_row_normalization(self) -> None:
        self._check(preserve=True, with_bias=True)


if __name__ == "__main__":
    unittest.main()
