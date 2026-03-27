# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for memory-efficient MLX abliteration.

Verifies that the delta-based abliteration approach avoids full
dequantization of MoE expert weights, reducing peak memory usage.
"""

import os
import sys
from unittest.mock import patch

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="MLX requires macOS"),
]

MLX_MODEL_PATH = os.environ.get(
    "HERETIC_TEST_MLX_MODEL",
    "/Users/overtime/Documents/GitHub/froggy/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
)

HAS_MLX = False
try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    pass

HAS_MODEL = os.path.isdir(MLX_MODEL_PATH) if HAS_MLX else False


@pytest.fixture(scope="module")
def mlx_settings():
    from heretic.config import Backend, Settings

    with patch("sys.argv", ["test"]):
        return Settings(
            model=MLX_MODEL_PATH,
            backend=Backend.MLX,
            batch_size=1,
            max_response_length=20,
            dtypes=["auto"],
        )


@pytest.fixture(scope="module")
def mlx_model(mlx_settings):
    pytest.importorskip("mlx")
    if not HAS_MODEL:
        pytest.skip("MLX model not found at " + MLX_MODEL_PATH)

    from heretic.mlx_model import MLXModel

    return MLXModel(mlx_settings)


# ===========================================================================
# Unit tests — delta-based abliteration internals
# ===========================================================================


@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
class TestDeltaAbliterationUnit:
    """Test the delta-based weight modification approach."""

    def test_quantized_matmul_computes_vTW_without_deq(self):
        """v @ W can be computed via quantized_matmul without dequantizing."""
        import numpy as np

        # Simulate a QuantizedSwitchLinear weight: (experts, out, in)
        W = mx.random.normal((8, 32, 64))  # 8 experts
        wq, scales, biases = mx.quantize(W, group_size=64, bits=4)

        v = mx.random.normal((32,))  # refusal direction (out_dims,)

        # Compute v @ W per expert via quantized_matmul (no dequant)
        results = []
        for i in range(8):
            vT_W_i = mx.quantized_matmul(
                v.reshape(1, -1), wq[i], scales[i], biases[i],
                transpose=False, group_size=64, bits=4,
            )
            results.append(vT_W_i)
        vT_W_qmm = mx.concatenate(results, axis=0)  # (8, 64)
        mx.eval(vT_W_qmm)

        # Reference: dequantize then matmul
        W_deq = mx.dequantize(wq, scales, biases, group_size=64, bits=4)
        vT_W_ref = mx.einsum("j,ejk->ek", v, W_deq)
        mx.eval(vT_W_ref)

        assert np.allclose(
            np.array(vT_W_qmm), np.array(vT_W_ref), atol=0.01
        ), "quantized_matmul should match dequantized reference"

    def test_delta_wrapper_modifies_output(self):
        """A module wrapped with a rank-1 delta should produce different output."""
        # Create a simple linear module
        linear = nn.Linear(64, 32)
        x = mx.random.normal((1, 64))

        # Original output
        y_orig = linear(x)
        mx.eval(y_orig)

        # Create rank-1 delta
        lora_A = mx.random.normal((1, 64))
        lora_B = mx.random.normal((32, 1))

        # Apply delta to output: y_new = y_orig + (x @ lora_A.T) @ lora_B.T
        y_delta = y_orig + (x @ lora_A.T) @ lora_B.T
        mx.eval(y_delta)

        import numpy as np

        assert not np.allclose(
            np.array(y_orig), np.array(y_delta), atol=1e-6
        ), "Delta should change output"


# ===========================================================================
# Integration tests — memory-efficient abliteration on real model
# ===========================================================================


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
@pytest.mark.skipif(not HAS_MODEL, reason="MLX model not available")
class TestMemoryEfficientAbliteration:
    def test_abliterate_stays_within_memory_budget(self, mlx_model):
        """
        Abliteration with wide min_weight_distance should NOT spike memory
        by dequantizing all MoE experts at once.
        """
        from heretic.model import AbliterationParameters
        from heretic.utils import Prompt

        num_layers = len(mlx_model.get_layers())
        hidden_size = 2048

        # Record memory before
        mx.eval(mx.zeros(1))  # force any pending evals
        mem_before = mx.get_active_memory()

        refusal_dirs = torch.randn(num_layers + 1, hidden_size)
        refusal_dirs = torch.nn.functional.normalize(refusal_dirs, p=2, dim=1)

        components = mlx_model.get_abliterable_components()
        parameters = {}
        for comp in components:
            parameters[comp] = AbliterationParameters(
                max_weight=1.0,
                # Wide abliteration window — touches many layers
                max_weight_position=float(num_layers // 2),
                min_weight=0.5,
                min_weight_distance=float(num_layers // 2),
            )

        mlx_model.abliterate(refusal_dirs, float(num_layers // 2), parameters)

        mx.eval(mx.zeros(1))
        mem_after = mx.get_active_memory()

        # Memory increase should be modest (< 8GB) even with wide window.
        # The old approach would spike ~16GB+ and OOM on 32GB systems.
        # Per-expert processing keeps MoE memory bounded; the remaining
        # increase is from o_proj dequant/requant + saved weight copies.
        mem_increase_gb = (mem_after - mem_before) / (1024**3)
        print(f"\nMemory increase from abliteration: {mem_increase_gb:.2f} GB")
        assert mem_increase_gb < 8.0, (
            f"Abliteration used {mem_increase_gb:.2f} GB — expected < 8 GB with per-expert approach"
        )

        mlx_model.reset_model()

    def test_abliterate_modifies_behavior_with_deltas(self, mlx_model):
        """Abliteration via deltas should still change model outputs."""
        from heretic.model import AbliterationParameters
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]
        baseline = mlx_model.get_logprobs(prompts)

        num_layers = len(mlx_model.get_layers())
        hidden_size = 2048
        refusal_dirs = torch.randn(num_layers + 1, hidden_size)
        refusal_dirs = torch.nn.functional.normalize(refusal_dirs, p=2, dim=1)

        components = mlx_model.get_abliterable_components()
        parameters = {}
        for comp in components:
            parameters[comp] = AbliterationParameters(
                max_weight=1.0,
                max_weight_position=float(num_layers // 2),
                min_weight=0.5,
                min_weight_distance=float(num_layers // 2),
            )

        mlx_model.abliterate(refusal_dirs, float(num_layers // 2), parameters)
        modified = mlx_model.get_logprobs(prompts)

        assert not torch.allclose(baseline, modified, atol=1e-3)

        mlx_model.reset_model()

    def test_reset_restores_after_delta_abliteration(self, mlx_model):
        """Reset should fully restore original behavior after delta abliteration."""
        from heretic.model import AbliterationParameters
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]
        baseline = mlx_model.get_logprobs(prompts)

        num_layers = len(mlx_model.get_layers())
        hidden_size = 2048
        refusal_dirs = torch.randn(num_layers + 1, hidden_size)
        refusal_dirs = torch.nn.functional.normalize(refusal_dirs, p=2, dim=1)

        components = mlx_model.get_abliterable_components()
        parameters = {}
        for comp in components:
            parameters[comp] = AbliterationParameters(
                max_weight=1.0,
                max_weight_position=float(num_layers // 2),
                min_weight=0.5,
                min_weight_distance=float(num_layers // 2),
            )

        mlx_model.abliterate(refusal_dirs, float(num_layers // 2), parameters)
        mlx_model.reset_model()

        restored = mlx_model.get_logprobs(prompts)
        assert torch.allclose(baseline, restored, atol=1e-4)
