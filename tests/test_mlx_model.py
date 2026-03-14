# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the MLX model backend.

Unit tests use lightweight mocks. Integration tests (marked @pytest.mark.integration)
require a real MLX model on disk and Apple Silicon hardware.
"""

import math
import os
import sys
from unittest.mock import patch

import pytest
import torch

# Skip entire module on non-macOS or missing mlx
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlx_settings():
    """Minimal Settings object pointing at the MLX model."""
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
    """Load the MLX model once for all integration tests in this module."""
    pytest.importorskip("mlx")
    if not HAS_MODEL:
        pytest.skip("MLX model not found at " + MLX_MODEL_PATH)

    from heretic.mlx_model import MLXModel

    return MLXModel(mlx_settings)


# ===========================================================================
# UNIT TESTS — Config & Backend detection
# ===========================================================================


class TestBackendConfig:
    def test_backend_enum_exists(self):
        from heretic.config import Backend

        assert hasattr(Backend, "AUTO")
        assert hasattr(Backend, "PYTORCH")
        assert hasattr(Backend, "MLX")

    def test_settings_has_backend_field(self):
        from heretic.config import Backend, Settings

        with patch("sys.argv", ["test"]):
            s = Settings(model="test-model")
        assert s.backend == Backend.AUTO  # default

    def test_settings_accepts_mlx_backend(self):
        from heretic.config import Backend, Settings

        with patch("sys.argv", ["test"]):
            s = Settings(model="test-model", backend=Backend.MLX)
        assert s.backend == Backend.MLX


# ===========================================================================
# UNIT TESTS — MLXModel interface contract
# ===========================================================================


@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
class TestMLXModelInterface:
    """Verify MLXModel has all the methods/attributes the main loop expects."""

    def test_class_exists(self):
        from heretic.mlx_model import MLXModel

        assert MLXModel is not None

    def test_has_required_methods(self):
        from heretic.mlx_model import MLXModel

        required = [
            "get_layers",
            "get_layer_modules",
            "get_abliterable_components",
            "get_responses",
            "get_responses_batched",
            "get_residuals",
            "get_residuals_batched",
            "get_logprobs",
            "get_logprobs_batched",
            "abliterate",
            "reset_model",
            "get_merged_model",
            "stream_chat_response",
        ]
        for method in required:
            assert hasattr(MLXModel, method), f"Missing method: {method}"

    def test_detect_mlx_model(self):
        """Backend detection should identify MLX models by config.json model_type."""
        from heretic.mlx_model import is_mlx_model

        if HAS_MODEL:
            assert is_mlx_model(MLX_MODEL_PATH) is True
        # Non-existent path should return False
        assert is_mlx_model("/nonexistent/path") is False


# ===========================================================================
# INTEGRATION TESTS — Require real model + Apple Silicon
# ===========================================================================


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
@pytest.mark.skipif(not HAS_MODEL, reason="MLX model not available")
class TestMLXModelLoading:
    def test_model_loads_successfully(self, mlx_model):
        assert mlx_model is not None
        assert mlx_model.tokenizer is not None

    def test_has_correct_layer_count(self, mlx_model):
        layers = mlx_model.get_layers()
        # Qwen3-Coder-30B-A3B has 48 layers
        assert len(layers) == 48

    def test_tokenizer_has_pad_token(self, mlx_model):
        assert mlx_model.tokenizer.pad_token is not None

    def test_tokenizer_padding_side(self, mlx_model):
        assert mlx_model.tokenizer.padding_side == "left"

    def test_response_prefix_initialized(self, mlx_model):
        assert isinstance(mlx_model.response_prefix, str)

    def test_needs_reload_initialized(self, mlx_model):
        assert mlx_model.needs_reload is False


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
@pytest.mark.skipif(not HAS_MODEL, reason="MLX model not available")
class TestMLXModelLayerIntrospection:
    def test_get_layer_modules_returns_dict(self, mlx_model):
        modules = mlx_model.get_layer_modules(0)
        assert isinstance(modules, dict)
        assert len(modules) > 0

    def test_layer_modules_have_expected_components(self, mlx_model):
        components = mlx_model.get_abliterable_components()
        assert "attn.o_proj" in components
        assert "mlp.down_proj" in components

    def test_attn_o_proj_is_module(self, mlx_model):
        modules = mlx_model.get_layer_modules(0)
        assert "attn.o_proj" in modules
        o_proj_list = modules["attn.o_proj"]
        assert len(o_proj_list) == 1
        # Should be an mlx.nn.Module
        assert isinstance(o_proj_list[0], nn.Module)

    def test_moe_layer_has_multiple_down_proj(self, mlx_model):
        """MoE layers should expose the switch_mlp.down_proj as a module."""
        # Check a MoE layer (layer 0 should be MoE for Qwen3-Coder-30B)
        modules = mlx_model.get_layer_modules(0)
        if "mlp.down_proj" in modules:
            down_proj_list = modules["mlp.down_proj"]
            # For MoE it's the SwitchLinear, for dense it's regular Linear
            assert len(down_proj_list) >= 1


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
@pytest.mark.skipif(not HAS_MODEL, reason="MLX model not available")
class TestMLXModelGeneration:
    def test_get_responses_returns_strings(self, mlx_model):
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Say hi")]
        responses = mlx_model.get_responses(prompts, skip_special_tokens=True)
        assert isinstance(responses, list)
        assert len(responses) == 1
        assert isinstance(responses[0], str)
        assert len(responses[0]) > 0

    def test_get_responses_batched(self, mlx_model):
        from heretic.utils import Prompt

        prompts = [
            Prompt(system="You are helpful.", user="Say hi"),
            Prompt(system="You are helpful.", user="Say bye"),
        ]
        responses = mlx_model.get_responses_batched(prompts, skip_special_tokens=True)
        assert len(responses) == 2
        for r in responses:
            assert isinstance(r, str)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
@pytest.mark.skipif(not HAS_MODEL, reason="MLX model not available")
class TestMLXModelResiduals:
    def test_get_residuals_returns_torch_tensor(self, mlx_model):
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]
        residuals = mlx_model.get_residuals(prompts)
        assert isinstance(residuals, torch.Tensor)

    def test_residuals_shape(self, mlx_model):
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]
        residuals = mlx_model.get_residuals(prompts)
        num_layers = len(mlx_model.get_layers())
        # Shape: (batch, num_layers + 1, hidden_size)
        # +1 because embeddings are included
        assert residuals.shape[0] == 1  # batch
        assert residuals.shape[1] == num_layers + 1  # layers + embeddings
        assert residuals.shape[2] == 2048  # hidden_size for Qwen3-Coder-30B

    def test_residuals_dtype_float32(self, mlx_model):
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]
        residuals = mlx_model.get_residuals(prompts)
        assert residuals.dtype == torch.float32

    def test_residuals_batched(self, mlx_model):
        from heretic.utils import Prompt

        prompts = [
            Prompt(system="You are helpful.", user="Hello"),
            Prompt(system="You are helpful.", user="World"),
        ]
        residuals = mlx_model.get_residuals_batched(prompts)
        assert residuals.shape[0] == 2


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
@pytest.mark.skipif(not HAS_MODEL, reason="MLX model not available")
class TestMLXModelLogprobs:
    def test_get_logprobs_returns_torch_tensor(self, mlx_model):
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]
        logprobs = mlx_model.get_logprobs(prompts)
        assert isinstance(logprobs, torch.Tensor)

    def test_logprobs_shape(self, mlx_model):
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]
        logprobs = mlx_model.get_logprobs(prompts)
        # Shape: (batch, vocab_size)
        assert logprobs.shape[0] == 1
        # Qwen3 vocab size is 151936
        assert logprobs.shape[1] > 100000

    def test_logprobs_are_valid(self, mlx_model):
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]
        logprobs = mlx_model.get_logprobs(prompts)
        # Log probs should be <= 0
        assert logprobs.max().item() <= 0.0 + 1e-5
        # Should sum to ~1 in probability space
        probs_sum = logprobs.exp().sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-3)

    def test_logprobs_batched(self, mlx_model):
        from heretic.utils import Prompt

        prompts = [
            Prompt(system="You are helpful.", user="Hello"),
            Prompt(system="You are helpful.", user="World"),
        ]
        logprobs = mlx_model.get_logprobs_batched(prompts)
        assert logprobs.shape[0] == 2


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
@pytest.mark.skipif(not HAS_MODEL, reason="MLX model not available")
class TestMLXModelAbliteration:
    def test_abliterate_modifies_behavior(self, mlx_model):
        """Abliteration with nonzero weight should change model outputs."""
        from heretic.model import AbliterationParameters
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]

        # Get baseline logprobs
        baseline = mlx_model.get_logprobs(prompts)

        # Create a fake refusal direction (random unit vector)
        num_layers = len(mlx_model.get_layers())
        hidden_size = 2048
        refusal_dirs = torch.randn(num_layers + 1, hidden_size)
        refusal_dirs = torch.nn.functional.normalize(refusal_dirs, p=2, dim=1)

        # Abliterate with a moderate weight
        components = mlx_model.get_abliterable_components()
        parameters = {}
        for comp in components:
            parameters[comp] = AbliterationParameters(
                max_weight=1.0,
                max_weight_position=float(num_layers // 2),
                min_weight=0.5,
                min_weight_distance=float(num_layers // 4),
            )

        mlx_model.abliterate(refusal_dirs, float(num_layers // 2), parameters)

        # Logprobs should differ after abliteration
        modified = mlx_model.get_logprobs(prompts)
        assert not torch.allclose(baseline, modified, atol=1e-3)

        # Reset for other tests
        mlx_model.reset_model()

    def test_reset_restores_original_behavior(self, mlx_model):
        """After reset, model should produce same outputs as before abliteration."""
        from heretic.model import AbliterationParameters
        from heretic.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]

        # Get baseline
        baseline = mlx_model.get_logprobs(prompts)

        # Abliterate
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
                min_weight_distance=float(num_layers // 4),
            )

        mlx_model.abliterate(refusal_dirs, float(num_layers // 2), parameters)

        # Reset
        mlx_model.reset_model()

        # Should match baseline
        restored = mlx_model.get_logprobs(prompts)
        assert torch.allclose(baseline, restored, atol=1e-4)

    def test_abliterate_per_layer_direction(self, mlx_model):
        """Abliteration with per-layer directions (direction_index=None)."""
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
                min_weight_distance=float(num_layers // 4),
            )

        # direction_index=None means per-layer directions
        mlx_model.abliterate(refusal_dirs, None, parameters)

        modified = mlx_model.get_logprobs(prompts)
        assert not torch.allclose(baseline, modified, atol=1e-3)

        mlx_model.reset_model()


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
@pytest.mark.skipif(not HAS_MODEL, reason="MLX model not available")
class TestMLXModelChat:
    def test_stream_chat_response(self, mlx_model):
        chat = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello in one word."},
        ]
        response = mlx_model.stream_chat_response(chat)
        assert isinstance(response, str)
        assert len(response) > 0


@pytest.mark.integration
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
@pytest.mark.skipif(not HAS_MODEL, reason="MLX model not available")
class TestMLXModelSave:
    def test_get_merged_model_returns_object(self, mlx_model):
        merged = mlx_model.get_merged_model()
        assert merged is not None
