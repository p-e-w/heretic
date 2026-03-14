# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests that verify the PyTorch code path is unaffected when MLX is unavailable.

Uses unittest.mock to simulate MLX not being installed.
"""

import builtins
import importlib
import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helper: make `import mlx` / `import mlx_lm` raise ImportError
# ---------------------------------------------------------------------------

_real_import = builtins.__import__


def _import_no_mlx(name, *args, **kwargs):
    """Drop-in __import__ replacement that blocks mlx and mlx_lm."""
    if name == "mlx" or name.startswith("mlx.") or name == "mlx_lm" or name.startswith("mlx_lm."):
        raise ImportError(f"Mocked: No module named '{name}'")
    return _real_import(name, *args, **kwargs)


# ---------------------------------------------------------------------------
# 1. Lazy MLX imports don't crash when mlx is absent
# ---------------------------------------------------------------------------


class TestLazyImportsSafe:
    """The mlx_model module itself must be importable even without mlx."""

    def test_mlx_model_module_imports(self):
        """Importing heretic.mlx_model must not raise when mlx is missing."""
        # The module-level code only does lazy imports via functions, plus
        # standard-library / torch imports, so this should always succeed.
        from heretic import mlx_model  # noqa: F401

    def test_import_mlx_function_raises(self):
        """_import_mlx() must raise ImportError when mlx is absent."""
        from heretic.mlx_model import _import_mlx

        with patch("builtins.__import__", side_effect=_import_no_mlx):
            with pytest.raises(ImportError):
                _import_mlx()

    def test_import_mlx_lm_function_raises(self):
        """_import_mlx_lm() must raise ImportError when mlx_lm is absent."""
        from heretic.mlx_model import _import_mlx_lm

        with patch("builtins.__import__", side_effect=_import_no_mlx):
            with pytest.raises(ImportError):
                _import_mlx_lm()


# ---------------------------------------------------------------------------
# 2. resolve_backend() falls back to PYTORCH when MLX is unavailable
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    """Create a Settings instance without CLI arg parsing interfering."""
    from heretic.config import Settings

    with patch("sys.argv", ["test"]):
        return Settings(**overrides)


class TestResolveBackendFallback:
    """resolve_backend() must return PYTORCH when MLX is not available."""

    def test_auto_backend_falls_back_to_pytorch(self):
        from heretic.config import Backend

        settings = _make_settings(model="/tmp/fake-model", backend=Backend.AUTO)

        with patch("heretic.mlx_model.is_mlx_available", return_value=False), \
             patch("heretic.mlx_model.is_mlx_model", return_value=False):
            from heretic.main import resolve_backend
            result = resolve_backend(settings)

        assert result == Backend.PYTORCH

    def test_explicit_pytorch_backend(self):
        from heretic.config import Backend

        settings = _make_settings(model="/tmp/fake-model", backend=Backend.PYTORCH)

        from heretic.main import resolve_backend
        result = resolve_backend(settings)

        assert result == Backend.PYTORCH

    def test_explicit_mlx_backend_raises_when_unavailable(self):
        from heretic.config import Backend

        settings = _make_settings(model="/tmp/fake-model", backend=Backend.MLX)

        with patch("heretic.mlx_model.is_mlx_available", return_value=False):
            from heretic.main import resolve_backend
            with pytest.raises(RuntimeError, match="MLX backend requested"):
                resolve_backend(settings)


# ---------------------------------------------------------------------------
# 3. PyTorch Model class import chain works
# ---------------------------------------------------------------------------


class TestPytorchModelImport:
    """The standard PyTorch Model class must remain importable."""

    def test_model_class_importable(self):
        from heretic.model import Model
        assert Model is not None

    def test_model_is_a_class(self):
        from heretic.model import Model
        assert isinstance(Model, type)

    def test_abliteration_parameters_importable(self):
        from heretic.model import AbliterationParameters
        assert AbliterationParameters is not None


# ---------------------------------------------------------------------------
# 4. is_mlx_available() returns False when mlx is mocked as missing
# ---------------------------------------------------------------------------


class TestIsMlxAvailable:
    def test_returns_false_when_mlx_missing(self):
        from heretic.mlx_model import is_mlx_available

        with patch("builtins.__import__", side_effect=_import_no_mlx):
            assert is_mlx_available() is False

    def test_does_not_raise_when_mlx_missing(self):
        from heretic.mlx_model import is_mlx_available

        with patch("builtins.__import__", side_effect=_import_no_mlx):
            # Must not raise -- it should catch ImportError internally
            result = is_mlx_available()
            assert result is False


# ---------------------------------------------------------------------------
# 5. is_mlx_model() returns False when mlx is mocked as missing
# ---------------------------------------------------------------------------


class TestIsMlxModel:
    def test_returns_false_when_mlx_missing_and_path_is_dir(self, tmp_path):
        """Even with a valid-looking model dir, is_mlx_model should return
        False when mlx_lm cannot be imported."""
        import json

        config = {
            "model_type": "llama",
            "hidden_size": 4096,
            "vocab_size": 32000,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        # Create a fake safetensors file (no pytorch bin)
        (tmp_path / "model.safetensors").write_bytes(b"fake")

        from heretic.mlx_model import is_mlx_model

        with patch("builtins.__import__", side_effect=_import_no_mlx):
            assert is_mlx_model(str(tmp_path)) is False

    def test_returns_false_for_nonexistent_path(self):
        from heretic.mlx_model import is_mlx_model

        assert is_mlx_model("/nonexistent/path/to/model") is False

    def test_returns_false_for_file_not_dir(self, tmp_path):
        fake_file = tmp_path / "not_a_dir.bin"
        fake_file.write_bytes(b"data")

        from heretic.mlx_model import is_mlx_model

        assert is_mlx_model(str(fake_file)) is False
