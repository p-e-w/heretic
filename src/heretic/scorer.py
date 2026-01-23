# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

from abc import ABC
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from pydantic import BaseModel

from heretic.plugin import Plugin
from heretic.utils import Prompt

if TYPE_CHECKING:
    from torch import Tensor

    from .config import Settings as HereticSettings
    from .model import Model

FinishReason = Literal["len", "eos", "unk", "empty"]


@dataclass
class Score:
    """
    Result of evaluating a scorer.

    - `value`: scalar value used for optimization (if enabled)
    - `display`: string shown to the user in logs/console
    """

    name: str
    value: float
    display: str


@dataclass(frozen=True)
class Response:
    """
    A single model response to a single prompt.
    """

    prompt: Prompt
    text: str
    finish_reason: FinishReason
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """
    Runtime context passed to scorers

    Provides scorer-safe access to the model.

    Scorers must use `get_responses(...)`, `get_logits(...)`, etc.
    Direct access to the underlying Model is intentionally not exposed.
    """

    settings: "HereticSettings"
    model: InitVar["Model"]

    _model: "Model" = field(init=False, repr=False)

    _responses_cache: dict[tuple[tuple[str, str], ...], list[Response]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self, model: "Model") -> None:
        self._model = model

    def _cache_key(self, prompts: list[Prompt]) -> tuple[tuple[str, str], ...]:
        return tuple((p.system, p.user) for p in prompts)

    def get_responses(self, prompts: list[Prompt]) -> list[Response]:
        """Get model responses (cached within this context)."""
        key = self._cache_key(prompts)
        if key not in self._responses_cache:
            self._responses_cache[key] = self._model.get_responses_batched(prompts)
        return self._responses_cache[key]

    def get_logits(self, prompts: list[Prompt]) -> "Tensor":
        return self._model.get_logits_batched(prompts)

    def get_residuals(self, prompts: list[Prompt]) -> "Tensor":
        return self._model.get_residuals_batched(prompts)


class Scorer(Plugin, ABC):
    """
    Abstract base class for scorer plugins.

    Scorers evaluate model behavior and return a Score.

    Example: counting refusals, measuring KL divergence, etc.
    """

    @classmethod
    def validate_contract(cls) -> None:
        """
        Validate the scorer contract.

        - Scorer plugins must not define a constructor (`__init__`). Initialization is
          handled by `Scorer.__init__` and an optional `start(ctx)` hook.
        - Scorer plugins may define a nested `PluginSettings` model (pydantic.BaseModel).
        """
        super().validate_contract()

        if "__init__" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not define __init__(). "
                "Use an optional start(ctx) method for scorer-specific initialization."
            )

    def __init__(
        self,
        settings: "HereticSettings",
        plugin_settings: BaseModel | None = None,
    ):
        super().__init__(plugin_settings=plugin_settings)

        # Scorers that define a nested `PluginSettings` model should always receive
        # validated plugin settings from the evaluator.
        settings_model = getattr(self.__class__, "PluginSettings", None)
        if settings_model is not None:
            if plugin_settings is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires plugin settings to be validated"
                )
            if not isinstance(plugin_settings, settings_model):
                raise TypeError(
                    f"{self.__class__.__name__}.plugin_settings must be an instance of "
                    f"{settings_model.__name__}"
                )

        self.heretic_settings = settings

    @property
    def model(self) -> NoReturn:  # type: ignore[override]
        raise AttributeError(
            "Direct access to the underlying Model is intentionally not exposed to scorers. "
            "Use the passed Context (e.g. `ctx.get_responses(...)`) inside `get_score(...)` / `start(ctx)`."
        )

    def start(self, ctx: Context) -> None:
        """
        Optional scorer initialization hook.

        Override this in subclasses to do one-time setup (e.g. load prompts, compute
        baselines).
        """
        return None

    def get_score(self, ctx: Context) -> Score:
        """
        Evaluate this scorer given the evaluation context.

        Override this method in subclasses. Use `self.make_result()` to build
        the return value with settings-derived defaults.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_score()"
        )

    def make_result(self, value: float, display: str | None = None) -> Score:
        """
        Helper to build Score with settings-derived defaults.

        Args:
            value: The numeric score value.
            display: Human-readable string. Defaults to str(value).

        Returns:
            Score with the class name as default (pending further labelling in the evaluator)
        """
        return Score(
            name=self.__class__.__name__,
            value=value,
            display=display if display is not None else str(value),
        )
