# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NoReturn

from pydantic import BaseModel

from heretic.plugin import Plugin
from heretic.utils import Prompt, load_prompts

if TYPE_CHECKING:
    from torch import Tensor

    from .config import DatasetSpecification
    from .config import Settings as HereticSettings
    from .model import Model

FinishReason = Literal["len", "eos", "unk", "empty"]


@dataclass
class Score:
    """
    Result of evaluating a scorer.

    - `value`: scalar value used for optimization (if enabled)
    - `cli_display`: formatted value shown to the user in logs/console
    - `md_display`: formatted value in the HF model card
    """

    name: str
    value: float
    cli_display: str
    md_display: str


class Context:
    """
    Runtime context passed to scorers

    Provides scorer-safe access to the model.

    Scorers must use `get_responses(...)`, `get_logits(...)`, etc.
    Direct access to the underlying Model is intentionally not exposed.
    """

    def __init__(self, settings: HereticSettings, model: Model) -> None:
        self._model = model
        self._settings = settings
        self._responses_cache: dict[tuple[tuple[str, str], ...], list[str]] = {}
    def _cache_key(self, prompts: list[Prompt]) -> tuple[tuple[str, str], ...]:
        return tuple((p.system, p.user) for p in prompts)

    def get_responses(self, prompts: list[Prompt]) -> list[str]:
        """Get model responses (cached within this context)."""
        key = self._cache_key(prompts)
        if key not in self._responses_cache:
            self._responses_cache[key] = self._model.get_responses_batched(prompts)
        return self._responses_cache[key]

    def get_logits(self, prompts: list[Prompt]) -> "Tensor":
        return self._model.get_logits_batched(prompts)

    def get_residuals(self, prompts: list[Prompt]) -> "Tensor":
        return self._model.get_residuals_batched(prompts)

    def load_prompts(self, specification: DatasetSpecification):
        return load_prompts(self._settings, specification)


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
          handled by `Scorer.__init__` and an optional `init(ctx)` method.
        - Scorer plugins may define `settings: <BaseModelSubclass>` to declare a settings schema.
        """
        if "__init__" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not define __init__(). "
                "Use an optional start(ctx) method for scorer-specific initialization."
            )

    def __init__(
        self,
        heretic_settings: HereticSettings,
        settings: BaseModel | None = None,
    ):
        super().__init__(settings=settings)

        # Scorers that declare a settings schema should always receive
        # validated plugin settings from the evaluator.
        settings_model = self.__class__.get_settings_model()
        if settings_model is not None:
            if settings is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires settings to be validated"
                )
            if not isinstance(settings, settings_model):
                raise TypeError(
                    f"{self.__class__.__name__}.settings must be an instance of "
                    f"{settings_model.__name__}"
                )

        self.heretic_settings = heretic_settings

    @property
    def model(self) -> NoReturn:  # type: ignore[override]
        raise AttributeError(
            "Direct access to the underlying Model is intentionally not exposed to scorers. "
            "Use the passed Context (e.g. `ctx.get_responses(...)`) inside `get_score(...)` / `start(ctx)`."
        )

    def init(self, ctx: Context) -> None:
        """
        Runs before the scorer starts scoring responses.

        Override this in subclasses to do one-time setup (e.g. load prompts, compute
        baselines).
        """
        return None

    def get_score(self, ctx: Context) -> Score:
        """
        Return a `Score` given the evaluation context.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_score()"
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        """
        Calculates a baseline score.

        Defaults to the current `get_score(...)` implementation and can be
        overridden by scorers that need a distinct baseline.
        """
        return self.get_score(ctx)
