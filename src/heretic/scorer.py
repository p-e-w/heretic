# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>


from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from heretic.plugin import Plugin
from heretic.utils import Prompt

if TYPE_CHECKING:
    from .config import Settings as HereticSettings
    from .model import Model

FinishReason = Literal["len", "eos", "unk", "empty"]


@dataclass(frozen=True)
class Score:
    """
    Result of evaluating a scorer/metric.

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


@dataclass
class Context:
    """
    Runtime context passed to scorers

    Provides access to settings, the model and some convenience functions.
    """

    settings: "HereticSettings"
    model: "Model"

    _responses_cache: dict[tuple[tuple[str, str], ...], list[Response]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _cache_key(self, prompts: list[Prompt]) -> tuple[tuple[str, str], ...]:
        return tuple((p.system, p.user) for p in prompts)

    def responses(self, prompts: list[Prompt]) -> list[Response]:
        """Get model responses (cached within this context)."""
        key = self._cache_key(prompts)
        if key not in self._responses_cache:
            self._responses_cache[key] = self.model.get_responses_batched(prompts)
        return self._responses_cache[key]


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
          handled by `Scorer.__init__` and optional `start()` hook.
        - Scorer plugins may define a nested `Settings` model (pydantic.BaseModel).
        """
        super().validate_contract()

        if "__init__" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not define __init__(). "
                "Use an optional start() method for scorer-specific initialization."
            )

    def __init__(
        self,
        settings: "HereticSettings",
        model: "Model",
        plugin_settings: BaseModel | None = None,
        instance_name: str | None = None,
    ):
        super().__init__(plugin_settings=plugin_settings)

        # Scorers that define a nested `Settings` model should always receive
        # validated plugin settings from the evaluator.
        settings_model = getattr(self.__class__, "Settings", None)
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
        self.model = model

        if instance_name:
            self.instance_name = f"{self.__class__.__name__}.{instance_name}"
        else:
            self.instance_name = None

        # if the plugin declares a `settings` field,
        # put the validated settings object there
        annotations = getattr(self.__class__, "__annotations__", {}) or {}
        if "settings" in annotations:
            setattr(self, "settings", plugin_settings)

        # Optional scorer-specific initialization hook
        start = getattr(self, "start", None)
        if callable(start):
            start()

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
            value: The numeric metric value.
            display: Human-readable string. Defaults to str(value).

        Returns:
            Score with scorer-derived name.
        """
        return Score(
            name=self.instance_name if self.instance_name else self.__class__.__name__,
            value=value,
            display=display if display is not None else str(value),
        )

    def get_primary_prompt_count(self) -> int | None:
        """
        Optional helper for UIs/README generation.

        If this scorer evaluates on a primary prompt set (e.g. refusal-count on
        "bad evaluation prompts"), return its size. Otherwise return None.
        """
        return None
