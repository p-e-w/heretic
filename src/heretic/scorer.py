from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal

from pydantic import BaseModel, Field

from heretic.plugin import Plugin

if TYPE_CHECKING:
    from .config import Settings as HereticSettings
    from .model import Model

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
    use_in_optimizer: bool
    direction: Literal["minimize", "maximize"]


@dataclass
class Response:
    response_text: str | None = None
    prompt_text: str | None = None

    finish_reason: Literal["len", "eos", "unk", "empty"] | None = None

    # Tokenization
    input_ids: List[int] | None = None
    response_ids: List[int] | None = None
    response_tokens: List[str] | None = None
    response_offsets: List[tuple[int, int]] | None = None

    # Logprobs / uncertainty
    token_logprobs: List[float] | None = None
    token_logits: List[float] | None = None

    # Hidden states / residuals (optional, heavy)
    hidden_states: List[List[List[float]]] | None = None
    residuals_last_token_per_layer: List[List[float]] | None = None


@dataclass
class EvaluationContext:
    """
    Runtime context passed to scorers during evaluation.

    Provides access to prompts, baseline logprobs, and the model.
    Scorers needing residuals can access them via `model.good_residuals`.
    """

    settings: "HereticSettings"
    model: "Model"
    good_prompts: list[str]
    bad_prompts: list[str]
    base_good_logprobs: Any  # Tensor

    _bad_responses: list[Response] | None = field(default=None, init=False, repr=False)

    def bad_responses(self) -> list[Response]:
        """
        Lazily generate responses for `bad_prompts` once and cache them.
        Scorers call this to avoid recomputing expensive generations.
        """
        if self._bad_responses is None:
            self._bad_responses = self.model.get_responses_batched(self.bad_prompts)
        return self._bad_responses


class Scorer(Plugin, ABC):
    """
    Abstract base class for scorer plugins.

    Scorers evaluate model behavior and return a MetricResult.

    Example: counting refusals, measuring KL divergence, etc.
    """

    class Settings(Plugin.Settings):
        """Scorer-specific settings with optimizer configuration."""

        use_in_optimizer: bool = Field(
            default=True,
            description="If true, this scorer's value is used as an Optuna objective.",
        )
        direction: Literal["minimize", "maximize"] = Field(
            default="minimize",
            description="Whether Optuna should minimize or maximize this scorer's value.",
        )
        label: str | None = Field(
            default=None,
            description="Optional display label for this scorer/metric.",
        )

    def __init__(
        self,
        settings: "HereticSettings",
        model: "Model",
        plugin_settings: BaseModel | None = None,
    ):
        super().__init__(plugin_settings=plugin_settings)
        self.settings = settings
        self.model = model

    def evaluate(self, ctx: EvaluationContext) -> Score:
        """
        Evaluate this scorer given the evaluation context.

        Override this method in subclasses. Use `self.make_result()` to build
        the return value with settings-derived defaults.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate()"
        )

    def make_result(self, value: float, display: str | None = None) -> Score:
        """
        Helper to build MetricResult with settings-derived defaults.

        Args:
            value: The numeric metric value.
            display: Human-readable string. Defaults to str(value).

        Returns:
            MetricResult with name/direction/use_in_optimizer from plugin_settings.
        """
        ps = self.plugin_settings
        return Score(
            name=getattr(ps, "label", None) or self.name if ps else self.name,
            value=value,
            display=display if display is not None else str(value),
            use_in_optimizer=getattr(ps, "use_in_optimizer", True) if ps else True,
            direction=getattr(ps, "direction", "minimize") if ps else "minimize",
        )

    @staticmethod
    def required_response_metadata_fields() -> set[str]:
        """
        Response-level metadata fields needed by this scorer.
        Override to request fields like 'response_text', 'token_logprobs', etc.
        """
        return set()

    @staticmethod
    def required_context_metadata_fields() -> set[str]:
        """
        Context-level metadata fields needed by this scorer.
        Override to request fields like 'good_residuals', 'bad_residuals', etc.
        """
        return set()
