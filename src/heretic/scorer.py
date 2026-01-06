from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from heretic.plugin import Plugin
from heretic.utils import Prompt

if TYPE_CHECKING:
    from .config import Settings as HereticSettings
    from .model import Model

FinishReason = Literal["len", "eos", "unk", "empty"]


@dataclass(frozen=True)
class ResponseText:
    prompt: Prompt
    response_text: str
    finish_reason: FinishReason


@dataclass(frozen=True)
class ResponseTokenization:
    input_ids: list[int]
    response_ids: list[int]
    response_tokens: list[str]


@dataclass(frozen=True)
class ResponseTokenScores:
    token_logprobs: list[float]
    token_logits: list[float]


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


@dataclass(frozen=True)
class Response:
    """
    A model response with structured metadata.

    Metadata is split into grouped objects; within each group, all fields are
    fully populated (no `None`s).
    """

    text: ResponseText
    tokenization: ResponseTokenization
    token_scores: ResponseTokenScores

    # Convenience accessors (non-optional, since groups are fully populated).
    @property
    def response_text(self) -> str:
        return self.text.response_text

    @property
    def prompt_text(self) -> str:
        # Convenience: most call sites treat "prompt text" as the user prompt.
        return self.text.prompt.user

    @property
    def prompt(self) -> Prompt:
        return self.text.prompt

    @property
    def finish_reason(self) -> FinishReason:
        return self.text.finish_reason

    @property
    def input_ids(self) -> list[int]:
        return self.tokenization.input_ids

    @property
    def response_ids(self) -> list[int]:
        return self.tokenization.response_ids

    @property
    def response_tokens(self) -> list[str]:
        return self.tokenization.response_tokens

    @property
    def token_logprobs(self) -> list[float]:
        return self.token_scores.token_logprobs

    @property
    def token_logits(self) -> list[float]:
        return self.token_scores.token_logits


@dataclass
class EvaluationContext:
    """
    Runtime context passed to scorers during evaluation.

    Provides access to prompts, baseline logprobs, and the model.
    Scorers needing residuals can access them via `model.good_residuals`.
    """

    settings: "HereticSettings"
    model: "Model"
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
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

    # General helpers (Prompt-based) for plugin flexibility.
    def responses(self, prompts: list[Prompt]) -> list[Response]:
        return self.model.get_responses_batched(prompts)

    def response_text(self, prompts: list[Prompt]) -> list[ResponseText]:
        return [r.text for r in self.responses(prompts)]

    def response_tokenization(self, prompts: list[Prompt]) -> list[ResponseTokenization]:
        return [r.tokenization for r in self.responses(prompts)]

    def response_token_scores(self, prompts: list[Prompt]) -> list[ResponseTokenScores]:
        return [r.token_scores for r in self.responses(prompts)]

    # Convenience accessors returning fully-populated grouped metadata
    def bad_response(self, index: int) -> Response:
        return self.bad_responses()[index]

    def bad_response_text(self) -> list[ResponseText]:
        return [r.text for r in self.bad_responses()]

    def bad_response_text_at(self, index: int) -> ResponseText:
        return self.bad_response(index).text

    def bad_response_tokenization(self) -> list[ResponseTokenization]:
        return [r.tokenization for r in self.bad_responses()]

    def bad_response_tokenization_at(self, index: int) -> ResponseTokenization:
        return self.bad_response(index).tokenization

    def bad_response_token_scores(self) -> list[ResponseTokenScores]:
        return [r.token_scores for r in self.bad_responses()]

    def bad_response_token_scores_at(self, index: int) -> ResponseTokenScores:
        return self.bad_response(index).token_scores


class Scorer(Plugin, ABC):
    """
    Abstract base class for scorer plugins.

    Scorers evaluate model behavior and return a MetricResult.

    Example: counting refusals, measuring KL divergence, etc.
    """

    class Settings(BaseModel):
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
        Override to request grouped metadata:
        - 'token_scores' to enable token logprobs/logits
        """
        return set()

    @staticmethod
    def required_context_metadata_fields() -> set[str]:
        """
        Context-level metadata fields needed by this scorer.
        Override to request fields like 'good_residuals', 'bad_residuals', etc.
        """
        return set()
