from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from optuna.study import StudyDirection
from pydantic import BaseModel

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
    - `direction`: what Optuna should do with the score
    """

    name: str
    value: float
    display: str
    direction: StudyDirection


@dataclass(frozen=True)
class Response:
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

    Provides access to settings, the model and some convenience functions.
    """

    settings: "HereticSettings"
    model: "Model"

    _responses_cache: dict[tuple[tuple[str, str], ...], list[Response]] = field(
        default_factory=dict, init=False, repr=False
    )
    _responses_cache_has_token_scores: set[tuple[tuple[str, str], ...]] = field(
        default_factory=set, init=False, repr=False
    )

    def _cache_key(self, prompts: list[Prompt]) -> tuple[tuple[str, str], ...]:
        return tuple((p.system, p.user) for p in prompts)

    def get_responses(self, prompts: list[Prompt]) -> list[Response]:
        """
        Get model responses (text + tokenization). Token scores may be empty unless
        `get_token_scores()` was used.
        """
        key = self._cache_key(prompts)
        if key not in self._responses_cache:
            self._responses_cache[key] = self.model.get_responses_batched(prompts)
        return self._responses_cache[key]

    def get_responses_with_token_scores(self, prompts: list[Prompt]) -> list[Response]:
        """Get model responses including per-token chosen logits/logprobs."""
        key = self._cache_key(prompts)
        if (
            key not in self._responses_cache
            or key not in self._responses_cache_has_token_scores
        ):
            self._responses_cache[key] = (
                self.model.get_responses_batched_with_token_scores(prompts)
            )
            self._responses_cache_has_token_scores.add(key)
        return self._responses_cache[key]

    def get_response_text(self, prompts: list[Prompt]) -> list[ResponseText]:
        return [r.text for r in self.get_responses(prompts)]

    def get_tokenization(self, prompts: list[Prompt]) -> list[ResponseTokenization]:
        return [r.tokenization for r in self.get_responses(prompts)]

    def get_token_scores(self, prompts: list[Prompt]) -> list[ResponseTokenScores]:
        return [r.token_scores for r in self.get_responses_with_token_scores(prompts)]

    def responses(self, prompts: list[Prompt]) -> list[Response]:
        return self.get_responses(prompts)

    def response_text(self, prompts: list[Prompt]) -> list[ResponseText]:
        return self.get_response_text(prompts)

    def response_tokenization(
        self, prompts: list[Prompt]
    ) -> list[ResponseTokenization]:
        return self.get_tokenization(prompts)

    def response_token_scores(self, prompts: list[Prompt]) -> list[ResponseTokenScores]:
        return self.get_token_scores(prompts)


class Scorer(Plugin, ABC):
    """
    Abstract base class for scorer plugins.

    Scorers evaluate model behavior and return a MetricResult.

    Example: counting refusals, measuring KL divergence, etc.
    """

    def __init__(
        self,
        settings: "HereticSettings",
        model: "Model",
        plugin_settings: BaseModel | None = None,
        direction: StudyDirection = StudyDirection.MINIMIZE,
        scale: float = 1.0,
        instance_name: str | None = None,
    ):
        super().__init__(plugin_settings=plugin_settings)
        self.settings = settings
        self.model = model
        self.direction = direction
        self.scale = scale
        if instance_name:
            self.instance_name = f"{self.__class__.__name__}.{instance_name}"
        else:
            self.instance_name = None
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
        return Score(
            name=self.instance_name if self.instance_name else self.__class__.__name__,
            value=value,
            display=display if display is not None else str(value),
            direction=self.direction,
        )

    def get_primary_prompt_count(self) -> int | None:
        """
        Optional helper for UIs/README generation.

        If this scorer evaluates on a primary prompt set (e.g. refusal-count on
        "bad evaluation prompts"), return its size. Otherwise return None.
        """
        return None
