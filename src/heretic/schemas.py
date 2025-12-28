from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal

if TYPE_CHECKING:
    from .config import Settings
    from .model import Model

@dataclass(frozen=True)
class MetricResult:
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

    settings: "Settings"
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
