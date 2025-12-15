from dataclasses import dataclass
from typing import Any, Dict, List, Literal


@dataclass
class ContextMetadata:
    # model and generation params
    generation_params: Dict[str, Any] | None = None
    
    # Internal vectors
    good_residuals: List[List[List[float]]] | None = None
    bad_residuals: List[List[List[float]]] | None = None


@dataclass
class Response:
    # basic prompt stuff
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
