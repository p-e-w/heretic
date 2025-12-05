from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class GenerationStep:
    step_index: int
    token_id: int
    token: str
    logprob: float | None
    topk: List[Dict[str, float]] | None = None
    entropy: float | None = None

    # Optional internals
    last_hidden_state: List[float] | None = None
    residuals_per_layer: List[List[float]] | None = None
    attention_summary: Dict[str, Any] | None = None


@dataclass
class GenerationTrace:
    steps: List[GenerationStep]
    finish_reason: str | None = None


@dataclass
class ContextMetadata:
    # model and generation params
    system_prompt: str | None = None
    model_name: str | None = None
    generation_params: Dict[str, Any] | None = None


@dataclass
class ResponseMetadata:
    # basic prompt stuff
    prompt_text: str | None = None

    # multi-turn in the future?
    conversation_id: str | None = None
    turn_index: int | None = None
    role: str | None = None

    finish_reason: str | None = None

    # Tokenization
    input_ids: List[int] | None = None
    response_ids: List[int] | None = None
    response_tokens: List[str] | None = None
    response_offsets: List[tuple[int, int]] | None = None

    # Logprobs / uncertainty
    token_logprobs: List[float] | None = None
    token_logits: List[float] | None = None

    # Embeddings
    response_embedding: List[float] | None = None
    prompt_embedding: List[float] | None = None

    # Hidden states / residuals (optional, heavy)
    last_hidden_states: List[List[float]] | None = None
    residuals_last_token_per_layer: List[List[float]] | None = None

    # Arbitrary plugin-specific extra
    extra: Dict[str, Any] | None = None

    generation_steps: List[GenerationTrace] | None = None
