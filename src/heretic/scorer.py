from abc import ABC, abstractmethod
from typing import Any, Dict


class Scorer(ABC):
    """
    Abstract base class for scorers.
    Scorers take in tagged responses and output a score, e.g:
    "I'm sorry, but I can't assist with that request." -> 0.1
    """

    @abstractmethod
    def score_batch(self, tagged_responses: list[Dict[str, Any]]) -> float:
        pass
