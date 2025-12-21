from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from heretic.schemas import ContextMetadata, Response
from heretic.plugin import Plugin

if TYPE_CHECKING:
    from .config import Settings
    from .model import Model


class Scorer(Plugin, ABC):
    """
    Abstract base class for scorer plugins.

    Scorers take in LLM responses + optional metadata and output a score, e.g:
    "I'm sorry, but I can't assist with that request." -> 1.0 (refusal)
    """

    def __init__(
        self,
        settings: "Settings",
        model: "Model",
        context_metadata: ContextMetadata,
        plugin_settings=None,
    ):
        super().__init__(plugin_settings=plugin_settings)
        self.settings = settings
        self.model = model
        self.context_metadata = context_metadata

    @abstractmethod
    def score_batch(self, responses: list[Response]) -> float:
        """
        Scores a batch of responses.

        Args:
            responses: A list of response objects with metadata.
        Returns:
            A single float score for the whole batch.
        """
        raise NotImplementedError

    @staticmethod
    def required_response_metadata_fields() -> set[str]:
        """
        Response-level metadata fields needed by this scorer.
        """
        return set()

    @staticmethod
    def required_context_metadata_fields() -> set[str]:
        """
        Context-level metadata fields needed by this scorer.
        """
        return set()
