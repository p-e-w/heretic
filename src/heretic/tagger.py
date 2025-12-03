# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from abc import ABC, abstractmethod
from heretic.schemas import ResponseMetadata
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from .config import Settings
    from .model import Model


class Tagger(ABC):
    """
    Abstract base class for tagger plugins.
    Taggers take in LLM responses + optional metadata and output a map of attributes, e.g:
    "I'm sorry, but I can't assist with that request." -> {"helpfulness": 0.1, "friendliness": 0.2}
    """

    def __init__(self, settings: "Settings", model: "Model"):
        self.settings = settings
        self.model = model

    @abstractmethod
    def tag_batch(
        self, responses: list[str], metadata: list[ResponseMetadata]
    ) -> list[Dict[str, Any]]:
        """
        tags a batch of responses.

        Args:
            responses: A list of model responses.
            metadata: A list of response metadata (optional)
        Returns:
            A list of dicts, where the keys are the tag names and the values are the tag values.
        """
        pass
