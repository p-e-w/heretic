# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings
    from .model import Model


class RefusalDetector(ABC):
    """
    Abstract base class for refusal detectors.
    """

    def __init__(self, settings: "Settings", model: "Model"):
        self.settings = settings
        self.model = model

    @abstractmethod
    def detect_batch(self, responses: list[str]) -> list[bool]:
        """
        Detect refusals in a batch of responses.

        Args:
            responses: A list of model responses.

        Returns:
            A list of booleans, where True indicates a refusal.
        """
        pass

    def detect(self, response: str) -> bool:
        """
        Detect if a single response is a refusal.

        Args:
            response: The model response string.

        Returns:
            True if the response is a refusal, False otherwise.
        """
        return self.detect_batch([response])[0]
