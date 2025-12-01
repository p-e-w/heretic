# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from abc import ABC, abstractmethod


class Plugin(ABC):
    @abstractmethod
    def score(self, responses: list[str]) -> list[float]:
        """
        Return a score (0.0 to 1.0) for each response.
        The meaning of the score depends on the plugin, but generally
        higher scores indicate a stronger presence of the target attribute.
        """
        pass

    @property
    def minimize(self) -> bool:
        """
        Whether the score should be minimized (True) or maximized (False).
        Default is True (minimize the score).
        """
        return True
