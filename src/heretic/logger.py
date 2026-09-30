# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from .plugin import Plugin, ResponseRecord
from .scorer import Score

LoggerPhase = Literal["baseline", "evaluation", "model_evaluation"]


@dataclass(frozen=True)
class LoggerEvent:
    """Information emitted after one scorer evaluation phase."""

    session_id: str
    phase: LoggerPhase
    trial_number: int | None
    parameters: dict[str, Any]
    scores: tuple[tuple[str, Score], ...]
    responses: tuple[ResponseRecord, ...]


class Logger(Plugin, ABC):
    """Base class for plugins that observe Heretic evaluation events."""

    @abstractmethod
    def log(self, event: LoggerEvent) -> None:
        """Consume one evaluation event."""
