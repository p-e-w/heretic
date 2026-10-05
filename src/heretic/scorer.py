# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from .config import Settings as HereticSettings
from .plugin import Context, Plugin


@dataclass
class Score:
    """
    Result of evaluating a scorer.

    - `value`: scalar value used for optimization (if enabled).
    - `rich_display`: formatted Rich markup shown to the user in logs/console.
    - `md_display`: formatted value in the HF model card.
    """

    value: float
    rich_display: str
    md_display: str


class Scorer(Plugin, ABC):
    """
    Abstract base class for scorer plugins.

    Scorers evaluate model behavior and return a Score.

    Examples: Counting refusals, measuring KL divergence, etc.
    """

    @property
    def score_name(self) -> str:
        """
        The name of the `Score` object returned by `get_score()`.
        This is what shows up in the CLI and Markdown metrics on HF.
        """
        return self.__class__.__name__

    def __init__(
        self,
        heretic_settings: HereticSettings,
        settings: BaseModel | None = None,
    ) -> None:
        super().__init__(heretic_settings=heretic_settings, settings=settings)

    @abstractmethod
    def get_score(self, ctx: Context) -> Score:
        """
        Return a `Score` given the evaluation context.
        The `value` of the `Score` must be of the order of magnitude 1
        to ensure that all scores are comparable during co-optimization.
        """

    def get_baseline_score(self, ctx: Context) -> Score:
        """
        Calculates a baseline score.

        Defaults to the current `get_score(...)` implementation and can be
        overridden by scorers that need a distinct baseline.
        """
        return self.get_score(ctx)

    def get_holdout_score(self, ctx: Context) -> Score | None:
        """
        Scores the model on prompts held out from optimization, or returns None
        if the scorer has no holdout set (the default).

        Called once for the baseline, and once for the selected trial after
        trial selection. Never called during optimization.
        """
        return None

    def get_holdout_gap(self, score: Score, holdout_score: Score) -> Score:
        """
        Returns the holdout score minus the score on the optimized-on prompts.
        Override this to display the gap in the scorer's own units.
        """
        gap = holdout_score.value - score.value
        return Score(
            value=gap,
            rich_display=f"[bold]{gap:+.4f}[/]",
            md_display=f"{gap:+.4f}",
        )
