# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from heretic.plugin import Context, Plugin

from .config import Settings as HereticSettings


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

    Example: counting refusals, measuring KL divergence, etc.
    """

    @property
    def score_name(self) -> str:
        """
        The name of the `Score` object returned by `get_score()`.
        This is what shows up in the CLI and Markdown metrics on HF.
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def reproducible(self) -> bool:
        """
        Whether runs using this scorer can be reproduced bit-for-bit.

        Set to False when the scorer's behavior is not deterministic or depends on
        state outside the pinned config, for example:
        - It calls an external service (e.g. an LLM judge over the OpenAI API).
        - It reads credentials or config from the environment (env vars, files).
        - It is otherwise non-deterministic (network, wall-clock, unseeded RNG).

        Defaults to False; override to True in your scorer class if any of the
        above DO NOT apply.
        """
        return False

    def __init__(
        self,
        heretic_settings: HereticSettings,
        settings: BaseModel | None = None,
    ):
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
