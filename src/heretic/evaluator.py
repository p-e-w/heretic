# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .config import ObjectiveDirection, Settings
from .model import Model
from .scorer import EvaluationContext, Score, Scorer
from .utils import load_plugin, print


class Evaluator:
    """
    Manages evaluation of the model using configured scorer plugins.

    Loads prompts, establishes baseline metrics, and runs scorers during optimization.
    """

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        print()
        print("Loading scorers...")
        self.scorers = self._load_scorers()

        # Establish baseline metrics (pre-abliteration)
        self.baseline_metrics = self.evaluate()
        self._print_baseline()

    def _print_baseline(self) -> None:
        """Print baseline metrics summary."""
        for m in self.baseline_metrics:
            print(f"* Baseline {m.name}: [bold]{m.display}[/]")

    def _get_plugin_namespace(self, namespace: str) -> dict[str, Any]:
        """Returns the config dict from the `[<namespace>]` TOML table."""
        extra = self.settings.model_extra or {}
        value = extra.get(namespace)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(
                f"Plugin namespace [{namespace}] must be a table/object, got {type(value).__name__}"
            )
        return value

    def _load_scorers(self) -> list[Scorer]:
        """Load and instantiate all configured scorer plugins."""
        scorer_configs = self.settings.scorers
        # the scaling factor and optimization direction (maximize, minimize, none)
        # is set at the top level
        if not scorer_configs:
            raise ValueError("No scorers configured. Set 'scorers' in config.toml")

        scorer_classes: list[type[Scorer]] = []

        # resolve plugin classes from names and validate
        for plugin_name, _, _ in scorer_configs:
            scorer_cls = load_plugin(name=plugin_name, base_class=Scorer)
            scorer_cls.validate_contract()
            scorer_classes.append(scorer_cls)
            print(f"* Loaded: [bold]{scorer_cls.__name__}[/bold]")

        scorers: list[Scorer] = []
        # instantiate scorers
        for index, scorer_cls in enumerate(scorer_classes):
            plugin_config = self._get_plugin_namespace(scorer_cls.name)
            plugin_settings: BaseModel | None = scorer_cls.validate_settings(
                plugin_config
            )
            direction: ObjectiveDirection = scorer_configs[index][1]
            scale: float = float(scorer_configs[index][2])
            scorers.append(
                scorer_cls(
                    settings=self.settings,
                    model=self.model,
                    plugin_settings=plugin_settings,
                    direction=direction,
                    scale=scale,
                )
            )
        return scorers

    def evaluate(self) -> list[Score]:
        """
        Run all scorers and return their metrics.

        Returns:
            List of MetricResult from each scorer.
        """
        ctx = EvaluationContext(settings=self.settings, model=self.model)
        return [scorer.evaluate(ctx) for scorer in self.scorers]

    def get_objectives(self, metrics: list[Score]) -> list[Score]:
        """Filter metrics to only those used in optimization."""
        return [m for m in metrics if m.direction != "ignore"]

    def get_objective_values(self, metrics: list[Score]) -> tuple[float, ...]:
        """Extract objective values as a tuple for Optuna."""
        values: list[float] = []
        for scorer, m in zip(self.scorers, metrics):
            if m.direction == "ignore":
                continue
            values.append(float(m.value) * float(getattr(scorer, "scale", 1.0)))
        return tuple(values)

    def get_objective_directions(self, metrics: list[Score]) -> list[str]:
        """Get optimization directions for objectives."""
        return [m.direction for m in self.get_objectives(metrics)]

    def get_baseline_refusals(self) -> int:
        """Get baseline refusal count (for backwards compat in main.py)."""
        for scorer, m in zip(self.scorers, self.baseline_metrics):
            if scorer.name == "RefusalRate":
                return int(m.value)
        return 0
