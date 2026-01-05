# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .config import Settings
from .model import Model
from .scorer import EvaluationContext, Score, Scorer
from .utils import load_plugin, load_prompts, print


class Evaluator:
    """
    Manages evaluation of the model using configured scorer plugins.

    Loads prompts, establishes baseline metrics, and runs scorers during optimization.
    """

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_good_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

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
        scorer_names = self.settings.scorers
        if not scorer_names:
            raise ValueError(
                "No scorers configured. Set 'scorers' in config.toml, e.g.:\n"
                'scorers = ["heretic.scorers.count_refusals.CountRefusals", "heretic.scorers.kl_divergence.KLDivergence"]'
            )

        scorer_classes: list[type[Scorer]] = []
        for name in scorer_names:
            scorer_cls = load_plugin(name=name, base_class=Scorer)
            scorer_cls.validate_contract()
            scorer_classes.append(scorer_cls)
            print(f"* Loaded: [bold]{scorer_cls.__name__}[/bold]")

        # Aggregate response metadata requirements across all scorers
        response_fields: set[str] = set()
        for scorer_cls in scorer_classes:
            response_fields |= set(scorer_cls.required_response_metadata_fields())

        self.model.set_requested_metadata_fields(response_fields)

        scorers: list[Scorer] = []
        for scorer_cls in scorer_classes:
            plugin_config = self._get_plugin_namespace(scorer_cls.name)
            plugin_settings: BaseModel | None = scorer_cls.validate_settings(
                plugin_config
            )
            scorers.append(
                scorer_cls(
                    settings=self.settings,
                    model=self.model,
                    plugin_settings=plugin_settings,
                )
            )
        return scorers

    def evaluate(self) -> list[Score]:
        """
        Run all scorers and return their metrics.

        Returns:
            List of MetricResult from each scorer.
        """
        ctx = EvaluationContext(
            settings=self.settings,
            model=self.model,
            good_prompts=self.good_prompts,
            bad_prompts=self.bad_prompts,
            base_good_logprobs=self.base_good_logprobs,
        )
        return [scorer.evaluate(ctx) for scorer in self.scorers]

    def get_objectives(self, metrics: list[Score]) -> list[Score]:
        """Filter metrics to only those used in optimization."""
        return [m for m in metrics if m.use_in_optimizer]

    def get_objective_values(self, metrics: list[Score]) -> tuple[float, ...]:
        """Extract objective values as a tuple for Optuna."""
        return tuple(m.value for m in self.get_objectives(metrics))

    def get_objective_directions(self, metrics: list[Score]) -> list[str]:
        """Get optimization directions for objectives."""
        return [m.direction for m in self.get_objectives(metrics)]

    def get_baseline_refusals(self) -> int:
        """Get baseline refusal count (for backwards compat in main.py)."""
        for m in self.baseline_metrics:
            if m.name == "CountRefusals":
                return int(m.value)
        return 0
