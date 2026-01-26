# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

from typing import Any

from optuna.study import StudyDirection
from pydantic import BaseModel

from .config import ScorerConfig, Settings
from .model import Model
from .plugin import get_plugin_namespace, load_plugin
from .scorer import Context, Score, Scorer
from .utils import deep_merge_dicts, print


class Evaluator:
    settings: Settings
    model: Model
    """
    Manages evaluation of the model using configured scorer plugins.

    Loads scorers, establishes baseline scores, and runs scorers during optimization.
    """

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model
        self._scorer_configs: list[ScorerConfig] = list(settings.scorers)

        print()
        print("Loading scorers...")
        self.scorers = self._load_scorers()
        self._setup_scorers()

        # Establish baseline scores (pre-abliteration)
        self.baseline_scores = self.get_scores()
        self._print_baseline()

    def _setup_scorers(self) -> None:
        """
        Optional scorer initialization hook.
        """
        ctx = Context(settings=self.settings, model=self.model)

        for scorer in self.scorers:
            scorer.setup(ctx)

    def _print_baseline(self) -> None:
        """Print baseline scores summary."""
        for s in self.baseline_scores:
            print(f"* Baseline {s.name}: [bold]{s.cli_display}[/]")

    def _get_scorer_settings_raw(
        self, *, scorer_cls: type[Scorer], instance_name: str | None
    ) -> dict[str, Any]:
        """
        Build the raw settings dict for a scorer class and optional instance.

        Config rules:
        - Base settings live in `[scorer.ClassName]` (applies to all instances)
        - Instance overrides live in `[scorer.ClassName_<instance_name>]` (preferred)
        - Only merge/validate keys that exist in the scorer Settings schema
        """
        class_name = scorer_cls.__name__
        canonical_ns = f"scorer.{class_name}"
        raw_class_table = get_plugin_namespace(self.settings.model_extra, canonical_ns)

        if instance_name is not None and "." in instance_name:
            raise ValueError(
                f"Invalid instance_name '{instance_name}' for scorer {class_name}: '.' is not allowed"
            )

        raw_instance_table: dict[str, Any] = {}
        if instance_name:
            instance_ns = f"scorer.{class_name}_{instance_name}"
            raw_instance_table = get_plugin_namespace(
                self.settings.model_extra, instance_ns
            )

        settings_model = scorer_cls.get_settings_model()
        if settings_model is None:
            # No settings schema: nothing to merge/validate.
            return {}

        allowed_keys = set(settings_model.model_fields.keys())
        base_filtered = {k: v for k, v in raw_class_table.items() if k in allowed_keys}
        instance_filtered = {
            k: v for k, v in raw_instance_table.items() if k in allowed_keys
        }
        return deep_merge_dicts(base_filtered, instance_filtered)

    def _load_scorers(self) -> list[Scorer]:
        """Load and instantiate all configured scorer plugins."""
        scorer_configs = self._scorer_configs
        # the scaling factor and optimization direction (maximize, minimize, none)
        # is set at the top level
        if not scorer_configs:
            raise ValueError("No scorers configured. Set 'scorers' in config.toml")

        scorer_classes: list[type[Scorer]] = []

        # resolve plugin classes from names and validate
        for cfg in scorer_configs:
            scorer_cls = load_plugin(name=cfg.plugin, base_class=Scorer)
            scorer_cls.validate_contract()
            scorer_classes.append(scorer_cls)

            print(
                f"* Loaded: [bold]{scorer_cls.__name__} {'- ' + cfg.instance_name if cfg.instance_name else ''}[/bold]"
            )

        scorers: list[Scorer] = []
        self._scorer_instance_labels: list[str | None] = []
        scorer_names: set[str] = set()
        # instantiate scorers
        for index, scorer_cls in enumerate(scorer_classes):
            instance_name = scorer_configs[index].instance_name or None

            raw_settings = self._get_scorer_settings_raw(
                scorer_cls=scorer_cls, instance_name=instance_name
            )
            scorer_settings: BaseModel | None = scorer_cls.validate_settings(
                raw_settings
            )

            scorer = scorer_cls(
                heretic_settings=self.settings,
                settings=scorer_settings,
            )

            # External labeling key: ensures multiple instances can coexist
            scorer_key = (
                scorer_cls.__name__
                if not instance_name
                else f"{scorer_cls.__name__}.{instance_name}"
            )
            if scorer_key in scorer_names:
                raise ValueError(
                    f"Duplicate scorer instance name: {scorer_key}. "
                    "Give each instance a unique `instance_name`."
                )
            scorer_names.add(scorer_key)

            scorers.append(scorer)
            self._scorer_instance_labels.append(instance_name)
        return scorers

    def get_scores(self) -> list[Score]:
        """
        Run all scorers and return their scores
        If there are multiple instances of the same scorer, the `Score`'s `name`
        is labeled externally as `<base> - <instance_name>`.

        Returns:
            List of Score from each scorer.
        """
        ctx = Context(settings=self.settings, model=self.model)
        scores: list[Score] = []
        for scorer, label in zip(self.scorers, self._scorer_instance_labels):
            s = scorer.get_score(ctx)
            if label:
                # Add label externally
                s.name = f"{s.name} - {label}"
            scores.append(s)
        return scores

    def get_objectives(self, scores: list[Score]) -> list[Score]:
        """Filter scores to only those used in optimization."""
        return [
            s
            for cfg, s in zip(self._scorer_configs, scores)
            if cfg.direction != StudyDirection.NOT_SET
        ]

    def get_objective_values(self, scores: list[Score]) -> tuple[float, ...]:
        """Extract objective values as a tuple for Optuna."""
        values: list[float] = []
        for cfg, s in zip(self._scorer_configs, scores):
            if cfg.direction == StudyDirection.NOT_SET:
                continue
            values.append(float(s.value) * float(cfg.scale))
        return tuple(values)

    def get_objective_directions(self) -> list[StudyDirection]:
        """Get optimization directions for objectives."""
        return [
            cfg.direction
            for cfg in self._scorer_configs
            if cfg.direction != StudyDirection.NOT_SET
        ]
