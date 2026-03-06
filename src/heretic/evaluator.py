# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from typing import Any

from optuna.study import StudyDirection
from pydantic import BaseModel

from .config import ScorerConfig, Settings
from .model import Model
from .plugin import get_plugin_namespace, load_plugin
from .scorer import Context, Score, Scorer
from .utils import deep_merge_dicts, parse_study_direction, print


class Evaluator:
    """
    Manages evaluation of the model using configured scorer plugins.

    Loads scorers, establishes baseline scores, and runs scorers during optimization.
    """

    settings: Settings
    model: Model

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model
        self._scorer_configs: list[ScorerConfig] = list(settings.scorers)

        print()
        print("Loading and initializing scorers...")
        self.scorers = self._load_and_init_scorers()

        # Establish baseline scores (pre-abliteration)
        self.baseline_scores = self.get_baseline_scores()
        self._print_baseline()

    def _load_and_init_scorers(self) -> list[Scorer]:
        """
        Load and instantiate all configured scorer plugins,
        then runs their initialization hooks.
        """
        scorer_configs = self._scorer_configs
        # The scaling factor and optimization direction (maximize, minimize, none) is set at the top level.
        if not scorer_configs:
            raise ValueError("No scorers configured. Set 'scorers' in config.toml")

        scorer_classes: list[type[Scorer]] = []

        # Resolve plugin classes from names and validate.
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
        # Instantiate scorers.
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

            # External labeling key: ensures multiple instances can coexist/
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

        # Run scorer init hooks.
        ctx = Context(settings=self.settings, model=self.model)

        for scorer in scorers:
            scorer.init(ctx)

        return scorers

    def _print_baseline(self) -> None:
        """Print baseline scores summary."""
        for name, score in zip(self.get_score_names(), self.baseline_scores):
            print(f"* Baseline {name}: [bold]{score.cli_display}[/]")

    def _format_score_name(self, scorer: Scorer, instance_name: str | None) -> str:
        if instance_name:
            return f"{scorer.score_name} - {instance_name}"
        return scorer.score_name

    def get_score_names(self) -> list[str]:
        """
        Return stable display names for scores in scorer order.
        """
        return [
            self._format_score_name(scorer, label)
            for scorer, label in zip(self.scorers, self._scorer_instance_labels)
        ]

    def _get_scorer_settings_raw(
        self, *, scorer_cls: type[Scorer], instance_name: str | None
    ) -> dict[str, Any]:
        """
        Build the raw settings dict for a scorer class and optional instance.

        Config rules:
        - Base settings live in `[scorer.ClassName]` (applies to all instances)
        - Instance overrides live in `[scorer.ClassName_<instance_name>]` (preferred)
        - Only merge/validate keys that exist in the scorer Settings schema.
        """
        settings_model = scorer_cls.get_settings_model()
        if settings_model is None:
            # No settings schema: nothing to merge/validate.
            return {}

        class_name = scorer_cls.__name__
        if instance_name and "." in instance_name:
            raise ValueError(
                f"Invalid instance_name '{instance_name}' for scorer {class_name}: '.' is not allowed"
            )

        namespaces = [f"scorer.{class_name}"]
        if instance_name:
            namespaces.append(f"scorer.{class_name}_{instance_name}")

        merged_settings: dict[str, Any] = {}
        allowed_keys = set(settings_model.model_fields.keys())

        for ns in namespaces:
            raw_table = get_plugin_namespace(self.settings.model_extra, ns)
            filtered = {k: v for k, v in raw_table.items() if k in allowed_keys}
            merged_settings = deep_merge_dicts(merged_settings, filtered)

        return merged_settings

    def get_scores(self) -> list[Score]:
        """
        Run all scorers and return their scores
        Returns:
            List of Score from each scorer.
        """
        ctx = Context(settings=self.settings, model=self.model)
        scores: list[Score] = []
        for scorer in self.scorers:
            scores.append(scorer.get_score(ctx))
        return scores

    def get_baseline_scores(self) -> list[Score]:
        """
        Run all scorers and return their baseline scores
        Returns:
            List of Score from each scorer.
        """
        ctx = Context(settings=self.settings, model=self.model)
        scores: list[Score] = []
        for scorer in self.scorers:
            scores.append(scorer.get_baseline_score(ctx))
        return scores

    def get_objective_names(self) -> list[str]:
        """
        Return objective names for scores used in optimization.
        """
        return [
            name
            for cfg, name in zip(self._scorer_configs, self.get_score_names())
            if parse_study_direction(cfg.direction) != StudyDirection.NOT_SET
        ]

    def get_objectives(self, scores: list[Score]) -> list[Score]:
        """Filter scores to only those used in optimization."""
        return [
            s
            for cfg, s in zip(self._scorer_configs, scores)
            if parse_study_direction(cfg.direction) != StudyDirection.NOT_SET
        ]

    def get_objective_values(self, scores: list[Score]) -> tuple[float, ...]:
        """Extract objective values as a tuple for Optuna."""
        values: list[float] = []
        for cfg, s in zip(self._scorer_configs, scores):
            if parse_study_direction(cfg.direction) == StudyDirection.NOT_SET:
                continue
            values.append(float(s.value))
        return tuple(values)

    def get_objective_directions(self) -> list[StudyDirection]:
        """Get optimization directions for objectives."""
        return [
            parse_study_direction(cfg.direction)
            for cfg in self._scorer_configs
            if parse_study_direction(cfg.direction) != StudyDirection.NOT_SET
        ]
