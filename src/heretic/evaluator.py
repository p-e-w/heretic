# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from dataclasses import dataclass
from typing import Any

from optuna.study import StudyDirection
from pydantic import BaseModel

from .config import ScorerConfig, Settings
from .model import Model
from .plugin import get_plugin_namespace, load_plugin
from .scorer import Context, Score, Scorer
from .utils import deep_merge_dicts, parse_study_direction, print


@dataclass
class ScorerEntry:
    scorer: Scorer
    name: str
    config: ScorerConfig


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
        self._scorer_entries: list[ScorerEntry] = []

        print()
        print("Loading and initializing scorers...")
        self._load_and_init_scorers()

        # Establish baseline scores (pre-abliteration)
        self.baseline_scores = self.get_baseline_scores()
        self._print_baseline()

    def _load_and_init_scorers(self) -> None:
        """
        Load and instantiate all configured scorer plugins,
        then runs their initialization hooks.
        """
        scorer_configs = list(self.settings.scorers)
        if not scorer_configs:
            raise ValueError("No scorers configured. Set 'scorers' in config.toml")

        scorer_names: set[str] = set()

        # Resolve plugin classes from names and validate.
        for cfg in scorer_configs:
            scorer_cls = load_plugin(name=cfg.plugin, base_class=Scorer)
            scorer_cls.validate_contract()

            print(
                f"* Loaded: [bold]{scorer_cls.__name__} {'- ' + cfg.instance_name if cfg.instance_name else ''}[/bold]"
            )

            # Instantiate scorers.
            instance_name = cfg.instance_name or None

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

            scorer_instance_name = (
                f"{scorer.score_name} - {instance_name}"
                if instance_name
                else scorer.score_name
            )
            self._scorer_entries.append(
                ScorerEntry(scorer=scorer, config=cfg, name=scorer_instance_name)
            )

        # Run scorer init hooks.
        ctx = Context(settings=self.settings, model=self.model)

        for entry in self._scorer_entries:
            entry.scorer.init(ctx)

    def _print_baseline(self) -> None:
        """Print baseline scores summary."""
        for name, score in self.baseline_scores:
            print(f"* Baseline {name}: [bold]{score.cli_display}[/]")

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

    def get_scores(self) -> list[tuple[str, Score]]:
        """
        Run all scorers and return their scores and names

        Returns:
            List of `Score` from each scorer and its name.
        """
        ctx = Context(settings=self.settings, model=self.model)
        return [
            (entry.name, entry.scorer.get_score(ctx)) for entry in self._scorer_entries
        ]

    def get_baseline_scores(self) -> list[tuple[str, Score]]:
        """
        Run all scorers and return their baseline scores and names

        Returns:
            List of `Score` from each scorer and its name.
        """
        ctx = Context(settings=self.settings, model=self.model)
        return [
            (entry.name, entry.scorer.get_baseline_score(ctx))
            for entry in self._scorer_entries
        ]

    def get_objective_names(self) -> list[str]:
        """Return objective names for scores used in optimization."""
        return [
            entry.name
            for entry in self._scorer_entries
            if parse_study_direction(entry.config.direction) != StudyDirection.NOT_SET
        ]

    def get_objective_values(
        self, scores: list[tuple[str, Score]]
    ) -> tuple[float, ...]:
        """Extract objective values as a tuple for Optuna."""
        objective_names = set(self.get_objective_names())
        return tuple(score.value for name, score in scores if name in objective_names)

    def get_objective_directions(self) -> list[StudyDirection]:
        """Get optimization directions for objectives."""
        return [
            parse_study_direction(entry.config.direction)
            for entry in self._scorer_entries
            if parse_study_direction(entry.config.direction) != StudyDirection.NOT_SET
        ]
