# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

from typing import Any

from optuna.study import StudyDirection
from pydantic import BaseModel

from .config import ObjectiveDirection, Settings
from .model import Model
from .scorer import EvaluationContext, Score, Scorer
from .utils import print
from .plugin import load_plugin


class Evaluator:
    settings: Settings
    model: Model
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

    def _deep_merge_dicts(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively merge two dicts.

        Values from `override` take precedence. Nested dicts are merged recursively.
        """
        merged: dict[str, Any] = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = self._deep_merge_dicts(merged[k], v)  # type: ignore[arg-type]
            else:
                merged[k] = v
        return merged

    def _get_scorer_settings_raw(
        self, *, scorer_cls: type[Scorer], instance_name: str | None
    ) -> dict[str, Any]:
        """
        Build the raw settings dict for a scorer class and optional instance.

        Config rules:
        - Base settings live in `[ClassName]` (applies to all instances)
        - Instance overrides live in `[ClassName.<instance_name>]`
        - Only merge/validate keys that exist in the scorer Settings schema
        """
        class_name = scorer_cls.__name__
        raw_class_table = self._get_plugin_namespace(class_name)

        if instance_name is not None and "." in instance_name:
            raise ValueError(
                f"Invalid instance_name '{instance_name}' for scorer {class_name}: '.' is not allowed"
            )

        raw_instance_table: dict[str, Any] = {}
        if instance_name:
            candidate = raw_class_table.get(instance_name)
            if candidate is None:
                raw_instance_table = {}
            elif isinstance(candidate, dict):
                raw_instance_table = candidate
            else:
                raise TypeError(
                    f"Plugin namespace [{class_name}.{instance_name}] must be a table/object, got {type(candidate).__name__}"
                )

        settings_model = getattr(scorer_cls, "Settings", None)
        if settings_model is None:
            # No settings schema: nothing to merge/validate.
            return {}

        allowed_keys = set(settings_model.model_fields.keys())
        base_filtered = {k: v for k, v in raw_class_table.items() if k in allowed_keys}
        instance_filtered = {
            k: v for k, v in raw_instance_table.items() if k in allowed_keys
        }
        return self._deep_merge_dicts(base_filtered, instance_filtered)

    def _load_scorers(self) -> list[Scorer]:
        """Load and instantiate all configured scorer plugins."""
        scorer_configs = self.settings.scorers
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
            print(f"* Loaded: [bold]{scorer_cls.__name__}[/bold]")

        scorers: list[Scorer] = []
        scorer_names: set[str] = set()
        # instantiate scorers
        for index, scorer_cls in enumerate(scorer_classes):
            direction: ObjectiveDirection = scorer_configs[index].direction
            scale: float = float(scorer_configs[index].scale)
            instance_name = scorer_configs[index].instance_name or None

            raw_settings = self._get_scorer_settings_raw(
                scorer_cls=scorer_cls, instance_name=instance_name
            )
            plugin_settings: BaseModel | None = scorer_cls.validate_settings(raw_settings)

            scorer = scorer_cls(
                settings=self.settings,
                model=self.model,
                plugin_settings=plugin_settings,
                direction=direction,
                scale=scale,
                instance_name=instance_name
            )

            scorer_name = (
                scorer_cls.__name__
                if not instance_name
                else f"{scorer_cls.__name__}.{instance_name}"
            )
            if scorer_name in scorer_names:
                raise ValueError(
                    f"Duplicate scorer instance name: {scorer_name}. "
                    "Give each instance a unique `instance_name`."
                )
            scorer_names.add(scorer_name)

            scorers.append(scorer)
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
        return [m for m in metrics if m.direction != StudyDirection.NOT_SET]

    def get_objective_values(self, metrics: list[Score]) -> tuple[float, ...]:
        """Extract objective values as a tuple for Optuna."""
        values: list[float] = []
        for scorer, m in zip(self.scorers, metrics):
            if m.direction == StudyDirection.NOT_SET:
                continue
            values.append(float(m.value) * float(getattr(scorer, "scale", 1.0)))
        return tuple(values)

    def get_objective_directions(self, metrics: list[Score]) -> list[StudyDirection]:
        """Get optimization directions for objectives."""
        directions: list[StudyDirection] = []
        for m in self.get_objectives(metrics):
            directions.append(m.direction)
        return directions

    def get_baseline_refusals(self) -> int:
        """Get baseline refusal count (for backwards compat in main.py)."""
        for scorer, m in zip(self.scorers, self.baseline_metrics):
            if scorer.__class__.__name__ == "RefusalRate":
                return int(m.value)
        return 0
