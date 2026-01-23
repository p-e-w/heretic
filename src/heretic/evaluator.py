# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

from typing import Any

from optuna.study import StudyDirection
from pydantic import BaseModel

from .config import ScorerConfig, Settings
from .model import Model
from .plugin import load_plugin
from .scorer import Context, Score, Scorer
from .utils import print


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
        self._scorer_configs: list[ScorerConfig] = list(settings.scorers)

        print()
        print("Loading scorers...")
        self.scorers = self._load_scorers()
        self._start_scorers()

        # Establish baseline metrics (pre-abliteration)
        self.baseline_metrics = self.get_scores()
        self._print_baseline()

    def _start_scorers(self) -> None:
        """
        Optional scorer initialization hook.
        """
        ctx = Context(settings=self.settings, model=self.model)

        for scorer in self.scorers:
            scorer.start(ctx)

    def _print_baseline(self) -> None:
        """Print baseline metrics summary."""
        for m in self.baseline_metrics:
            print(f"* Baseline {m.name}: [bold]{m.display}[/]")

    def _get_plugin_namespace(self, namespace: str) -> dict[str, Any]:
        """
        Returns the config dict from the `[<namespace>]` TOML table.
        """
        extra = self.settings.model_extra or {}
        cur: Any = extra
        for part in namespace.split("."):
            if not isinstance(cur, dict):
                return {}
            cur = cur.get(part)

        if cur is None:
            return {}
        if not isinstance(cur, dict):
            raise TypeError(
                f"Plugin namespace [{namespace}] must be a table/object, got {type(cur).__name__}"
            )
        return cur

    def _deep_merge_dicts(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
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
        - Base settings live in `[scorer.ClassName]` (applies to all instances)
        - Instance overrides live in `[scorer.ClassName_<instance_name>]` (preferred)
        - Only merge/validate keys that exist in the scorer Settings schema
        """
        class_name = scorer_cls.__name__
        canonical_ns = f"scorer.{class_name}"
        raw_class_table = self._get_plugin_namespace(canonical_ns)

        if instance_name is not None and "." in instance_name:
            raise ValueError(
                f"Invalid instance_name '{instance_name}' for scorer {class_name}: '.' is not allowed"
            )

        raw_instance_table: dict[str, Any] = {}
        if instance_name:
            instance_ns = f"scorer.{class_name}_{instance_name}"
            raw_instance_table = self._get_plugin_namespace(instance_ns)

        settings_model = getattr(scorer_cls, "PluginSettings", None)
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
            print(f"* Loaded: [bold]{scorer_cls.__name__}[/bold]")

        scorers: list[Scorer] = []
        self._scorer_instance_labels: list[str | None] = []
        scorer_names: set[str] = set()
        # instantiate scorers
        for index, scorer_cls in enumerate(scorer_classes):
            instance_name = scorer_configs[index].instance_name or None

            raw_settings = self._get_scorer_settings_raw(
                scorer_cls=scorer_cls, instance_name=instance_name
            )
            plugin_settings: BaseModel | None = scorer_cls.validate_settings(
                raw_settings
            )

            scorer = scorer_cls(
                settings=self.settings,
                plugin_settings=plugin_settings,
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

    def get_objectives(self, metrics: list[Score]) -> list[Score]:
        """Filter metrics to only those used in optimization."""
        return [
            m
            for cfg, m in zip(self._scorer_configs, metrics)
            if cfg.direction != StudyDirection.NOT_SET
        ]

    def get_objective_values(self, metrics: list[Score]) -> tuple[float, ...]:
        """Extract objective values as a tuple for Optuna."""
        values: list[float] = []
        for cfg, m in zip(self._scorer_configs, metrics):
            if cfg.direction == StudyDirection.NOT_SET:
                continue
            values.append(float(m.value) * float(cfg.scale))
        return tuple(values)

    def get_objective_directions(self) -> list[StudyDirection]:
        """Get optimization directions for objectives."""
        return [
            cfg.direction
            for cfg in self._scorer_configs
            if cfg.direction != StudyDirection.NOT_SET
        ]

    def get_baseline_refusals(self) -> int:
        """Get baseline refusal count (for backwards compat in main.py)."""
        for scorer, m in zip(self.scorers, self.baseline_metrics):
            if scorer.__class__.__name__ == "RefusalRate":
                return int(m.value)
        return 0
