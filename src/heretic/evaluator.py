# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from optuna.study import StudyDirection
from pydantic import BaseModel

from .config import DatasetSpecification, LoggerConfig, ScorerConfig, Settings
from .logger import Logger, LoggerEvent, LoggerPhase
from .model import Model
from .plugin import get_plugin_namespace, is_builtin_plugin, load_plugin
from .scorer import Context, Score, Scorer
from .utils import deep_merge_dicts, parse_study_direction, print


@dataclass
class ScorerEntry:
    scorer: Scorer
    name: str
    config: ScorerConfig


@dataclass
class LoggerEntry:
    logger: Logger
    name: str
    config: LoggerConfig


class Evaluator:
    """
    Manages evaluation of the model using configured scorer plugins.

    Loads scorers, establishes baseline scores, and runs scorers during optimization.
    """

    settings: Settings
    model: Model

    def __init__(
        self, settings: Settings, model: Model, *, session_id: str | None = None
    ):
        self.settings = settings
        self.model = model
        self.session_id = session_id or uuid4().hex
        self._scorer_entries: list[ScorerEntry] = []
        self._logger_entries: list[LoggerEntry] = []

        print()
        print("Loading and initializing scorers...")
        self._load_and_init_scorers()

        print()
        print("Loading and initializing loggers...")
        self._load_and_init_loggers()

        print()
        print("Getting baseline scores...")
        self.baseline_scores = self.get_baseline_scores()
        for name, score in self.baseline_scores:
            print(f"* Baseline [bold]{name}:[/] [green]{score.rich_display}[/]")

    def _load_and_init_scorers(self) -> None:
        """
        Load and instantiate all configured scorer plugins,
        then runs their initialization hooks.
        """
        scorer_configs = self.settings.scorers
        if not scorer_configs:
            raise ValueError("No scorers configured. Set 'scorers' in config.toml")

        scorer_keys: set[str] = set()

        # Resolve plugin classes from names and validate.
        for config in scorer_configs:
            scorer_cls = load_plugin(name=config.plugin, base_class=Scorer)
            scorer_cls.validate_contract()

            print(
                f"* Loaded: [bold]{scorer_cls.__name__} {'- ' + config.instance_name if config.instance_name else ''}[/bold]"
            )

            # Instantiate scorers.
            instance_name = config.instance_name or None

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

            # External labeling key: ensures multiple instances can coexist.
            # Uses underscore to match the TOML namespace format (`scorer.<Class>_<instance>`).
            scorer_key = (
                scorer_cls.__name__
                if not instance_name
                else f"{scorer_cls.__name__}_{instance_name}"
            )
            if scorer_key in scorer_keys:
                raise ValueError(
                    f"Duplicate scorer instance name: {scorer_key}. "
                    "Give each instance a unique `instance_name`."
                )
            scorer_keys.add(scorer_key)

            scorer_instance_name = (
                f"{scorer.score_name} - {instance_name}"
                if instance_name
                else scorer.score_name
            )
            self._scorer_entries.append(
                ScorerEntry(scorer=scorer, config=config, name=scorer_instance_name)
            )

        # Run scorer init hooks.
        ctx = Context(
            settings=self.settings,
            model=self.model,
            session_id=self.session_id,
            phase="initialization",
        )

        for entry in self._scorer_entries:
            entry.scorer.init(ctx.for_scorer(entry.name))

    def _load_and_init_loggers(self) -> None:
        """Load logger plugins and run their initialization hooks."""
        logger_configs = self.settings.loggers
        if not logger_configs:
            return

        logger_keys: set[str] = set()
        for config in logger_configs:
            logger_cls = load_plugin(name=config.plugin, base_class=Logger)
            logger_cls.validate_contract()

            print(
                f"* Loaded: [bold]{logger_cls.__name__} "
                f"{'- ' + config.instance_name if config.instance_name else ''}[/bold]"
            )

            instance_name = config.instance_name or None
            raw_settings = self._get_logger_settings_raw(
                logger_cls=logger_cls,
                instance_name=instance_name,
            )
            logger_settings = logger_cls.validate_settings(raw_settings)
            logger = logger_cls(
                heretic_settings=self.settings,
                settings=logger_settings,
            )

            logger_key = (
                logger_cls.__name__
                if not instance_name
                else f"{logger_cls.__name__}_{instance_name}"
            )
            if logger_key in logger_keys:
                raise ValueError(
                    f"Duplicate logger instance name: {logger_key}. "
                    "Give each instance a unique `instance_name`."
                )
            logger_keys.add(logger_key)
            logger_name = (
                f"{logger_cls.__name__} - {instance_name}"
                if instance_name
                else logger_cls.__name__
            )
            self._logger_entries.append(
                LoggerEntry(logger=logger, name=logger_name, config=config)
            )

        ctx = Context(
            settings=self.settings,
            model=self.model,
            session_id=self.session_id,
            phase="initialization",
            _capture_responses=bool(self._logger_entries),
        )
        for entry in self._logger_entries:
            entry.logger.init(ctx)

    def get_dataset_specifications(self) -> list[DatasetSpecification]:
        """
        Collect the dataset specifications declared in the settings of all
        loaded scorers.
        """
        specifications = []
        for entry in self._scorer_entries:
            if entry.scorer.settings is None:
                continue
            for value in dict(entry.scorer.settings).values():
                if isinstance(value, DatasetSpecification):
                    specifications.append(value)
        return specifications

    def _get_scorer_settings_raw(
        self, *, scorer_cls: type[Scorer], instance_name: str | None
    ) -> dict[str, Any]:
        """
        Build the raw settings dict for a scorer class and optional instance.

        Config rules:
        - Base settings live in `[scorer.ClassName]` (applies to all instances).
        - Instance overrides live in `[scorer.ClassName_<instance_name>]` (preferred).
        - Only merge/validate keys that exist in the scorer Settings schema.
        """
        settings_model = scorer_cls.get_settings_model()
        if settings_model is None:
            # No settings schema: nothing to merge/validate.
            return {}

        class_name = scorer_cls.__name__

        namespaces = [f"scorer.{class_name}"]
        if instance_name:
            namespaces.append(f"scorer.{class_name}_{instance_name}")

        merged_settings: dict[str, Any] = {}
        allowed_keys = set(settings_model.model_fields.keys())

        for namespace in namespaces:
            raw_table = get_plugin_namespace(self.settings.model_extra, namespace)
            filtered = {k: v for k, v in raw_table.items() if k in allowed_keys}
            merged_settings = deep_merge_dicts(merged_settings, filtered)

        return merged_settings

    def _get_logger_settings_raw(
        self, *, logger_cls: type[Logger], instance_name: str | None
    ) -> dict[str, Any]:
        """Build the raw settings dict for a logger class and optional instance."""
        settings_model = logger_cls.get_settings_model()
        if settings_model is None:
            return {}

        class_name = logger_cls.__name__
        namespaces = [f"logger.{class_name}"]
        if instance_name:
            namespaces.append(f"logger.{class_name}_{instance_name}")

        merged_settings: dict[str, Any] = {}
        allowed_keys = set(settings_model.model_fields.keys())
        for namespace in namespaces:
            raw_table = get_plugin_namespace(self.settings.model_extra, namespace)
            filtered = {k: v for k, v in raw_table.items() if k in allowed_keys}
            merged_settings = deep_merge_dicts(merged_settings, filtered)
        return merged_settings

    def all_scorers_reproducible(self) -> bool:
        """
        Returns True if all scorers are reproducible,
        False if not.
        """
        return all(entry.scorer.reproducible for entry in self._scorer_entries)

    def all_scorers_builtin(self) -> bool:
        """
        Returns True if all scorers are built-in,
        i.e included in Heretic by default.
        """
        return all(
            is_builtin_plugin(entry.config.plugin) for entry in self._scorer_entries
        )

    def _new_context(self, phase: LoggerPhase) -> Context:
        return Context(
            settings=self.settings,
            model=self.model,
            session_id=self.session_id,
            phase=phase,
            _capture_responses=bool(self._logger_entries),
        )

    def _log_event(
        self,
        *,
        ctx: Context,
        phase: LoggerPhase,
        trial_number: int | None,
        parameters: dict[str, Any],
        scores: list[tuple[str, Score]],
    ) -> None:
        if not self._logger_entries:
            return

        event = LoggerEvent(
            session_id=self.session_id,
            phase=phase,
            trial_number=trial_number,
            parameters=parameters,
            scores=tuple(scores),
            responses=ctx.response_records,
        )
        for entry in self._logger_entries:
            # Logger plugins are observers. Give each one an isolated snapshot so
            # mutating a Score or another nested event value cannot affect the
            # objective values returned to the optimizer or another logger.
            entry.logger.log(deepcopy(event))

    def get_scores(
        self,
        *,
        trial_number: int | None = None,
        parameters: dict[str, Any] | None = None,
        phase: LoggerPhase = "evaluation",
    ) -> list[tuple[str, Score]]:
        """
        Run all scorers and return their scores and names

        Returns:
            List of `Score` from each scorer and its name.
        """
        ctx = self._new_context(phase)
        scores = [
            (entry.name, entry.scorer.get_score(ctx.for_scorer(entry.name)))
            for entry in self._scorer_entries
        ]
        self._log_event(
            ctx=ctx,
            phase=phase,
            trial_number=trial_number,
            parameters={} if parameters is None else parameters,
            scores=scores,
        )
        return scores

    def get_baseline_scores(self) -> list[tuple[str, Score]]:
        """
        Run all scorers and return their baseline scores and names

        Returns:
            List of `Score` from each scorer and its name.
        """
        ctx = self._new_context("baseline")
        scores = [
            (
                entry.name,
                entry.scorer.get_baseline_score(ctx.for_scorer(entry.name)),
            )
            for entry in self._scorer_entries
        ]
        self._log_event(
            ctx=ctx,
            phase="baseline",
            trial_number=None,
            parameters={},
            scores=scores,
        )
        return scores

    def get_paired_score_records(
        self, scores: list[tuple[str, Score]]
    ) -> list[dict[str, Any]]:
        """
        Pair each trial score with its baseline into one serializable record.

        `scores` (from `get_scores()`) and `self.baseline_scores` are both ordered
        by `_scorer_entries`, so they align positionally.
        """
        records: list[dict[str, Any]] = []
        for (name, score), (baseline_name, baseline) in zip(
            scores, self.baseline_scores
        ):
            assert name == baseline_name, (
                f"Score/baseline order mismatch: {name!r} != {baseline_name!r}"
            )
            records.append(
                {
                    "name": name,
                    "score": dict(score.__dict__),
                    "baseline": dict(baseline.__dict__),
                }
            )
        return records

    def _objective_entries(self) -> list[ScorerEntry]:
        """
        Scorer entries that participate in optimization, in canonical order.
        Single source of truth for which scorers are objectives and in what
        order. Every objective-derived list (names, directions, values) is built
        from this so they stay positionally aligned: Optuna matches the objective
        values returned each trial to the study `directions` by index, so a length
        or order mismatch here would silently corrupt the optimization.
        """
        return [
            entry
            for entry in self._scorer_entries
            if parse_study_direction(entry.config.optimization)
            != StudyDirection.NOT_SET
        ]

    def get_objective_names(self) -> list[str]:
        """Return objective names for scores used in optimization."""
        return [entry.name for entry in self._objective_entries()]

    def get_objective_values(
        self, scores: list[tuple[str, Score]]
    ) -> tuple[float, ...]:
        """
        Extract objective values as a tuple for Optuna.

        Ordered by `_objective_entries()` so the result aligns by index with
        `get_objective_names()` and `get_objective_directions()`.
        """
        score_by_name = {name: score for name, score in scores}
        return tuple(
            score_by_name[entry.name].value for entry in self._objective_entries()
        )

    def get_objective_directions(self) -> list[StudyDirection]:
        """Get optimization directions for objectives."""
        return [
            parse_study_direction(entry.config.optimization)
            for entry in self._objective_entries()
        ]
