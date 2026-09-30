# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import create_trial

from heretic.config import LoggerConfig, ScorerConfig, Settings
from heretic.evaluator import Evaluator, LoggerEntry
from heretic.logger import Logger, LoggerEvent
from heretic.loggers.jsonl import JSONL
from heretic.loggers.jsonl import Settings as JSONLSettings
from heretic.model import Model
from heretic.plugin import Context, ResponseRecord, ResponseSource
from heretic.scorer import Score
from heretic.utils import (
    Prompt,
    _sanitize_checkpoint_for_reproduction,
    _strip_logger_settings,
    generate_config_toml,
    generate_reproduce_json,
)


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.response_calls = 0

    def get_responses_batched(
        self, _prompts: list[Prompt], *, skip_special_tokens: bool
    ) -> list[str]:
        assert skip_special_tokens is True
        self.response_calls += 1
        return list(self.responses)


class MutatingLogger(Logger):
    def log(self, event: LoggerEvent) -> None:
        event.scores[0][1].value = 99.0


class JsonlLoggerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompts = [
            Prompt(system="system α", user="first prompt"),
            Prompt(system="system β", user="second prompt"),
        ]
        self.responses = ["I’m sorry\nこんにちは", "Answer — done"]

    def _settings(self, output_path: Path) -> Settings:
        scorer_plugin = "heretic.scorers.keyword_rate.KeywordRate"
        settings_data: dict[str, Any] = {
            "model": "fake-model",
            "scorers": [
                ScorerConfig(
                    plugin=scorer_plugin,
                    optimization="none",
                    instance_name="bad",
                ),
                ScorerConfig(
                    plugin=scorer_plugin,
                    optimization="none",
                    instance_name="good",
                ),
            ],
            "loggers": [
                LoggerConfig(plugin="heretic.loggers.jsonl.JSONL"),
            ],
            "scorer": {
                "KeywordRate_bad": {"response_category": "bad"},
                "KeywordRate_good": {"response_category": "good"},
            },
            "logger": {"JSONL": {"path": output_path}},
        }
        with patch.object(sys, "argv", ["test_logger"]):
            return Settings.model_validate(settings_data)

    def _evaluator(
        self, output_path: Path, model: FakeModel, session_id: str
    ) -> Evaluator:
        settings = self._settings(output_path)
        with patch.object(Context, "load_prompts", return_value=self.prompts):
            return Evaluator(settings, cast(Model, model), session_id=session_id)

    def _records(self, output_path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ]

    def _minimal_settings(self) -> Settings:
        with patch.object(sys, "argv", ["test_logger"]):
            return Settings(model="fake-model")

    def test_baseline_and_trial_log_cached_responses_without_duplicate_inference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "responses.jsonl"
            model = FakeModel(self.responses)
            evaluator = self._evaluator(output_path, model, session_id="session-a")

            # Both scorer instances share one Context cache in the baseline phase.
            self.assertEqual(model.response_calls, 1)
            baseline_records = self._records(output_path)
            self.assertEqual(len(baseline_records), 2)
            self.assertEqual(
                baseline_records[0]["scorers"],
                [
                    {
                        "name": "Refusals - bad",
                        "dataset": "mlabonne/harmful_behaviors",
                        "category": "bad",
                    },
                    {
                        "name": "Refusals - good",
                        "dataset": "mlabonne/harmful_behaviors",
                        "category": "good",
                    },
                ],
            )
            self.assertEqual(baseline_records[0]["trial_number"], None)
            self.assertEqual(baseline_records[0]["phase"], "baseline")
            self.assertEqual(baseline_records[0]["response"], self.responses[0])

            scores = evaluator.get_scores(
                trial_number=17,
                parameters={"abliteration": {"layer": 3}},
            )

            # A new evaluation context performs one generation, not one per scorer.
            self.assertEqual(model.response_calls, 2)
            self.assertEqual([score.value for _, score in scores], [0.5, 0.5])

            records = self._records(output_path)
            self.assertEqual(len(records), 4)
            self.assertEqual(
                [(record["phase"], record["trial_number"]) for record in records],
                [
                    ("baseline", None),
                    ("baseline", None),
                    ("evaluation", 17),
                    ("evaluation", 17),
                ],
            )
            self.assertEqual(records[2]["response"], self.responses[0])
            self.assertEqual(records[3]["prompt"], "second prompt")

    def test_logger_appends_across_sessions_and_uses_optuna_trial_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "responses.jsonl"
            first_model = FakeModel(self.responses)
            first = self._evaluator(output_path, first_model, session_id="session-a")
            first.get_scores(trial_number=3)

            second_model = FakeModel(self.responses)
            second = self._evaluator(output_path, second_model, session_id="session-b")
            second.get_scores(trial_number=4)

            records = self._records(output_path)
            self.assertEqual(len(records), 8)
            self.assertEqual(
                {record["session_id"] for record in records},
                {"session-a", "session-b"},
            )
            self.assertEqual(
                {record["trial_number"] for record in records},
                {None, 3, 4},
            )

    def test_write_errors_are_not_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "responses"
            output_directory.mkdir()

            with self.assertRaises(IsADirectoryError):
                self._evaluator(
                    output_directory,
                    FakeModel(self.responses),
                    session_id="session-a",
                )

    def _event(self) -> LoggerEvent:
        return LoggerEvent(
            session_id="session-a",
            phase="evaluation",
            trial_number=1,
            parameters={},
            scores=(
                (
                    "Refusals",
                    Score(value=0.5, rich_display="0.5", md_display="0.5"),
                ),
            ),
            responses=(
                ResponseRecord(
                    prompt=Prompt(system="system", user="prompt"),
                    response="response",
                    sources=(
                        ResponseSource(
                            scorer="Refusals",
                            dataset="dataset",
                            category="bad",
                        ),
                    ),
                ),
            ),
        )

    def test_jsonl_separates_valid_unterminated_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "responses.jsonl"
            output_path.write_text('{"old": 1}', encoding="utf-8")

            logger = JSONL(
                heretic_settings=self._minimal_settings(),
                settings=JSONLSettings(path=output_path),
            )
            logger.log(self._event())

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0], {"old": 1})
            self.assertEqual(records[1]["response"], "response")

    def test_jsonl_rejects_malformed_final_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "responses.jsonl"
            malformed = '{"old": 1}\nnot-json\n'
            output_path.write_text(malformed, encoding="utf-8")

            logger = JSONL(
                heretic_settings=self._minimal_settings(),
                settings=JSONLSettings(path=output_path),
            )
            with self.assertRaisesRegex(ValueError, "final JSONL record"):
                logger.log(self._event())

            self.assertEqual(output_path.read_text(encoding="utf-8"), malformed)

    def test_jsonl_serializes_instances_using_the_same_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "responses.jsonl"
            old_record = {"old": "x" * 9000}
            output_path.write_text(json.dumps(old_record), encoding="utf-8")
            alias = Path(directory) / "alias.jsonl"
            alias.symlink_to(output_path)
            barrier = threading.Barrier(8)
            settings = self._minimal_settings()

            def append(index: int) -> None:
                logger = JSONL(
                    heretic_settings=settings,
                    settings=JSONLSettings(path=alias if index % 2 else output_path),
                )
                barrier.wait(timeout=10)
                logger.log(self._event())

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(append, range(8)))
            records = self._records(output_path)
            self.assertEqual(records[0], old_record)
            self.assertEqual(len(records), 9)
            self.assertTrue(
                all(record["response"] == "response" for record in records[1:])
            )

    def test_logger_cannot_mutate_scores_returned_to_optimizer(self) -> None:
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.session_id = "session-a"
        evaluator._logger_entries = [
            LoggerEntry(
                logger=MutatingLogger(heretic_settings=self._minimal_settings()),
                name="MutatingLogger",
                config=LoggerConfig(plugin="test_logger.MutatingLogger"),
            ),
        ]
        score = Score(value=0.5, rich_display="0.5", md_display="0.5")

        evaluator._log_event(
            ctx=SimpleNamespace(response_records=()),
            phase="evaluation",
            trial_number=1,
            parameters={},
            scores=[("Refusals", score)],
        )

        self.assertEqual(score.value, 0.5)

    def test_reproduction_config_excludes_local_logger_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "private" / "responses.jsonl"
            config = generate_config_toml(self._settings(output_path))

            self.assertNotIn("responses.jsonl", config)
            self.assertNotIn("loggers", config)
            self.assertNotIn("logger", config)

    def test_reproduction_json_excludes_local_logger_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "private" / "responses.jsonl"
            trial = create_trial(
                value=0.5,
                user_attrs={
                    "direction_index": 0,
                    "parameters": {},
                    "scores": [],
                },
            )
            reproduction = json.loads(
                generate_reproduce_json(
                    self._settings(output_path),
                    trial,
                    timestamp="2026-09-18T00:00:00",
                    uploaded_model_hashes={},
                    include_system_information=False,
                )
            )

            self.assertNotIn("loggers", reproduction["settings"])
            self.assertNotIn("logger", reproduction["settings"])
            self.assertNotIn("responses.jsonl", json.dumps(reproduction))

    def test_reproduction_settings_sanitization_drops_incoming_logger_config(
        self,
    ) -> None:
        incoming = {
            "model": "fake-model",
            "loggers": [{"plugin": "heretic.loggers.jsonl.JSONL"}],
            "logger": {"JSONL": {"path": "/private/responses.jsonl"}},
        }

        sanitized = _strip_logger_settings(incoming)
        with patch.object(sys, "argv", ["test_logger"]):
            settings = Settings.model_validate(sanitized)

        self.assertEqual(settings.loggers, [])
        self.assertNotIn("logger", settings.model_extra or {})

    def test_exported_journal_is_sanitized_and_remains_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint_path = directory_path / "checkpoint.jsonl"
            exported_path = directory_path / "reproduce" / "checkpoint.jsonl"
            first_settings = self._settings(directory_path / "first.jsonl")
            second_settings = self._settings(directory_path / "second.jsonl")

            storage = JournalStorage(JournalFileBackend(str(checkpoint_path)))
            study = optuna.create_study(
                storage=storage,
                study_name="heretic",
                directions=["minimize"],
            )
            study.set_user_attr("settings", first_settings.model_dump_json())
            study.optimize(lambda _trial: 0.5, n_trials=1)
            study.set_user_attr("settings", second_settings.model_dump_json())
            original_checkpoint = checkpoint_path.read_bytes()

            _sanitize_checkpoint_for_reproduction(checkpoint_path, exported_path)

            self.assertEqual(checkpoint_path.read_bytes(), original_checkpoint)
            exported_text = exported_path.read_text(encoding="utf-8")
            self.assertNotIn("first.jsonl", exported_text)
            self.assertNotIn("second.jsonl", exported_text)
            self.assertNotIn("heretic.loggers.jsonl.JSONL", exported_text)

            exported_storage = JournalStorage(JournalFileBackend(str(exported_path)))
            exported_studies = exported_storage.get_all_studies()
            self.assertEqual(len(exported_studies), 1)
            exported_attrs = exported_storage.get_study_user_attrs(
                exported_studies[0]._study_id
            )
            exported_settings = json.loads(exported_attrs["settings"])
            self.assertNotIn("loggers", exported_settings)
            self.assertNotIn("logger", exported_settings)

            resumed = optuna.load_study(
                study_name="heretic",
                storage=exported_storage,
            )
            resumed.optimize(lambda _trial: 0.25, n_trials=1)
            self.assertEqual([trial.number for trial in resumed.trials], [0, 1])

    def test_exported_journal_rejects_malformed_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint_path = directory_path / "checkpoint.jsonl"
            exported_path = directory_path / "reproduce" / "checkpoint.jsonl"
            checkpoint_path.write_text('{"op_code": 0}\nnot-json\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                _sanitize_checkpoint_for_reproduction(
                    checkpoint_path,
                    exported_path,
                )

            self.assertFalse(exported_path.exists())


if __name__ == "__main__":
    unittest.main()
