# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from typing import Any, cast
from unittest import mock

import optuna

from heretic.config import Settings
from heretic.evaluator import Evaluator
from heretic.main import rescore_top_trials_on_merged_models
from heretic.model import Model
from heretic.modifier import Modifier
from heretic.plugin import Context
from heretic.scorer import Score

# Matches the order in which the scorers are evaluated below.
OBJECTIVE_NAMES = ["KL divergence", "Refusal rate"]

# Live (trial-time) scores. The values are mutually non-dominating, so every
# trial lands on the Pareto front, sorted ascending by score tuple.
LIVE_SCORES = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)]

# Scores returned by the fake evaluator, in call order (the function re-scores
# the front trials in live-sorted order). Deliberately ordered so that ranking
# by merged scores flips the first two trials relative to the live ranking.
MERGED_SCORES = [(101.0, 50.0), (100.0, 49.0), (103.0, 52.0), (104.0, 51.0)]


def make_score_records(kld: float, refusals: float) -> list[dict[str, Any]]:
    return [
        {
            "name": "KL divergence",
            "score": {
                "value": kld,
                "rich_display": f"{kld:.4f}",
                "md_display": f"{kld:.4f}",
            },
            "baseline": {},
        },
        {
            "name": "Refusal rate",
            "score": {
                "value": refusals,
                "rich_display": f"{refusals:.1f}",
                "md_display": f"{refusals:.1f}",
            },
            "baseline": {},
        },
    ]


def make_scores(values: tuple[float, float]) -> list[tuple[str, Score]]:
    return [
        ("KL divergence", Score(value=values[0], rich_display="", md_display="")),
        ("Refusal rate", Score(value=values[1], rich_display="", md_display="")),
    ]


class FakeParameters:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FakeParameters":
        return cls(data)


class FakeModel:
    def __init__(self):
        self.model = object()  # The live model, as Model.model would be.
        self.merge_calls = 0

    def get_merged_model(self):
        self.merge_calls += 1
        return object()  # A distinct merged model on every call.


class FakeModifier:
    parameters_class = FakeParameters

    def __init__(self):
        self.reset_calls = 0
        self.modified_parameters = []

    def reset_model(self, ctx: Context) -> None:
        self.reset_calls += 1

    def modify_model(self, ctx: Context, parameters: Any) -> None:
        self.modified_parameters.append(parameters)


class FakeEvaluator:
    def __init__(self, results: list[Any]):
        # `results[i]` is returned by the i-th `get_scores()` call;
        # an Exception instance is raised instead of returned.
        self.results = list(results)
        self.get_scores_calls = 0

    def get_scores(self) -> list[tuple[str, Score]]:
        result = self.results.pop(0)
        self.get_scores_calls += 1
        if isinstance(result, Exception):
            raise result
        return make_scores(result)

    def get_paired_score_records(
        self, scores: list[tuple[str, Score]]
    ) -> list[dict[str, Any]]:
        return [
            {"name": name, "score": dict(score.__dict__), "baseline": {}}
            for name, score in scores
        ]


def build_study() -> optuna.study.Study:
    study = optuna.create_study(directions=["minimize", "minimize"])

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        kld, refusals = LIVE_SCORES[trial.number]
        trial.set_user_attr("index", trial.number + 1)
        trial.set_user_attr("parameters", {"strength": 0.1 * (trial.number + 1)})
        trial.set_user_attr("scores", make_score_records(kld, refusals))
        return kld, refusals

    study.optimize(objective, n_trials=len(LIVE_SCORES))
    return study


def sorted_pareto_trials(study: optuna.study.Study) -> list[optuna.trial.FrozenTrial]:
    # Mirrors the live-score ordering used in main.run().
    return sorted(
        study.best_trials,
        key=lambda trial: tuple(
            next(
                (
                    score["score"]["value"]
                    for score in trial.user_attrs["scores"]
                    if score["name"] == name
                ),
                None,
            )
            for name in OBJECTIVE_NAMES
        ),
    )


class RescoreTopTrialsOnMergedModelsTests(unittest.TestCase):
    def run_rescoring(
        self,
        evaluator: FakeEvaluator,
        n_merged_score_trials: int,
    ) -> tuple[
        optuna.study.Study,
        FakeModel,
        FakeModifier,
        list[optuna.trial.FrozenTrial],
        dict[int, list[dict[str, Any]]],
    ]:
        study = build_study()
        sorted_trials = sorted_pareto_trials(study)
        self.assertEqual(
            [trial.number for trial in sorted_trials],
            [0, 1, 2, 3],
            "test setup broken: front is not in expected live order",
        )

        # Settings parses CLI arguments on instantiation (cli_parse_args),
        # which would choke on pytest's own argv, so neutralize it here.
        with mock.patch("sys.argv", ["heretic"]):
            settings = Settings(
                model="dummy-model", n_merged_score_trials=n_merged_score_trials
            )
        model = FakeModel()
        modifier = FakeModifier()

        # The fakes duck-type the real interfaces; `cast` keeps the type
        # checker happy without dragging the heavyweight classes into it.
        merged_scores_by_trial = rescore_top_trials_on_merged_models(
            study=study,
            sorted_trials=sorted_trials,
            settings=settings,
            model=cast(Model, model),
            evaluator=cast(Evaluator, evaluator),
            modifier=cast(Modifier, modifier),
            objective_names=OBJECTIVE_NAMES,
        )
        return study, model, modifier, sorted_trials, merged_scores_by_trial

    def test_top_trials_are_rescored_on_merged_models(self) -> None:
        evaluator = FakeEvaluator(MERGED_SCORES)
        study, model, modifier, sorted_trials, merged_scores_by_trial = (
            self.run_rescoring(evaluator, n_merged_score_trials=2)
        )

        # Exactly the top-K trials were re-scored, with the model restored
        # to the live state afterwards.
        self.assertEqual(model.merge_calls, 2)
        self.assertEqual(evaluator.get_scores_calls, 2)
        self.assertEqual(sorted(merged_scores_by_trial), [0, 1])
        self.assertIsInstance(model.model, object)
        self.assertEqual(modifier.reset_calls, 2)
        self.assertEqual(
            [parameters.data for parameters in modifier.modified_parameters],
            [{"strength": 0.1}, {"strength": 0.2}],
        )

        # The merged score records were persisted to the study.
        self.assertEqual(sorted(study.user_attrs["merged_scores"]), [0, 1])
        self.assertEqual(merged_scores_by_trial[0][0]["score"]["value"], 101.0)

        # Only part of the front has merged scores, so trials keep their
        # live-score ranking (merging live and merged values into one
        # ranking would compare incommensurable numbers).
        self.assertEqual(
            [trial.number for trial in sorted_trials],
            [0, 1, 2, 3],
        )

    def test_failed_merged_evaluation_skips_trial(self) -> None:
        evaluator = FakeEvaluator([RuntimeError("merge failed"), MERGED_SCORES[1]])
        study, model, _modifier, sorted_trials, merged_scores_by_trial = (
            self.run_rescoring(evaluator, n_merged_score_trials=2)
        )

        # The failing trial was skipped, the second one was still re-scored,
        # and the failure did not propagate.
        self.assertEqual(model.merge_calls, 2)
        self.assertEqual(sorted(merged_scores_by_trial), [1])
        self.assertEqual(sorted(study.user_attrs["merged_scores"]), [1])

        # Not every trial has merged scores, so the live ranking is kept.
        self.assertEqual(
            [trial.number for trial in sorted_trials],
            [0, 1, 2, 3],
        )

    def test_zero_disables_rescoring(self) -> None:
        evaluator = FakeEvaluator(MERGED_SCORES)
        study, model, _, sorted_trials, merged_scores_by_trial = self.run_rescoring(
            evaluator, n_merged_score_trials=0
        )

        self.assertEqual(merged_scores_by_trial, {})
        self.assertEqual(model.merge_calls, 0)
        self.assertNotIn("merged_scores", study.user_attrs)
        self.assertEqual(
            [trial.number for trial in sorted_trials],
            [0, 1, 2, 3],
        )

    def test_front_smaller_than_k_rescores_entire_front(self) -> None:
        evaluator = FakeEvaluator(MERGED_SCORES)
        study, model, _, sorted_trials, merged_scores_by_trial = self.run_rescoring(
            evaluator, n_merged_score_trials=10
        )

        self.assertEqual(model.merge_calls, 4)
        self.assertEqual(sorted(merged_scores_by_trial), [0, 1, 2, 3])

        # Every trial on the front has merged scores, so the front is
        # re-ranked by them: trial 1 now comes first.
        self.assertEqual(
            [trial.number for trial in sorted_trials],
            [1, 0, 2, 3],
        )
        self.assertEqual(sorted(study.user_attrs["merged_scores"]), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
