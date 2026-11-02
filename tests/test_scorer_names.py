# SPDX-License-Identifier: AGPL-3.0-or-later
"""Final metric labels must not alias distinct optimization objectives."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from heretic.config import InitOnlySettings
from heretic.evaluator import Evaluator
from heretic.scorer import Score
from heretic.scorers.bert_classifier import BERTClassifier
from heretic.scorers.kl_divergence import KLDivergence

BERT_PLUGIN = "heretic.scorers.bert_classifier.BERTClassifier"
KL_PLUGIN = "heretic.scorers.kl_divergence.KLDivergence"


class ScorerNameTests(unittest.TestCase):
    def setUp(self):
        self.initialized = []
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        for scorer in (BERTClassifier, KLDivergence):
            self.patches.enter_context(
                patch.object(
                    scorer,
                    "init",
                    autospec=True,
                    side_effect=lambda instance, ctx: self.initialized.append(
                        type(instance).__name__
                    ),
                )
            )
        self.patches.enter_context(
            patch.object(
                BERTClassifier,
                "get_score",
                autospec=True,
                return_value=Score(value=0.25, rich_display="0.25", md_display="0.25"),
            )
        )
        self.patches.enter_context(
            patch.object(
                KLDivergence,
                "get_score",
                autospec=True,
                return_value=Score(value=0.75, rich_display="0.75", md_display="0.75"),
            )
        )

    def settings(self, classifier_name, kl_instance=None):
        return InitOnlySettings(
            model="test/base",
            scorers=[
                {"plugin": BERT_PLUGIN, "optimization": "minimize"},
                {
                    "plugin": KL_PLUGIN,
                    "optimization": "minimize",
                    "instance_name": kl_instance,
                },
            ],
            scorer={"BERTClassifier": {"score_name": classifier_name}},
        )

    def test_custom_classifier_label_cannot_alias_another_scorer(self):
        with self.assertRaisesRegex(ValueError, "Duplicate score name"):
            Evaluator(self.settings("KL divergence"), SimpleNamespace())
        self.assertEqual(self.initialized, [])

    def test_instance_suffix_cannot_create_a_duplicate_final_label(self):
        with self.assertRaisesRegex(ValueError, "Duplicate score name"):
            Evaluator(
                self.settings("KL divergence - x", kl_instance="x"), SimpleNamespace()
            )
        self.assertEqual(self.initialized, [])

    def test_distinct_labels_preserve_independent_objective_values(self):
        evaluator = Evaluator(self.settings("Classifier refusals"), SimpleNamespace())
        self.assertEqual(len(self.initialized), 2)
        self.assertEqual(
            evaluator.get_objective_names(), ["Classifier refusals", "KL divergence"]
        )
        self.assertEqual(
            evaluator.get_objective_values(evaluator.get_scores()), (0.25, 0.75)
        )

    def test_distinct_instances_of_one_scorer_remain_supported(self):
        settings = InitOnlySettings(
            model="test/base",
            scorers=[
                {
                    "plugin": BERT_PLUGIN,
                    "optimization": "minimize",
                    "instance_name": "one",
                },
                {
                    "plugin": BERT_PLUGIN,
                    "optimization": "minimize",
                    "instance_name": "two",
                },
            ],
        )
        evaluator = Evaluator(settings, SimpleNamespace())
        self.assertEqual(len(self.initialized), 2)
        self.assertEqual(
            evaluator.get_objective_names(),
            [
                "Refusals (Classifier) - one",
                "Refusals (Classifier) - two",
            ],
        )


if __name__ == "__main__":
    unittest.main()
