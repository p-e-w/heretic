import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F

from heretic.config import InitOnlySettings
from heretic.plugin import Context
from heretic.scorers.kl_divergence import KLDivergence, Settings
from heretic.utils import Prompt


class KLPrecisionTests(unittest.TestCase):
    def scorer(self, baseline):
        context = Mock(spec=Context)
        context.load_prompts.return_value = [Prompt("", "fixture")]
        context.get_logits.return_value = baseline
        scorer = KLDivergence(InitOnlySettings(model="unused"), Settings())
        scorer.init(context)
        return scorer, context

    def reference(self, baseline, current):
        return F.kl_div(
            F.log_softmax(current.double(), dim=-1),
            F.log_softmax(baseline.double(), dim=-1),
            reduction="batchmean",
            log_target=True,
        ).item()

    def test_bfloat16_scores_preserve_nearby_trial_order(self):
        baseline = torch.tensor([[0.5, 0.0]], dtype=torch.bfloat16)
        scorer, context = self.scorer(baseline)
        scores = []
        for difference in [0.015625, 0.03125]:
            current = torch.tensor([[0.5 + difference, 0.0]], dtype=torch.bfloat16)
            context.get_logits.return_value = current
            score = scorer.get_score(context).value
            self.assertAlmostEqual(score, self.reference(baseline, current), delta=1e-7)
            scores.append(score)
        self.assertLess(scores[0], scores[1])
        self.assertGreater(scores[0], 0.0)

    def test_identical_low_precision_logits_score_zero(self):
        for dtype in [torch.bfloat16, torch.float16]:
            with self.subTest(dtype=dtype):
                baseline = torch.tensor([[0.5, 0.0]], dtype=dtype)
                scorer, context = self.scorer(baseline)
                self.assertEqual(scorer.get_score(context).value, 0.0)

    def test_full_precision_logits_remain_unchanged(self):
        for dtype in [torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                baseline = torch.tensor([[0.5, 0.0]], dtype=dtype)
                current = torch.tensor([[0.7, 0.0]], dtype=dtype)
                scorer, context = self.scorer(baseline)
                context.get_logits.return_value = current
                expected = F.kl_div(
                    F.log_softmax(current, dim=-1),
                    F.log_softmax(baseline, dim=-1),
                    reduction="batchmean",
                    log_target=True,
                ).item()
                self.assertEqual(scorer.get_score(context).value, expected)

    def test_bfloat16_metric_does_not_prefer_the_farther_trial(self):
        baseline = torch.tensor([[0.5, 0.0]], dtype=torch.bfloat16)
        scorer, context = self.scorer(baseline)
        values = []
        references = []
        for value in [0.515625, 0.53125]:
            current = torch.tensor([[value, 0.0]], dtype=torch.bfloat16)
            context.get_logits.return_value = current
            values.append(scorer.get_score(context).value)
            references.append(self.reference(baseline, current))
        self.assertLess(references[0], references[1])
        self.assertLess(values[0], values[1])
