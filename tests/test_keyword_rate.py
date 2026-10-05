# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace
from typing import Any

from heretic.scorers.keyword_rate import KeywordRate, Settings
from heretic.utils import Prompt

PROMPTS = [Prompt(system="", user=f"prompt {i}") for i in range(20)]


def init_scorer(holdout_fraction: float, seed: int = 0) -> KeywordRate:
    scorer = KeywordRate(
        heretic_settings=SimpleNamespace(seed=seed),  # ty: ignore[invalid-argument-type]
        settings=Settings(holdout_fraction=holdout_fraction),
    )
    ctx: Any = SimpleNamespace(load_prompts=lambda specification: list(PROMPTS))
    scorer.init(ctx)
    return scorer


class HoldoutSplitTests(unittest.TestCase):
    def test_no_holdout_by_default(self) -> None:
        scorer = init_scorer(0.0)

        self.assertEqual(scorer.prompts, PROMPTS)
        self.assertEqual(scorer.holdout_prompts, [])
        self.assertIsNone(scorer.get_holdout_score(SimpleNamespace()))  # ty: ignore[invalid-argument-type]

    def test_split_is_disjoint_and_complete(self) -> None:
        scorer = init_scorer(0.25)

        self.assertEqual(len(scorer.prompts), 15)
        self.assertEqual(len(scorer.holdout_prompts), 5)
        self.assertCountEqual(scorer.prompts + scorer.holdout_prompts, PROMPTS)

    def test_split_is_determined_by_seed(self) -> None:
        self.assertEqual(
            init_scorer(0.25, seed=1).holdout_prompts,
            init_scorer(0.25, seed=1).holdout_prompts,
        )
        self.assertNotEqual(
            init_scorer(0.25, seed=1).holdout_prompts,
            init_scorer(0.25, seed=2).holdout_prompts,
        )

    def test_rejects_fraction_that_empties_a_part(self) -> None:
        for holdout_fraction in [0.01, 0.99]:
            with self.subTest(holdout_fraction=holdout_fraction):
                with self.assertRaises(ValueError):
                    init_scorer(holdout_fraction)


if __name__ == "__main__":
    unittest.main()
