# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from typing import cast

from heretic.config import Settings
from heretic.plugin import Context
from heretic.scorers.keyword_rate import KeywordRate
from heretic.scorers.keyword_rate import Settings as KeywordRateSettings
from heretic.scorers.kl_divergence import KLDivergence
from heretic.scorers.kl_divergence import Settings as KLDivergenceSettings


class EmptyPromptContext:
    logits_requested = False

    def load_prompts(self, specification) -> list:
        return []

    def get_logits(self, prompts):
        self.logits_requested = True
        raise AssertionError("empty prompt validation should happen before logits")


class ScorerPromptValidationTests(unittest.TestCase):
    def test_keyword_rate_rejects_empty_prompt_set_during_init(self):
        scorer = KeywordRate(
            heretic_settings=cast(Settings, object()),
            settings=KeywordRateSettings(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "KeywordRate scorer requires at least one prompt",
        ):
            scorer.init(cast(Context, EmptyPromptContext()))

    def test_kl_divergence_rejects_empty_prompt_set_before_getting_logits(self):
        scorer = KLDivergence(
            heretic_settings=cast(Settings, object()),
            settings=KLDivergenceSettings(),
        )
        ctx = EmptyPromptContext()

        with self.assertRaisesRegex(
            ValueError,
            "KLDivergence scorer requires at least one prompt",
        ):
            scorer.init(cast(Context, ctx))

        self.assertFalse(ctx.logits_requested)


if __name__ == "__main__":
    unittest.main()
