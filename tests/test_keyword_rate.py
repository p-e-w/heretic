# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import io
import unittest
from typing import cast
from unittest.mock import patch

from rich.console import Console

from heretic.config import Settings as HereticSettings
from heretic.scorer import Context
from heretic.scorers.keyword_rate import KeywordRate, Settings
from heretic.utils import Prompt


class FakeContext:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def get_responses(self, prompts: list[Prompt]) -> list[str]:
        return self.responses


class KeywordRateTests(unittest.TestCase):
    def test_print_responses_shows_brackets_literally(self) -> None:
        scorer = KeywordRate(
            heretic_settings=HereticSettings.model_construct(),
            settings=Settings(print_responses=True),
        )
        scorer.prompts = [
            Prompt(
                system="You are a [helpful] assistant.",
                user="Name a common misconception about [bird migration].",
            ),
            Prompt(
                system="You are a helpful assistant.",
                user="What does the [/INST] token mean?",
            ),
        ]
        ctx = FakeContext(["Use arr[i] here.", "It ends the [/INST] block."])

        output = io.StringIO()
        console = Console(file=output, highlight=False, width=200)

        with patch("heretic.scorers.keyword_rate.print", console.print):
            score = scorer.get_score(cast(Context, ctx))

        self.assertEqual(score.md_display, "0/2")
        for text in [
            "You are a [helpful] assistant.",
            "[bird migration]",
            "What does the [/INST] token mean?",
            "Use arr[i] here.",
            "It ends the [/INST] block.",
        ]:
            self.assertIn(text, output.getvalue())


if __name__ == "__main__":
    unittest.main()
