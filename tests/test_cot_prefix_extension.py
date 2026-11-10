# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from heretic.main import get_cot_prefix_extension


class CotPrefixExtensionTests(unittest.TestCase):
    def test_extends_closed_block_with_missing_whitespace(self) -> None:
        # Template-injected CoT (e.g. LiquidAI LFM): responses continue right
        # after the closed block with a common newline that completes the prefix.
        # This is the case the recheck exists for, and it must keep working.
        self.assertEqual(
            get_cot_prefix_extension(["\n\nSure, here is", "\n\nOf course"]),
            "\n\n",
        )

    def test_returns_empty_string_for_fully_diverging_responses(self) -> None:
        # Nothing is common, so nothing may be appended.
        self.assertEqual(get_cot_prefix_extension(["Yes.", "No."]), "")

    def test_rejects_reasoning_text_common_prefix(self) -> None:
        # Qwen3-style self-generated CoT (issue #487): after the closed block
        # the model re-enters reasoning, and the reasoning beginnings are
        # near-identical across prompts. Appending them would keep the model
        # in CoT mode while writing its final answer.
        responses = [
            f"\nOkay, the user asks about {topic}. Let me think this through."
            for topic in ("cats", "dogs", "cars")
        ]
        # The reasoning continuation must not become part of the prefix.
        self.assertIsNone(get_cot_prefix_extension(responses))

    def test_rejects_common_prefix_containing_cot_boundary(self) -> None:
        # A common prefix that reaches across a </think> boundary necessarily
        # contains non-whitespace characters, so it must be rejected as well.
        responses = [
            "\nOkay, let me think. </think>\n\nHere is the answer about cats",
            "\nOkay, let me think. </think>\n\nHere is the answer about dogs",
        ]
        self.assertIsNone(get_cot_prefix_extension(responses))


if __name__ == "__main__":
    unittest.main()
