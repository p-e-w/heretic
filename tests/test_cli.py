# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from heretic.cli import normalize_cli_args


class NormalizeCliArgsTests(unittest.TestCase):
    def test_inserts_model_flag_before_trailing_positional_model(self):
        self.assertEqual(
            normalize_cli_args(["--quantization", "bnb_4bit", "Qwen/Qwen3"]),
            ["--quantization", "bnb_4bit", "--model", "Qwen/Qwen3"],
        )

    def test_inserts_model_flag_before_leading_positional_model(self):
        self.assertEqual(
            normalize_cli_args(["Qwen/Qwen3", "--n-trials", "10"]),
            ["--model", "Qwen/Qwen3", "--n-trials", "10"],
        )

    def test_does_not_treat_option_value_as_positional_model(self):
        self.assertEqual(
            normalize_cli_args(["--n-trials", "10"]),
            ["--n-trials", "10"],
        )

    def test_preserves_explicit_model_option(self):
        self.assertEqual(
            normalize_cli_args(["--model", "Qwen/Qwen3", "--n-trials", "10"]),
            ["--model", "Qwen/Qwen3", "--n-trials", "10"],
        )

    def test_preserves_explicit_model_equals_option(self):
        self.assertEqual(
            normalize_cli_args(["--model=Qwen/Qwen3", "--n-trials", "10"]),
            ["--model=Qwen/Qwen3", "--n-trials", "10"],
        )

    def test_adds_placeholder_model_for_reproduce_mode(self):
        self.assertEqual(
            normalize_cli_args(["--reproduce", "reproduce.json"]),
            ["--reproduce", "reproduce.json", "--model", ""],
        )

    def test_adds_placeholder_model_for_collect_reproducibles_mode(self):
        self.assertEqual(
            normalize_cli_args(["--collect-reproducibles=out"]),
            ["--collect-reproducibles=out", "--model", ""],
        )


if __name__ == "__main__":
    unittest.main()
