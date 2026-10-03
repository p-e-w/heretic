# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import sys
import unittest
from unittest.mock import patch

from heretic.cli import is_help_invocation, normalize_cli_args
from heretic.config import Settings


def _parse_settings(args: list[str]) -> Settings:
    normalized_args = normalize_cli_args(args)

    with patch.object(sys, "argv", ["heretic", *normalized_args]):
        try:
            return Settings()  # ty:ignore[missing-argument]
        except SystemExit as error:
            raise AssertionError(
                f"Settings CLI parsing exited with status {error.code}; "
                f"normalized args: {normalized_args!r}"
            ) from error


class SettingsCliIntegrationTests(unittest.TestCase):
    def test_parses_nullable_bool_value_before_positional_model(self) -> None:
        settings = _parse_settings(["--upload-repo-private", "true", "MODEL"])

        self.assertIs(settings.upload_repo_private, True)
        self.assertEqual(settings.model, "MODEL")

    def test_parses_nested_option_value_before_positional_model(self) -> None:
        settings = _parse_settings(["--good-prompts.dataset", "local.txt", "MODEL"])

        self.assertEqual(settings.good_prompts.dataset, "local.txt")
        self.assertEqual(settings.model, "MODEL")

    def test_parses_nested_option_after_parent_without_value(self) -> None:
        settings = _parse_settings(
            ["--good-prompts", "--good-prompts.dataset", "local.txt", "MODEL"]
        )

        self.assertEqual(settings.good_prompts.dataset, "local.txt")
        self.assertEqual(settings.model, "MODEL")

    def test_parses_dash_prefixed_positional_model_after_terminator(self) -> None:
        settings = _parse_settings(["--", "-local-model"])

        self.assertEqual(settings.model, "-local-model")

    def test_option_terminator_protects_model_option_like_model(self) -> None:
        settings = _parse_settings(["--", "--model=literal"])

        self.assertEqual(settings.model, "--model=literal")

    def test_option_terminator_protects_non_processing_mode_like_model(self) -> None:
        settings = _parse_settings(["--", "--reproduce"])

        self.assertEqual(settings.model, "--reproduce")
        self.assertIsNone(settings.reproduce)


class HelpInvocationTests(unittest.TestCase):
    def test_option_terminator_protects_help_like_model(self) -> None:
        self.assertFalse(is_help_invocation(["--", "--help"]))

    def test_option_terminator_hides_help_after_explicit_model(self) -> None:
        self.assertFalse(is_help_invocation(["--model", "MODEL", "--", "--help"]))


class NormalizeCliArgsTests(unittest.TestCase):
    def test_replaces_option_terminator_before_positional_model(self) -> None:
        self.assertEqual(
            normalize_cli_args(["--", "Qwen/Qwen3"]),
            ["--model", "Qwen/Qwen3"],
        )

    def test_inserts_model_flag_before_trailing_positional_model(self) -> None:
        self.assertEqual(
            normalize_cli_args(["--quantization", "bnb_4bit", "Qwen/Qwen3"]),
            ["--quantization", "bnb_4bit", "--model", "Qwen/Qwen3"],
        )

    def test_inserts_model_flag_before_leading_positional_model(self) -> None:
        self.assertEqual(
            normalize_cli_args(["Qwen/Qwen3", "--n-trials", "10"]),
            ["--model", "Qwen/Qwen3", "--n-trials", "10"],
        )

    def test_does_not_treat_option_value_as_positional_model(self) -> None:
        self.assertEqual(
            normalize_cli_args(["--n-trials", "10"]),
            ["--n-trials", "10"],
        )

    def test_preserves_explicit_model_option(self) -> None:
        self.assertEqual(
            normalize_cli_args(["--model", "Qwen/Qwen3", "--n-trials", "10"]),
            ["--model", "Qwen/Qwen3", "--n-trials", "10"],
        )

    def test_preserves_explicit_model_equals_option(self) -> None:
        self.assertEqual(
            normalize_cli_args(["--model=Qwen/Qwen3", "--n-trials", "10"]),
            ["--model=Qwen/Qwen3", "--n-trials", "10"],
        )

    def test_adds_placeholder_model_for_reproduce_mode(self) -> None:
        self.assertEqual(
            normalize_cli_args(["--reproduce", "reproduce.json"]),
            ["--reproduce", "reproduce.json", "--model", ""],
        )

    def test_adds_placeholder_model_for_collect_reproducibles_mode(self) -> None:
        self.assertEqual(
            normalize_cli_args(["--collect-reproducibles=out"]),
            ["--collect-reproducibles=out", "--model", ""],
        )


if __name__ == "__main__":
    unittest.main()
