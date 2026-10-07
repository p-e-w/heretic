# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from argparse import Action
from typing import Any

from pydantic_settings import CliSettingsSource

from .config import Settings


def _get_value_option_nargs() -> dict[str, str | int | None]:
    option_nargs: dict[str, str | int | None] = {}

    def add_argument_method(parser: Any, *option_strings: str, **kwargs: Any) -> Action:
        action = parser.add_argument(*option_strings, **kwargs)

        if action.nargs != 0:
            for option_string in action.option_strings:
                option_nargs[option_string] = action.nargs

        return action

    CliSettingsSource(
        Settings,
        cli_parse_args=False,
        cli_implicit_flags=True,
        cli_kebab_case=True,
        add_argument_method=add_argument_method,
    )

    return option_nargs


VALUE_OPTION_NARGS = _get_value_option_nargs()

MODEL_OPTION = "--model"
NON_PROCESSING_MODE_OPTIONS = {
    "--collect-reproducibles",
    "--reproduce",
}


def _has_option(args: list[str], option: str) -> bool:
    for arg in args:
        if arg == "--":
            break

        if arg == option or arg.startswith(f"{option}="):
            return True

    return False


def _has_model_option(args: list[str]) -> bool:
    return _has_option(args, MODEL_OPTION)


def _is_non_processing_mode(args: list[str]) -> bool:
    return any(_has_option(args, option) for option in NON_PROCESSING_MODE_OPTIONS)


def _find_positional_model_index(args: list[str]) -> int | None:
    skip_next = False

    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        if arg == "--":
            return index if index + 1 < len(args) else None

        if arg.startswith("-"):
            option = arg.split("=", 1)[0]
            if "=" not in arg and option in VALUE_OPTION_NARGS:
                nargs = VALUE_OPTION_NARGS[option]
                if nargs != "?" or (
                    index + 1 < len(args) and not args[index + 1].startswith("-")
                ):
                    skip_next = True
            continue

        return index

    return None


def normalize_cli_args(args: list[str]) -> list[str]:
    """
    Normalize Heretic's positional model shorthand into explicit pydantic CLI args.
    """

    normalized_args = list(args)

    if _has_model_option(normalized_args):
        return normalized_args

    if _is_non_processing_mode(normalized_args):
        return [*normalized_args, MODEL_OPTION, ""]

    model_index = _find_positional_model_index(normalized_args)
    if model_index is not None:
        if normalized_args[model_index] == "--":
            model = normalized_args[model_index + 1]
            if model.startswith("-"):
                normalized_args[model_index : model_index + 2] = [
                    f"{MODEL_OPTION}={model}"
                ]
            else:
                normalized_args[model_index] = MODEL_OPTION
        else:
            normalized_args.insert(model_index, MODEL_OPTION)

    return normalized_args


def is_help_invocation(args: list[str]) -> bool:
    for arg in normalize_cli_args(args):
        if arg == "--":
            break

        if arg in ("-h", "--help"):
            return True

    return False
