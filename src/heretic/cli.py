# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import types
from typing import Any, get_args, get_origin

from .config import Settings


def _cli_option_name(field_name: str) -> str:
    return f"--{field_name.replace('_', '-')}"


def _is_bool_annotation(annotation: Any) -> bool:
    if annotation is bool:
        return True

    origin = get_origin(annotation)
    if origin in (types.UnionType,):
        return bool in get_args(annotation)

    return False


VALUE_OPTIONS = {
    _cli_option_name(field_name)
    for field_name, field in Settings.model_fields.items()
    if not _is_bool_annotation(field.annotation)
}

MODEL_OPTION = "--model"
NON_PROCESSING_MODE_OPTIONS = {
    "--collect-reproducibles",
    "--reproduce",
}


def _has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


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
            return index + 1 if index + 1 < len(args) else None

        if arg.startswith("-"):
            option = arg.split("=", 1)[0]
            if "=" not in arg and option in VALUE_OPTIONS:
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
        normalized_args.insert(model_index, MODEL_OPTION)

    return normalized_args
