# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import gc
import getpass
from dataclasses import asdict
from importlib.metadata import version
from typing import TypeVar

import questionary
import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import load_dataset
from optuna import Trial
from questionary import Choice
from rich.console import Console

from .config import DatasetSpecification, Settings

print = Console(highlight=False).print


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def load_prompts(specification: DatasetSpecification) -> list[str]:
    dataset = load_dataset(specification.dataset, split=specification.split)
    return list(dataset[specification.column])


T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def empty_cache():
    # Collecting garbage is not an idempotent operation, and to avoid OOM errors,
    # gc.collect() has to be called both before and after emptying the backend cache.
    # See https://github.com/p-e-w/heretic/pull/17 for details.
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()
    elif is_sdaa_available():
        torch.sdaa.empty_cache()
    elif is_musa_available():
        torch.musa.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()


def get_trial_parameters(trial: Trial) -> dict[str, str]:
    params = {}

    direction_index = trial.user_attrs["direction_index"]
    params["direction_index"] = (
        "per layer" if (direction_index is None) else f"{direction_index:.2f}"
    )

    for component, parameters in trial.user_attrs["parameters"].items():
        for name, value in asdict(parameters).items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[str],
) -> str:
    model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"

    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version("heretic-llm")}

## Abliteration parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in get_trial_parameters(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.2f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""


def is_notebook() -> bool:
    import os
    import sys

    # 1. Check for Google Colab environment variables
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return True

    # 2. Check for Google Colab module
    if "google.colab" in sys.modules:
        return True

    # 3. Check for Kaggle environment variables
    if (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_DATA_PROXY_TOKEN" in os.environ
    ):
        return True

    # 4. Check IPython shell type
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False

        shell_name = shell.__class__.__name__
        # Standard Jupyter/Colab shells
        if shell_name in ["ZMQInteractiveShell", "Shell"]:
            return True

        # Some other environments might have different shell names but still be notebooks
        if "google.colab" in str(shell.__class__):
            return True

    except (ImportError, NameError, AttributeError):
        return False

    return False


def prompt_select(message: str, choices: list, style=None):
    if is_notebook():
        print()
        print(message)
        real_choices = []
        for i, choice in enumerate(choices, 1):
            if isinstance(choice, Choice):
                print(f"[{i}] {choice.title}")
                real_choices.append(choice.value)
            else:
                print(f"[{i}] {choice}")
                real_choices.append(choice)

        print()
        while True:
            print(f"Enter number (1-{len(real_choices)}): ", end="")
            selection = input()
            if not selection:
                return None
            if selection.isdigit() and 1 <= int(selection) <= len(real_choices):
                return real_choices[int(selection) - 1]
            print("Please enter a valid number")
    else:
        return questionary.select(message, choices=choices, style=style).ask()


def prompt_text(message: str, default: str = None, qmark: str = "?") -> str:
    if is_notebook():
        prompt = f"{message} [{default}]: " if default else f"{message}: "
        if qmark == ">":  # Chat mode
            prompt = "User: "

        print(prompt, end="")
        value = input()

        # Handle exit commands in notebook chat mode
        if qmark == ">" and value.strip().lower() in ["/exit", "exit", "quit"]:
            return ""

        return value if value else default
    else:
        return questionary.text(
            message, default=default or "", qmark=qmark
        ).unsafe_ask()


def prompt_password(message: str) -> str:
    if is_notebook():
        print(f"{message} ", end="")
        return getpass.getpass("")
    else:
        return questionary.password(message).ask()


def prompt_path(message: str) -> str:
    if is_notebook():
        print(f"{message} ", end="")
        return input()
    else:
        return questionary.path(message).ask()
