# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import contextlib
import gc
import getpass
import importlib.metadata
import os
import platform
import random
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, TypeVar

import huggingface_hub
import numpy as np
import questionary
import pynvml
import tomli_w
import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import DatasetDict, ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from optuna import Trial
from psutil import Process
from questionary import Choice, Style
from rich.console import Console

from .config import DatasetSpecification, Settings

print = Console(highlight=False).print


def get_accelerator_info(rich: bool = True) -> str:
    """The single source of truth for hardware detection and reporting."""
    b = "[bold]" if rich else ""
    _b = "[/]" if rich else ""
    y = "[bold yellow]" if rich else ""

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        total_vram = sum(torch.cuda.mem_get_info(i)[1] for i in range(count))
        cuda_ver = torch.version.cuda
        driver_ver = "Unknown"
        with contextlib.suppress(Exception):
            pynvml.nvmlInit()
            driver_ver = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()

        report = f"Detected {b}{count}{_b} CUDA device(s) ({total_vram / (1024**3):.2f} GB total VRAM)\n"
        report += f"CUDA Version: {b}{cuda_ver}{_b}\n"
        report += f"Driver Version: {b}{driver_ver}{_b}\n"

        for i in range(count):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.mem_get_info(i)[1] / (1024**3)
            report += f"* GPU {i}: {b}{name}{_b} ({vram:.2f} GB)\n"
    elif is_xpu_available():
        count = torch.xpu.device_count()
        driver_ver = "Unknown"
        with contextlib.suppress(Exception):
            driver_ver = torch.xpu.get_driver_version()

        report = f"Detected {b}{count}{_b} XPU device(s)\n"
        report += f"Driver Version: {b}{driver_ver}{_b}\n"
        for i in range(count):
            report += f"* XPU {i}: {b}{torch.xpu.get_device_name(i)}{_b}\n"
    elif is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore
        report = f"Detected {b}{count}{_b} MLU device(s):\n"
        for i in range(count):
            report += f"* MLU {i}: {b}{torch.mlu.get_device_name(i)}{_b}\n"  # ty:ignore
    elif is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore
        report = f"Detected {b}{count}{_b} SDAA device(s):\n"
        for i in range(count):
            report += f"* SDAA {i}: {b}{torch.sdaa.get_device_name(i)}{_b}\n"  # ty:ignore
    elif is_musa_available():
        count = torch.musa.device_count()  # ty:ignore
        report = f"Detected {b}{count}{_b} MUSA device(s):\n"
        for i in range(count):
            report += f"* MUSA {i}: {b}{torch.musa.get_device_name(i)}{_b}\n"  # ty:ignore
    elif is_npu_available():
        report = f"NPU detected (CANN version: {b}{torch.version.cann}{_b})\n"  # ty:ignore
    elif torch.backends.mps.is_available():
        report = f"Detected {b}1{_b} MPS device (Apple Metal)\n"
    else:
        report = (
            f"{y}No GPU or other accelerator detected. Operations will be slow.{_b}\n"
        )

    return report.strip()


def print_memory_usage():
    def p(label: str, size_in_bytes: int):
        print(f"[grey50]{label}: [bold]{size_in_bytes / (1024**3):.2f} GB[/][/]")

    p("Resident system RAM", Process().memory_info().rss)

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        allocated = sum(torch.cuda.memory_allocated(device) for device in range(count))
        reserved = sum(torch.cuda.memory_reserved(device) for device in range(count))
        p("Allocated GPU VRAM", allocated)
        p("Reserved GPU VRAM", reserved)
    elif is_xpu_available():
        count = torch.xpu.device_count()
        allocated = sum(torch.xpu.memory_allocated(device) for device in range(count))
        reserved = sum(torch.xpu.memory_reserved(device) for device in range(count))
        p("Allocated XPU memory", allocated)
        p("Reserved XPU memory", reserved)
    elif torch.backends.mps.is_available():
        p("Allocated MPS memory", torch.mps.current_allocated_memory())
        p("Driver (reserved) MPS memory", torch.mps.driver_allocated_memory())


def is_notebook() -> bool:
    # Check for specific environment variables (Colab, Kaggle).
    # This is necessary because when running as a subprocess (e.g. !heretic),
    # get_ipython() might not be available or might not reflect the notebook environment.
    if os.getenv("COLAB_GPU") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return True

    # Check IPython shell type (for library usage).
    try:
        from IPython import get_ipython  # ty:ignore[unresolved-import]

        shell = get_ipython()
        if shell is None:
            return False

        shell_name = shell.__class__.__name__
        if shell_name in ["ZMQInteractiveShell", "Shell"]:
            return True

        if "google.colab" in str(shell.__class__):
            return True

        return False
    except (ImportError, NameError, AttributeError):
        return False


def prompt_select(message: str, choices: list[Any]) -> Any:
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

        while True:
            try:
                selection = input("Enter number: ")
                index = int(selection) - 1
                if 0 <= index < len(real_choices):
                    return real_choices[index]
                print(
                    f"[red]Please enter a number between 1 and {len(real_choices)}[/]"
                )
            except ValueError:
                print("[red]Invalid input. Please enter a number.[/]")
    else:
        return questionary.select(
            message,
            choices=choices,
            style=Style([("highlighted", "reverse")]),
        ).ask()


def prompt_text(
    message: str,
    default: str = "",
    qmark: str = "?",
    unsafe: bool = False,
) -> str:
    if is_notebook():
        print()
        result = input(f"{message} [{default}]: " if default else f"{message}: ")
        return result if result else default
    else:
        question = questionary.text(message, default=default, qmark=qmark)
        if unsafe:
            return question.unsafe_ask()
        else:
            return question.ask()


def prompt_path(message: str) -> str:
    if is_notebook():
        return prompt_text(message)
    else:
        return questionary.path(message, only_directories=True).ask()


def prompt_password(message: str) -> str:
    if is_notebook():
        print()
        return getpass.getpass(message)
    else:
        return questionary.password(message).ask()


def prompt_confirm(message: str, default: bool = True) -> bool:
    if is_notebook():
        print()
        choices = "[Y/n]" if default else "[y/N]"
        result = input(f"{message} {choices} ").strip().lower()
        if not result:
            return default
        return result in ("y", "yes")
    else:
        return questionary.confirm(message, default=default).ask()


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


@dataclass
class Prompt:
    system: str
    user: str


def load_prompts(
    settings: Settings,
    specification: DatasetSpecification,
) -> list[Prompt]:
    path = specification.dataset
    split_str = specification.split

    if os.path.isdir(path):
        if Path(path, DATASET_STATE_JSON_FILENAME).exists():
            # Dataset saved with datasets.save_to_disk; needs special handling.
            # Path should be the subdirectory for a particular split.
            dataset = load_from_disk(path)
            assert not isinstance(dataset, DatasetDict), (
                "Loading dataset dicts is not supported"
            )
            # Parse the split instructions.
            instruction = ReadInstruction.from_spec(split_str)
            # Associate the split with its number of examples (lines).
            split_name = str(dataset.split)
            name2len = {split_name: len(dataset)}
            # Convert the instructions to absolute indices and select the first one.
            abs_instruction = instruction.to_absolute(name2len)[0]
            # Get the dataset by applying the indices.
            dataset = dataset[abs_instruction.from_ : abs_instruction.to]
        else:
            # Path is a local directory.
            dataset = load_dataset(
                path,
                split=split_str,
                # Don't require the number of examples (lines) per split to be pre-defined.
                verification_mode=VerificationMode.NO_CHECKS,
                # But also don't use cached data, as the dataset may have changed on disk.
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )
    else:
        # Probably a repository path; let load_dataset figure it out.
        dataset = load_dataset(path, split=split_str)

    prompts = list(dataset[specification.column])

    if specification.prefix:
        prompts = [f"{specification.prefix} {prompt}" for prompt in prompts]

    if specification.suffix:
        prompts = [f"{prompt} {specification.suffix}" for prompt in prompts]

    system_prompt = (
        settings.system_prompt
        if specification.system_prompt is None
        else specification.system_prompt
    )

    return [
        Prompt(
            system=system_prompt,
            user=prompt,
        )
        for prompt in prompts
    ]


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
        torch.mlu.empty_cache()  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        torch.sdaa.empty_cache()  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        torch.musa.empty_cache()  # ty:ignore[unresolved-attribute]
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
        for name, value in parameters.items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[Prompt],
) -> str:
    if Path(settings.model).exists():
        # Hide the path, which may contain private information.
        model_link = "a model"
    else:
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
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.4f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""


def generate_config_toml(settings: Settings) -> str:
    """Serializes the full Settings object to TOML."""
    return tomli_w.dumps(settings.model_dump(exclude_none=True))


def generate_requirements_txt() -> str:
    """Collects installed packages with exact versions."""
    dists = importlib.metadata.distributions()
    reqs = []
    for dist in dists:
        with contextlib.suppress(Exception):
            reqs.append(f"{dist.metadata['Name']}=={dist.version}")

    reqs = sorted(set(reqs), key=lambda x: x.lower())
    return "\n".join(reqs) + "\n"


def generate_environment_txt() -> str:
    """Collects OS, Python, and PyTorch/GPU information."""
    lines = [
        "Environment Snapshot",
        "====================",
        f"OS: {platform.platform()} ({platform.machine()})",
        f"Python: {platform.python_version()}",
        f"Heretic Version: {version('heretic-llm')}",
        "",
        "PyTorch & Accelerators",
        "----------------------",
        f"PyTorch Version: {torch.__version__}",
        get_accelerator_info(rich=False),
    ]

    return "\n".join(lines) + "\n"


def set_reproducibility(seed: int):
    """Sets the seed for all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_reproduce_readme(settings: Settings, checkpoint_filename: str) -> str:
    """Generates a README.md for the reproduce/ folder."""
    return f"""# Reproduction Guide

This directory contains the necessary information and assets to reproduce the results obtained during this Heretic run.

## Contents

- **config.toml**: The exact configuration used, including the seed `{settings.seed}`.
- **environment.txt**: Details about the OS, Python, and hardware used.
- **requirements.txt**: The exact versions of all installed Python packages.
- **{checkpoint_filename}**: The Optuna study journal containing the history of all trials.

## How to Reproduce

1. Ensure your hardware and environment match the specifications in `environment.txt`.
2. Install the exact package versions listed in `requirements.txt`.
3. Run Heretic on the same model using the provided seed (e.g., `heretic --model {settings.model} --seed {settings.seed}`).
"""


def create_reproduce_folder(
    path: Path, settings: Settings, checkpoint_path: str | Path
) -> None:
    reproduce_dir = path / "reproduce"
    reproduce_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = Path(checkpoint_path).name

    (reproduce_dir / "config.toml").write_text(
        generate_config_toml(settings), encoding="utf-8"
    )
    (reproduce_dir / "requirements.txt").write_text(
        generate_requirements_txt(), encoding="utf-8"
    )
    (reproduce_dir / "environment.txt").write_text(
        generate_environment_txt(), encoding="utf-8"
    )
    (reproduce_dir / "README.md").write_text(
        generate_reproduce_readme(settings, checkpoint_filename), encoding="utf-8"
    )

    # Copy Optuna study journal
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        (reproduce_dir / checkpoint_file.name).write_bytes(checkpoint_file.read_bytes())


def upload_reproduce_folder(
    repo_id: str,
    settings: Settings,
    token: str,
    checkpoint_path: str | Path,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        create_reproduce_folder(tmp_path, settings, checkpoint_path=checkpoint_path)

        reproduce_dir = tmp_path / "reproduce"
        for file_path in reproduce_dir.iterdir():
            if file_path.is_file():
                huggingface_hub.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"reproduce/{file_path.name}",
                    repo_id=repo_id,
                    token=token,
                )
