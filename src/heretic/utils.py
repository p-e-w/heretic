# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import getpass
import json
import os
from datetime import datetime, timezone
import platform
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import huggingface_hub
import numpy as np
import questionary
import tomli_w
import torch
from transformers import AutoConfig
from datasets import DatasetDict, ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from optuna import Trial
from psutil import Process
from questionary import Choice, Style
from rich.console import Console
from rich.text import Text

from .config import DatasetSpecification, Settings
from .system import (
    get_accelerator_info,
    get_accelerator_info_dict,
    get_cpu_info,
    get_cpu_info_dict,
    get_heretic_version_info,
    get_python_env_info,
    get_python_env_info_dict,
    get_requirements_dict,
    is_xpu_available,
)

print = Console(highlight=False).print


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

    version_info = get_heretic_version_info()
    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version_info.version}

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
    """Collects direct project dependencies as a formatted string."""
    requirements = get_requirements_dict()
    sorted_requirements = sorted(
        [f"{name}=={version}" for name, version in requirements.items()],
        key=lambda x: x.lower(),
    )
    return "\n".join(sorted_requirements) + "\n"


def generate_environment_txt() -> str:
    """Collects OS, Python, CPU, Heretic, and PyTorch/GPU information."""
    version_info = get_heretic_version_info()
    version = f"v{version_info.version}" if version_info.version else "Unknown version"
    origin = f" (Origin: {version_info.origin})" if version_info.origin else ""

    return f"""Environment Snapshot
====================
OS: {platform.platform()} ({platform.machine()})
CPU: {get_cpu_info()}
Python: {get_python_env_info()}
Heretic: {version}{origin}

PyTorch & Accelerators
----------------------
PyTorch Version: {torch.__version__}
{Text.from_markup(get_accelerator_info(include_warnings=False)).plain}
"""


def set_seed(seed: int):
    """Sets the seed for all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_reproduce_readme(
    settings: Settings,
    checkpoint_filename: str,
    trial: Trial,
    timestamp: str | None = None,
    base_model_commit: str | None = None,
    uploaded_model_hashes: dict[str, str] | None = None,
) -> str:
    """Generates a README.md for the reproduce/ folder."""
    torch_version = torch.__version__
    install_hint = f"pip install torch=={torch_version}"
    if "+" in torch_version:
        suffix = torch_version.split("+")[1]
        if suffix:
            install_hint += f" --index-url https://download.pytorch.org/whl/{suffix}"

    heterogeneous_warning = ""
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 1:
            device_names = {torch.cuda.get_device_name(i) for i in range(count)}
            if len(device_names) > 1:
                heterogeneous_warning = """
> [WARNING!]
> **Heterogeneous GPUs Detected!**
> This system uses multiple non-identical GPUs. When operations are distributed across different GPUs (e.g. via `device_map='auto'`), non-deterministic behavior can occur. **Reproducibility ***cannot*** be guaranteed in this environment.**
"""

    version_info = get_heretic_version_info()
    origin_warning = ""
    if not version_info.is_standard_pypi:
        if version_info.origin and version_info.origin.startswith("Git"):
            repo_info = version_info.origin.split("Git (")[1].strip(")")
            origin_warning = f"""
> [NOTE]
> **Git Installation Detected**
> This system installed `heretic-llm` from source repository: `{repo_info}`.
> To reproduce these results, you must install Heretic from this exact repository and commit.
"""
        elif version_info.origin == "Local":
            origin_warning = """
> [WARNING!]
> **Local Code Detected!**
> This system installed `heretic-llm` from a local directory or wheel. Uncommitted or experimental code may have been executed. **Reproducibility ***cannot*** be guaranteed in this environment.**
"""
        else:
            origin_warning = """
> [WARNING!]
> **Non-Standard Installation Detected!**
> This system installed `heretic-llm` from an unknown non-standard source. **Reproducibility ***cannot*** be guaranteed in this environment.**
"""

    timestamp_str = f"- **Timestamp (UTC):** `{timestamp}`\n" if timestamp else ""
    commit_str = (
        f"- **Base Model Commit:** `{base_model_commit}`\n" if base_model_commit else ""
    )

    hashes_str = ""
    if uploaded_model_hashes:
        hashes_str = (
            "## Uploaded Model Checksums\n\n| Filename | SHA256 |\n| :--- | :--- |\n"
        )
        for filename, sha256 in uploaded_model_hashes.items():
            hashes_str += f"| `{filename}` | `{sha256}` |\n"
        hashes_str += "\n"

    return f"""# Reproduction Guide

This directory contains the necessary information and assets to reproduce the results obtained during this Heretic run.{heterogeneous_warning}{origin_warning}

## Selected Trial

- **Base Model:** `{settings.model}`
{commit_str}{timestamp_str}- **Trial number:** `#{trial.user_attrs["index"]}`

{hashes_str}## Contents

- **config.toml**: The exact configuration used, including the seed `{settings.seed}`.
- **environment.txt**: Details about the OS, Python, Heretic, PyTorch, Driver and hardware used.
- **requirements.txt**: The exact versions of all installed Python packages.
- **{checkpoint_filename}**: The Optuna study journal containing the history of all trials.

## How to Reproduce

1. Ensure your hardware and environment match the specifications in `environment.txt`.
2. Install the exact package versions listed in `requirements.txt`.
3. Place the provided `config.toml` in your working directory and run `heretic` without any additional arguments.

> Make sure to install correct PyTorch version from `environment.txt`. 
> e.g., `{install_hint}`
"""


def generate_reproduce_json(
    settings: Settings,
    trial: Trial,
    timestamp: str | None = None,
    base_model_commit: str | None = None,
    uploaded_model_hashes: dict[str, str] | None = None,
) -> str:
    """Generates a reproduce.json file for the reproduce/ folder."""
    version_info = get_heretic_version_info()
    data = {
        "base_model": {
            "id": settings.model,
            "commit_hash": base_model_commit,
        },
        "system": {
            "os": {"platform": platform.platform(), "machine": platform.machine()},
            "cpu": get_cpu_info_dict(),
            "python": get_python_env_info_dict(),
            "heretic": {
                "version": version_info.version,
                "is_standard_pypi": version_info.is_standard_pypi,
                "metadata": version_info.metadata,
            },
            "pytorch_version": torch.__version__,
            "accelerator": get_accelerator_info_dict(),
        },
        "requirements": get_requirements_dict(),
        "settings": settings.model_dump(exclude_none=True),
        "trial": {
            "index": trial.user_attrs.get("index"),
            "parameters": trial.user_attrs.get("parameters"),
        },
        "timestamp_utc": timestamp,
        "uploaded_model_hashes": uploaded_model_hashes or {},
    }
    return json.dumps(data, indent=4)


def create_reproduce_folder(
    path: Path,
    settings: Settings,
    checkpoint_path: str | Path,
    trial: Trial,
    uploaded_model_hashes: dict[str, str] | None = None,
) -> None:
    reproduce_dir = path / "reproduce"
    reproduce_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = Path(checkpoint_path).name

    timestamp = datetime.now(timezone.utc).isoformat()
    base_model_commit = None
    if not Path(settings.model).exists():
        try:
            config = AutoConfig.from_pretrained(
                settings.model, trust_remote_code=settings.trust_remote_code
            )
            base_model_commit = getattr(config, "_commit_hash", None)
        except Exception:
            pass

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
        generate_reproduce_readme(
            settings,
            checkpoint_filename,
            trial,
            timestamp=timestamp,
            base_model_commit=base_model_commit,
            uploaded_model_hashes=uploaded_model_hashes,
        ),
        encoding="utf-8",
    )
    (reproduce_dir / "reproduce.json").write_text(
        generate_reproduce_json(
            settings,
            trial,
            timestamp=timestamp,
            base_model_commit=base_model_commit,
            uploaded_model_hashes=uploaded_model_hashes,
        ),
        encoding="utf-8",
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
    trial: Trial,
) -> None:
    uploaded_model_hashes = {}
    try:
        api = huggingface_hub.HfApi()
        info = api.model_info(repo_id=repo_id, files_metadata=True, token=token)
        weight_extensions = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
        for file in info.siblings:
            if file.rfilename.endswith(weight_extensions):
                sha256 = getattr(file, "lfs", {}).get("sha256")
                if sha256:
                    uploaded_model_hashes[file.rfilename] = sha256
    except Exception as e:
        print(f"[yellow]Warning: Could not fetch uploaded model hashes: {e}[/]")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        create_reproduce_folder(
            tmp_path,
            settings,
            checkpoint_path=checkpoint_path,
            trial=trial,
            uploaded_model_hashes=uploaded_model_hashes,
        )

        reproduce_dir = tmp_path / "reproduce"
        for file_path in reproduce_dir.iterdir():
            if file_path.is_file():
                huggingface_hub.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"reproduce/{file_path.name}",
                    repo_id=repo_id,
                    token=token,
                )
