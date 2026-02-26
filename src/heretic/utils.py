# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import contextlib
import gc
import getpass
import importlib.metadata
import json
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
import safetensors.torch
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


def get_accelerator_info() -> dict[str, Any]:
    """Consolidates hardware detection logic into a structured dictionary."""
    info: dict[str, Any] = {
        "type": None,
        "count": 0,
        "devices": [],
        "cuda_version": None,
        "cudnn_version": None,
        "cuda_capability": None,
    }

    if torch.cuda.is_available():
        info["type"] = "cuda"
        info["count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
        with contextlib.suppress(Exception):
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["cuda_capability"] = torch.cuda.get_device_capability()

        for i in range(info["count"]):
            name = torch.cuda.get_device_name(i)
            with contextlib.suppress(Exception):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append(
                    {"name": name, "total_memory": props.total_memory}
                )
    elif is_xpu_available():
        info["type"] = "xpu"
        info["count"] = torch.xpu.device_count()
        for i in range(info["count"]):
            info["devices"].append({"name": torch.xpu.get_device_name(i)})
    elif is_mlu_available():
        info["type"] = "mlu"
        info["count"] = torch.mlu.device_count()  # ty:ignore
        for i in range(info["count"]):
            info["devices"].append({"name": torch.mlu.get_device_name(i)})  # ty:ignore
    elif is_sdaa_available():
        info["type"] = "sdaa"
        info["count"] = torch.sdaa.device_count()  # ty:ignore
        for i in range(info["count"]):
            info["devices"].append({"name": torch.sdaa.get_device_name(i)})  # ty:ignore
    elif is_musa_available():
        info["type"] = "musa"
        info["count"] = torch.musa.device_count()  # ty:ignore
        for i in range(info["count"]):
            info["devices"].append({"name": torch.musa.get_device_name(i)})  # ty:ignore
    elif is_npu_available():
        info["type"] = "npu"
        info["cann_version"] = torch.version.cann  # ty:ignore
    elif torch.backends.mps.is_available():
        info["type"] = "mps"
        info["count"] = 1
    else:
        info["type"] = "cpu"

    return info


def print_memory_usage():
    def p(label: str, size_in_bytes: int):
        print(f"[grey50]{label}: [bold]{size_in_bytes / (1024**3):.2f} GB[/][/]")

    p("Resident system RAM", Process().memory_info().rss)

    acc = get_accelerator_info()
    if acc["type"] == "cuda":
        allocated = sum(
            torch.cuda.memory_allocated(device) for device in range(acc["count"])
        )
        reserved = sum(
            torch.cuda.memory_reserved(device) for device in range(acc["count"])
        )
        p("Allocated GPU VRAM", allocated)
        p("Reserved GPU VRAM", reserved)
    elif acc["type"] == "xpu":
        allocated = sum(
            torch.xpu.memory_allocated(device) for device in range(acc["count"])
        )
        reserved = sum(
            torch.xpu.memory_reserved(device) for device in range(acc["count"])
        )
        p("Allocated XPU memory", allocated)
        p("Reserved XPU memory", reserved)
    elif acc["type"] == "mps":
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
    ]

    acc = get_accelerator_info()
    if acc["type"] == "cuda":
        lines.append(f"CUDA Version: {acc['cuda_version']}")
        if acc["cudnn_version"]:
            lines.append(f"CuDNN Version: {acc['cudnn_version']}")
        if acc["cuda_capability"]:
            lines.append(f"CUDA Driver Cap: {acc['cuda_capability']}")

        lines.append(f"CUDA Devices ({acc['count']}):")
        for i, device in enumerate(acc["devices"]):
            lines.append(
                f"  * GPU {i}: {device['name']} ({device['total_memory'] / (1024**3):.2f} GB)"
            )
    elif acc["type"] in ["xpu", "mlu", "sdaa", "musa"]:
        lines.append(f"{acc['type'].upper()} Devices ({acc['count']}):")
        for i, device in enumerate(acc["devices"]):
            lines.append(f"  * {acc['type'].upper()} {i}: {device['name']}")
    elif acc["type"] == "npu":
        lines.append(f"NPU detected (CANN version: {acc['cann_version']})")
    elif acc["type"] == "mps":
        lines.append("MPS Device (Apple Metal)")
    else:
        lines.append("No GPU or accelerator detected.")

    return "\n".join(lines) + "\n"


def set_reproducibility(seed: int, deterministic: bool):
    """Sets the seed for all RNGs and optionally enables strict deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Make PyTorch deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: This may raise an error for some non-deterministic operations
        torch.use_deterministic_algorithms(True, warn_only=True)


def generate_reproduce_readme(
    settings: Settings, checkpoint_filename: str | None = None
) -> str:
    """Generates a README.md for the reproduce/ folder."""
    checkpoint_desc = (
        f"- **{checkpoint_filename}**: The Optuna study journal containing the history of all trials."
        if checkpoint_filename
        else ""
    )
    return f"""# Reproduction Guide

This directory contains the necessary information and assets to reproduce the results obtained during this Heretic run.

## Contents

- **config.toml**: The exact configuration used, including the seed `{settings.seed}`.
- **environment.txt**: Details about the OS, Python, and hardware used.
- **requirements.txt**: The exact versions of all installed Python packages.
- **random_states.safetensors**: A secure snapshot of the RNG states (Python, NumPy, PyTorch CPU/CUDA).
{checkpoint_desc}

## How to Reproduce

1. Ensure you have the same versions of the models and datasets as specified in `config.toml`.
2. Install the exact package versions listed in `requirements.txt`.
3. Run Heretic using the provided `config.toml`. 
4. For bit-perfect reproduction, ensure you are using compatible hardware as described in `environment.txt`.

While the random states provide a deep snapshot for verification, specifying the `seed` in your configuration is usually sufficient to repeat the optimization path.
"""


def get_random_states() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Captures RNG states natively into Safetensors formats."""
    # PyTorch Tensors.
    tensors = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
        for i, state in enumerate(cuda_states):
            tensors[f"torch_cuda_{i}"] = state

    # Python / NumPy (saved as Metadata JSON).
    metadata = {"python_random": json.dumps(random.getstate())}

    # np.random.get_state() returns a tuple: (str, ndarray, int, int, float)
    # We need to convert the ndarray to a list so json.dumps can serialize it.
    np_state = np.random.get_state()
    serializable_np_state = tuple(
        x.tolist() if isinstance(x, np.ndarray) else x  # ty:ignore[no-matching-overload]
        for x in np_state
    )
    metadata["numpy_random"] = json.dumps(serializable_np_state)

    return tensors, metadata


def create_reproduce_folder(
    path: Path, settings: Settings, checkpoint_path: str | Path | None = None
) -> None:
    reproduce_dir = path / "reproduce"
    reproduce_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = Path(checkpoint_path).name if checkpoint_path else None

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

    tensors, metadata = get_random_states()
    safetensors.torch.save_file(
        tensors, reproduce_dir / "random_states.safetensors", metadata=metadata
    )

    # Copy Optuna study journal if provided
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            (reproduce_dir / checkpoint_file.name).write_bytes(
                checkpoint_file.read_bytes()
            )


def upload_reproduce_folder(
    repo_id: str,
    settings: Settings,
    token: str,
    checkpoint_path: str | Path | None = None,
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
