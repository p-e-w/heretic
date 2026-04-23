# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import getpass
import json
import os
import platform
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import huggingface_hub
import numpy as np
import questionary
import tomli_w
import torch
from datasets import DatasetDict, ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from optuna import Trial
from psutil import Process
from questionary import Choice, Style
from rich.console import Console

from .config import DatasetSpecification, Settings
from .system import (
    get_accelerator_info_dict,
    get_cpu_info_dict,
    get_heretic_version_info,
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


def is_hf_path(path: str) -> bool:
    """Checks whether a path likely refers to a Hugging Face repository."""

    return (
        not path.startswith("/")
        and not path.endswith("/")
        and path.count("/") == 1
        and "\\" not in path
        and not Path(path).exists()
    )


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

    if is_hf_path(path):
        dataset = load_dataset(
            path,
            revision=specification.commit,
            split=split_str,
        )
    else:
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
            # Path should be a local directory.
            dataset = load_dataset(
                path,
                split=split_str,
                # Don't require the number of examples (lines) per split to be pre-defined.
                verification_mode=VerificationMode.NO_CHECKS,
                # But also don't use cached data, as the dataset may have changed on disk.
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )

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
    contains_reproducibility_information: bool,
) -> str:
    if is_hf_path(settings.model):
        model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"
    else:
        # Hide the path, which may contain private information.
        model_link = "a model"

    version_info = get_heretic_version_info()

    if contains_reproducibility_information:
        reproducibility_instructions = """
> [!TIP]
> **This model is reproducible!**
>
> See the [README](reproduce/README.md) in the `reproduce` directory for more information.
"""
    else:
        reproducibility_instructions = ""

    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version_info.version}
{reproducibility_instructions}
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
| **Refusals** | {trial.user_attrs["refusals"]}/{trial.user_attrs["n_bad_prompts"]} | {
        trial.user_attrs["base_refusals"]
    }/{trial.user_attrs["n_bad_prompts"]} |

-----

"""


def generate_config_toml(settings: Settings) -> str:
    """Serializes the full Settings object to TOML."""

    return tomli_w.dumps(settings.model_dump(exclude_none=True))


def generate_requirements_txt() -> str:
    """Collects direct project dependencies as a formatted string."""

    requirements = [
        f"{package}=={version}" for package, version in get_requirements_dict().items()
    ]
    return "\n".join(requirements) + "\n"


def set_seed(seed: int):
    """Sets the seed for all RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_hf_link(
    path: str,
    commit: str | None = None,
    is_dataset: bool = False,
) -> str:
    prefix = "datasets/" if is_dataset else ""
    base_url = f"https://huggingface.co/{prefix}{path}"
    link = f"[{path}]({base_url})"

    if commit:
        commit_url = f"{base_url}/commit/{commit}"
        link += f" (Commit: [`{commit[:7]}`]({commit_url}))"

    return link


def generate_reproduce_readme(
    settings: Settings,
    checkpoint_filename: str,
    trial: Trial,
    include_system_information: bool,
) -> str:
    """Generates the contents of a README.md for the reproduce/ folder."""

    heterogeneous_warning = ""

    if include_system_information:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 1:
                device_names = {torch.cuda.get_device_name(i) for i in range(count)}
                if len(device_names) > 1:
                    heterogeneous_warning = """
> [!WARNING]
> **Heterogeneous GPUs**
>
> This model was generated using multiple non-identical GPUs. When operations are distributed across different GPUs
> (e.g. via `device_map='auto'`), non-deterministic behavior can occur.
>
> Reproducibility *cannot* be guaranteed in this environment.
"""

        cpu = get_cpu_info_dict()
        python_env = get_python_env_info_dict()

        accelerators = get_accelerator_info_dict()
        if accelerators["type"] is None:
            accelerator_report = "**No GPU or other accelerator detected.**"
        else:
            devices = accelerators["devices"]
            total_vram = sum(device.get("vram_gb", 0) for device in devices)
            vram_suffix = f" ({total_vram:.2f} GB total VRAM)" if total_vram > 0 else ""
            accelerator_lines = [
                f"- **{accelerators['type']}:** Detected {len(devices)} device(s){vram_suffix}"
            ]

            if accelerators.get("api_name") and accelerators.get("api_version"):
                accelerator_lines.append(
                    f"  - **{accelerators['api_name']}:** {accelerators['api_version']}"
                )

            if accelerators.get("driver_version"):
                accelerator_lines.append(
                    f"  - **Driver Version:** {accelerators['driver_version']}"
                )

            accelerator_lines.append("- **Devices:**")
            for i, device in enumerate(devices):
                vram = f" ({device['vram_gb']:.2f} GB)" if device.get("vram_gb") else ""
                accelerator_lines.append(
                    f"  - **{accelerators['type']} {i}:** {device['name']}{vram}"
                )
            accelerator_report = "\n".join(accelerator_lines)

        system_report = f"""## System

- **Python:** {python_env["version"]} ({python_env["implementation"]}, {python_env["compiler"]}) [{python_env["environment"]}]
- **Operating system:** {platform.platform()} ({platform.machine()})
- **CPU:** {cpu["brand"] or "Unknown"}

### Accelerators

{accelerator_report}

"""
        system_instructions = (
            "1. Ensure your system matches the specifications in the **System** section above. "
            "Exact reproducibility is only guaranteed if all aspects of your system are identical to the one the model was originally generated on.\n"
        )
    else:
        system_report = ""
        system_instructions = ""

    version_info = get_heretic_version_info()
    origin_warning = ""
    if not version_info.is_standard_pypi:
        if version_info.origin and version_info.origin.startswith("Git"):
            repo_info = version_info.origin.split("Git (")[1].rstrip(")")
            origin_warning = f"""
> [!IMPORTANT]
> **Git installation**
>
> This system installed Heretic from a Git repository: {repo_info}
>
> To reproduce the model, you must install Heretic from this exact repository and commit.
"""
        elif version_info.origin == "Local":
            origin_warning = """
> [!WARNING]
> **Local code**
>
> This system installed Heretic from a local directory or wheel. Uncommitted or experimental code may have been executed.
>
> Reproducibility *cannot* be guaranteed in this environment.
"""
        else:
            origin_warning = """
> [!WARNING]
> **Non-standard installation**
>
> This system installed Heretic from an unknown non-standard source.
>
> Reproducibility *cannot* be guaranteed in this environment.
"""

    pytorch_version = torch.__version__
    pytorch_install_command = f"pip install torch=={pytorch_version}"
    if "+" in pytorch_version:
        suffix = pytorch_version.split("+")[1]
        if suffix:
            pytorch_install_command += (
                f" --index-url https://download.pytorch.org/whl/{suffix}"
            )

    return f"""# Reproduction guide

This directory contains the necessary information and assets to reproduce the results obtained during this Heretic run.{heterogeneous_warning}{origin_warning}

## Models

- **Base model:** {format_hf_link(settings.model, settings.model_commit)}

## Datasets

- **Good prompts:** {format_hf_link(settings.good_prompts.dataset, settings.good_prompts.commit, is_dataset=True)}
- **Bad prompts:** {format_hf_link(settings.bad_prompts.dataset, settings.bad_prompts.commit, is_dataset=True)}
- **Good evaluation prompts:** {format_hf_link(settings.good_evaluation_prompts.dataset, settings.good_evaluation_prompts.commit, is_dataset=True)}
- **Bad evaluation prompts:** {format_hf_link(settings.bad_evaluation_prompts.dataset, settings.bad_evaluation_prompts.commit, is_dataset=True)}

## Selected trial

- **Trial number:** {trial.user_attrs["index"]}
- **KL divergence:** {trial.user_attrs["kl_divergence"]:.6f}
- **Refusals:** {trial.user_attrs["refusals"]}/{trial.user_attrs["n_bad_prompts"]}

{system_report}## Environment

- **Heretic:** v{version_info.version}{f" (Origin: {version_info.origin})" if version_info.origin else ""}
- **PyTorch:** {pytorch_version}
- **Other dependencies:** See [`requirements.txt`](requirements.txt).

## Contents of this directory

- [`requirements.txt`](requirements.txt): The exact versions of all Python packages.
- [`config.toml`](config.toml): The exact configuration used, including the RNG seed.
- [`{checkpoint_filename}`]({checkpoint_filename}): The Optuna study journal containing the history of all trials.
- [`SHA256SUMS`](SHA256SUMS): Cryptographic hashes for all weight files.
- [`reproduce.json`](reproduce.json): A machine-readable file containing all reproducibility information.

## How to reproduce

{system_instructions}1. Install the exact version of Heretic indicated in the **Environment** section above, from its original source.
1. Install the packages listed in `requirements.txt`: `pip install -r requirements.txt`
1. Install the correct version of PyTorch: `{pytorch_install_command}`
1. Place the provided `config.toml` in your working directory.
1. Run Heretic without any additional arguments: `heretic`
1. Wait for the run to finish, then select trial **{trial.user_attrs["index"]}** and export the model.
1. Verify that the weight files have been exactly reproduced by comparing their SHA-256 hashes against those in `SHA256SUMS`: `sha256sum -c SHA256SUMS` (or look at the hashes online if you uploaded to Hugging Face)

> [!TIP]
> To use the included Optuna study journal `{checkpoint_filename}`, place it in the checkpoints directory (usually `checkpoints/`) before running Heretic.
>
> This allows you to export other models from the Pareto front, or to run additional trials without having to re-run the stored trials.
"""


def generate_reproduce_json(
    settings: Settings,
    trial: Trial,
    timestamp: str,
    uploaded_model_hashes: dict[str, str],
    include_system_information: bool,
) -> str:
    """Generates the contents of a reproduce.json file for the reproduce/ folder."""

    version_info = get_heretic_version_info()

    data = {
        "version": "1",  # Version number of the reproduce.json file format, to allow for future changes.
        "timestamp": timestamp,
        "system": None,  # Defined here to preserve insertion order.
        "environment": {
            "heretic": {
                "version": version_info.version,
                "is_standard_pypi": version_info.is_standard_pypi,
                "metadata": version_info.metadata,
            },
            "pytorch_version": torch.__version__,
            "requirements": get_requirements_dict(),
        },
        "settings": settings.model_dump(),
        "parameters": {
            "direction_index": trial.user_attrs["direction_index"],
            "abliteration_parameters": trial.user_attrs["parameters"],
        },
        "metrics": {
            "kl_divergence": trial.user_attrs["kl_divergence"],
            "refusals": trial.user_attrs["refusals"],
            "base_refusals": trial.user_attrs["base_refusals"],
            "n_bad_prompts": trial.user_attrs["n_bad_prompts"],
        },
        "hashes": uploaded_model_hashes,
    }

    if include_system_information:
        data["system"] = {
            "python": get_python_env_info_dict(),
            "os": {
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "cpu": get_cpu_info_dict(),
            "accelerators": get_accelerator_info_dict(),
        }
    else:
        del data["system"]

    return json.dumps(data, indent=4)


def generate_sha256sums(hashes: dict[str, str]) -> str:
    """Generates GNU Coreutils compatible SHA256SUMS file content."""

    lines = []

    for filename, sha256 in sorted(hashes.items()):
        # Use '*' to indicate binary mode for model weights.
        lines.append(f"{sha256} *{filename}")

    return "\n".join(lines) + "\n"


def create_reproduce_folder(
    path: Path,
    settings: Settings,
    checkpoint_path: str | Path,
    trial: Trial,
    uploaded_model_hashes: dict[str, str],
    include_system_information: bool,
):
    reproduce_dir = path / "reproduce"
    reproduce_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = Path(checkpoint_path).name

    # Fetch commit hash for the base model.
    settings.model_commit = huggingface_hub.model_info(settings.model).sha

    # Fetch commit hashes for all HF datasets to ensure reproducibility.
    for spec in [
        settings.good_prompts,
        settings.bad_prompts,
        settings.good_evaluation_prompts,
        settings.bad_evaluation_prompts,
    ]:
        spec.commit = huggingface_hub.dataset_info(spec.dataset).sha

    # Strip microseconds and timezone for a clean format.
    timestamp = (
        datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None).isoformat()
    )

    (reproduce_dir / "requirements.txt").write_text(
        generate_requirements_txt(),
        encoding="utf-8",
    )

    (reproduce_dir / "config.toml").write_text(
        generate_config_toml(settings),
        encoding="utf-8",
    )

    if uploaded_model_hashes:
        (reproduce_dir / "SHA256SUMS").write_text(
            generate_sha256sums(uploaded_model_hashes),
            encoding="utf-8",
        )

    (reproduce_dir / "reproduce.json").write_text(
        generate_reproduce_json(
            settings,
            trial,
            timestamp=timestamp,
            uploaded_model_hashes=uploaded_model_hashes,
            include_system_information=include_system_information,
        ),
        encoding="utf-8",
    )

    (reproduce_dir / "README.md").write_text(
        generate_reproduce_readme(
            settings,
            checkpoint_filename,
            trial,
            include_system_information=include_system_information,
        ),
        encoding="utf-8",
    )

    # Copy Optuna study journal.
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        (reproduce_dir / checkpoint_file.name).write_bytes(checkpoint_file.read_bytes())


def upload_reproduce_folder(
    repo_id: str,
    settings: Settings,
    token: str,
    checkpoint_path: str | Path,
    trial: Trial,
    include_system_information: bool,
):
    api = huggingface_hub.HfApi()
    info = api.model_info(repo_id=repo_id, files_metadata=True, token=token)

    if not info.siblings:
        raise RuntimeError("Could not fetch uploaded model hashes.")

    # For weights, we only care about safetensors.
    weight_extensions = (".safetensors",)

    uploaded_model_hashes = {}

    for file in info.siblings:
        if file.rfilename.endswith(weight_extensions):
            sha256 = getattr(file, "lfs", {}).get("sha256")
            if not sha256:
                raise RuntimeError("Could not fetch uploaded model hashes.")
            uploaded_model_hashes[file.rfilename] = sha256

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        create_reproduce_folder(
            tmp_path,
            settings,
            checkpoint_path=checkpoint_path,
            trial=trial,
            uploaded_model_hashes=uploaded_model_hashes,
            include_system_information=include_system_information,
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
