# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import platform
import random
import shutil
from dataclasses import asdict
from enum import IntEnum
from pathlib import Path
from typing import Any, cast
from urllib.request import urlopen

import cpuinfo
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import (
    GatedRepoError,
    disable_progress_bars,
    enable_progress_bars,
)
from questionary import Choice
from rich.table import Table

from .system import (
    get_accelerator_info_dict,
    get_heretic_version_info,
    get_requirements_dict,
)
from .utils import print, prompt_select


def collect_reproducibles(path: str):
    print(
        f"Collecting [bold]reproduce.json[/] files from Hugging Face and storing them in [bold]{path}[/]..."
    )
    print()

    api = HfApi()

    models = api.list_models(
        filter=["heretic", "reproducible"],
        sort="created_at",
        expand=["gated", "tags"],
    )

    found = 0
    downloaded = 0

    # We're only downloading tiny files, so the progress bars are just noise.
    disable_progress_bars()

    try:
        for model in models:
            # Ignore repositories containing quantizations.
            if model.tags is not None and "gguf" in model.tags:
                continue

            if model.gated:
                try:
                    api.auth_check(model.id, repo_type="model")
                except GatedRepoError:
                    continue

            print(f"[bold]{model.id}[/]...", end="")

            user, repository = model.id.split("/")

            paths_info = api.get_paths_info(
                model.id,
                "reproduce/reproduce.json",
                expand=True,
            )
            # The reproduce.json file might not exist in the repository
            # despite the relevant tags being present.
            if not paths_info:
                print(" [yellow]no reproduce.json found[/]")
                continue

            found += 1

            commit_hash = paths_info[0].last_commit.oid

            file_path = (
                Path(path)
                / "huggingface.co"
                / user
                / f"{repository}-{commit_hash[:7]}.json"
            )
            if file_path.exists():
                print(" already stored")
                continue

            cache_path = hf_hub_download(
                model.id,
                "reproduce/reproduce.json",
            )

            file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache_path, file_path)
            print(" [green]downloaded[/]")

            downloaded += 1
    finally:
        enable_progress_bars()

    print()
    print(f"Found: [bold]{found}[/] files")
    print(f"Downloaded: [bold]{downloaded}[/] files")
    print(f"Already stored: [bold]{found - downloaded}[/] files")


def load_reproduction_information(path: str) -> dict[str, Any]:
    if path.lower().startswith(("http://", "https://")):
        # The path is a URL on the web.

        # Obtain raw download URL.
        path = path.replace("/blob/", "/raw/")  # Hugging Face, GitHub
        path = path.replace("/src/branch/", "/raw/branch/")  # Codeberg

        json_str = urlopen(path).read().decode("utf-8")
    else:
        # The path is (assumed to be) a local file system path.
        json_str = Path(path).read_text(encoding="utf-8")

    return json.loads(json_str)


class MismatchSeverity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __rich__(self) -> str:
        match self:
            case MismatchSeverity.LOW:
                return "[green]low[/]"
            case MismatchSeverity.MEDIUM:
                return "[yellow]medium[/]"
            case MismatchSeverity.HIGH:
                return "[red]high[/]"
            case MismatchSeverity.CRITICAL:
                return "[bold red]critical[/]"
            case _:
                raise ValueError(f"unknown MismatchSeverity value: {self}")


def get_package_mismatch_severity(package_name: str) -> MismatchSeverity:
    if package_name in [
        "heretic-llm",
    ]:
        return MismatchSeverity.CRITICAL
    elif package_name in [
        "torch",
        "transformers",
    ]:
        return MismatchSeverity.HIGH
    elif package_name in [
        "accelerate",
        "bitsandbytes",
        "kernels",
        "optuna",
        "peft",
        "tokenizers",
        "triton",
    ]:
        return MismatchSeverity.MEDIUM
    else:
        return MismatchSeverity.LOW


def format_version_information(version_information: dict[str, Any]) -> str:
    version = version_information["version"]
    metadata = version_information["metadata"]

    if "type" in metadata:
        match metadata["type"]:
            case "pypi":
                return version
            case "git":
                return f"{version}-git+{metadata['url']}@{metadata['commit_hash']}"
            case "local":
                # Append a random number to ensure that two local installations
                # are always considered to be different versions.
                return f"{version}-local-{random.randint(2**16, 2**17)}"
            case _:
                raise ValueError(
                    f"unknown metadata.type value in version information: {metadata['type']}"
                )
    else:
        return f"{version}-unknown-{random.randint(2**16, 2**17)}"


def check_environment(reproduction_information: dict[str, Any]) -> bool:
    mismatch_severity: MismatchSeverity | None = None

    system_mismatches = []
    package_mismatches = []

    def verify(
        mismatch_list: list[tuple[str, Any, Any, MismatchSeverity]],
        name: str,
        this: Any,
        original: Any,
        severity: MismatchSeverity,
    ):
        nonlocal mismatch_severity
        if this != original:
            mismatch_list.append((name, this, original, severity))
            if mismatch_severity is None:
                mismatch_severity = severity
            else:
                mismatch_severity = max(severity, mismatch_severity)

    if "system" in reproduction_information:
        system = reproduction_information["system"]

        verify(
            system_mismatches,
            "Python version",
            platform.python_version(),
            system["python"]["version"],
            MismatchSeverity.LOW,
        )

        verify(
            system_mismatches,
            "Operating system",
            platform.platform(),
            system["os"]["platform"],
            MismatchSeverity.LOW,
        )

        verify(
            system_mismatches,
            "CPU",
            cpuinfo.get_cpu_info().get("brand_raw"),
            system["cpu"]["brand"],
            MismatchSeverity.LOW,
        )

        accelerators = get_accelerator_info_dict()

        verify(
            system_mismatches,
            "Accelerator type",
            accelerators["type"],
            system["accelerators"]["type"],
            MismatchSeverity.HIGH,
        )

        if (
            accelerators["type"]
            and accelerators["type"] == system["accelerators"]["type"]
        ):
            verify(
                system_mismatches,
                accelerators["api_name"],
                accelerators["api_version"],
                system["accelerators"]["api_version"],
                MismatchSeverity.MEDIUM,
            )
            verify(
                system_mismatches,
                "Driver version",
                accelerators["driver_version"],
                system["accelerators"]["driver_version"],
                MismatchSeverity.MEDIUM,
            )
            verify(
                system_mismatches,
                "Devices",
                "\n".join([device["name"] for device in accelerators["devices"]]),
                "\n".join(
                    [device["name"] for device in system["accelerators"]["devices"]]
                ),
                MismatchSeverity.MEDIUM,
            )

    else:
        print(
            (
                "[yellow]The provided JSON file does not contain system information. "
                "Some system parameters can affect reproducibility, but due to the lack of system information, "
                "Heretic is unable to verify that those parameters match the original environment. "
                "Reproduction may or may not produce a byte-for-byte identical model.[/]"
            )
        )

    requirements = get_requirements_dict()
    requirements["heretic-llm"] = format_version_information(
        asdict(get_heretic_version_info())
    )
    requirements["torch"] = torch.__version__

    original_requirements = reproduction_information["environment"]["requirements"]
    original_requirements["heretic-llm"] = format_version_information(
        reproduction_information["environment"]["heretic"]
    )
    original_requirements["torch"] = reproduction_information["environment"][
        "pytorch_version"
    ]

    package_names = sorted(requirements.keys() | original_requirements.keys())

    for package_name in package_names:
        verify(
            package_mismatches,
            package_name,
            requirements.get(package_name),
            original_requirements.get(package_name),
            get_package_mismatch_severity(package_name),
        )

    if system_mismatches or package_mismatches:
        print()
        print(
            (
                "[yellow]Your local environment doesn't perfectly match the environment "
                "used to produce the original model. The following components differ:[/]"
            )
        )

    if system_mismatches:
        table = Table()
        table.add_column("Component")
        table.add_column("This system", overflow="fold")
        table.add_column("Original system", overflow="fold")
        table.add_column("Severity", width=8)

        for component, this, original, severity in system_mismatches:
            table.add_row(f"[bold]{component}[/]", this, original, severity)

        print()
        print("[bold]System Mismatches[/]")
        print(table)

    if package_mismatches:
        table = Table()
        table.add_column("Package")
        table.add_column("This system", overflow="fold")
        table.add_column("Original system", overflow="fold")
        table.add_column("Severity", width=8)

        for package, this, original, severity in package_mismatches:
            table.add_row(f"[bold]{package}[/]", this, original, severity)

        print()
        print("[bold]Package Mismatches[/]")
        print(table)

    if system_mismatches or package_mismatches:
        print()
        print(
            (
                f"There is a {cast(MismatchSeverity, mismatch_severity).__rich__()} chance "
                "that reproduction won't produce a byte-for-byte identical model. "
                "However, the resulting model will very likely still behave similarly "
                "to the original model."
            )
        )

        print()
        choice = prompt_select(
            "How would you like to proceed?",
            [
                Choice(
                    title="Attempt to reproduce the model anyway",
                    value=True,
                ),
                Choice(
                    title="Exit program",
                    value=False,
                ),
            ],
        )

        return choice
    else:
        # There are no mismatches at all, so there is nothing to confirm.
        return True
