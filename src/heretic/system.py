# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import gc
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import cpuinfo
import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)


def empty_cache():
    """Clears the backend cache and collects garbage."""

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


def get_nvidia_driver_version() -> str | None:
    """Gets the NVIDIA driver version using nvidia-smi."""

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip().split("\n")[0]
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None


def get_amdgpu_driver_version() -> str | None:
    """Gets the AMD GPU (ROCm) driver and suite version info."""

    # 1. Try amd-smi (modern standard for ROCm 6.0+)
    try:
        output = subprocess.check_output(
            ["amd-smi", "version"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if output.strip():
            return output.strip().replace("\n", " | ")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 2. Try rocm-smi --showdriverversion
    try:
        output = subprocess.check_output(
            ["rocm-smi", "--showdriverversion"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in output.split("\n"):
            if "Driver version" in line:
                return line.split(":")[-1].strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 3. Try /sys/module/amdgpu/version (Linux kernel driver version)
    try:
        if platform.system() == "Linux":
            version_path = "/sys/module/amdgpu/version"
            if os.path.exists(version_path):
                with open(version_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
    except Exception:
        pass

    return None


def get_xpu_driver_version() -> str | None:
    """Gets the Intel XPU driver version."""

    try:
        output = subprocess.check_output(
            ["xpu-smi", "discovery"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in output.split("\n"):
            if "Driver Version" in line:
                return line.split(":")[-1].strip()
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_npu_driver_version() -> str | None:
    """Gets the Huawei NPU driver version."""

    try:
        output = subprocess.check_output(
            ["npu-smi", "info", "-t", "board", "-i", "0"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in output.split("\n"):
            if "Software Version" in line:
                return line.split()[-1].strip()
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_mps_driver_version() -> str | None:
    """Gets the Apple Silicon (MPS) driver version via macOS version."""

    try:
        output = subprocess.check_output(
            ["sw_vers", "-productVersion"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class HereticVersionInfo:
    """Detailed information about the heretic-llm installation."""

    version: str
    origin: str | None
    is_standard_pypi: bool
    metadata: dict[str, Any]


def get_heretic_version_info() -> HereticVersionInfo:
    """Detects version and installation source (PyPI, Git, Local) of heretic-llm."""

    package_name = "heretic-llm"
    origin_metadata: dict[str, Any] = {"type": "unknown"}
    # This package must be installed for this code to run.
    distribution = importlib.metadata.distribution(package_name)

    base_version = distribution.version.lstrip("v")

    try:
        direct_url_content = distribution.read_text("direct_url.json")
    except Exception:
        direct_url_content = None

    if not direct_url_content:
        # Standard PyPI installation.
        origin_metadata["type"] = "pypi"

        return HereticVersionInfo(
            version=base_version,
            origin="PyPI",
            is_standard_pypi=True,
            metadata=origin_metadata,
        )

    data = json.loads(direct_url_content)

    # Check for Git source.
    if "vcs_info" in data and data["vcs_info"].get("vcs") == "git":
        vcs_info = data["vcs_info"]
        commit_hash = vcs_info.get("commit_id", "unknown")
        repo_url = data.get("url", "unknown_repo")
        requested_revision = vcs_info.get("requested_revision")

        if requested_revision:
            origin_str = (
                f"Git ({repo_url}@{requested_revision} - commit: {commit_hash})"
            )
        else:
            origin_str = f"Git ({repo_url} @ {commit_hash})"

        origin_metadata.update(
            {
                "type": "git",
                "url": repo_url,
                "commit_hash": commit_hash,
                "requested_revision": requested_revision,
            }
        )

        return HereticVersionInfo(
            version=base_version,
            origin=origin_str,
            is_standard_pypi=False,
            metadata=origin_metadata,
        )

    # Check for local file/wheel directory.
    if "url" in data and data["url"].startswith("file://"):
        origin_metadata["type"] = "local"

        return HereticVersionInfo(
            version=base_version,
            origin="Local",
            is_standard_pypi=False,
            metadata=origin_metadata,
        )

    return HereticVersionInfo(
        version=base_version,
        origin=None,
        is_standard_pypi=False,
        metadata=origin_metadata,
    )


def get_accelerator_info_dict() -> dict[str, Any]:
    """Retrieves raw accelerator info (CUDA, ROCm, etc) directly into structured keys."""

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        is_rocm = getattr(torch.version, "hip", None) is not None

        # ROCm (AMD) and CUDA (NVIDIA) share the same API in PyTorch.
        # We distinguish them by checking for the HIP version.
        info: dict[str, Any] = {
            "type": "ROCm" if is_rocm else "CUDA",
            "api_name": "HIP Version" if is_rocm else "CUDA Version",
            "api_version": torch.version.hip if is_rocm else torch.version.cuda,  # ty:ignore[unresolved-attribute]
            "driver_version": get_amdgpu_driver_version()
            if is_rocm
            else get_nvidia_driver_version(),
            "devices": [],
        }

        for i in range(count):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.mem_get_info(i)[1] / (1024**3)
            info["devices"].append({"name": name, "vram_gb": round(vram, 2)})

        return info

    if is_xpu_available():
        count = torch.xpu.device_count()  # ty:ignore[unresolved-attribute]
        return {
            "type": "XPU",
            "api_name": None,
            "api_version": None,
            "driver_version": get_xpu_driver_version(),
            "devices": [{"name": torch.xpu.get_device_name(i)} for i in range(count)],  # ty:ignore[unresolved-attribute]
        }

    if is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
        return {
            "type": "MLU",
            "api_name": None,
            "api_version": None,
            "driver_version": None,
            "devices": [{"name": torch.mlu.get_device_name(i)} for i in range(count)],  # ty:ignore[unresolved-attribute]
        }

    if is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore[unresolved-attribute]
        return {
            "type": "SDAA",
            "api_name": None,
            "api_version": None,
            "driver_version": None,
            "devices": [{"name": torch.sdaa.get_device_name(i)} for i in range(count)],  # ty:ignore[unresolved-attribute]
        }

    if is_musa_available():
        count = torch.musa.device_count()  # ty:ignore[unresolved-attribute]
        return {
            "type": "MUSA",
            "api_name": None,
            "api_version": None,
            "driver_version": None,
            "devices": [{"name": torch.musa.get_device_name(i)} for i in range(count)],  # ty:ignore[unresolved-attribute]
        }

    if is_npu_available():
        return {
            "type": "NPU",
            "api_name": "CANN Version",
            "api_version": torch.version.cann,  # ty:ignore[unresolved-attribute]
            "driver_version": get_npu_driver_version(),
            "devices": [],  # Multi-NPU is less common.
        }

    if torch.backends.mps.is_available():
        return {
            "type": "MPS",
            "api_name": None,
            "api_version": None,
            "driver_version": get_mps_driver_version(),
            "devices": [{"name": "Apple Metal"}],
        }

    return {"type": None}


def get_accelerator_info(include_warnings: bool = True) -> str:
    """Convenience wrapper for hardware detection and console-friendly formatting."""

    info = get_accelerator_info_dict()

    if info["type"] is None:
        suffix = " Operations will be slow." if include_warnings else ""
        return (
            f"[bold yellow]No GPU or other accelerator detected.{suffix}[/]\n".strip()
        )

    devices = info["devices"]
    count = len(devices)
    total_vram = sum(d.get("vram_gb", 0) for d in devices)

    vram_suffix = f" ({total_vram:.2f} GB total VRAM)" if total_vram > 0 else ""
    report = f"Detected [bold]{count or 1}[/] {info['type']} device(s){vram_suffix}\n"

    if info.get("api_name") and info.get("api_version"):
        report += f"{info['api_name']}: [bold]{info['api_version']}[/]\n"

    driver = info.get("driver_version") or "Unknown"
    report += f"Driver Version: [bold]{driver}[/]\n"

    for i, dev in enumerate(devices):
        vram = f" ({dev['vram_gb']:.2f} GB)" if dev.get("vram_gb") else ""
        report += f"* {info['type']} {i}: [bold]{dev['name']}[/]{vram}\n"

    return report.strip()


def get_cpu_info_dict() -> dict[str, str | int | None]:
    """Gets granular CPU identifiers using the py-cpuinfo library."""

    info = cpuinfo.get_cpu_info()

    return {
        "brand": info.get("brand_raw"),
        "vendor": info.get("vendor_id_raw"),
        "family": info.get("family"),
        "model": info.get("model"),
        "stepping": info.get("stepping"),
    }


def get_cpu_info() -> str:
    """Gets the CPU brand name."""

    info = get_cpu_info_dict()
    parts = []
    parts.append(
        f"Family {info['family']}, Model {info['model']}, Stepping {info['stepping']}"
    )

    details = f" ({'; '.join(parts)})" if parts else ""
    brand = info["brand"] or "Unknown CPU"
    return f"{brand}{details}"


def get_python_env_info_dict() -> dict[str, str]:
    implementation = platform.python_implementation()
    compiler = platform.python_compiler()

    # Check for Conda.
    if "CONDA_PREFIX" in os.environ:
        env_type = "Conda"
    # Check for Virtualenv/Venv.
    elif hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
        env_type = "Virtualenv/Venv"
    else:
        env_type = "System"

    return {
        "version": platform.python_version(),
        "implementation": implementation,
        "compiler": compiler,
        "environment": env_type,
    }


def get_python_env_info() -> str:
    """Detects the type of Python environment (Conda, Venv, etc.) and build info."""

    info = get_python_env_info_dict()
    return f"{info['version']} ({info['implementation']}, {info['compiler']}) [{info['environment']}]"


def get_package_version(name: str) -> str:
    """Gets the installed version of a package, stripping local suffixes like +cu128."""

    # Normalize name: pip considers hyphens and underscores equivalent.
    normalized_name = name.lower().replace("_", "-")
    version_str = importlib.metadata.version(normalized_name)
    return version_str.split("+")[0] if "+" in version_str else version_str


def get_requirements_dict() -> dict[str, str]:
    """Recursively finds all direct and transitive dependencies of heretic-llm and core libraries."""

    # We start with heretic-llm and the core compute libraries.
    # PyTorch is not listed as a dependency in the heretic-llm package
    # because installation is hardware-specific and must be done manually.
    packages_to_check = ["heretic-llm", "torch", "torchaudio", "torchvision"]

    visited = set()
    required_packages = set()

    while packages_to_check:
        package = packages_to_check.pop(0)
        # Normalize name: pip considers hyphens and underscores equivalent.
        normalized_package = package.lower().replace("_", "-")
        if normalized_package in visited:
            continue
        visited.add(normalized_package)

        try:
            distribution = importlib.metadata.distribution(normalized_package)
            required_packages.add(normalized_package)
            if distribution.requires:
                for requirement in distribution.requires:
                    # Requirements can include environment markers like '; extra == "hf"'
                    # or version constraints. We should ignore optional 'extra' dependencies
                    # to keep the reproduction environment clean and relevant.
                    if ";" in requirement and "extra ==" in requirement:
                        continue

                    # We just want the base package name.
                    match = re.match(r"^([a-zA-Z0-9_\-]+)", requirement)
                    if match:
                        dep_name = match.group(0).lower().replace("_", "-")
                        if dep_name not in visited:
                            packages_to_check.append(dep_name)
        except importlib.metadata.PackageNotFoundError:
            # If a package is listed as a dependency but not installed, we skip it.
            continue

    required_packages_sorted = sorted(required_packages)

    # Lookup versions for all discovered packages.
    dependencies = {}
    version_info = get_heretic_version_info()

    for package in required_packages_sorted:
        # If heretic-llm was installed from source (Git/Local), exclude it
        # from requirements.txt to prevent pip from downloading an unrelated
        # version from PyPI during reproduction.
        if package == "heretic-llm" and not version_info.is_standard_pypi:
            continue

        dependencies[package] = get_package_version(package)

    return dependencies
