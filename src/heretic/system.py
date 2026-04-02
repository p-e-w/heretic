# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import cpuinfo
import psutil
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

    version: str | None
    origin: str | None
    is_standard_pypi: bool
    metadata: dict[str, Any]


def get_heretic_version_info() -> HereticVersionInfo:
    """Detects version and installation source (PyPI, Git, Local) of heretic-llm."""
    package_name = "heretic-llm"
    origin_metadata: dict[str, Any] = {"type": "unknown"}
    try:
        distribution = importlib.metadata.distribution(package_name)
    except importlib.metadata.PackageNotFoundError:
        return HereticVersionInfo(
            version=None,
            origin=None,
            is_standard_pypi=False,
            metadata=origin_metadata,
        )

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

    try:
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

    except json.JSONDecodeError:
        pass

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
        total_vram = sum(torch.cuda.mem_get_info(i)[1] for i in range(count))
        is_rocm = getattr(torch.version, "hip", None) is not None

        # ROCm (AMD) and CUDA (NVIDIA) share the same API in PyTorch.
        # We distinguish them by checking for the HIP version.
        info: dict[str, Any] = {
            "type": "ROCm" if is_rocm else "CUDA",
            "count": count,
            "total_vram_gb": round(total_vram / (1024**3), 2),
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
            "count": count,
            "driver_version": get_xpu_driver_version(),
            "devices": [{"name": torch.xpu.get_device_name(i)} for i in range(count)],  # ty:ignore[unresolved-attribute]
        }

    if is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
        return {
            "type": "MLU",
            "count": count,
            "devices": [{"name": torch.mlu.get_device_name(i)} for i in range(count)],  # ty:ignore[unresolved-attribute]
        }

    if is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore[unresolved-attribute]
        return {
            "type": "SDAA",
            "count": count,
            "devices": [{"name": torch.sdaa.get_device_name(i)} for i in range(count)],  # ty:ignore[unresolved-attribute]
        }

    if is_musa_available():
        count = torch.musa.device_count()  # ty:ignore[unresolved-attribute]
        return {
            "type": "MUSA",
            "count": count,
            "devices": [{"name": torch.musa.get_device_name(i)} for i in range(count)],  # ty:ignore[unresolved-attribute]
        }

    if is_npu_available():
        return {
            "type": "NPU",
            "count": 1,
            "cann_version": torch.version.cann,  # ty:ignore[unresolved-attribute]
            "driver_version": get_npu_driver_version(),
        }

    if torch.backends.mps.is_available():
        return {
            "type": "MPS",
            "count": 1,
            "driver_version": get_mps_driver_version(),
        }

    return {"type": "None"}


def get_accelerator_info(include_warnings: bool = True) -> str:
    """The single source of truth for hardware detection and formatting."""
    info = get_accelerator_info_dict()

    if info["type"] == "None":
        suffix = " Operations will be slow." if include_warnings else ""
        return (
            f"[bold yellow]No GPU or other accelerator detected.{suffix}[/]\n".strip()
        )

    if info["type"] in ("CUDA", "ROCm"):
        api_label = "HIP Version" if info["type"] == "ROCm" else "CUDA Version"
        driver = info.get("driver_version") or "Unknown"
        report = f"Detected [bold]{info['count']}[/] {info['type']} device(s) ({info['total_vram_gb']:.2f} GB total VRAM)\n"
        report += f"{api_label}: [bold]{info['api_version']}[/]\n"
        report += f"Driver Version: [bold]{driver}[/]\n"
        for i, dev in enumerate(info["devices"]):
            report += f"* GPU {i}: [bold]{dev['name']}[/] ({dev['vram_gb']:.2f} GB)\n"
        return report.strip()

    if info["type"] == "XPU":
        driver = info.get("driver_version") or "Unknown"
        report = f"Detected [bold]{info['count']}[/] XPU device(s)\n"
        report += f"Driver Version: [bold]{driver}[/]\n"
        for i, dev in enumerate(info["devices"]):
            report += f"* XPU {i}: [bold]{dev['name']}[/]\n"
        return report.strip()

    if info["type"] in ("MLU", "SDAA", "MUSA"):
        report = f"Detected [bold]{info['count']}[/] {info['type']} device(s):\n"
        for i, dev in enumerate(info["devices"]):
            report += f"* {info['type']} {i}: [bold]{dev['name']}[/]\n"
        return report.strip()

    if info["type"] == "NPU":
        driver = info.get("driver_version") or "Unknown"
        report = (
            f"Detected NPU device(s) (CANN version: [bold]{info['cann_version']}[/])\n"
        )
        report += f"Driver Version: [bold]{driver}[/]\n"
        return report.strip()

    if info["type"] == "MPS":
        driver = info.get("driver_version") or "Unknown"
        report = "Detected [bold]1[/] MPS device (Apple Metal)\n"
        report += f"Driver Version (macOS): [bold]{driver}[/]\n"
        return report.strip()

    return ""


def get_cpu_info_dict() -> dict[str, Any]:
    """Gets granular CPU identifiers using the py-cpuinfo library."""
    info = cpuinfo.get_cpu_info()

    capability = None
    try:
        capability = str(torch.backends.cpu.get_cpu_capability())
    except Exception:
        pass

    return {
        "brand": info.get("brand_raw"),
        "vendor": info.get("vendor_id_raw"),
        "family": info.get("family"),
        "model": info.get("model"),
        "stepping": info.get("stepping"),
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "speed": info.get("hz_advertised_friendly"),
        "capability": capability,
    }


def get_cpu_info() -> str:
    """Gets the CPU brand name and instruction set capability."""
    info = get_cpu_info_dict()
    parts = []
    if info["family"]:
        parts.append(
            f"Family {info['family']}, Model {info['model']}, Stepping {info['stepping']}"
        )
    if info["cores"] and info["threads"]:
        parts.append(f"{info['cores']} Cores, {info['threads']} Threads")
    if info["speed"]:
        parts.append(info["speed"])

    details = f" ({'; '.join(parts)})" if parts else ""
    brand = info["brand"] or "Unknown CPU"
    capability = f" [Capability: {info['capability']}]" if info["capability"] else ""
    return f"{brand}{details}{capability}"


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
