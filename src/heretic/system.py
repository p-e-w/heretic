# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
import platform
import subprocess
import sys
from typing import Any

import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)


def get_nvidia_driver_version() -> str:
    """Gets the NVIDIA driver version using nvidia-smi."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip().split("\n")[0]
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return "Unknown"


def get_xpu_driver_version() -> str:
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
        return "Unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unknown"


def get_npu_driver_version() -> str:
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
        return "Unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unknown"


def get_mps_driver_version() -> str:
    """Gets the Apple Silicon (MPS) driver version via macOS version."""
    try:
        output = subprocess.check_output(
            ["sw_vers", "-productVersion"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unknown"


def get_amdgpu_driver_version() -> str:
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

    return "Unknown"


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
        report = f"Detected [bold]{info['count']}[/] {info['type']} device(s) ({info['total_vram_gb']:.2f} GB total VRAM)\n"
        report += f"{api_label}: [bold]{info['api_version']}[/]\n"
        report += f"Driver Version: [bold]{info['driver_version']}[/]\n"
        for i, dev in enumerate(info["devices"]):
            report += f"* GPU {i}: [bold]{dev['name']}[/] ({dev['vram_gb']:.2f} GB)\n"
        return report.strip()

    if info["type"] == "XPU":
        report = f"Detected [bold]{info['count']}[/] XPU device(s)\n"
        report += f"Driver Version: [bold]{info['driver_version']}[/]\n"
        for i, dev in enumerate(info["devices"]):
            report += f"* XPU {i}: [bold]{dev['name']}[/]\n"
        return report.strip()

    if info["type"] in ("MLU", "SDAA", "MUSA"):
        report = f"Detected [bold]{info['count']}[/] {info['type']} device(s):\n"
        for i, dev in enumerate(info["devices"]):
            report += f"* {info['type']} {i}: [bold]{dev['name']}[/]\n"
        return report.strip()

    if info["type"] == "NPU":
        report = (
            f"Detected NPU device(s) (CANN version: [bold]{info['cann_version']}[/])\n"
        )
        report += f"Driver Version: [bold]{info['driver_version']}[/]\n"
        return report.strip()

    if info["type"] == "MPS":
        report = "Detected [bold]1[/] MPS device (Apple Metal)\n"
        report += f"Driver Version (macOS): [bold]{info['driver_version']}[/]\n"
        return report.strip()

    return ""


def get_cpu_info_dict() -> dict[str, str]:
    brand = platform.processor()
    try:
        if platform.system() == "Windows":
            brand = (
                subprocess.check_output(
                    [
                        "powershell",
                        "-Command",
                        "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name",
                    ],
                    text=True,
                )
                .strip()
                .split("\n")[0]
            )
        elif platform.system() == "Linux":
            brand = subprocess.check_output(
                "grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2",
                shell=True,
                text=True,
            ).strip()
        elif platform.system() == "Darwin":
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass

    capability = "Unknown"
    try:
        capability = torch.backends.cpu.get_cpu_capability()
    except Exception:
        pass

    return {"brand": brand, "capability": str(capability)}


def get_cpu_info() -> str:
    """Gets the CPU brand name and instruction set capability."""
    info = get_cpu_info_dict()
    return f"{info['brand']} (Capability: {info['capability']})"


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
