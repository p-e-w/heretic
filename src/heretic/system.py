# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
import platform
import subprocess
import sys

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


def get_accelerator_info(include_warnings: bool = True) -> str:
    """The single source of truth for hardware detection and reporting."""

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        total_vram = sum(torch.cuda.mem_get_info(i)[1] for i in range(count))

        # ROCm (AMD) and CUDA (NVIDIA) share the same API in PyTorch.
        # We distinguish them by checking for the HIP version.
        is_rocm = getattr(torch.version, "hip", None) is not None

        if is_rocm:
            label = "ROCm"
            api_version_label = "HIP Version"
            api_version = torch.version.hip  # ty:ignore[unresolved-attribute]
            driver_version = get_amdgpu_driver_version()
        else:
            label = "CUDA"
            api_version_label = "CUDA Version"
            api_version = torch.version.cuda
            driver_version = get_nvidia_driver_version()

        report = f"Detected [bold]{count}[/] {label} device(s) ({total_vram / (1024**3):.2f} GB total VRAM)\n"
        report += f"{api_version_label}: [bold]{api_version}[/]\n"
        report += f"Driver Version: [bold]{driver_version}[/]\n"

        for i in range(count):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.mem_get_info(i)[1] / (1024**3)
            report += f"* GPU {i}: [bold]{name}[/] ({vram:.2f} GB)\n"
    elif is_xpu_available():
        count = torch.xpu.device_count()  # ty:ignore[unresolved-attribute]
        driver_version = get_xpu_driver_version()

        report = f"Detected [bold]{count}[/] XPU device(s)\n"
        report += f"Driver Version: [bold]{driver_version}[/]\n"
        for i in range(count):
            report += f"* XPU {i}: [bold]{torch.xpu.get_device_name(i)}[/]\n"  # ty:ignore[unresolved-attribute]
    elif is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
        report = f"Detected [bold]{count}[/] MLU device(s):\n"
        for i in range(count):
            report += f"* MLU {i}: [bold]{torch.mlu.get_device_name(i)}[/]\n"  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore[unresolved-attribute]
        report = f"Detected [bold]{count}[/] SDAA device(s):\n"
        for i in range(count):
            report += f"* SDAA {i}: [bold]{torch.sdaa.get_device_name(i)}[/]\n"  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        count = torch.musa.device_count()  # ty:ignore[unresolved-attribute]
        report = f"Detected [bold]{count}[/] MUSA device(s):\n"
        for i in range(count):
            report += f"* MUSA {i}: [bold]{torch.musa.get_device_name(i)}[/]\n"  # ty:ignore[unresolved-attribute]
    elif is_npu_available():
        driver_version = get_npu_driver_version()
        report = (
            f"Detected NPU device(s) (CANN version: [bold]{torch.version.cann}[/])\n"  # ty:ignore[unresolved-attribute]
        )
        report += f"Driver Version: [bold]{driver_version}[/]\n"
    elif torch.backends.mps.is_available():
        driver_version = get_mps_driver_version()
        report = "Detected [bold]1[/] MPS device (Apple Metal)\n"
        report += f"Driver Version (macOS): [bold]{driver_version}[/]\n"
    else:
        suffix = " Operations will be slow." if include_warnings else ""
        report = f"[bold yellow]No GPU or other accelerator detected.{suffix}[/]\n"

    return report.strip()


def get_cpu_info() -> str:
    """Gets the CPU brand name and instruction set capability."""
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

    return f"{brand} (Capability: {capability})"


def get_python_env_info() -> str:
    """Detects the type of Python environment (Conda, Venv, etc.) and build info."""
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

    return f"{platform.python_version()} ({implementation}, {compiler}) [{env_type}]"
