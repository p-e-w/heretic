# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import os
import platform
import re
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
    """Gets granular CPU identifiers (brand, family, model, stepping, manufacturer, cores, threads, speed)."""
    brand = platform.processor()
    family = "Unknown"
    model = "Unknown"
    stepping = "Unknown"
    vendor = "Unknown"
    cores = "Unknown"
    threads = "Unknown"
    speed_mhz = "Unknown"

    try:
        if platform.system() == "Windows":
            # Get granular details from WMI.
            output = subprocess.check_output(
                [
                    "powershell",
                    "-Command",
                    "Get-CimInstance Win32_Processor | Select-Object Name, Caption, Manufacturer, MaxClockSpeed, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json",
                ],
                text=True,
            ).strip()
            if output:
                data = json.loads(output)
                # If there are multiple processors, take the first one.
                if isinstance(data, list):
                    data = data[0]

                brand = data.get("Name", brand).strip()
                vendor = data.get("Manufacturer", "Unknown")
                cores = str(data.get("NumberOfCores", "Unknown"))
                threads = str(data.get("NumberOfLogicalProcessors", "Unknown"))
                speed_mhz = str(data.get("MaxClockSpeed", "Unknown"))
                # Caption usually looks like: "Intel64 Family 6 Model 154 Stepping 4"
                caption = data.get("Caption", "")
                if caption:
                    match = re.search(
                        r"Family\s+(\d+)\s+Model\s+(\d+)\s+Stepping\s+(\d+)", caption
                    )
                    if match:
                        family, model, stepping = match.groups()

        elif platform.system() == "Linux":
            # Direct parse of /proc/cpuinfo.
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                content = f.read()
                logical_count = 0
                for line in content.split("\n"):
                    if ":" not in line:
                        continue
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()

                    if key == "processor":
                        logical_count += 1
                    elif key == "model name":
                        brand = val
                    elif key == "vendor_id":
                        vendor = val
                    elif key == "cpu family":
                        family = val
                    elif key == "model":
                        model = val
                    elif key == "stepping":
                        stepping = val
                    elif key == "cpu cores":
                        cores = val
                    elif key == "cpu MHz":
                        # This is current speed, not always max, but standard for /proc/cpuinfo.
                        speed_mhz = val

                threads = str(logical_count) if logical_count > 0 else "Unknown"

        elif platform.system() == "Darwin":
            brand = (
                subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                .decode()
                .strip()
            )
            vendor = "Apple"
            cores = (
                subprocess.check_output(["sysctl", "-n", "machdep.cpu.core_count"])
                .decode()
                .strip()
            )
            threads = (
                subprocess.check_output(["sysctl", "-n", "machdep.cpu.thread_count"])
                .decode()
                .strip()
            )
    except Exception:
        pass

    capability = "Unknown"
    try:
        capability = str(torch.backends.cpu.get_cpu_capability())
    except Exception:
        pass

    return {
        "brand": brand,
        "vendor": vendor,
        "family": family,
        "model": model,
        "stepping": stepping,
        "cores": cores,
        "threads": threads,
        "speed_mhz": speed_mhz,
        "capability": capability,
    }


def get_cpu_info() -> str:
    """Gets the CPU brand name and instruction set capability."""
    info = get_cpu_info_dict()
    parts = []
    if info["family"] != "Unknown":
        parts.append(
            f"Family {info['family']}, Model {info['model']}, Stepping {info['stepping']}"
        )
    if info["cores"] != "Unknown" and info["threads"] != "Unknown":
        parts.append(f"{info['cores']} Cores, {info['threads']} Threads")
    if info["speed_mhz"] != "Unknown":
        parts.append(f"{info['speed_mhz']} MHz")

    details = f" ({'; '.join(parts)})" if parts else ""
    return f"{info['brand']}{details} [Capability: {info['capability']}]"


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
