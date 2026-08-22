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


def _is_torch_xla_available() -> bool:
    """Check if torch_xla is available without initializing it."""
    try:
        import torch_xla  # noqa: F401
        return True
    except ImportError:
        return False


# Public alias for __init__.py export
is_torch_xla_available = _is_torch_xla_available


def detect_tpu() -> bool:
    """Detect if running on TPU (PyTorch/XLA)."""
    if not _is_torch_xla_available():
        return False
    # Check PJRT_DEVICE env var (set by TPU VMs)
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    # Check if XLA device is available
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device_hw(xm.xla_device()) == "TPU"
    except Exception:
        return False


def _get_tpu_core_count_from_env() -> int:
    """Total TPU chips: product(TPU_CHIPS_PER_HOST_BOUNDS) * product(TPU_HOST_BOUNDS)."""
    import math

    total = 1
    for var in ("TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"):
        val = os.environ.get(var)
        if val:
            try:
                total *= math.prod(int(dim) for dim in val.split(","))
            except ValueError:
                pass
    return total


def _ensure_spmd_if_multichip(enable: bool = False) -> None:
    """Enable SPMD mode before the XLA client initializes.

    torch_xla 2.8: when a single process drives multiple TPU chips, SPMD must
    be active from the moment the XLA client starts. Setting XLA_USE_SPMD=1
    in the environment (before any device access) makes the client initialize
    in SPMD mode; calling use_spmd() after client init goes through the
    "force replication" path, which segfaults in PjRtComputationClient::
    ExecuteReplicated (a plain sharded matmul is enough to reproduce).
    Safe to call repeatedly and on non-TPU hosts (no-op).

    Only called with enable=True for the single-process multi-core FSDP path.
    Single-core runs must stay in plain multi-device mode: the SPMD virtual
    device breaks memory probing (memory_info asserts on SPMD:0) and its
    deviceless tensor fetch accumulates per step until executions start
    failing with null tensor data ("Check failed: data->tensor_data").
    """
    if not enable or not _is_torch_xla_available():
        return
    try:
        os.environ.setdefault("XLA_USE_SPMD", "1")
        import torch_xla.runtime as xr

        if xr.is_spmd():
            return
        if xr.process_count() == 1 and _get_tpu_core_count_from_env() > 1:
            xr.use_spmd()
    except Exception:
        pass


def get_xla_device(core_id: int = 0, enable_spmd: bool = False) -> torch.device:
    """Get the XLA device for the given core ID."""
    if not _is_torch_xla_available():
        raise RuntimeError("torch_xla not available")
    _ensure_spmd_if_multichip(enable=enable_spmd)
    import torch_xla.core.xla_model as xm
    return xm.xla_device(n=core_id)


def get_xla_device_count() -> int:
    """Get the number of available XLA devices (TPU cores)."""
    if not _is_torch_xla_available():
        return 0
    try:
        import torch_xla.runtime as xr

        count = xr.global_device_count()
        if count > 1:
            return count
        # torch_xla 2.8 in SPMD mode (single process, multi-chip) reports one
        # virtual device; derive the physical core count from TPU env vars.
        from_env = _get_tpu_core_count_from_env()
        return from_env if from_env > 1 else count
    except Exception:
        return 0


def setup_tpu_environment(enable_spmd: bool = False) -> None:
    """Set up environment variables for optimal TPU performance."""
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_BF16", "1")
    os.environ.setdefault("XLA_DOWNCAST_BF16", "1")
    # SPMD must be active before the XLA client initializes (see
    # _ensure_spmd_if_multichip); set the flag early for the FSDP path.
    if enable_spmd:
        os.environ.setdefault("XLA_USE_SPMD", "1")
    # Disable tokenizers parallelism to avoid warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def mark_step() -> None:
    """Mark a step for XLA lazy execution. Call after each forward pass on TPU.

    wait=True is critical on torch_xla 2.8: with the default wait=False,
    execution is dispatched asynchronously and tensor device data is not
    attached yet when control returns. Downstream ops like torch.cat that
    inspect shapes eagerly (result_type -> dim) then hit
    "Check failed: data->tensor_data" inside XLATensor::shape(). Waiting
    makes each step fully materialize before returning.
    """
    if not _is_torch_xla_available():
        return
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step(wait=True)
    except Exception:
        pass


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
    elif _is_torch_xla_available():
        # On TPU, clear the XLA compilation cache to free system RAM.
        # XLA caches compiled HLO graphs which accumulate across trials.
        try:
            import torch_xla.runtime as xr
            xr.reset_spmd()
        except Exception:
            pass
        try:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        except Exception:
            pass
        gc.collect()

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


def get_tpu_info_dict() -> dict[str, Any]:
    """Get TPU-specific information."""
    if not detect_tpu():
        return {"type": None}

    try:
        import torch_xla
        import torch_xla.runtime as xr

        device_count = get_xla_device_count()
        devices = []
        for i in range(device_count):
            devices.append({
                "name": f"TPU Core {i}",
                "ordinal": i,
                "device": f"xla:{i}",
            })

        return {
            "type": "TPU",
            "api_name": "PJRT",
            "api_version": getattr(torch_xla, "__version__", "unknown"),
            "driver_version": None,
            "devices": devices,
            "world_size": xr.global_ordinal() + 1 if xr.process_count() > 1 else 1,
        }
    except Exception as e:
        return {"type": "TPU", "error": str(e)}


def get_accelerator_info_dict() -> dict[str, Any]:
    """Retrieves raw accelerator info (CUDA, ROCm, etc) directly into structured keys."""

    # Check TPU first (before CUDA since TPU VMs may have CUDA visible)
    if detect_tpu():
        return get_tpu_info_dict()

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
