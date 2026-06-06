# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# ruff: noqa: E402

import sys
import os

if sys.platform == "win32":
    # Reconfigure stdout/stderr to UTF-8 for Windows terminals that default to cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

if "-h" not in sys.argv and "--help" not in sys.argv:
    try:
        from importlib.metadata import version
        sys.stdout.write(f"\033[36m█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀\033[0m  v{version('heretic-llm')}\n")
        sys.stdout.write("\033[36m█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░\033[0m\n")
        sys.stdout.write("\033[36m▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀\033[0m  \033[4;34mhttps://github.com/p-e-w/heretic\033[0m\n\n")
        sys.stdout.flush()
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Windows AMD ROCm Auto-Installer
    # Runs before any ML libraries are imported to avoid locking PyTorch DLLs.
    # Only active on Windows; silently skipped on other platforms.
    # -------------------------------------------------------------------------
    if sys.platform == "win32":
        import subprocess

        # Pinned wheel versions — update these together when upgrading ROCm/torch.
        _TORCH_VERSION = "2.9.1"
        _ROCM_VERSION  = "7.13.0"
        _ROCM_TAG      = "rocm7.13.0"   # tag embedded in the torch wheel filename
        _AMD_BASE      = "https://repo.amd.com/rocm/whl"

        # ------------------------------------------------------------------
        # 1. Determine whether torch is CPU-only or ROCm-enabled.
        # ------------------------------------------------------------------
        is_cpu = True
        try:
            from importlib.metadata import version as _pkg_version
            _torch_ver = _pkg_version("torch")
            if "+rocm" in _torch_ver or "+cu" in _torch_ver:
                is_cpu = False
        except Exception:
            # Fallback: ask a subprocess so we don't accidentally import torch here.
            try:
                _out = subprocess.check_output(
                    [sys.executable, "-c", "import torch; print(torch.version.hip)"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                if _out.strip() and _out.strip().lower() != "none":
                    is_cpu = False
            except (FileNotFoundError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired, OSError):
                pass

        # ------------------------------------------------------------------
        # 2. If ROCm torch is active, ensure bitsandbytes is already patched.
        # ------------------------------------------------------------------
        if not is_cpu:
            import importlib.util
            try:
                _spec = importlib.util.find_spec("bitsandbytes")
                if _spec is not None and _spec.submodule_search_locations:
                    _bnb_dir  = _spec.submodule_search_locations[0]
                    _dll_dest = os.path.join(_bnb_dir, "libbitsandbytes_rocm83.dll")
                    if not os.path.exists(_dll_dest):
                        _main_dir    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        _script_path = os.path.join(_main_dir, "scripts", "patch_bitsandbytes.py")
                        if not os.path.exists(_script_path):
                            _script_path = os.path.join(os.getcwd(), "scripts", "patch_bitsandbytes.py")
                        if os.path.exists(_script_path):
                            subprocess.check_call([sys.executable, _script_path])
            except Exception:
                pass  # Non-fatal: the main installer below will also run the patch script.

        # ------------------------------------------------------------------
        # 3. Detect the installed GPU and its RDNA generation.
        # ------------------------------------------------------------------
        has_amd       = False
        gpu_name      = ""
        arch_target   = "gfx103X-all"
        lib_suffix    = "gfx103x_all"
        generation_name = "RDNA2 (Radeon RX 6000-series / gfx103X)"
        is_forced     = False

        force_arch = os.environ.get("HERETIC_FORCE_ARCH", "").lower()
        if force_arch in ("gfx1030", "gfx103x", "rdna2"):
            has_amd = True
            arch_target = "gfx103X-all";  lib_suffix = "gfx103x_all"
            generation_name = "RDNA2 (Radeon RX 6000-series / gfx103X)"
            is_forced = True
        elif force_arch in ("gfx1100", "gfx110x", "rdna3"):
            has_amd = True
            arch_target = "gfx110X-all";  lib_suffix = "gfx110x_all"
            generation_name = "RDNA3 (Radeon RX 7000-series / gfx110X)"
            is_forced = True
        elif force_arch in ("gfx1200", "gfx120x", "rdna4"):
            has_amd = True
            arch_target = "gfx120X-all";  lib_suffix = "gfx120x_all"
            generation_name = "RDNA4 (Radeon RX 9000-series / gfx120X)"
            is_forced = True
        elif force_arch == "cpu":
            has_amd = False
            is_forced = True

        if not is_forced:
            # Prefer list-form subprocess (no shell injection surface).
            # Fall back to PowerShell if wmic is absent (removed in Win 11 24H2+).
            try:
                gpu_name = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    text=True, stderr=subprocess.DEVNULL, timeout=10,
                )
                has_amd = any(b in gpu_name for b in ("AMD", "Radeon"))
            except (FileNotFoundError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired, OSError):
                try:
                    gpu_name = subprocess.check_output(
                        ["powershell", "-Command",
                         "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                        text=True, stderr=subprocess.DEVNULL, timeout=10,
                    )
                    has_amd = any(b in gpu_name for b in ("AMD", "Radeon"))
                except (FileNotFoundError, subprocess.CalledProcessError,
                        subprocess.TimeoutExpired, OSError):
                    pass

            if has_amd:
                import re as _re
                # Default is RDNA2; override for RDNA3 / RDNA4 if the GPU name matches.
                if _re.search(r'\b(7900|7800|7700|7600|780M|760M|740M|W7900|W7800|W7700|W7600|W7500)\b', gpu_name, _re.IGNORECASE):
                    arch_target = "gfx110X-all";  lib_suffix = "gfx110x_all"
                    generation_name = "RDNA3 (Radeon RX 7000-series / gfx110X)"
                elif _re.search(r'\b(9000|9070|9060|R9700|W8900|W8800|W8600)\b', gpu_name, _re.IGNORECASE):
                    arch_target = "gfx120X-all";  lib_suffix = "gfx120x_all"
                    generation_name = "RDNA4 (Radeon RX 9000-series / gfx120X)"

        # ------------------------------------------------------------------
        # 4. Detect which architecture the currently installed torch targets.
        # ------------------------------------------------------------------
        installed_suffix = None
        if not is_cpu:
            try:
                import json
                from importlib.metadata import distribution as _dist
                _direct_url = _dist("torch").read_text("direct_url.json")
                if _direct_url:
                    _url = json.loads(_direct_url).get("url", "")
                    if   "gfx103" in _url.lower(): installed_suffix = "gfx103x_all"
                    elif "gfx110" in _url.lower(): installed_suffix = "gfx110x_all"
                    elif "gfx120" in _url.lower(): installed_suffix = "gfx120x_all"
            except Exception:
                pass

        if not installed_suffix:
            for _suffix in ("gfx103x_all", "gfx110x_all", "gfx120x_all"):
                try:
                    from importlib.metadata import version as _pkg_version
                    _pkg_version(f"rocm-sdk-libraries-{_suffix.replace('_', '-')}")
                    installed_suffix = _suffix
                    break
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # 5. Decide whether an install is needed and prompt the user.
        # ------------------------------------------------------------------
        needs_install = False
        warning_msg   = ""

        if has_amd:
            if is_cpu:
                needs_install = True
                warning_msg = (
                    f"WARNING: An AMD GPU ({generation_name}) was detected, "
                    "but PyTorch CPU-only is currently active."
                )
            elif installed_suffix and installed_suffix != lib_suffix:
                needs_install = True
                curr_gen = ("RDNA2" if "gfx103" in installed_suffix
                            else "RDNA3" if "gfx110" in installed_suffix
                            else "RDNA4")
                warning_msg = (
                    f"WARNING: An AMD GPU ({generation_name}) was detected, "
                    f"but the currently installed ROCm packages target {curr_gen}."
                )

        if needs_install:
            import questionary
            from questionary import Choice

            print(warning_msg)
            print(
                f"Would you like Heretic to automatically configure the AMD ROCm "
                f"packages for {generation_name} and enable 4-bit quantization support?"
            )

            choices = [
                Choice(title=f"Yes, auto-configure AMD ROCm for {generation_name} & patch bitsandbytes", value="install"),
                Choice(title="No, keep current configuration", value="skip"),
            ]

            # Notebooks lack an interactive TTY; fall back to numbered text input.
            is_nb = bool(os.getenv("COLAB_GPU") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
            if not is_nb:
                try:
                    from IPython import get_ipython as _get_ipython
                    is_nb = _get_ipython() is not None
                except Exception:
                    pass

            def _text_menu(choices):
                """Numbered fallback menu for non-interactive / notebook environments."""
                print("\nSelect action:")
                for i, c in enumerate(choices, 1):
                    print(f"  [{i}] {c.title}")
                while True:
                    try:
                        raw = input("Enter number: ").strip()
                        if not raw:
                            return choices[0].value
                        idx = int(raw) - 1
                        if 0 <= idx < len(choices):
                            return choices[idx].value
                        print(f"Please enter a number between 1 and {len(choices)}.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                    except EOFError:
                        return choices[0].value

            choice = "skip"
            if is_nb:
                choice = _text_menu(choices)
            else:
                try:
                    choice = questionary.select(
                        "Select action:",
                        choices=choices,
                        style=questionary.Style([("highlighted", "reverse")]),
                    ).ask()
                    if choice is None:
                        choice = "skip"
                except Exception:
                    # questionary requires a real TTY; fall back to plain text.
                    choice = _text_menu(choices)

            if choice == "install":
                print(f"\nConfiguring AMD ROCm PyTorch and SDK wheels for {generation_name}.")
                print("This may take several minutes on first run (subsequent runs use the local cache)...\n")

                # --- Strategy: swap in a pre-generated arch-specific pyproject + lock pair,
                # then let `uv sync` install the correct wheels cleanly.
                #
                # This permanently fixes the install: every subsequent `uv run heretic`
                # syncs from the correct lock and installs nothing (already satisfied).
                #
                # Falls back to direct `uv pip install` if the variant files are not
                # present (e.g. non-git / pip-installed heretic).
                _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if not os.path.isfile(os.path.join(_repo_root, "pyproject.toml")):
                    _repo_root = os.getcwd()

                _rdna_map = {"gfx103x_all": "rdna2", "gfx110x_all": "rdna3", "gfx120x_all": "rdna4"}
                _rdna     = _rdna_map.get(lib_suffix, "rdna3")

                _pyproject_variant = os.path.join(_repo_root, f"pyproject.{_rdna}.toml")
                _lock_variant      = os.path.join(_repo_root, f"uv.lock.{_rdna}")

                if os.path.isfile(_pyproject_variant) and os.path.isfile(_lock_variant):
                    # ---- Preferred path: swap pyproject + lock, then uv sync ----
                    import shutil
                    print(f"Activating {_rdna.upper()} configuration...")
                    try:
                        shutil.copy2(_pyproject_variant, os.path.join(_repo_root, "pyproject.toml"))
                        shutil.copy2(_lock_variant,      os.path.join(_repo_root, "uv.lock"))
                    except OSError as _copy_err:
                        print(f"ERROR: Could not swap configuration files: {_copy_err}")
                        sys.exit(1)

                    try:
                        subprocess.check_call(["uv", "sync"])
                        print("ROCm PyTorch and SDK wheels configured successfully!")
                    except subprocess.CalledProcessError as _sync_err:
                        print(f"ERROR: uv sync failed (exit {_sync_err.returncode}).")
                        print("Please run 'uv sync' manually, then restart Heretic.")
                        sys.exit(1)

                    # Patch bitsandbytes for Windows ROCm support.
                    try:
                        _script_path = os.path.join(_repo_root, "scripts", "patch_bitsandbytes.py")
                        if not os.path.exists(_script_path):
                            _script_path = os.path.join(os.getcwd(), "scripts", "patch_bitsandbytes.py")
                        if os.path.exists(_script_path):
                            subprocess.check_call([sys.executable, _script_path])
                        else:
                            _env = os.environ.copy()
                            _env["PYTHONPATH"] = os.getcwd() + os.pathsep + _env.get("PYTHONPATH", "")
                            subprocess.check_call(
                                [sys.executable, "-m", "scripts.patch_bitsandbytes"],
                                env=_env,
                            )
                    except subprocess.CalledProcessError as _patch_err:
                        print(
                            f"WARNING: Patch script failed (exit {_patch_err.returncode}). "
                            "Run 'python scripts/patch_bitsandbytes.py' manually."
                        )
                    except (FileNotFoundError, OSError) as _patch_err:
                        print(f"WARNING: Could not run patch script: {_patch_err}")

                    print("\nROCm environment configured. Relaunching Heretic...\n")
                    # Plain `uv run heretic` — no --no-sync needed. uv will sync,
                    # find the lock already satisfied, install nothing, and start.
                    _relaunch_args = sys.argv[1:]
                    try:
                        subprocess.check_call(["uv", "run", "heretic"] + _relaunch_args)
                        sys.exit(0)
                    except (subprocess.CalledProcessError, OSError) as _err:
                        print(f"ERROR: Could not relaunch Heretic: {_err}")
                        print(
                            f"Please restart manually: uv run heretic {' '.join(_relaunch_args)}"
                        )
                        sys.exit(1)

                else:
                    # ---- Fallback path: direct wheel install (non-git / pip-installed) ----
                    print("(Variant lock files not found; falling back to direct wheel install)")

                    if sys.executable is None:
                        print("ERROR: Cannot determine Python executable path. Please install manually.")
                        sys.exit(1)

                    _py       = f"{sys.version_info.major}{sys.version_info.minor}"
                    _base     = f"{_AMD_BASE}/{arch_target}"
                    _torch_url    = (
                        f"{_base}/torch-{_TORCH_VERSION}%2B{_ROCM_TAG}"
                        f"-cp{_py}-cp{_py}-win_amd64.whl"
                    )
                    _sdk_core_url = f"{_base}/rocm_sdk_core-{_ROCM_VERSION}-py3-none-win_amd64.whl"
                    _sdk_libs_url = f"{_base}/rocm_sdk_libraries_{lib_suffix}-{_ROCM_VERSION}-py3-none-win_amd64.whl"

                    has_uv = False
                    try:
                        subprocess.check_call(
                            ["uv", "--version"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=10,
                        )
                        has_uv = True
                    except (FileNotFoundError, subprocess.CalledProcessError,
                            subprocess.TimeoutExpired, OSError):
                        pass

                    install_cmd = (
                        ["uv", "pip", "install",
                         "--python", sys.executable,
                         "--extra-index-url", _base + "/",
                         "--index-strategy", "unsafe-best-match",
                         _torch_url, _sdk_core_url, _sdk_libs_url]
                        if has_uv else
                        [sys.executable, "-m", "pip", "install",
                         "--extra-index-url", _base + "/",
                         _torch_url, _sdk_core_url, _sdk_libs_url]
                    )

                    try:
                        subprocess.check_call(install_cmd)
                        print("ROCm PyTorch and SDK wheels configured successfully!")

                        # Patch bitsandbytes.
                        try:
                            _script_path = os.path.join(_repo_root, "scripts", "patch_bitsandbytes.py")
                            if not os.path.exists(_script_path):
                                _script_path = os.path.join(os.getcwd(), "scripts", "patch_bitsandbytes.py")
                            if os.path.exists(_script_path):
                                subprocess.check_call([sys.executable, _script_path])
                            else:
                                _env = os.environ.copy()
                                _env["PYTHONPATH"] = os.getcwd() + os.pathsep + _env.get("PYTHONPATH", "")
                                subprocess.check_call(
                                    [sys.executable, "-m", "scripts.patch_bitsandbytes"],
                                    env=_env,
                                )
                        except subprocess.CalledProcessError as _patch_err:
                            print(
                                f"WARNING: Patch script failed (exit {_patch_err.returncode}). "
                                "Run 'python scripts/patch_bitsandbytes.py' manually."
                            )
                        except (FileNotFoundError, OSError) as _patch_err:
                            print(f"WARNING: Could not run patch script: {_patch_err}")

                        print("\nROCm environment configured. Relaunching Heretic...\n")
                        _relaunch_args = sys.argv[1:]
                        try:
                            subprocess.check_call(
                                ["uv", "run", "--no-sync", "heretic"] + _relaunch_args
                            )
                            sys.exit(0)
                        except OSError as _exec_err:
                            print(f"ERROR: Could not relaunch Heretic: {_exec_err}")
                            print(
                                f"Please restart manually: "
                                f"uv run heretic {' '.join(_relaunch_args)}"
                            )
                            sys.exit(1)

                    except subprocess.CalledProcessError as _inst_err:
                        print(f"ERROR: Installation failed (exit {_inst_err.returncode}).")
                        print("Falling back to CPU-only execution...")
                    except OSError as _inst_err:
                        print(f"ERROR: Installation failed: {_inst_err}")
                        print("Falling back to CPU-only execution...")

from .config import Settings


def _is_help_invocation() -> bool:
    args = sys.argv[1:]
    return "-h" in args or "--help" in args


# Parse and handle CLI help before importing heavyweight ML/runtime dependencies.
if _is_help_invocation():
    Settings()  # ty:ignore[missing-argument]

# FIXME: Rich progress bars are currently disabled because of rendering issues
#        when used from multiple threads in parallel (e.g. by huggingface_hub).
"""
from .progress import patch_tqdm

# This patches tqdm class definitions, which must happen
# before any other module imports tqdm.
patch_tqdm()
"""

import logging

if sys.platform == "win32":
    # bitsandbytes calls `rocminfo` (a Linux-only tool) at import time to
    # detect GPU architecture on ROCm. On Windows this raises FileNotFoundError
    # and logs noisy ERROR messages that are harmless when quantization is
    # disabled.
    logging.getLogger("bitsandbytes").setLevel(logging.CRITICAL)
import math
import os
import random
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from os.path import commonprefix
from pathlib import Path
from typing import Any

import huggingface_hub
import numpy as np
import questionary
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import ModelCard, ModelCardData
from pydantic import ValidationError
from questionary import Choice, Style
from rich.table import Table
from rich.traceback import install

from .analyzer import Analyzer
from .config import QuantizationMethod
from .evaluator import Evaluator
from .model import AbliterationParameters, Model, get_model_class
from .reproduce import collect_reproducibles
from .system import empty_cache, get_accelerator_info
from .utils import (
    format_duration,
    get_readme_intro,
    get_trial_parameters,
    is_hf_path,
    load_prompts,
    print,
    print_memory_usage,
    prompt_password,
    prompt_path,
    prompt_select,
    prompt_text,
    set_seed,
    upload_reproduce_folder,
)


def obtain_merge_strategy(settings: Settings, model: Model) -> str | None:
    """
    Prompts the user for how to proceed with saving the model.
    Provides info to the user if the model is quantized on memory use.
    Returns "merge", "adapter", or None (if cancelled/invalid).
    """

    if settings.quantization == QuantizationMethod.BNB_4BIT:
        print()
        print(
            "Model was loaded with quantization. Merging requires reloading the base model."
        )
        print(
            "[yellow]WARNING: CPU merging requires dequantizing the entire model to system RAM.[/]"
        )
        print("[yellow]This can lead to system freezes if you run out of memory.[/]")

        try:
            # Estimate memory requirements by loading the model structure on the "meta" device.
            # This doesn't consume actual RAM but allows us to inspect the parameter count/dtype.
            #
            # Suppress warnings during meta device loading (e.g., "Some weights were not initialized").
            # These are expected and harmless since we're only inspecting model structure, not running inference.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=model.trusted_models.get(settings.model),
                    **model.revision_kwargs,
                )
                footprint_bytes = meta_model.get_memory_footprint()
                footprint_gb = footprint_bytes / (1024**3)
                print(
                    f"[yellow]Estimated RAM required (excluding overhead): [bold]~{footprint_gb:.2f} GB[/][/]"
                )
        except Exception:
            # Fallback if meta loading fails (e.g. owing to custom model code
            # or bitsandbytes quantization config issues on the meta device).
            print(
                "[yellow]Rule of thumb: You need approximately 3x the parameter count in GB RAM.[/]"
            )
            print(
                "[yellow]Example: A 27B model requires ~80GB RAM. A 70B model requires ~200GB RAM.[/]"
            )
        print()

    strategy = prompt_select(
        "How do you want to proceed?",
        choices=[
            Choice(
                title="Merge LoRA into full model"
                + (
                    ""
                    if settings.quantization == QuantizationMethod.NONE
                    else " (requires sufficient RAM)"
                ),
                value="merge",
            ),
            Choice(
                title="Save LoRA adapter only (can be merged later)",
                value="adapter",
            ),
        ],
    )

    return strategy


def run():
    # Enable expandable segments to reduce memory fragmentation on multi-GPU setups.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    if (
        # There is at least one argument (argv[0] is the program name).
        len(sys.argv) > 1
        # Heretic is being invoked in standard (model processing) mode.
        and "--collect-reproducibles" not in sys.argv
        # No model has been explicitly provided.
        and "--model" not in sys.argv
        # The last argument is a parameter value rather than a flag (such as "--help").
        and not sys.argv[-1].startswith("-")
    ):
        # Assume the last argument is the model.
        sys.argv.insert(-1, "--model")

    # Work around the "model" argument being required
    # when Heretic is invoked in a non-processing mode.
    if "--collect-reproducibles" in sys.argv and "--model" not in sys.argv:
        sys.argv.extend(["--model", ""])

    try:
        # The required argument "model" must be provided by the user,
        # either on the command line or in the configuration file.
        settings = Settings()  # ty:ignore[missing-argument]
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error_detail in error.errors():
            print(
                f"[bold]{error_detail['loc'][0]}[/]: [yellow]{error_detail['msg']}[/]"
            )

        print()
        print(
            "Run [bold]heretic --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

    if settings.collect_reproducibles is not None:
        collect_reproducibles(settings.collect_reproducibles)
        return

    if settings.seed is None:
        settings.seed = random.randint(0, 2**32 - 1)

    set_seed(settings.seed)

    print(get_accelerator_info())

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch sizes,
    # resulting in many computation graphs being compiled. Raising the limit (default = 8)
    # avoids errors from TorchDynamo assuming that something is wrong because we
    # recompile too often.
    torch._dynamo.config.cache_size_limit = 64

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # Another library that generates warning spam.
    logging.getLogger("lm_eval").setLevel(logging.ERROR)

    import optuna
    from optuna import Trial, TrialPruned
    from optuna.exceptions import ExperimentalWarning
    from optuna.samplers import TPESampler
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    from optuna.study import StudyDirection
    from optuna.trial import TrialState

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    os.makedirs(settings.study_checkpoint_dir, exist_ok=True)

    study_checkpoint_file = os.path.join(
        settings.study_checkpoint_dir,
        "".join(
            [(c if (c.isalnum() or c in ["_", "-"]) else "--") for c in settings.model]
        )
        + ".jsonl",
    )

    lock_obj = JournalFileOpenLock(study_checkpoint_file)
    backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    try:
        existing_study = storage.get_all_studies()[0]
    except IndexError:
        existing_study = None

    if existing_study is not None and settings.evaluate_model is None:
        choices = []

        if existing_study.user_attrs["finished"]:
            print()
            print(
                (
                    "[green]You have already processed this model.[/] "
                    "You can show the results from the previous run, allowing you to export models or to run additional trials. "
                    "Alternatively, you can ignore the previous run and start from scratch. "
                    "This will delete the checkpoint file and all results from the previous run."
                )
            )
            choices.append(
                Choice(
                    title="Show the results from the previous run",
                    value="continue",
                )
            )
        else:
            print()
            print(
                (
                    "[yellow]You have already processed this model, but the run was interrupted.[/] "
                    "You can continue the previous run from where it stopped. This will override any specified settings. "
                    "Alternatively, you can ignore the previous run and start from scratch. "
                    "This will delete the checkpoint file and all results from the previous run."
                )
            )
            choices.append(
                Choice(
                    title="Continue the previous run",
                    value="continue",
                )
            )

        choices.append(
            Choice(
                title="Ignore the previous run and start from scratch",
                value="restart",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        print()
        choice = prompt_select("How would you like to proceed?", choices)

        if choice == "continue":
            settings = Settings.model_validate_json(
                existing_study.user_attrs["settings"]
            )
        elif choice == "restart":
            os.unlink(study_checkpoint_file)
            backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
            storage = JournalStorage(backend)
        elif choice is None or choice == "":
            return

    model = Model(settings)
    print()
    print_memory_usage()

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]

            try:
                # Warmup run to build the computation graph so that part isn't benchmarked.
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    # Even a batch size of 1 already fails.
                    # We cannot recover from this.
                    raise

                print(f"[red]Failed[/] ({error})")
                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    if settings.response_prefix is None:
        print()
        print("Checking for common response prefix...")
        prefix_check_prompts = good_prompts[:100] + bad_prompts[:100]
        responses = model.get_responses_batched(prefix_check_prompts)

        # Despite being located in os.path, commonprefix actually performs
        # a naive string operation without any path-specific logic,
        # which is exactly what we need here. Trailing spaces are removed
        # to avoid issues where multiple different tokens that all start
        # with a space character lead to the common prefix ending with
        # a space, which would result in an uncommon tokenization.
        settings.response_prefix = commonprefix(responses).rstrip(" ")

        if settings.response_prefix:
            print(f"* Prefix found: [bold]{settings.response_prefix!r}[/]")

            for cot_initializer, closed_cot_block in settings.chain_of_thought_skips:
                if settings.response_prefix.startswith(cot_initializer):
                    settings.response_prefix = closed_cot_block
                    print(
                        f"* Closed Chain-of-Thought block: [bold]{settings.response_prefix!r}[/]"
                    )

                    # When using a Chain-of-Thought skip, we need to check that the prefix
                    # is actually complete (e.g. not missing a trailing newline).
                    print("* Rechecking with prefix...")
                    responses = model.get_responses_batched(prefix_check_prompts)
                    additional_prefix = commonprefix(responses).rstrip(" ")
                    if additional_prefix:
                        settings.response_prefix += additional_prefix
                        print(
                            f"* Extended prefix found: [bold]{settings.response_prefix!r}[/]"
                        )

                    break
        else:
            print("* None found")

    evaluator = Evaluator(settings, model)

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/]...")
        settings.model = settings.evaluate_model
        model.reset_model()
        print("* Evaluating...")
        evaluator.get_score()
        return

    print()
    print("Calculating per-layer refusal directions...")

    needs_full_residuals = settings.print_residual_geometry or settings.plot_residuals

    if needs_full_residuals:
        print("* Obtaining residuals for good prompts...")
        good_residuals = model.get_residuals_batched(good_prompts)
        print("* Obtaining residuals for bad prompts...")
        bad_residuals = model.get_residuals_batched(bad_prompts)

        good_means = good_residuals.mean(dim=0)
        bad_means = bad_residuals.mean(dim=0)

        analyzer = Analyzer(settings, model, good_residuals, bad_residuals)

        if settings.print_residual_geometry:
            analyzer.print_residual_geometry()

        if settings.plot_residuals:
            analyzer.plot_residuals()

        # We don't need the full residuals after computing their means and analyzing geometry.
        del good_residuals, bad_residuals, analyzer
    else:
        print("* Obtaining residual mean for good prompts...")
        good_means = model.get_residuals_mean(good_prompts)
        print("* Obtaining residual mean for bad prompts...")
        bad_means = model.get_residuals_mean(bad_prompts)

    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if settings.orthogonalize_direction:
        # Implements https://huggingface.co/blog/grimjim/projected-abliteration
        # Adjust the refusal directions so that only the component that is
        # orthogonal to the good direction is subtracted during abliteration.
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = (
            refusal_directions - projection_vector.unsqueeze(1) * good_directions
        )
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)
        del good_directions, projection_vector

    del good_means, bad_means

    # Clear cache before starting the optimization study.
    # This should free up memory from the objects released with the del statements above.
    empty_cache()

    trial_index = 0
    start_index = 0
    start_time = time.perf_counter()

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(model.get_layers()) - 1

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                0.8,
                1.5,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * last_layer_index,
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

        print()
        print(
            f"Running trial [bold]{trial_index}[/] of [bold]{settings.n_trials}[/]..."
        )
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Resetting model...")
        model.reset_model()
        print("* Abliterating...")
        model.abliterate(refusal_directions, direction_index, parameters)
        print("* Evaluating...")
        score, kl_divergence, refusals = evaluator.get_score()

        elapsed_time = time.perf_counter() - start_time
        remaining_time = (elapsed_time / (trial_index - start_index)) * (
            settings.n_trials - trial_index
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        if trial_index < settings.n_trials:
            print(
                f"[grey50]Estimated remaining time: [bold]{format_duration(remaining_time)}[/][/]"
            )
        print_memory_usage()

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)
        trial.set_user_attr("base_refusals", evaluator.base_refusals)
        trial.set_user_attr("n_bad_prompts", len(evaluator.bad_prompts))

        return score

    def objective_wrapper(trial: Trial) -> tuple[float, float]:
        try:
            return objective(trial)
        except KeyboardInterrupt:
            # Stop the study gracefully on Ctrl+C.
            trial.study.stop()
            raise TrialPruned()

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=settings.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
            seed=settings.seed,
        ),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
        storage=storage,
        study_name="heretic",
        load_if_exists=True,
    )

    study.set_user_attr("settings", settings.model_dump_json())
    study.set_user_attr("finished", False)

    def count_completed_trials() -> int:
        # Count number of complete trials to compute trials to run.
        return sum([(1 if t.state == TrialState.COMPLETE else 0) for t in study.trials])

    start_index = trial_index = count_completed_trials()
    if start_index > 0:
        print()
        print("Resuming existing study.")

    try:
        study.optimize(
            objective_wrapper,
            n_trials=settings.n_trials - count_completed_trials(),
        )
    except KeyboardInterrupt:
        # This additional handler takes care of the small chance that KeyboardInterrupt
        # is raised just between trials, which wouldn't be caught by the handler
        # defined in objective_wrapper above.
        pass

    if count_completed_trials() == settings.n_trials:
        study.set_user_attr("finished", True)

    while True:
        # If no trials at all have been evaluated, the study must have been stopped
        # by pressing Ctrl+C while the first trial was running. In this case, we just
        # re-raise the interrupt to invoke the standard handler defined below.
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            raise KeyboardInterrupt

        # Get the Pareto front of trials. We can't use study.best_trials directly
        # as get_score() doesn't return the pure KL divergence and refusal count.
        # Note: Unlike study.best_trials, this does not handle objective constraints.
        sorted_trials = sorted(
            completed_trials,
            key=lambda trial: (
                trial.user_attrs["refusals"],
                trial.user_attrs["kl_divergence"],
            ),
        )
        min_divergence = math.inf
        best_trials = []
        for trial in sorted_trials:
            kl_divergence = trial.user_attrs["kl_divergence"]
            if kl_divergence < min_divergence:
                min_divergence = kl_divergence
                best_trials.append(trial)

        choices = [
            Choice(
                title=(
                    f"[Trial {trial.user_attrs['index']:>3}] "
                    f"Refusals: {trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                    f"KL divergence: {trial.user_attrs['kl_divergence']:.4f}"
                ),
                value=trial,
            )
            for trial in best_trials
        ]

        choices.append(
            Choice(
                title="Run additional trials",
                value="continue",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        print()
        print("[bold green]Optimization finished![/]")
        print()
        print(
            (
                "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
                "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
                "chat with it to test how well it works, or run standard benchmarks on it. "
                "You can return to this menu later to select a different trial. "
                "[yellow]Note that KL divergence values above 0.5 usually indicate significant damage to the original model's capabilities.[/]"
            )
        )

        while True:
            print()
            trial = prompt_select("Which trial do you want to use?", choices)

            if trial == "continue":
                while True:
                    try:
                        n_additional_trials = prompt_text(
                            "How many additional trials do you want to run?"
                        )
                        if n_additional_trials is None or n_additional_trials == "":
                            n_additional_trials = 0
                            break
                        n_additional_trials = int(n_additional_trials)
                        if n_additional_trials > 0:
                            break
                        print("[red]Please enter a number greater than 0.[/]")
                    except ValueError:
                        print("[red]Please enter a number.[/]")

                if n_additional_trials == 0:
                    continue

                settings.n_trials += n_additional_trials
                study.set_user_attr("settings", settings.model_dump_json())
                study.set_user_attr("finished", False)

                try:
                    study.optimize(
                        objective_wrapper,
                        n_trials=settings.n_trials - count_completed_trials(),
                    )
                except KeyboardInterrupt:
                    pass

                if count_completed_trials() == settings.n_trials:
                    study.set_user_attr("finished", True)

                break

            elif trial is None or trial == "":
                return

            print()
            print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
            print("* Parameters:")
            for name, value in get_trial_parameters(trial).items():
                print(f"  * {name} = [bold]{value}[/]")

            # Per https://github.com/huggingface/peft/issues/868#issuecomment-1820642893 once a LoRA is merged it's
            # expected to be empty. Provide a utility function to restore the previous LoRA-ified state.
            def reset_trial_model() -> None:
                print("* Resetting model...")
                model.reset_model()
                print("* Abliterating...")
                model.abliterate(
                    refusal_directions,
                    trial.user_attrs["direction_index"],
                    {
                        k: AbliterationParameters(**v)
                        for k, v in trial.user_attrs["parameters"].items()
                    },
                )

            reset_trial_model()

            while True:
                print()
                action = prompt_select(
                    "What do you want to do with the decensored model?",
                    [
                        "Save the model to a local folder",
                        "Upload the model to Hugging Face",
                        "Chat with the model",
                        "Benchmark the model",
                        "Return to the trial selection menu",
                    ],
                )

                if action is None or action == "Return to the trial selection menu":
                    break

                # All actions are wrapped in a try/except block so that if an error occurs,
                # another action can be tried, instead of the program crashing and losing
                # the optimized model.
                try:
                    match action:
                        case "Save the model to a local folder":
                            save_directory = prompt_path("Path to the folder:")
                            if not save_directory:
                                continue

                            strategy = obtain_merge_strategy(settings, model)
                            if strategy is None:
                                continue

                            if strategy == "adapter":
                                print("Saving LoRA adapter...")
                                model.model.save_pretrained(
                                    save_directory,
                                    max_shard_size=settings.max_shard_size,
                                )
                            else:
                                print("Saving merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.save_pretrained(
                                    save_directory,
                                    max_shard_size=settings.max_shard_size,
                                )
                                del merged_model
                                empty_cache()
                                model.tokenizer.save_pretrained(save_directory)
                                if model.processor is not None:
                                    model.processor.save_pretrained(save_directory)
                                reset_trial_model()

                            print(f"Model saved to [bold]{save_directory}[/].")

                        case "Upload the model to Hugging Face":
                            # We don't use huggingface_hub.login() because that stores the token on disk,
                            # and since this program will often be run on rented or shared GPU servers,
                            # it's better to not persist credentials.
                            token = huggingface_hub.get_token()
                            if not token:
                                token = prompt_password("Hugging Face access token:")
                            if not token:
                                continue

                            user = huggingface_hub.whoami(token)
                            fullname = user.get(
                                "fullname",
                                user.get("name", "unknown user"),
                            )
                            email = user.get("email", "no email found")
                            print(f"Logged in as [bold]{fullname} ({email})[/]")

                            repo_id = prompt_text(
                                "Name of repository:",
                                default=f"{user['name']}/{Path(settings.model).name}-heretic",
                            )

                            visibility = prompt_select(
                                "Should the repository be public or private?",
                                [
                                    "Public",
                                    "Private",
                                ],
                            )
                            if visibility is None:
                                continue
                            private = visibility == "Private"

                            strategy = obtain_merge_strategy(settings, model)
                            if strategy is None:
                                continue

                            # Reproducibility requires that the model and all datasets
                            # are available on the Hugging Face Hub (not local paths).
                            datasets = [
                                settings.good_prompts.dataset,
                                settings.bad_prompts.dataset,
                                settings.good_evaluation_prompts.dataset,
                                settings.bad_evaluation_prompts.dataset,
                            ]
                            is_reproducible = is_hf_path(settings.model) and all(
                                is_hf_path(dataset) for dataset in datasets
                            )

                            if is_reproducible:
                                print(
                                    (
                                        "Heretic can add information to the repository that allows others to reproduce the model. "
                                        "This is optional, but valuable to the community as both a learning tool and to preserve computational work already done. "
                                        "Guaranteeing reproducibility requires basic system information (Python and OS version, CPU and GPU/accelerator info) "
                                        "as tensor operations can give different results in different system environments. "
                                        "[bold]The information does not include any file system paths or other private data.[/]"
                                    )
                                )
                                reproducibility_information = prompt_select(
                                    "Which reproducibility information do you want to add?",
                                    [
                                        Choice(
                                            title="Full: Settings, package versions, and system information",
                                            value="full",
                                        ),
                                        Choice(
                                            title="Basic: Settings and package versions",
                                            value="basic",
                                        ),
                                        Choice(
                                            title="Don't add any reproducibility information",
                                            value="none",
                                        ),
                                    ],
                                )
                                if reproducibility_information is None:
                                    continue
                            else:
                                reproducibility_information = "none"

                            if strategy == "adapter":
                                print("Uploading LoRA adapter...")
                                model.model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    max_shard_size=settings.max_shard_size,
                                    token=token,
                                )
                            else:
                                print("Uploading merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    max_shard_size=settings.max_shard_size,
                                    token=token,
                                )
                                del merged_model
                                empty_cache()
                                model.tokenizer.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                                if model.processor is not None:
                                    model.processor.push_to_hub(
                                        repo_id,
                                        private=private,
                                        token=token,
                                    )
                                reset_trial_model()

                            if is_hf_path(settings.model):
                                card = ModelCard.load(settings.model)
                            else:
                                card_path = (
                                    Path(settings.model)
                                    / huggingface_hub.constants.REPOCARD_NAME
                                )
                                if card_path.exists():
                                    card = ModelCard.load(card_path)
                                else:
                                    card = None

                            if card is not None:
                                if card.data is None:
                                    card.data = ModelCardData()
                                if card.data.tags is None:
                                    card.data.tags = []
                                card.data.tags.append("heretic")
                                card.data.tags.append("uncensored")
                                card.data.tags.append("decensored")
                                card.data.tags.append("abliterated")
                                if reproducibility_information != "none":
                                    card.data.tags.append("reproducible")
                                card.text = (
                                    get_readme_intro(
                                        settings,
                                        trial,
                                        reproducibility_information != "none",
                                    )
                                    + card.text
                                )
                                card.push_to_hub(repo_id, token=token)

                            if reproducibility_information != "none":
                                # Set the number of trials to the number of actual completed trials
                                # for the reproduction configuration.
                                settings.n_trials = count_completed_trials()

                                upload_reproduce_folder(
                                    repo_id,
                                    settings,
                                    token,
                                    checkpoint_path=study_checkpoint_file,
                                    trial=trial,
                                    include_system_information=(
                                        reproducibility_information == "full"
                                    ),
                                )

                            print(f"Model uploaded to [bold]{repo_id}[/].")

                        case "Chat with the model":
                            print()
                            print(
                                "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                            )

                            chat = [
                                {"role": "system", "content": settings.system_prompt},
                            ]

                            while True:
                                try:
                                    message = prompt_text(
                                        "User:",
                                        qmark=">",
                                        unsafe=True,
                                    )
                                    if not message:
                                        break
                                    chat.append({"role": "user", "content": message})

                                    print("[bold]Assistant:[/] ", end="")
                                    response = model.stream_chat_response(chat)
                                    chat.append(
                                        {"role": "assistant", "content": response}
                                    )
                                except (KeyboardInterrupt, EOFError):
                                    # Ctrl+C/Ctrl+D
                                    break

                        case "Benchmark the model":
                            benchmarks = questionary.checkbox(
                                "Which benchmarks do you want to run?",
                                [
                                    Choice(
                                        title=f"{benchmark.name}: {benchmark.description}",
                                        value=benchmark,
                                    )
                                    for benchmark in settings.benchmarks
                                ],
                                style=Style([("highlighted", "reverse")]),
                            ).ask()
                            if not benchmarks:
                                continue

                            scope = prompt_select(
                                (
                                    "Do you want to benchmark the original model along with the decensored model? "
                                    "Benchmarking both models allows you to compare the scores, but it takes twice as much time."
                                ),
                                [
                                    "Benchmark only the decensored model",
                                    "Benchmark both models",
                                ],
                            )
                            if scope is None:
                                continue
                            benchmark_original_model = scope == "Benchmark both models"
                            import lm_eval
                            from lm_eval.models.huggingface import HFLM

                            hflm = HFLM(
                                pretrained=model.model,  # ty:ignore[invalid-argument-type]
                                tokenizer=model.tokenizer,  # ty:ignore[invalid-argument-type]
                                batch_size="auto",
                            )

                            table = Table()
                            table.add_column("Benchmark")
                            table.add_column("Metric")
                            if benchmark_original_model:
                                table.add_column("This model", justify="right")
                                table.add_column("Original model", justify="right")
                            else:
                                table.add_column("Value", justify="right")

                            try:
                                first_benchmark = True

                                for benchmark in benchmarks:
                                    print(
                                        f"Running benchmark [bold]{benchmark.name}[/]..."
                                    )

                                    def get_results() -> dict[str, Any]:
                                        results = lm_eval.simple_evaluate(
                                            model=hflm,
                                            tasks=[benchmark.task],
                                        )
                                        return results["results"][benchmark.task]

                                    results = get_results()
                                    if benchmark_original_model:
                                        with model.model.disable_adapter():  # ty:ignore[call-non-callable]
                                            original_results = get_results()

                                    first_row = True

                                    for metric, value in results.items():
                                        if metric != "alias":
                                            if first_row and not first_benchmark:
                                                if benchmark_original_model:
                                                    table.add_row("", "", "", "")
                                                else:
                                                    table.add_row("", "", "")

                                            def format_value(value: Any) -> str:
                                                if isinstance(
                                                    value,
                                                    (float, np.floating),
                                                ):
                                                    return f"{value:.4f}"
                                                else:
                                                    return f"{value}"

                                            cells = [
                                                benchmark.name if first_row else "",
                                                metric,
                                                format_value(value),
                                            ]
                                            if benchmark_original_model:
                                                cells.append(
                                                    format_value(
                                                        original_results[metric]
                                                    )
                                                )
                                            table.add_row(*cells)

                                            first_row = False
                                            first_benchmark = False
                            except KeyboardInterrupt:
                                pass

                            # The benchmark run might have been cancelled by the user
                            # before any benchmark was completed, so we only print results
                            # if there actually are some.
                            if table.rows:
                                print(table)

                except Exception as error:
                    print(f"[red]Error: {error}[/]")


def main():
    # Install Rich traceback handler.
    install()

    try:
        run()
    except BaseException as error:
        # Transformers appears to handle KeyboardInterrupt (or BaseException)
        # internally in some places, which can re-raise a different error in the handler,
        # masking the root cause. We therefore check both the error itself and its context.
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
