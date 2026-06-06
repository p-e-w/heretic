"""
setup_rocm.py — Heretic Windows ROCm pre-flight setup.

Run this ONCE after cloning, before launching heretic:

    python scripts/setup_rocm.py

It will:
  1. Detect your AMD GPU architecture (RDNA2 / RDNA3 / RDNA4).
  2. Swap in the matching pyproject.<arch>.toml + uv.lock.<arch>.
  3. Run `uv sync --frozen --no-install-project` to install ROCm wheels.
  4. Patch bitsandbytes for Windows ROCm DLL support.
  5. Write a .heretic_rocm_arch marker so subsequent launches skip setup.

Because this runs as plain `python` (not as heretic.exe), there is no
Windows file-lock on the heretic executable — uv sync can freely update it.
"""

import os
import re
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Constants — update together when upgrading ROCm / torch.
# ---------------------------------------------------------------------------
_TORCH_VERSION = "2.9.1"
_ROCM_VERSION  = "7.13.0"
_ROCM_TAG      = "rocm7.13.0"
_AMD_BASE      = "https://repo.amd.com/rocm/whl"

# ---------------------------------------------------------------------------
# 1. Locate repo root (the directory that contains pyproject.toml).
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root  = os.path.dirname(_script_dir)
if not os.path.isfile(os.path.join(_repo_root, "pyproject.toml")):
    _repo_root = os.getcwd()

# ---------------------------------------------------------------------------
# 2. Detect GPU arch (same logic as main.py).
# ---------------------------------------------------------------------------
has_amd         = False
gpu_name        = ""
arch_target     = "gfx103X-all"
lib_suffix      = "gfx103x_all"
generation_name = "RDNA2 (Radeon RX 6000-series / gfx103X)"

force_arch = os.environ.get("HERETIC_FORCE_ARCH", "").lower()
is_forced  = False

if force_arch in ("gfx1030", "gfx103x", "rdna2"):
    has_amd = True; arch_target = "gfx103X-all"; lib_suffix = "gfx103x_all"
    generation_name = "RDNA2 (Radeon RX 6000-series / gfx103X)"; is_forced = True
elif force_arch in ("gfx1100", "gfx110x", "rdna3"):
    has_amd = True; arch_target = "gfx110X-all"; lib_suffix = "gfx110x_all"
    generation_name = "RDNA3 (Radeon RX 7000-series / gfx110X)"; is_forced = True
elif force_arch in ("gfx1200", "gfx120x", "rdna4"):
    has_amd = True; arch_target = "gfx120X-all"; lib_suffix = "gfx120x_all"
    generation_name = "RDNA4 (Radeon RX 9000-series / gfx120X)"; is_forced = True
elif force_arch == "cpu":
    has_amd = False; is_forced = True

if not is_forced:
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
        if re.search(r'\b(7900|7800|7700|7600|780M|760M|740M|W7900|W7800|W7700|W7600|W7500)\b',
                     gpu_name, re.IGNORECASE):
            arch_target = "gfx110X-all"; lib_suffix = "gfx110x_all"
            generation_name = "RDNA3 (Radeon RX 7000-series / gfx110X)"
        elif re.search(r'\b(9000|9070|9060|R9700|W8900|W8800|W8600)\b',
                       gpu_name, re.IGNORECASE):
            arch_target = "gfx120X-all"; lib_suffix = "gfx120x_all"
            generation_name = "RDNA4 (Radeon RX 9000-series / gfx120X)"

# ---------------------------------------------------------------------------
# 3. Check marker — skip if already configured for this arch.
# ---------------------------------------------------------------------------
_rdna_map    = {"gfx103x_all": "rdna2", "gfx110x_all": "rdna3", "gfx120x_all": "rdna4"}
_rdna        = _rdna_map.get(lib_suffix, "rdna2")
_marker_path = os.path.join(_repo_root, ".heretic_rocm_arch")

if has_amd and os.path.isfile(_marker_path):
    try:
        with open(_marker_path, "r", encoding="utf-8") as _mf:
            if _mf.read().strip() == _rdna:
                print(f"[setup_rocm] Already configured for {generation_name}. Nothing to do.")
                sys.exit(0)
    except OSError:
        pass

if not has_amd:
    print("[setup_rocm] No AMD GPU detected. Nothing to configure.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# 4. Prompt the user.
# ---------------------------------------------------------------------------
print(f"\nDetected GPU: {generation_name}")
print(f"This script will install ROCm PyTorch + SDK wheels and patch bitsandbytes.\n")

try:
    import questionary
    from questionary import Choice
    choice = questionary.select(
        "Configure AMD ROCm for heretic?",
        choices=[
            Choice(f"Yes — full setup with 4-bit quantization support (recommended)", "install"),
            Choice(f"Yes — ROCm only, skip bitsandbytes patch (no 4-bit quantization)", "install_no_bnb"),
            Choice("No — keep CPU-only configuration", "skip"),
        ],
        style=questionary.Style([("highlighted", "reverse")]),
    ).ask()
    if choice is None:
        choice = "skip"
except Exception:
    print("Select action:")
    print("  [1] Yes — full setup with 4-bit quantization support (recommended)")
    print("  [2] Yes — ROCm only, skip bitsandbytes patch")
    print("  [3] No — keep CPU-only configuration")
    while True:
        try:
            raw = input("Enter number [1]: ").strip()
            if not raw:
                choice = "install"; break
            idx = int(raw)
            if idx == 1: choice = "install"; break
            if idx == 2: choice = "install_no_bnb"; break
            if idx == 3: choice = "skip"; break
            print("Please enter 1, 2, or 3.")
        except (ValueError, EOFError):
            choice = "install"; break

if choice == "skip":
    print("Keeping CPU-only configuration.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# 5. Swap pyproject.toml + uv.lock and run `uv sync`.
# ---------------------------------------------------------------------------
print(f"\nActivating {_rdna.upper()} configuration...")

_pyproject_variant = os.path.join(_repo_root, f"pyproject.{_rdna}.toml")
_lock_variant      = os.path.join(_repo_root, f"uv.lock.{_rdna}")

if os.path.isfile(_pyproject_variant) and os.path.isfile(_lock_variant):
    # Preferred path: swap pre-generated arch-specific files, then uv sync.
    try:
        shutil.copy2(_pyproject_variant, os.path.join(_repo_root, "pyproject.toml"))
        shutil.copy2(_lock_variant,      os.path.join(_repo_root, "uv.lock"))
    except OSError as e:
        print(f"ERROR: Could not swap configuration files: {e}")
        sys.exit(1)

    print(f"Installing ROCm wheels for {generation_name}...")
    print("This may take several minutes on first run (subsequent runs use the local cache).\n")

    try:
        # --frozen: use the lock exactly; no re-resolution.
        # --no-install-project: skip reinstalling heretic-llm itself.
        #   Since this script runs BEFORE heretic.exe is launched,
        #   the exe is not locked — but we still skip it to save time
        #   (project code hasn't changed, only its deps).
        subprocess.check_call(["uv", "sync", "--frozen", "--no-install-project"],
                              cwd=_repo_root)
        print("\nROCm wheels installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: uv sync failed (exit {e.returncode}).")
        print("Please run 'uv sync' manually and then restart this script.")
        sys.exit(1)
    except FileNotFoundError:
        print("\nERROR: 'uv' not found. Please install uv: https://docs.astral.sh/uv/")
        sys.exit(1)

else:
    # Fallback: direct wheel install (non-git / pip-installed heretic).
    print("(Variant lock files not found — falling back to direct wheel install)")
    _py       = f"{sys.version_info.major}{sys.version_info.minor}"
    _base     = f"{_AMD_BASE}/{arch_target}"
    _torch_url    = f"{_base}/torch-{_TORCH_VERSION}%2B{_ROCM_TAG}-cp{_py}-cp{_py}-win_amd64.whl"
    _sdk_core_url = f"{_base}/rocm_sdk_core-{_ROCM_VERSION}-py3-none-win_amd64.whl"
    _sdk_libs_url = f"{_base}/rocm_sdk_libraries_{lib_suffix}-{_ROCM_VERSION}-py3-none-win_amd64.whl"

    print(f"Installing ROCm wheels for {generation_name}...")
    try:
        subprocess.check_call(
            ["uv", "pip", "install",
             "--extra-index-url", _base + "/",
             "--index-strategy", "unsafe-best-match",
             _torch_url, _sdk_core_url, _sdk_libs_url],
            cwd=_repo_root,
        )
        print("ROCm wheels installed successfully!")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: Installation failed: {e}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Patch bitsandbytes.
# ---------------------------------------------------------------------------
if choice == "install":
    _patch_script = os.path.join(_repo_root, "scripts", "patch_bitsandbytes.py")
    if os.path.isfile(_patch_script):
        print("\nPatching bitsandbytes for Windows ROCm DLL support...")
        try:
            subprocess.check_call([sys.executable, _patch_script], cwd=_repo_root)
        except subprocess.CalledProcessError as e:
            print(f"WARNING: bitsandbytes patch failed (exit {e.returncode}).")
            print("  4-bit quantization may not be available.")
            print("  You can retry manually: python scripts/patch_bitsandbytes.py")
    else:
        print(f"WARNING: Patch script not found at {_patch_script}")
else:
    print("Skipping bitsandbytes patch (4-bit quantization will not be available).")

# ---------------------------------------------------------------------------
# 7. Write arch marker.
# ---------------------------------------------------------------------------
try:
    with open(_marker_path, "w", encoding="utf-8") as _mf:
        _mf.write(_rdna)
except OSError:
    pass  # Non-fatal.

print(f"\nSetup complete! Run heretic with:\n  uv run --frozen heretic <model-id>\n")
