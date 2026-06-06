# Running heretic on Windows with an AMD GPU (RDNA 2/3/4)

This guide covers running `heretic` natively on **Windows 11** with **AMD RDNA2, RDNA3, and RDNA4 GPUs**
(RX 6000-series, RX 7000-series, and RX 9000-series — `gfx103X`, `gfx110X`, `gfx120X` architectures).

> **Status: Working ✅**  
> Verified on: AMD Radeon RX 6900 XT (gfx1030), Windows 11, Python 3.12, ROCm 7.13 / HIP 7.13.99004

---

## Prerequisites

- Windows 11 (22H2 or later)
- Python **3.12** (exact — `uv` will enforce this)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- AMD Adrenalin driver **24.x** or later (shipped with WDDM 3.x support)
- ~12 GB free disk space for ROCm SDK wheels

---

## Installation

### 1. Clone and enter the repository

```powershell
git clone https://github.com/Matlan1/heretic-win-AMD.git
cd heretic-win-AMD
```

### 2. Create venv and install standard dependencies

```powershell
uv sync
```
*(By default, this installs a lightweight CPU-only environment to keep the initial setup fast.)*

### 3. Run the AMD ROCm setup script

```powershell
uv run python scripts/setup_rocm.py
```

This script automatically:
1. Detects your AMD GPU architecture (RDNA2 / RDNA3 / RDNA4)
2. Swaps in the matching pre-generated `pyproject.rdnaX.toml` + `uv.lock.rdnaX`
3. Runs `uv sync --frozen` to install the correct ROCm PyTorch and SDK wheels
4. Patches `bitsandbytes` for Windows ROCm DLL support
5. Writes a `.heretic_rocm_arch` marker so setup is skipped on future runs

The first run may take several minutes (downloading ~3 GB of wheels).
**Subsequent runs start immediately** — the setup is permanent.

> **Why a separate script?** On Windows, `heretic.exe` cannot update its own dependencies
> via `uv sync` while it is running (Windows file-lock on running executables). The pre-flight
> script runs as plain Python — before heretic launches — avoiding this issue entirely.

---

### Alternative: Manual ROCm setup (non-git installs)

If you installed heretic via `pip install heretic-llm` (not via git clone), the variant files
won't be present. Install wheels directly:

```powershell
# For RDNA2 (Radeon RX 6000-series / gfx103X):
uv pip install --python .venv/Scripts/python.exe `
  --extra-index-url https://repo.amd.com/rocm/whl/gfx103X-all/ `
  --index-strategy unsafe-best-match `
  "https://repo.amd.com/rocm/whl/gfx103X-all/torch-2.9.1%2Brocm7.13.0-cp312-cp312-win_amd64.whl" `
  "https://repo.amd.com/rocm/whl/gfx103X-all/rocm_sdk_core-7.13.0-py3-none-win_amd64.whl" `
  "https://repo.amd.com/rocm/whl/gfx103X-all/rocm_sdk_libraries_gfx103x_all-7.13.0-py3-none-win_amd64.whl"

# For RDNA3 (Radeon RX 7000-series / gfx110X): replace gfx103X-all with gfx110X-all and gfx103x_all with gfx110x_all
# For RDNA4 (Radeon RX 9000-series / gfx120X): replace with gfx120X-all / gfx120x_all
```
*(Replace `cp312` with your Python version, e.g. `cp311` for Python 3.11.)*

Then patch bitsandbytes manually:
```powershell
uv run python scripts/patch_bitsandbytes.py
```

---

### Manual Installation (Alternative)

If you prefer to install the wheels and patch bitsandbytes manually, run the following:

#### A. Install AMD ROCm PyTorch and SDK wheels (Example for RDNA2)
```powershell
# For RDNA2 (gfx103X), use gfx103X-all and rocm_sdk_libraries_gfx103x_all
# For RDNA3 (gfx110X), use gfx110X-all and rocm_sdk_libraries_gfx110x_all
# For RDNA4 (gfx120X), use gfx120X-all and rocm_sdk_libraries_gfx120x_all
uv pip install `
  --extra-index-url https://repo.amd.com/rocm/whl/gfx103X-all/ `
  --index-strategy unsafe-best-match `
  "https://repo.amd.com/rocm/whl/gfx103X-all/torch-2.9.1%2Brocm7.13.0-cp312-cp312-win_amd64.whl" `
  "https://repo.amd.com/rocm/whl/gfx103X-all/rocm_sdk_core-7.13.0-py3-none-win_amd64.whl" `
  "https://repo.amd.com/rocm/whl/gfx103X-all/rocm_sdk_libraries_gfx103x_all-7.13.0-py3-none-win_amd64.whl"
```
*(Note: Replace `cp312` with your Python version, e.g. `cp311` for Python 3.11, and adjust target wheel directories/filenames for RDNA3/4 as noted in the comments).*

#### B. Patch bitsandbytes for 4-bit quantization support
```powershell
uv run python scripts/patch_bitsandbytes.py
```

Verify that the GPU is detected successfully:
```powershell
uv run python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
# Expected: AMD Radeon RX 6900 XT
# Expected: ['gfx1030', 'gfx1031', ...]
```

> **How auto-setup works:** `setup_rocm.py` detects your AMD GPU generation (RDNA2/3/4),
> copies the matching pre-generated `pyproject.rdnaX.toml` → `pyproject.toml` and `uv.lock.rdnaX` → `uv.lock`,
> then runs `uv sync --frozen --no-install-project` to install the correct ROCm wheels from the pre-built lock.
> After patching `bitsandbytes`, a `.heretic_rocm_arch` marker file is written to prevent the setup from
> running again. The setup is permanent until a `git pull` restores the default CPU lock files.

---

## Running heretic

After running `setup_rocm.py`, launch heretic with:

```powershell
uv run --frozen heretic <model-id>
```

Example with a small test model:

```powershell
uv run --frozen heretic Qwen/Qwen2.5-0.5B-Instruct
```

> **Why `--frozen`?** After ROCm setup, the `uv.lock` contains ROCm-specific wheels that are
> Windows-only. Without `--frozen`, uv tries to re-resolve the lock for all supported platforms
> (including platforms without ROCm triton wheels) and fails. `--frozen` tells uv to use the
> lock exactly as-is and skip re-resolution.

> **Important:** Always run from a proper **PowerShell** or **cmd** terminal — not from an IDE terminal or subprocess. heretic uses an interactive TUI that requires a real Windows console.

---

## Known Limitations on Windows

| Feature | Status | Notes |
|---|---|---|
| GPU inference (FP16/BF16) | ✅ Working | Full speed (gfx1030, gfx110X, gfx120X kernels supported) |
| `--quantization NONE` | ✅ Working | Works universally |
| `--quantization bnb_4bit` | ✅ Working (all) | RDNA2 uses the bundled `gfx1030` DLL natively. RDNA3/RDNA4 fall back to the same `gfx1030` DLL (functional but not architecture-optimised). Build from source for maximum RDNA3/4 performance — see below. |
| `torchvision` / `torchaudio` | ⚠️ Not installed | Not required by heretic; install separately if needed |

---

## How It Works (Technical Details)

To ensure maximum hardware coverage on Windows, heretic uses a dynamic setup pipeline:

1. **GPU Detection:** `setup_rocm.py` inspects the Windows system to determine which AMD Radeon GPU generation is installed (RDNA2, RDNA3, or RDNA4).
2. **Arch-Specific Config Swap:** The script copies `pyproject.rdnaX.toml` → `pyproject.toml` and `uv.lock.rdnaX` → `uv.lock` for the detected architecture. These pre-generated pairs are committed to the repository — one per arch — and each locks the correct ROCm PyTorch and SDK wheels as default (non-extra) dependencies.
3. **`uv sync --frozen --no-install-project`:** With the correct lock in place, uv installs the right arch-specific wheels cleanly. `--frozen` prevents re-resolution; `--no-install-project` skips reinstalling heretic itself (saving time, since project code hasn't changed). Every future `uv run --frozen heretic` starts immediately — the setup is permanent.
4. **bitsandbytes Patch:** `bitsandbytes` raises a `RuntimeError` at import time on Windows ROCm because standard packages do not bundle `libbitsandbytes_rocm*.dll`. The patch script copies a precompiled library to `libbitsandbytes_rocm83.dll` inside the package folder. It dynamically detects your GPU architecture (e.g. `gfx1100` for RDNA3) and looks for a matching `libbitsandbytes_rocm_<arch>.dll` in the `bin/` directory. If a specific architecture DLL does not exist, it falls back to the RDNA2 `libbitsandbytes_rocm_gfx1030.dll`.

---

## Compiling bitsandbytes for RDNA2, RDNA3, or RDNA4

If you want to compile the ROCm backend DLL for your exact architecture from source, follow these steps:

### 1. Prerequisites
* **Visual Studio 2022** with the **"Desktop development with C++"** workload installed.
* **AMD ROCm SDK for Windows** installed (v6.x or v7.x).
* **CMake** and **Ninja** installed and available in your system path.

### 2. Configure Build Environment
Open the **x64 Native Tools Command Prompt for VS 2022** (do not use PowerShell or normal cmd, as MSVC compiler variables are required). Set the ROCm paths (adjust paths if your installed version differs):
```cmd
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.13.0
set HIP_PATH=%ROCM_PATH%
set PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib;%PATH%
```

### 3. Clone and Configure bitsandbytes
```cmd
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
```
Run CMake with the target architecture corresponding to your GPU generation:
* For **RDNA2**: Use `-DBNB_ROCM_ARCH="gfx1030"`
* For **RDNA3**: Use `-DBNB_ROCM_ARCH="gfx1100"`
* For **RDNA4**: Use `-DBNB_ROCM_ARCH="gfx1200"`

Example (for RDNA3):
```cmd
cmake -G Ninja -B build -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1100" -DCMAKE_BUILD_TYPE=Release
```

### 4. Build the DLL
Compile the ROCm library DLL:
```cmd
cmake --build build --config Release
```
The compiled DLL (typically `libbitsandbytes_rocm.dll`) will be generated inside the build outputs directory. 

### 5. Install the DLL
Rename and copy the compiled DLL into your `heretic-win-AMD/bin/` folder using your architecture identifier:
* For **RDNA2**: Copy to `bin/libbitsandbytes_rocm_gfx1030.dll`
* For **RDNA3**: Copy to `bin/libbitsandbytes_rocm_gfx1100.dll`
* For **RDNA4**: Copy to `bin/libbitsandbytes_rocm_gfx1200.dll`

Once copied, re-run `uv run python scripts/patch_bitsandbytes.py`. The patcher will automatically detect your GPU architecture and use your compiled DLL.

---

## Troubleshooting

**`No solution found: rocm[libraries]==7.13.0`** — You ran `uv pip install` without the required
AMD index flags. Use the full command from the Alternative section (with `--extra-index-url`,
`--index-strategy unsafe-best-match`).

**`No solution found when resolving dependencies`** — You ran `uv run heretic` without `--frozen` after ROCm setup. The ROCm lock is Windows-only and cannot resolve on all platforms. Use `uv run --frozen heretic <model>` instead.

**`HIP error: invalid device function` / `hipErrorInvalidImage` / `hipErrorInvalidKernelFile`** — The wrong torch arch wheel is installed (e.g. RDNA3 wheels on an RDNA2 GPU). Delete `.heretic_rocm_arch` and re-run `uv run python scripts/setup_rocm.py` — it will detect the correct arch and install the right wheels.

**`NoConsoleScreenBufferError`** — You are running heretic from an IDE subprocess. Open a real
PowerShell/cmd window and run it there.

**`GPU not detected / torch.cuda.is_available() returns False`** — Make sure the correct arch-specific ROCm wheels are installed. Delete `.heretic_rocm_arch` and re-run `uv run python scripts/setup_rocm.py`, or install manually using direct wheel URLs as shown in the Alternative section above.
