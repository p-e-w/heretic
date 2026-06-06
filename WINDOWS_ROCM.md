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
*(Note: By default, this installs a lightweight CPU-only environment to keep the initial setup fast and prevent downloading incorrect GPU libraries).*

### 3. Automatically Set Up AMD ROCm & bitsandbytes

Simply run `heretic` using `uv run`. Heretic will automatically detect your AMD GPU, prompt you to install the correct ROCm packages for your architecture, and configure 4-bit bitsandbytes quantization:

```powershell
uv run heretic --model <model-id>
```

When prompted, choose **Yes**. Heretic will synchronize your environment using the appropriate GPU extra (e.g. `uv sync --extra rocm-rdna3`), run the bitsandbytes patch script, and restart the process automatically.

---

### Alternative: Install ROCm packages directly on first setup

If you want to skip the first-run prompt and install the GPU packages directly, specify your GPU architecture extra during your initial sync:

```powershell
# For RDNA2 (Radeon RX 6000-series / gfx103X):
uv sync --extra rocm-rdna2

# For RDNA3 (Radeon RX 7000-series / gfx110X):
uv sync --extra rocm-rdna3

# For RDNA4 (Radeon RX 9000-series / gfx120X):
uv sync --extra rocm-rdna4
```
*(Once completed, you can run `uv run heretic` normally without any extra flags).*

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
  --exclude-newer-package rocm=false `
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

> **Tip:** You can safely run using `uv run heretic` or `uv run python`. Thanks to the architecture-aware
> extras in `pyproject.toml`, `uv` will respect your installed ROCm wheels and will not overwrite them.

---

## Running heretic

Once installed, launch heretic using `uv run`:

```powershell
uv run heretic --model <model-id>
```

Example with a small test model:

```powershell
uv run heretic --model Qwen/Qwen2.5-0.5B-Instruct
```

> **Important:** Always run from a proper **PowerShell** or **cmd** terminal — not from an IDE terminal or subprocess. heretic uses an interactive TUI that requires a real Windows console.

---

## Known Limitations on Windows

| Feature | Status | Notes |
|---|---|---|
| GPU inference (FP16/BF16) | ✅ Working | Full speed (gfx1030, gfx110X, gfx120X kernels supported) |
| `--quantization NONE` | ✅ Working | Works universally |
| `--quantization bnb_4bit` | ⚠️ RDNA2 only | The precompiled DLL targets `gfx1030` (RDNA2). RDNA3/RDNA4 users require building `bitsandbytes` from source to run 4-bit. |
| `torchvision` / `torchaudio` | ⚠️ Not installed | Not required by heretic; install separately if needed |

---

## How It Works (Technical Details)

To ensure maximum hardware coverage on Windows, heretic uses a dynamic setup pipeline:

1. **Dynamic GPU Detection:** On startup, the launcher inspects the Windows system to determine which AMD Radeon GPU generation is installed (RDNA2, RDNA3, or RDNA4).
2. **Dynamic Wheel Selection:** The launcher downloads the corresponding PyTorch and ROCm SDK wheels from `repo.amd.com/rocm/whl/`. RDNA2 gets `gfx103X-all` wheels, RDNA3 gets `gfx110X-all`, and RDNA4 gets `gfx120X-all` wheels. This ensures that the installed PyTorch has native, optimized kernels matching your exact GPU architecture.
3. **bitsandbytes Patch:** `bitsandbytes` raises a `RuntimeError` at import time on Windows ROCm because standard packages do not bundle `libbitsandbytes_rocm*.dll`. heretic automatically patches `bitsandbytes` by copying a precompiled library to `libbitsandbytes_rocm83.dll` inside the package folder. The patch script dynamically detects your GPU architecture (e.g. `gfx1100` for RDNA3) and looks for a matching `libbitsandbytes_rocm_<arch>.dll` in the `bin/` directory. If a specific architecture DLL does not exist, it falls back to the RDNA2 `libbitsandbytes_rocm_gfx1030.dll`.

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
AMD index flags. Use the full command from step 3 (with `--extra-index-url`,
`--index-strategy unsafe-best-match`, and `--exclude-newer-package rocm=false`).

**`HIP error: invalid device function`** — The wrong torch wheel is installed (one without gfx1030
kernels). Re-run step 3 with `--force-reinstall` appended.

**`NoConsoleScreenBufferError`** — You are running heretic from an IDE subprocess. Open a real
PowerShell/cmd window and run it there.

**GPU not detected / `torch.cuda.is_available()` returns `False`** — Make sure you have installed the GPU package extras (e.g. `uv sync --extra rocm-rdna2`, `--extra rocm-rdna3`, or `--extra rocm-rdna4`) and not just the default CPU fallback sync.

