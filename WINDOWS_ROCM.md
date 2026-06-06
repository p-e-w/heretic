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

### 3. Automatically Set Up AMD ROCm & bitsandbytes

Simply run `heretic` once. Heretic will detect your AMD GPU, prompt you to install the ROCm PyTorch wheels automatically, download them, and configure 4-bit bitsandbytes quantization:

```powershell
.venv\Scripts\heretic --model <model-id>
```

When prompted, choose **Yes** to automatically install the ROCm PyTorch and SDK wheels, copy the required Windows ROCm DLL, and patch the environment.

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
.venv\Scripts\python.exe scripts/patch_bitsandbytes.py
```

Verify that the GPU is detected successfully:
```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
# Expected: AMD Radeon RX 6900 XT
# Expected: ['gfx1030', 'gfx1031', ...]
```

> **Why not `uv run python`?** `uv run` automatically re-syncs the virtual environment against
> `pyproject.toml` before executing, which would overwrite the ROCm torch wheel with the standard
> CPU-only build from PyPI. Always use `.venv\Scripts\python.exe` or `.venv\Scripts\heretic.exe`
> directly after installing the ROCm wheels.

---

## Running heretic

Once installed, launch heretic using the virtual environment's executable directly:

```powershell
.venv\Scripts\heretic.exe --model <model-id> --quantization NONE
```

Example with a small test model:

```powershell
.venv\Scripts\heretic.exe --model Qwen/Qwen2.5-0.5B-Instruct --quantization NONE
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
3. **bitsandbytes Patch:** `bitsandbytes` raises a `RuntimeError` at import time on Windows ROCm because standard packages do not bundle `libbitsandbytes_rocm*.dll`. heretic automatically patches `bitsandbytes` with a precompiled `libbitsandbytes_rocm83.dll`. Note that this bundled DLL was compiled for `gfx1030` (RDNA2), so 4-bit quantization is currently limited to RDNA2 hardware. Using FP16/BF16 (`--quantization NONE`) is fully supported and works universally on all RDNA2, RDNA3, and RDNA4 GPUs.

---

## Troubleshooting

**`No solution found: rocm[libraries]==7.13.0`** — You ran `uv pip install` without the required
AMD index flags. Use the full command from step 3 (with `--extra-index-url`,
`--index-strategy unsafe-best-match`, and `--exclude-newer-package rocm=false`).

**`HIP error: invalid device function`** — The wrong torch wheel is installed (one without gfx1030
kernels). Re-run step 3 with `--force-reinstall` appended.

**`NoConsoleScreenBufferError`** — You are running heretic from an IDE subprocess. Open a real
PowerShell/cmd window and run it there.

**GPU not detected / `torch.cuda.is_available()` returns `False`** — You ran the verify step with
`uv run python` instead of `.venv\Scripts\python.exe`. `uv run` re-syncs the venv and overwrites
the ROCm wheels with the CPU-only PyPI build. Re-run step 3 to reinstall the ROCm wheels, then
verify with `.venv\Scripts\python.exe` directly.
