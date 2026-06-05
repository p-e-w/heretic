# Running heretic on Windows with an AMD GPU (RDNA2 / gfx103X)

This guide covers running `heretic` natively on **Windows 11** with **AMD RDNA2 GPUs**
(RX 6700 XT, RX 6800, RX 6800 XT, RX 6900 XT, etc. — `gfx1030`/`gfx1031`/`gfx1032`).

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
git clone https://github.com/p-e-w/heretic.git
cd heretic
```

### 2. Create venv and install standard dependencies

```powershell
uv sync
```

### 3. Install AMD ROCm 7.13 PyTorch and SDK wheels

These wheels are served from AMD's official index and target the `gfx103X` RDNA2 family:

```powershell
uv pip install `
  "https://repo.amd.com/rocm/whl/gfx103X-all/torch-2.9.1%2Brocm7.13.0-cp312-cp312-win_amd64.whl" `
  "https://repo.amd.com/rocm/whl/gfx103X-all/rocm_sdk_core-7.13.0-py3-none-win_amd64.whl" `
  "https://repo.amd.com/rocm/whl/gfx103X-all/rocm_sdk_libraries_gfx103x_all-7.13.0-py3-none-win_amd64.whl"
```

> **Note:** `rocm_sdk_libraries` is ~250 MB and `rocm_sdk_core` is ~690 MB. The download will take a few minutes depending on your connection.

Verify the GPU is visible:

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
| GPU inference (FP16/BF16) | ✅ Working | Full speed, gfx1030 kernels confirmed |
| `--quantization NONE` | ✅ Working | |
| `--quantization bnb_4bit` | ❌ Not working | `bitsandbytes` does not ship ROCm Windows DLLs. Use `NONE`. |
| `torchvision` / `torchaudio` | ⚠️ Not installed | Not required by heretic; install separately if needed |

---

## How It Works (Technical Details)

The official AMD Python ROCm wheels for Windows (`repo.radeon.com`, ROCm 6.4.4) only target
**RDNA3/RDNA4** (`gfx1100`, `gfx1200`, etc.). RDNA2 (`gfx1030`) is excluded.

AMD's **TheRock** distribution (`repo.amd.com/rocm/whl/gfx103X-all/`) provides ROCm 7.13 wheels
compiled specifically for the `gfx103X` family and works on Windows via WDDM.

`bitsandbytes` raises a `RuntimeError` at import time on Windows ROCm because the Windows wheels
do not ship `libbitsandbytes_rocm*.dll`. This is harmless when `--quantization NONE` is used;
heretic handles it gracefully and raises a clear error only if 4-bit quantization is explicitly
requested.

---

## Troubleshooting

**`HIP error: invalid device function`** — The wrong torch wheel is installed (one without gfx1030
kernels). Re-run step 3 with `--force-reinstall`.

**`NoConsoleScreenBufferError`** — You are running heretic from an IDE subprocess. Open a real
PowerShell/cmd window and run it there.

**GPU not detected / `torch.cuda.is_available()` returns `False`** — You ran the verify step with
`uv run python` instead of `.venv\Scripts\python.exe`. `uv run` overwrites the ROCm wheels with
the CPU-only PyPI build. Re-run step 3 to reinstall the ROCm wheels, then verify with
`.venv\Scripts\python.exe` directly.
