# Running heretic on Windows with an AMD GPU (RDNA 2/3/4)

This guide covers running `heretic` natively on **Windows 11** with **AMD RDNA2, RDNA3, and RDNA4 GPUs**
(RX 6000-series, RX 7000-series, and RX 9000-series — `gfx103X`, `gfx110X`, `gfx120X` architectures).

> **Status: Working ✅**  
> Verified on: AMD Radeon RX 6900 XT (gfx1030), Windows 11, Python 3.12, ROCm 7.13 / HIP 7.13.99004

---

## Prerequisites

- Windows 11 (22H2 or later)
- Python **3.12** — `uv` will install and manage this automatically
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- AMD Adrenalin driver **24.x** or later
- ~12 GB free disk space for ROCm SDK wheels

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/Matlan1/heretic-win-AMD.git
cd heretic-win-AMD
```

### 2. Install base dependencies

```powershell
uv sync
```

This installs a lightweight CPU-only environment (~200 MB). Fast, no GPU packages yet.

### 3. Run the one-time ROCm setup

```powershell
uv run python scripts/setup_rocm.py
```

The script will:
1. Detect your AMD GPU generation (RDNA2 / RDNA3 / RDNA4) automatically
2. Ask for confirmation before downloading anything
3. Swap in the pre-generated `pyproject.toml` + `uv.lock` for your architecture
4. Install the matching ROCm PyTorch and SDK wheels (~3 GB, cached after first run)
5. Patch `bitsandbytes` for Windows ROCm DLL support
6. Write a `.heretic_rocm_arch` marker so setup is never repeated

**First run:** several minutes (downloading ~3 GB of wheels).  
**Every run after:** instant — setup is permanent.

---

## Running heretic

Identical to upstream — no extra flags:

```powershell
uv run heretic <model-id>
```

Example:

```powershell
uv run heretic Qwen/Qwen2.5-0.5B-Instruct
```

> **Important:** Always run from a real **PowerShell** or **cmd** window — not an IDE terminal or subprocess. heretic uses an interactive TUI that requires a proper Windows console.

---

## Known Limitations on Windows

| Feature | Status | Notes |
|---|---|---|
| GPU inference (FP16/BF16) | ✅ Working | Full speed on RDNA2, RDNA3, and RDNA4 |
| `--quantization NONE` | ✅ Working | Default — works on all GPUs |
| `--quantization bnb_4bit` | ✅ Working | RDNA2 uses the bundled `gfx1030` DLL. RDNA3/RDNA4 fall back to the same DLL (functional, not arch-optimised). Build from source for full RDNA3/4 performance — see below. |
| `torchvision` / `torchaudio` | ⚠️ Not included | Not required by heretic; install separately if needed |

---

## How It Works

The standard `uv.lock` shipped in the repository is CPU-only (plain PyPI torch). This keeps
`uv sync` fast and avoids downloading GPU packages that may not match the user's hardware.

`setup_rocm.py` replaces the CPU configuration with a GPU-specific one in three steps:

1. **Detect GPU generation** via `wmic` / PowerShell. Maps the GPU name to RDNA2, RDNA3, or RDNA4.
2. **Swap config files** — copies `pyproject.rdnaX.toml` → `pyproject.toml` and `uv.lock.rdnaX` → `uv.lock`. These pairs are pre-generated and committed to the repository, one per architecture. Torch is pinned as a direct wheel URL (not an index source) so that uv never needs to re-resolve dependencies.
3. **Sync and patch** — runs `uv sync --frozen --no-install-project` to install the ROCm wheels from the pre-built lock, then patches `bitsandbytes` by copying a precompiled `libbitsandbytes_rocm_<arch>.dll` into the package folder.

After setup, `uv run heretic` works identically to upstream — the pyproject and lock are consistent, so uv installs any missing packages and launches immediately.

---

## Compiling bitsandbytes from Source (Optional)

The bundled `libbitsandbytes_rocm_gfx1030.dll` (RDNA2) works as a fallback for all architectures. For best performance on RDNA3/RDNA4, compile a native DLL:

### Prerequisites
- **Visual Studio 2022** with the **"Desktop development with C++"** workload
- **AMD ROCm SDK for Windows** (v6.x or v7.x)
- **CMake** and **Ninja** on your system PATH

### Steps

**1.** Open the **x64 Native Tools Command Prompt for VS 2022** and set ROCm paths:

```cmd
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.13.0
set HIP_PATH=%ROCM_PATH%
set PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib;%PATH%
```

**2.** Clone and configure bitsandbytes:

```cmd
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
```

Pick your architecture:
- RDNA2 → `-DBNB_ROCM_ARCH="gfx1030"`
- RDNA3 → `-DBNB_ROCM_ARCH="gfx1100"`
- RDNA4 → `-DBNB_ROCM_ARCH="gfx1200"`

```cmd
cmake -G Ninja -B build -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1100" -DCMAKE_BUILD_TYPE=Release
```

**3.** Build:

```cmd
cmake --build build --config Release
```

**4.** Copy the resulting `libbitsandbytes_rocm.dll` into the repo's `bin/` folder:
- RDNA2 → `bin/libbitsandbytes_rocm_gfx1030.dll`
- RDNA3 → `bin/libbitsandbytes_rocm_gfx1100.dll`
- RDNA4 → `bin/libbitsandbytes_rocm_gfx1200.dll`

**5.** Re-run the patch script:

```powershell
uv run python scripts/patch_bitsandbytes.py
```

---

## Troubleshooting

**`No such file or directory: scripts/setup_rocm.py`** — Your clone is outdated. Run `git pull` and retry.

**`HIP error: invalid device function` / `hipErrorInvalidImage`** — The wrong arch wheels are installed (e.g. RDNA3 wheels on an RDNA2 GPU). Delete `.heretic_rocm_arch` and re-run `uv run python scripts/setup_rocm.py`.

**`WARNING: An AMD GPU was detected, but ROCm is not configured`** — Setup has not been run yet, or `.heretic_rocm_arch` was deleted. Run `uv run python scripts/setup_rocm.py`.

**`GPU not detected / torch.cuda.is_available() returns False`** — Verify your AMD Adrenalin driver is 24.x or later. Then delete `.heretic_rocm_arch` and re-run setup.

**`NoConsoleScreenBufferError`** — Run heretic from a real PowerShell or cmd window, not an IDE terminal.

**Setup ran but wrong arch was installed** — Override detection with the environment variable:
```powershell
$env:HERETIC_FORCE_ARCH = "rdna3"   # or rdna2, rdna4
uv run python scripts/setup_rocm.py
```
