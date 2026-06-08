# Contributing to heretic (Windows AMD ROCm fork)

Thank you for your interest! This is a Windows AMD ROCm fork of
[p-e-w/heretic](https://github.com/p-e-w/heretic), modified to run natively
on Windows 11 with AMD RDNA2, RDNA3, and RDNA4 GPUs.

---

## What we accept

| Type | Examples |
|---|---|
| **Windows ROCm fixes** | Crash fixes, DLL issues, ROCm detection, safetensors workarounds |
| **GPU generation support** | New RDNA architecture support, wheel URL updates |
| **Upstream syncs** | Pulling new features or fixes from [p-e-w/heretic](https://github.com/p-e-w/heretic) |
| **Documentation** | Setup guides, troubleshooting entries, README improvements |
| **Setup UX** | Improvements to `setup_rocm.py` and the first-run prompt |

For issues that are not Windows/AMD-specific, please open them in the
[upstream repository](https://github.com/p-e-w/heretic/issues) instead.

---

## Development setup

**Requirements:** Windows 11, Python 3.12, [uv](https://docs.astral.sh/uv/),
AMD RDNA2/3/4 GPU (only needed for GPU-path testing).

```powershell
git clone https://github.com/Matlan1/heretic-win-AMD.git
cd heretic-win-AMD
uv sync --dev
```

For GPU functionality, run the one-time ROCm setup:

```powershell
uv run python scripts/setup_rocm.py
```

After setup, test your changes with:

```powershell
uv run heretic <model-id>
```

---

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for formatting and
linting. Run these before committing:

```powershell
uv run ruff format .
uv run ruff check --fix .
```

CI enforces both — PRs with formatting or lint failures will not be merged.

---

## Pull requests

- **PR titles** must follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
  `fix:`, `feat:`, `docs:`, `chore:`, `refactor:`, etc.
  The semantic-pr CI check enforces this automatically.

- **Description**: explain what the change does. For bug fixes, include
  reproduction steps. For ROCm-specific fixes, note which GPU generation
  you tested on.

- **CI must pass**: ruff format check, ruff lint, and `uv build`.

---

## Merging upstream changes

When syncing from [p-e-w/heretic](https://github.com/p-e-w/heretic):

1. The Windows AMD patches live in a clearly marked block near the top of
   `src/heretic/main.py` (sections 4–6). Keep them intact when rebasing.
2. Check that the `requires-python` and dependency versions in
   `pyproject.toml` are propagated to the three arch variants
   (`pyproject.rdna2.toml`, `pyproject.rdna3.toml`, `pyproject.rdna4.toml`)
   if the upstream changes them.

---

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml).
Always include your GPU model, driver version, and the full error output.
