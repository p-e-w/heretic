# Contributing to Heretic

Thank you for your interest in contributing to Heretic! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended for dependency management)
- A CUDA-capable GPU (for testing model-related changes)

### Setting Up the Development Environment

1. Fork the repository on GitHub.

2. Clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/heretic.git
   cd heretic
   ```

3. Install dependencies using uv:

   ```bash
   uv sync --all-extras --dev
   ```

4. Verify the installation:

   ```bash
   uv run heretic --help
   ```

## Development Workflow

### Code Style

This project enforces code style through automated tooling. Please follow these conventions:

- **Formatting**: Code is formatted with [Ruff](https://docs.astral.sh/ruff/). Run `uv run ruff format .` before committing.
- **Linting**: Imports are sorted and code is linted with Ruff. Run `uv run ruff check --extend-select I .` to check.
- **Type checking**: All code is type-checked with [ty](https://github.com/astral-sh/ty). Run `uv run ty check .` to verify.
- **Naming**: Avoid abbreviations in identifier names unless they are widely understood (e.g. "KL divergence").
- **Comments**: Comments should start with a capital letter and end with a period, using correct grammar and spelling.
- **Type annotations**: Function and method signatures must be fully type-annotated, including the return type.
- **SPDX headers**: Every Python source file must start with an SPDX/Copyright header.
- **Config settings**: When adding new settings in `config.py`, also add them to `config.default.toml` with their default value and description as a comment. The order in `config.default.toml` should match `config.py`.

### Running Checks

Before submitting a pull request, ensure all checks pass locally:

```bash
# Format check
uv run ruff format --check .

# Lint and import sort check
uv run ruff check --output-format=github --extend-select I .

# Type check
uv run ty check --output-format=github --error-on-warning .

# Build verification
uv build
```

The CI pipeline runs these checks on Python 3.10, 3.11, 3.12, and 3.13.

### Making Changes

1. Create a new branch from `master`:

   ```bash
   git checkout -b <branch-name> master
   ```

2. Make your changes, following the code style guidelines above.

3. Commit your changes with a descriptive message.

4. Push to your fork and open a pull request against the `master` branch.

## Pull Request Guidelines

- **One change per PR**: Each pull request should implement one change, and one change only. If your changes are semantically independent, split them into separate PRs.
- **Do not modify unrelated code**: Do not change existing code unless the changes are directly related to the PR. This includes formatting and comment changes to unrelated files.
- **Semantic PR title**: PR titles are validated by CI. Use a [Conventional Commits](https://www.conventionalcommits.org/) prefix:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `perf:` for performance improvements
  - `docs:` for documentation changes
  - `build:` for build system changes
- **Describe your changes**: Include a summary of what your PR does and why in the PR description.
- **Link related issues**: If your PR addresses an issue, reference it in the description (e.g. "Closes #123").

## Reporting Issues

- Search [existing issues](https://github.com/p-e-w/heretic/issues) before opening a new one.
- Include your environment details (OS, Python version, GPU, PyTorch version) when reporting bugs.
- Provide the full error output and steps to reproduce the problem.

## License

By contributing to this project, you agree to release your contributions under the [AGPL-3.0-or-later](LICENSE) license.
