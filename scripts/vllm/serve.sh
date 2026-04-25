#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# Starts vLLM with an OpenAI-compatible HTTP API (default: http://localhost:8000).
# Requires: pip install -r requirements-vllm.txt (separate env recommended; GPU + CUDA).

set -euo pipefail

MODEL="${HERETIC_VLLM_MODEL:-p-e-w/gemma-3-12b-it-heretic}"

echo "Serving ${MODEL} — extra args are forwarded to vLLM."
exec vllm serve "${MODEL}" "$@"
