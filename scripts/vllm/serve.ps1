# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# Starts vLLM with an OpenAI-compatible HTTP API (default: http://localhost:8000).
# Requires: pip install -r requirements-vllm.txt (separate env recommended; GPU + CUDA).

$ErrorActionPreference = "Stop"

$model = if ($env:HERETIC_VLLM_MODEL) { $env:HERETIC_VLLM_MODEL } else { "p-e-w/gemma-3-12b-it-heretic" }

Write-Host "Serving $model — pass extra vLLM flags after the script name."
& vllm serve $model @args
