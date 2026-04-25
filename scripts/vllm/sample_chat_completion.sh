#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# POST /v1/chat/completions with one text + one image URL (requires vLLM server running).

set -euo pipefail

BASE_URL="${HERETIC_VLLM_URL:-http://localhost:8000}"
MODEL="${HERETIC_VLLM_MODEL:-p-e-w/gemma-3-12b-it-heretic}"

curl -sS -X POST "${BASE_URL%/}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data "$(cat <<EOF
{
  "model": "${MODEL}",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe this image in one sentence."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
          }
        }
      ]
    }
  ]
}
EOF
)"
