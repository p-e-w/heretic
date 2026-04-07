# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# POST /v1/chat/completions with one text + one image URL (requires vLLM server running).

$ErrorActionPreference = "Stop"

$baseUrl = if ($env:HERETIC_VLLM_URL) { $env:HERETIC_VLLM_URL.TrimEnd("/") } else { "http://localhost:8000" }
$model = if ($env:HERETIC_VLLM_MODEL) { $env:HERETIC_VLLM_MODEL } else { "p-e-w/gemma-3-12b-it-heretic" }

$body = [ordered]@{
    model    = $model
    messages = @(
        [ordered]@{
            role    = "user"
            content = @(
                [ordered]@{ type = "text"; text = "Describe this image in one sentence." }
                [ordered]@{
                    type      = "image_url"
                    image_url = [ordered]@{
                        url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                    }
                }
            )
        }
    )
}

$json = $body | ConvertTo-Json -Depth 20
Invoke-RestMethod -Uri "$baseUrl/v1/chat/completions" -Method Post -Body $json -ContentType "application/json; charset=utf-8"
