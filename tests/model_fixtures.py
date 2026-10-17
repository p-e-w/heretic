# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from typing import Any, cast

import torch
from transformers import (
    BatchEncoding,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from heretic.config import Settings
from heretic.model import Model
from heretic.utils import Prompt


class TinyModel(Model):
    """Real transformer blocks with small integer prompts and no network access."""

    def __init__(
        self, hybrid: bool = False, device: str = "cpu", multimodal: bool = False
    ):
        self.settings = Settings.model_construct(
            model="tiny-local", batch_size=2, seed=123, response_prefix=""
        )
        options: dict[str, Any] = {
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "pad_token_id": 0,
            "eos_token_id": 31,
        }
        torch.manual_seed(7)
        if hybrid or multimodal:
            config = Qwen3_5TextConfig(
                **options,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                layer_types=["linear_attention", "full_attention", "linear_attention"],
            )
            if multimodal:
                self.model = Qwen3_5ForConditionalGeneration(
                    Qwen3_5Config(
                        text_config=config,
                        vision_config=Qwen3_5VisionConfig(
                            depth=1,
                            hidden_size=32,
                            intermediate_size=48,
                            num_heads=4,
                            out_hidden_size=32,
                        ),
                        image_token_id=29,
                        video_token_id=28,
                        vision_start_token_id=27,
                        vision_end_token_id=26,
                    )
                )
            else:
                self.model = Qwen3_5ForCausalLM(config)
        else:
            self.model = LlamaForCausalLM(LlamaConfig(**options))
        cast(torch.nn.Module, self.model).to(device).eval()
        self.model.config.name_or_path = self.settings.model
        self.dtype = torch.float32
        self.needs_reload = False

    def generate(self, prompts: list[Prompt], **kwargs: Any):
        rows = [[int(token) for token in prompt.user.split()] for prompt in prompts]
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long, device=self.model.device)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, -len(row) :] = torch.tensor(row, device=ids.device)
            mask[index, -len(row) :] = 1
        inputs = BatchEncoding({"input_ids": ids, "attention_mask": mask})
        return inputs, cast(Any, self.model).generate(
            **inputs,
            **kwargs,
            do_sample=False,
            pad_token_id=0,
        )


PROMPTS = [
    Prompt(system="", user=text)
    for text in [
        "1 2 3",
        "4 5",
        "6 7 8 9",
        "10",
        "11 12 13",
    ]
]
