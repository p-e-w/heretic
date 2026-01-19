# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from typing import cast

import torch.nn.functional as F
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.scorer import Context, Score, Scorer
from heretic.utils import load_prompts, print


class KLDivergence(Scorer):
    """
    KL divergence between current model and baseline on "good" prompts.

    Measures how much the model's behavior has drifted from baseline.
    Lower is better (less damage).
    """

    class Settings(BaseModel):
        prompts: DatasetSpecification = Field(
            default=DatasetSpecification(
                dataset="mlabonne/harmless_alpaca",
                split="test[:100]",
                column="text",
            ),
            description="Prompt set used to measure drift from baseline.",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ps = cast(KLDivergence.Settings, self.plugin_settings)

        print()
        print(
            f"Loading KLDivergence evaluation prompts from [bold]{ps.prompts.dataset}[/]..."
        )
        self.prompts = load_prompts(self.settings, ps.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        print("* Obtaining baseline first-token probability distributions...")
        self._baseline_logprobs = self.model.get_logprobs_batched(self.prompts)

    def get_score(self, ctx: Context) -> Score:
        logprobs = ctx.model.get_logprobs_batched(self.prompts)
        kl = F.kl_div(
            logprobs,
            self._baseline_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        return self.make_result(kl, f"{kl:.4f}")
