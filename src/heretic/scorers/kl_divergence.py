# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>


import torch.nn.functional as F
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.scorer import Context, Score, Scorer
from heretic.utils import print


class Settings(BaseModel):
    prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Prompt set used to measure drift from baseline.",
    )


class KLDivergence(Scorer):
    """
    KL divergence between current model and baseline.

    Measures how much the model's behavior has drifted from baseline.
    Lower is better (less damage).
    """

    settings: Settings

    @property
    def score_name(self) -> str:
        return "KL divergence"

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading KLDivergence evaluation prompts from [bold]{self.settings.prompts.dataset}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        print("* Obtaining baseline first-token probability distributions...")
        baseline_logits = ctx.get_logits(self.prompts)

        self._baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)

    def get_score(self, ctx: Context) -> Score:
        logits = ctx.get_logits(self.prompts)
        logprobs = F.log_softmax(logits, dim=-1)
        # This occasionally returns a negative 0 floating point, so we force it to be positive
        # since KL divergence is always >= by definition.
        kl = abs(F.kl_div(
            logprobs,
            self._baseline_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item())
        return Score(
            name=self.score_name,
            value=kl,
            cli_display=f"{kl:.4f}",
            md_display=f"{kl:.4f}",
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        return Score(
            name=self.score_name,
            value=0,
            cli_display="0 (by definition)",
            md_display="0 *(by definition)*",
        )
