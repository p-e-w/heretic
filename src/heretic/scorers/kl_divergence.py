# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>


import torch.nn.functional as F
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.scorer import Context, Score, Scorer
from heretic.utils import load_prompts, print


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

    def setup(self, ctx: Context) -> None:
        print()
        print(
            f"Loading KLDivergence evaluation prompts from [bold]{self.settings.prompts.dataset}[/]..."
        )
        self.prompts = load_prompts(self.heretic_settings, self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        print("* Obtaining baseline first-token probability distributions...")
        baseline_logits = ctx.get_logits(self.prompts)

        self._baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)

    def get_score(self, ctx: Context) -> Score:
        logits = ctx.get_logits(self.prompts)
        logprobs = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(
            logprobs,
            self._baseline_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        return Score(
            name=self.__class__.__name__,
            value=kl,
            cli_display=f"{kl:.4f}",
            hf_display=f"{kl:.4f}"
        )
