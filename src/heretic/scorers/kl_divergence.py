from __future__ import annotations

from typing import cast

import torch.nn.functional as F
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.scorer import EvaluationContext, Score, Scorer
from heretic.utils import load_prompts, print


class KLDivergence(Scorer):
    """
    KL divergence between current model and baseline on "good" prompts.

    Measures how much the model's behavior has drifted from baseline.
    Lower is better (less damage).
    """

    name = "KLDivergence"

    class Settings(BaseModel):
        evaluation_prompts: DatasetSpecification = Field(
            default=DatasetSpecification(
                dataset="mlabonne/harmless_alpaca",
                split="test[:100]",
                column="text",
            ),
            description="Prompt set used to measure drift from baseline.",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ps = self.plugin_settings
        if ps is None:
            raise ValueError("KLDivergence requires plugin settings to be validated")
        ps = cast(KLDivergence.Settings, ps)

        print()
        print(
            f"Loading KLDivergence evaluation prompts from [bold]{ps.evaluation_prompts.dataset}[/]..."
        )
        self._eval_prompts = load_prompts(self.settings, ps.evaluation_prompts)
        print(f"* [bold]{len(self._eval_prompts)}[/] prompts loaded")

        print("* Obtaining baseline first-token probability distributions...")
        self._baseline_logprobs = self.model.get_logprobs_batched(self._eval_prompts)

    def evaluate(self, ctx: EvaluationContext) -> Score:
        logprobs = ctx.model.get_logprobs_batched(self._eval_prompts)
        kl = F.kl_div(
            logprobs,
            self._baseline_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        return self.make_result(kl, f"{kl:.4f}")
