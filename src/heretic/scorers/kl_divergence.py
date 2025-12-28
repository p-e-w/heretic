from __future__ import annotations

import torch.nn.functional as F

from heretic.schemas import EvaluationContext, MetricResult
from heretic.scorer import Scorer


class KLDivergence(Scorer):
    """
    KL divergence between current model and baseline on "good" prompts.

    Measures how much the model's behavior has drifted from baseline.
    Lower is better (less damage).
    """

    name = "KLDivergence"

    def evaluate(self, ctx: EvaluationContext) -> MetricResult:
        logprobs = ctx.model.get_logprobs_batched(ctx.good_prompts)
        kl = F.kl_div(
            logprobs,
            ctx.base_good_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        return self.make_result(kl, f"{kl:.4f}")
