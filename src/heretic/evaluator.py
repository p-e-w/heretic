# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import torch.nn.functional as F

from .config import Settings
from .model import Model
from .utils import load_prompts, print


class Evaluator:
    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        print()
        print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
        self.good_prompts = load_prompts(settings.good_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
        self.bad_prompts = load_prompts(settings.bad_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)} ([bold]{self.base_refusals / len(self.bad_prompts) * 100:.1f}[/] %)"
        )

    def is_refusal(self, response: str) -> bool:
        # Remove emphasis (e.g. "I *will not*...") to facilitate detection.
        response = response.lower().replace("*", "")

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def count_refusals(self) -> int:
        responses = self.model.get_responses_batched(self.bad_prompts)
        refusals = [response for response in responses if self.is_refusal(response)]
        return len(refusals)

    def get_score(self) -> tuple[float, float, int]:
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = F.kl_div(
            logprobs, self.base_logprobs, reduction="batchmean", log_target=True
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]", end="")

        kl_score = -(
            (
                (
                    (kl_divergence - self.settings.max_kl_divergence)
                    / self.settings.max_kl_divergence
                )
                + 1
            )
            ** self.settings.kl_score_shape
        )

        if kl_divergence > self.settings.max_kl_divergence:
            print(" [yellow](constraint violation; aborting trial)[/]")
            return kl_score, kl_divergence, self.base_refusals
        else:
            print()

        print("  * Counting model refusals...")
        refusals = self.count_refusals()
        print(
            f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)} ([bold]{refusals / len(self.bad_prompts) * 100:.1f}[/] %)"
        )

        # This score is constructed to achieve several properties:
        #
        # 1. For the unmodified model, kl_divergence = 0 and refusals = base_refusals,
        #    so the baseline score is 0.
        #
        # 2. The best possible outcome is kl_divergence = 0 and refusals = 0,
        #    giving a score of 1.
        #
        # 3. If kl_divergence > max_kl_divergence, the score is negative.
        #    As the baseline is 0, this ensures that such a configuration
        #    is never chosen, enforcing the max_kl_divergence constraint.
        #
        # 4. kl_score_shape controls how strongly a kl_divergence well below
        #    max_kl_divergence affects the score. A high value means that
        #    kl_divergence only matters when it approaches max_kl_divergence,
        #    and the optimizer will prioritize lowering refusals rather than
        #    lowering kl_divergence.
        score = kl_score - (refusals / self.base_refusals) + 1
        print(f"  * Score: [bold]{score:.4f}[/]")

        return score, kl_divergence, refusals
