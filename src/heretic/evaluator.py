# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import torch.nn.functional as F

from .config import Settings
from .model import Model
from .plugins import Plugin
from .utils import load_prompts, print


class Evaluator:
    def __init__(self, settings: Settings, model: Model, plugin: Plugin):
        self.settings = settings
        self.model = model
        self.plugin = plugin

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Calculating initial scores...")
        self.base_score = self.calculate_score()
        print(f"* Initial score: [bold]{self.base_score:.4f}[/]")

    def calculate_score(self) -> float:
        responses = self.model.get_responses_batched(self.bad_prompts)
        scores = self.plugin.score(responses)
        # We calculate the average score across all responses.
        # For refusal, this is the refusal rate (0.0 to 1.0).
        # For classifier, this is the average probability of the target label (0.0 to 1.0).
        avg_score = sum(scores) / len(scores)
        return avg_score

    def get_score(self) -> tuple[tuple[float, float], float, float]:
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.2f}[/]")

        print("  * Calculating steering score...")
        responses = self.model.get_responses_batched(self.bad_prompts)
        scores = self.plugin.score(responses)
        avg_score = sum(scores) / len(scores)
        
        print(f"  * Average Score: [bold]{avg_score:.4f}[/]")

        # Optimization Objective:
        # We want to minimize KL Divergence (stay close to original model).
        # We want to OPTIMIZE the steering metric.
        
        if self.settings.steering_mode == "refusal":
            # Minimize refusal rate.
            # Objective = (KL / scale) + (Refusal Rate / Base Refusal Rate)
            # Note: Base Refusal Rate might be 0, so we need to be careful.
            # The original code used (refusals / base_refusals).
            # If base_refusals is 0, we have a problem. But usually it's high.
            metric_term = avg_score / self.base_score if self.base_score > 0 else avg_score
        else:
            # Maximize target label (e.g. "joy").
            # So we minimize (1 - avg_score).
            # Objective = (KL / scale) + (1 - avg_score)
            metric_term = 1.0 - avg_score

        objective = (kl_divergence / self.settings.kl_divergence_scale) + metric_term

        return (objective, metric_term), kl_divergence, avg_score
