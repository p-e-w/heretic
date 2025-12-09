# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import torch.nn.functional as F

from .config import Settings
from .model import Model
from .scorer import Scorer
from .tagger import Tagger
from .utils import load_prompts, load_plugin, print


class Evaluator:
    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

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

        print()
        print("Loading tagger plugin...")
        self.tagger_plugin = self._load_tagger_plugin()
        self.model.set_requested_metadata_fields(
            self.tagger_plugin.required_response_metadata_fields()
        )
        print()
        print("Loading scorer plugin...")
        self.scorer_plugin = self._load_scorer_plugin()

        self.base_score = self.tag_and_score_batch()
        print(f"* Initial score: [bold]{self.base_score}[/]/{len(self.bad_prompts)}")

    def _load_tagger_plugin(self) -> Tagger:
        tagger = load_plugin(
            name=self.settings.tagger,
            base_class=Tagger,
            package="heretic",
            subpackage="taggers",
        )
        print(f"* Loaded tagger plugin: [bold]{tagger.__name__}[/bold]")
        self.model.set_requested_context_metadata_fields(
            tagger.required_context_metadata_fields()
        )
        return tagger(
            settings=self.settings,
            model=self.model,
            context_metadata=self.model.get_context_metadata(),
        )

    def _load_scorer_plugin(self) -> Scorer:
        scorer = load_plugin(
            name=self.settings.scorer,
            base_class=Scorer,
            package="heretic",
            subpackage="scorers",
        )
        print(f"* Loaded scorer plugin: [bold]{scorer.__name__}[/bold]")
        return scorer()

    def tag_and_score_batch(self) -> float:
        responses = self.model.get_responses_batched(self.bad_prompts)
        tags = self.tagger_plugin.tag_batch(responses=responses)
        score = self.scorer_plugin.score_batch(tags)
        return score

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        print("  * Counting model score...")
        refusals = self.tag_and_score_batch()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        score = (
            (kl_divergence / self.settings.kl_divergence_scale),
            (refusals / self.base_score),
        )

        return score, kl_divergence, refusals
