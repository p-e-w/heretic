# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from torch import Tensor

from .config import Settings
from .model import Model
from .utils import Prompt, load_prompts, print


class Evaluator:
    settings: Settings
    model: Model
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    base_logprobs: Tensor
    base_probs: Tensor
    base_isneginf: Tensor
    base_refusals: int

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        # The base distribution is constant across all trials, so precompute the
        # quantities that kl_divergence() needs from it once, rather than on every call.
        self.base_probs = self.base_logprobs.exp()
        self.base_isneginf = self.base_logprobs.isneginf()

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings, settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def is_refusal(self, response: str) -> bool:
        # Classify empty responses as refusals to avoid optimizing for them.
        if not response.strip():
            return True

        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def count_refusals(self) -> int:
        refusal_count = 0

        responses = self.model.get_responses_batched(
            self.bad_prompts,
            skip_special_tokens=True,
        )

        for prompt, response in zip(self.bad_prompts, responses):
            is_refusal = self.is_refusal(response)
            if is_refusal:
                refusal_count += 1

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                print(
                    f"[bold]Response:[/] [{'red' if is_refusal else 'green'}]{response}[/]"
                )

        if self.settings.print_responses:
            print()

        return refusal_count

    def kl_divergence(self, logprobs: Tensor) -> float:
        # Compute the KL divergence D(base || abliterated) of the first-token
        # distributions, defined as the sum over the vocabulary of
        # P_base * (log P_base - log P_abliterated), averaged over the evaluation
        # prompts. This is equivalent to F.kl_div(logprobs, base_logprobs,
        # reduction="batchmean", log_target=True), but robust to tokens that the
        # generation pipeline forces to zero probability.
        #
        # Some models declare `suppress_tokens` in their generation config (for
        # example, the multimodal Gemma-4 models suppress the end-of-image and
        # end-of-audio tokens). The generation pipeline sets those tokens' logits to
        # -inf in both logprob tensors, so the naive term becomes
        # 0 * (-inf - -inf) = 0 * nan = nan, which poisons the entire sum and yields a
        # NaN KL divergence (https://github.com/p-e-w/heretic/issues/346). Those
        # positions carry zero probability mass under the base distribution, so by the
        # standard convention 0 * log(0/q) = 0 they must contribute nothing.
        kl_terms = self.base_probs * (self.base_logprobs - logprobs)
        kl_terms = kl_terms.masked_fill(self.base_isneginf, 0.0)
        return (kl_terms.sum() / self.base_logprobs.shape[0]).item()

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = self.kl_divergence(logprobs)
        print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        print("  * Counting model refusals...")
        refusals = self.count_refusals()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        kl_divergence_scale = self.settings.kl_divergence_scale
        kl_divergence_target = self.settings.kl_divergence_target

        refusals_score = (
            refusals / self.base_refusals if self.base_refusals > 0 else float(refusals)
        )

        if kl_divergence >= kl_divergence_target:
            kld_score = kl_divergence / kl_divergence_scale
        else:
            kld_score = refusals_score * kl_divergence_target / kl_divergence_scale

        score = (
            kld_score,
            refusals_score,
        )

        return score, kl_divergence, refusals
