# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import logging

import torch.nn.functional as F
from torch import Tensor

from .config import Settings
from .model import Model
from .utils import Prompt, load_prompts, print

logger = logging.getLogger(__name__)


class Evaluator:
    settings: Settings
    model: Model
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    base_logprobs: Tensor
    base_refusals: int

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        # Track dual baselines for score consistency across LLM judge fallback
        self._base_refusals_llm: int | None = None
        self._base_refusals_substring: int = 0
        self._last_used_llm_judge: bool = False

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings, settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Counting model refusals...")
        base_responses = model.get_responses_batched(
            self.bad_prompts, skip_special_tokens=True,
        )

        # Always compute substring baseline
        self._base_refusals_substring = sum(
            1 for r in base_responses if self.is_refusal(r)
        )

        # Try LLM judge for baseline if enabled
        if settings.use_llm_judge:
            flags = self._try_llm_judge(base_responses)
            if flags is not None:
                self._base_refusals_llm = sum(flags)
                self.base_refusals = self._base_refusals_llm
                logger.info(
                    "Baseline: LLM judge=%d, substring=%d",
                    self._base_refusals_llm, self._base_refusals_substring,
                )
            else:
                self.base_refusals = self._base_refusals_substring
                logger.warning("Baseline LLM judge failed, using substring (%d)", self.base_refusals)
        else:
            self.base_refusals = self._base_refusals_substring

        if self.settings.print_responses:
            for prompt, response in zip(self.bad_prompts, base_responses):
                is_ref = self.is_refusal(response)
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                disp = "[italic]\\[empty][/]" if not response.strip() else response
                print(f"[bold]Response:[/] [{'red' if is_ref else 'green'}]{disp}[/]")
            print()

        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def _try_llm_judge(self, responses: list[str]) -> list[bool] | None:
        """Attempt LLM judge classification. Returns None on failure."""
        try:
            from .llm_judge import classify_refusals_batch

            prompt_texts = [p.user for p in self.bad_prompts]
            flags = classify_refusals_batch(prompt_texts, responses)
            if flags is not None:
                logger.info("LLM judge classified %d responses", len(flags))
            else:
                logger.warning("LLM judge returned None (all models exhausted)")
            return flags
        except ImportError:
            logger.warning("LLM judge module not available")
            return None
        except Exception:
            logger.warning("LLM judge unexpected error", exc_info=True)
            return None

    def is_refusal(self, response: str) -> bool:
        # Classify empty responses as refusals to avoid optimizing for them.
        if not response.strip():
            return True

        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won't" -> "won't").
        response = response.replace("\u2019", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def count_refusals(self) -> int:
        responses = self.model.get_responses_batched(
            self.bad_prompts,
            skip_special_tokens=True,
        )

        # Try LLM judge if enabled
        refusal_flags: list[bool] | None = None
        if self.settings.use_llm_judge:
            refusal_flags = self._try_llm_judge(responses)

        self._last_used_llm_judge = refusal_flags is not None

        refusal_count = 0
        for i, (prompt, response) in enumerate(zip(self.bad_prompts, responses)):
            is_refusal = (
                refusal_flags[i] if refusal_flags is not None
                else self.is_refusal(response)
            )
            if is_refusal:
                refusal_count += 1

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                disp = "[italic]\\[empty][/]" if not response.strip() else response
                print(
                    f"[bold]Response:[/] [{'red' if is_refusal else 'green'}]{disp}[/]"
                )

        if self.settings.print_responses:
            print()

        return refusal_count

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

        print("  * Counting model refusals...")
        refusals = self.count_refusals()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        kl_divergence_scale = self.settings.kl_divergence_scale
        kl_divergence_target = self.settings.kl_divergence_target

        # Use matching baseline to ensure score consistency:
        # LLM judge trial → LLM judge baseline, substring trial → substring baseline
        if self._last_used_llm_judge and self._base_refusals_llm is not None:
            base = self._base_refusals_llm
        else:
            base = self._base_refusals_substring

        refusals_score = (
            refusals / base if base > 0 else float(refusals)
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
