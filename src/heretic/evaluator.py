# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

import atexit
import logging
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError

import torch.nn.functional as F
from torch import Tensor

from .config import Settings
from .model import Model
from .utils import Prompt, load_prompts, print

logger = logging.getLogger(__name__)


class PendingScore:
    """Holds GPU results and a background LLM judge future for pipelined evaluation."""

    def __init__(
        self,
        evaluator: Evaluator,
        kl_divergence: float,
        responses: list[str],
        judge_future: Future[list[bool] | None] | None,
    ) -> None:
        self._evaluator = evaluator
        self.kl_divergence = kl_divergence
        self._responses = responses
        self._judge_future = judge_future

    def resolve(
        self, timeout: float | None = None
    ) -> tuple[tuple[float, float], float, int]:
        """Block until LLM judge completes and compute final score.

        Args:
            timeout: Maximum seconds to wait for the LLM judge future.
                     None means wait indefinitely. On timeout, falls back
                     to substring matching.
        """
        ev = self._evaluator

        refusal_flags: list[bool] | None = None
        if self._judge_future is not None:
            try:
                refusal_flags = self._judge_future.result(timeout=timeout)
            except TimeoutError:
                logger.warning(
                    "LLM judge timed out after %.1fs, falling back to substring",
                    timeout,
                )
            except Exception:
                logger.warning("Pipelined LLM judge raised", exc_info=True)

        ev._last_used_llm_judge = refusal_flags is not None

        refusals = 0
        for i, response in enumerate(self._responses):
            is_ref = (
                refusal_flags[i]
                if refusal_flags is not None
                else ev.is_refusal(response)
            )
            if is_ref:
                refusals += 1

            if ev.settings.print_responses:
                prompt = ev.bad_prompts[i]
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                disp = "[italic]\\[empty][/]" if not response.strip() else response
                print(f"[bold]Response:[/] [{'red' if is_ref else 'green'}]{disp}[/]")

        if ev.settings.print_responses:
            print()

        if ev._last_used_llm_judge and ev._base_refusals_llm is not None:
            base = ev._base_refusals_llm
        else:
            base = ev._base_refusals_substring

        refusals_score = refusals / base if base > 0 else float(refusals)
        kl_target = ev.settings.kl_divergence_target
        kl_scale = ev.settings.kl_divergence_scale

        if self.kl_divergence >= kl_target:
            kld_score = self.kl_divergence / kl_scale
        else:
            kld_score = refusals_score * kl_target / kl_scale

        score = (kld_score, refusals_score)
        return score, self.kl_divergence, refusals


class Evaluator:
    settings: Settings
    model: Model
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    base_logprobs: Tensor
    base_refusals: int

    def __init__(self, settings: Settings, model: Model) -> None:
        self.settings = settings
        self.model = model
        self._judge_executor = ThreadPoolExecutor(max_workers=1)
        atexit.register(self._judge_executor.shutdown, wait=False)

        # Track dual baselines for score consistency across LLM judge fallback
        self._base_refusals_llm: int | None = None
        self._base_refusals_substring: int = 0
        self._last_used_llm_judge: bool = False

        # Check LLM judge dependency upfront so users know immediately
        if settings.use_llm_judge:
            try:
                import httpx  # noqa: F401
            except ImportError:
                print(
                    "[bold yellow]WARNING: use_llm_judge is enabled but httpx is not installed.[/]"
                )
                print("[yellow]Install with: pip install heretic-llm\\[llm-judge][/]")
                print(
                    "[yellow]Falling back to substring matching for refusal classification.[/]"
                )
                settings.use_llm_judge = False

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
            self.bad_prompts,
            skip_special_tokens=True,
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
                    self._base_refusals_llm,
                    self._base_refusals_substring,
                )
            else:
                self.base_refusals = self._base_refusals_substring
                logger.warning(
                    "Baseline LLM judge failed, using substring (%d)",
                    self.base_refusals,
                )
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
                refusal_flags[i]
                if refusal_flags is not None
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

    def start_evaluation(self) -> PendingScore:
        """Run GPU work, submit LLM judge async, return pending score.

        The returned PendingScore can be resolved later (after the caller
        has started the next trial's GPU work) to get the final score.
        """
        # GPU: generate responses for bad prompts
        print("  * Counting model refusals...")
        responses = self.model.get_responses_batched(
            self.bad_prompts,
            skip_special_tokens=True,
        )

        # Submit LLM judge to background thread (non-blocking)
        judge_future: Future[list[bool] | None] | None = None
        if self.settings.use_llm_judge:
            judge_future = self._judge_executor.submit(
                self._try_llm_judge,
                responses,
            )

        # GPU: logprobs for good prompts (overlaps with LLM judge)
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        return PendingScore(self, kl_divergence, responses, judge_future)

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        """Synchronous evaluation (backward compatible)."""
        pending = self.start_evaluation()
        score, kl_divergence, refusals = pending.resolve()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")
        return score, kl_divergence, refusals
