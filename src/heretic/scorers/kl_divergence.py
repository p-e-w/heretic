# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import re

import torch.nn.functional as F
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import format_dataset_specification, is_hf_path, print


class Settings(BaseModel):
    reference_model: str | None = Field(
        default=None,
        description="Reference checkpoint for KL. Defaults to the starting model.",
    )
    reference_model_commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the reference checkpoint.",
    )
    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts used to measure KL divergence from original model.",
    )


class KLDivergence(Scorer):
    """
    KL divergence between current model and baseline.

    Measures how much the model's behavior has drifted from baseline.
    Lower is better (less damage).
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return self.settings.reference_model is None or bool(
            is_hf_path(self.settings.reference_model)
            and re.fullmatch(
                r"[0-9a-fA-F]{40}", self.settings.reference_model_commit or ""
            )
        )

    @property
    def score_name(self) -> str:
        return "KL divergence"

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading KL divergence evaluation prompts from [bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        print("* Obtaining baseline first-token probability distributions...")
        if self.settings.reference_model is None:
            baseline_logits = ctx.get_logits(self.prompts)
        else:
            with ctx.get_model().reference_model(
                self.settings.reference_model,
                self.settings.reference_model_commit,
            ) as reference:
                baseline_logits = (
                    reference.get_logits_batched(self.prompts).float().cpu()
                )

        self._baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)

    def get_score(self, ctx: Context) -> Score:
        logits = ctx.get_logits(self.prompts)
        if self.settings.reference_model is not None:
            logits = logits.float()
        logprobs = F.log_softmax(logits, dim=-1)

        kl_divergence = F.kl_div(
            logprobs,
            self._baseline_logprobs.to(logprobs.device),
            reduction="batchmean",
            log_target=True,
        ).item()

        return Score(
            value=kl_divergence,
            rich_display=f"[bold]{kl_divergence:.4f}[/]",
            md_display=f"{kl_divergence:.4f}",
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        if self.settings.reference_model is not None:
            return self.get_score(ctx)
        return Score(
            value=0,
            rich_display="[bold]0[/] [italic](by definition)[/]",
            md_display="0 *(by definition)*",
        )
