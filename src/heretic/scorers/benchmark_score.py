# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import lm_eval
from lm_eval.models.huggingface import HFLM
from pydantic import BaseModel, Field, NonNegativeInt

from heretic.scorer import Context, Score, Scorer


class Settings(BaseModel):
    score_name: str = Field(
        default="PIQA acc_norm",
        description="Name that describes what the configured benchmark score measures.",
    )

    task: str = Field(
        default="piqa",
        description="Task ID of the benchmark in the Language Model Evaluation Harness.",
    )

    metric: str = Field(
        default="acc_norm,none",
        description="Task metric to use as the benchmark score.",
    )

    batch_size: NonNegativeInt = Field(
        default=0,  # auto
        description="Number of input sequences to process in parallel (0 = auto).",
    )


class BenchmarkScore(Scorer):
    """
    Calculates the score of a benchmark from the Language Model Evaluation Harness.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def get_score(self, ctx: Context) -> Score:
        model = ctx.get_model()

        hflm = HFLM(
            pretrained=model.model,
            tokenizer=model.tokenizer,  # ty: ignore[invalid-argument-type]
            batch_size="auto"
            if self.settings.batch_size == 0
            else self.settings.batch_size,
        )

        results = lm_eval.simple_evaluate(
            model=hflm,
            tasks=[self.settings.task],
        )

        if self.settings.batch_size == 0 and hflm.batch_sizes:
            # Cache automatically determined batch size to avoid re-determining
            # the batch size on every future run.
            self.settings.batch_size = min(hflm.batch_sizes.values())

        benchmark_score = float(
            results["results"][self.settings.task][self.settings.metric]
        )

        return Score(
            value=benchmark_score,
            rich_display=f"[bold]{benchmark_score:.4f}[/]",
            md_display=f"{benchmark_score:.4f}",
        )
