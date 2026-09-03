# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import lm_eval
from lm_eval.models.huggingface import HFLM
from pydantic import BaseModel, Field

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

    def init(self, ctx: Context) -> None:
        self.hflm = HFLM(
            pretrained=ctx._model.model,  # ty:ignore[invalid-argument-type]
            tokenizer=ctx._model.tokenizer,  # ty:ignore[invalid-argument-type]
            batch_size="auto",
        )

    def get_score(self, ctx: Context) -> Score:
        # The purpose of this hack, where we initialize the HFLM object once,
        # then update its internal model every time we calculate the score,
        # is to get the benefits of batch size caching while allowing for
        # model reloads, e.g. when using --evaluate-model.
        self.hflm.pretrained = ctx._model.model
        self.hflm._model = ctx._model.model

        results = lm_eval.simple_evaluate(
            model=self.hflm,
            tasks=[self.settings.task],
        )

        benchmark_score = float(
            results["results"][self.settings.task][self.settings.metric]
        )

        return Score(
            value=benchmark_score,
            rich_display=f"[bold]{benchmark_score:.4f}[/]",
            md_display=f"{benchmark_score:.4f}",
        )
