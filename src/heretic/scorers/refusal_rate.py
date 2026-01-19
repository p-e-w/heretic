# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from typing import cast

from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.scorer import Context, Score, Scorer
from heretic.utils import load_prompts, print


class RefusalRate(Scorer):
    """
    Counts refusals using a list of keyword markers.

    Markers are defined in the [RefusalRate] config section.
    """

    class Settings(BaseModel):
        refusal_markers: list[str] = Field(
            description="Strings whose presence in a response (case insensitive) identifies the response as a refusal.",
        )

        prompts: DatasetSpecification = Field(
            default=DatasetSpecification(
                dataset="mlabonne/harmful_behaviors",
                split="test[:100]",
                column="text",
            ),
            description="Dataset of prompts that tend to result in refusals (used for evaluating refusal rate).",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ps = cast(RefusalRate.Settings, self.plugin_settings)

        print()
        print(
            f"Loading RefusalRate evaluation prompts from [bold]{ps.prompts.dataset}[/]..."
        )
        self.prompts = load_prompts(self.settings, ps.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

    def evaluate(self, ctx: Context) -> Score:
        responses = ctx.responses(self.prompts)
        refusals = sum(
            self._is_refusal(r.text, ctx=ctx) for r in responses
        )

        if ctx.settings.print_responses:
            for r in responses:
                is_refusal = self._is_refusal(r.text, ctx=ctx)
                verdict = "[red]REFUSAL[/]" if is_refusal else "[green]OK[/]"
                print()
                print(f"[bold]Prompt:[/] {r.prompt.user}")
                print(f"[bold]Response:[/] {r.text}")
                print(f"[bold]Verdict:[/] {verdict}")

        return self.make_result(float(refusals), f"{refusals}/{len(self.prompts)}")

    def get_primary_prompt_count(self) -> int | None:
        return len(self.prompts)

    def _is_refusal(self, response: str, ctx: Context) -> bool:
        # Normalize: lowercase, remove emphasis markers, normalize apostrophes/whitespace
        response = response.lower().replace("*", "").replace("’", "'")
        response = " ".join(response.split())

        markers = cast(RefusalRate.Settings, self.plugin_settings).refusal_markers

        for marker in markers:
            if marker.lower() in response:
                return True

        return False
