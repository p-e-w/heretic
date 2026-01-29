# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.scorer import Context, Score, Scorer
from heretic.utils import print


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

    print_responses: bool = Field(
        default=False,
        description="Whether to print prompt/response pairs when counting refusals.",
    )


class RefusalRate(Scorer):
    """
    Counts refusals using a list of keyword markers.

    Markers are defined in the [scorer.RefusalRate] config section.
    """

    settings: Settings

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading RefusalRate evaluation prompts from [bold]{self.settings.prompts.dataset}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

    def get_score(self, ctx: Context) -> Score:
        responses = ctx.get_responses(self.prompts)
        refusals = sum(self._is_refusal(r) for r in responses)

        if self.settings.print_responses:
            for index, response in enumerate(responses):
                prompt = self.prompts[index]
                is_refusal = self._is_refusal(response)
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                print(
                    f"[bold]Response:[/] [{'red' if is_refusal else 'green'}]{response}[/]"
                )

        return Score(
            name="Refusals",
            value=float(refusals),
            cli_display=f"{refusals}/{len(self.prompts)}",
            hf_display=f"{refusals}/{len(self.prompts)}",
        )

    def _is_refusal(self, response: str) -> bool:
        # Normalize: lowercase, remove emphasis markers, normalize apostrophes/whitespace
        response = response.lower().replace("*", "").replace("’", "'")
        response = " ".join(response.split())

        markers = self.settings.refusal_markers

        for marker in markers:
            if marker.lower() in response:
                return True

        return False
