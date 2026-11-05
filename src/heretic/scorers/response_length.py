# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from pydantic import BaseModel, Field, NonNegativeInt
from rich.markup import escape

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.scorer import Context, Score, Scorer
from heretic.utils import format_dataset_specification, print


class Settings(BaseModel):
    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts to evaluate the response length on.",
    )

    strip_responses: bool = Field(
        default=True,
        description="Whether to strip leading and trailing whitespace from responses before measuring their length.",
    )

    short_response_threshold: NonNegativeInt = Field(
        # Chosen because "No" is the shortest common valid response.
        default=2,
        description="Length in characters below which a response is classified as a 'short response'.",
    )

    short_response_score_length: float = Field(
        default=500,
        description=(
            "Length value artificially assigned to 'short responses' for the purpose of scoring. "
            "This can be used to penalize very short (and thus likely invalid) responses."
        ),
    )

    print_responses: bool = Field(
        default=False,
        description="Whether to print prompt/response pairs.",
    )


class ResponseLength(Scorer):
    """
    Measures the mean length of responses.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "Mean response length"

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading response length evaluation prompts from [bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

    def get_score(self, ctx: Context) -> Score:
        total_length = 0
        responses = ctx.get_responses(self.prompts)

        for prompt, response in zip(self.prompts, responses, strict=True):
            response_length = len(
                response.strip() if self.settings.strip_responses else response
            )
            if response_length < self.settings.short_response_threshold:
                response_length = self.settings.short_response_score_length
            total_length += response_length

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {escape(prompt.system)}")
                print(f"[bold]Prompt:[/] {escape(prompt.user)}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                else:
                    response = escape(response)
                print(f"[bold]Response:[/] {response}")

        if self.settings.print_responses:
            print()

        mean_length = float(total_length / len(self.prompts))

        return Score(
            value=mean_length,
            rich_display=f"[bold]{mean_length:.1f}[/]",
            md_display=f"{mean_length:.1f}",
        )
