from typing import cast

from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.scorer import EvaluationContext, Score, Scorer
from heretic.utils import load_prompts, print


class RefusalRate(Scorer):
    """
    Counts refusals using a list of keyword markers.

    Markers are defined in the [RefusalRate] config section.
    """

    class Settings(BaseModel):
        refusal_markers: list[str] = Field(
            description="Optional override for the global refusal markers.",
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

        ps = self.plugin_settings
        if ps is None:
            raise ValueError("RefusalRate requires plugin settings to be validated")
        ps = cast(RefusalRate.Settings, ps)

        print()
        print(
            f"Loading RefusalRate evaluation prompts from [bold]{ps.prompts.dataset}[/]..."
        )
        self.prompts = load_prompts(self.settings, ps.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

    def evaluate(self, ctx: EvaluationContext) -> Score:
        responses = ctx.get_responses(self.prompts)
        refusals = sum(
            self._is_refusal(r.text.response_text, ctx=ctx) for r in responses
        )

        if ctx.settings.print_responses:
            for r in responses:
                is_refusal = self._is_refusal(r.text.response_text, ctx=ctx)
                verdict = "[red]REFUSAL[/]" if is_refusal else "[green]OK[/]"
                print()
                print(f"[bold]Prompt:[/] {r.prompt_text}")
                print(f"[bold]Response:[/] {r.response_text}")
                print(f"[bold]Verdict:[/] {verdict}")

        return self.make_result(float(refusals), f"{refusals}/{len(self.prompts)}")

    def get_primary_prompt_count(self) -> int | None:
        return len(self.prompts)

    def _is_refusal(self, response: str, ctx: EvaluationContext) -> bool:
        # Normalize: lowercase, remove emphasis markers, normalize apostrophes/whitespace
        response = response.lower().replace("*", "").replace("’", "'")
        response = " ".join(response.split())

        ps = self.plugin_settings
        markers: list[str]
        if ps is not None:
            markers = cast(RefusalRate.Settings, ps).refusal_markers

        for marker in markers:
            if marker.lower() in response:
                return True

        return False
