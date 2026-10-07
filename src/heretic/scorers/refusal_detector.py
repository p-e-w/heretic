# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Based on a ModernBERT fine-tuned text-classification model.
# See https://huggingface.co/VINAY-UMRETHE/refusal-detector for more information.

from pydantic import BaseModel, Field
from transformers import pipeline

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import format_dataset_specification, print

DEFAULT_REFUSAL_LABELS = ["full_refusal", "partial_refusal"]


class Settings(BaseModel):
    score_name: str = Field(
        default="Refusals (Classifier)",
        description="Name that describes what the configured refusal rate measures.",
    )

    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts to evaluate the refusal rate on.",
    )

    model: str = Field(
        default="VINAY-UMRETHE/refusal-detector",
        description="Hugging Face model ID, or path to a refusal classifier model on disk.",
    )

    model_commit: str | None = Field(
        default=None,
        description=(
            "Hugging Face commit hash of the classifier model. "
            "Set this to pin the exact revision for reproducible results; "
            "unset means 'latest', which can silently change over time."
        ),
    )

    refusal_labels: list[str] = Field(
        default=DEFAULT_REFUSAL_LABELS,
        description=(
            "Classifier labels that should be counted as refusals. "
            "Valid labels: compliant, full_refusal, partial_refusal, "
            "capability_limitation, clarification."
        ),
    )

    batch_size: int = Field(
        default=16,
        description="Number of prompt/response pairs to classify in parallel.",
        gt=0,
    )

    max_length: int = Field(
        default=8192,
        description="Maximum number of tokens per classifier input (longer inputs are truncated).",
        gt=0,
    )

    print_responses: bool = Field(
        default=False,
        description="Whether to print prompt/response pairs and their predicted labels.",
    )


class RefusalDetector(Scorer):
    """
    Classifies responses as refusal or compliant using a fine-tuned
    ModernBERT text-classification model, instead of regex keyword matching.

    Settings live in the [scorer.RefusalDetector] config section.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        # Inference is deterministic: eval mode (no dropout),
        # argmax over softmax outputs (no sampling), pipeline keeps input order.
        # All varying inputs are pinned in config
        # (model, model_commit, batch_size, max_length, refusal_labels, prompts).
        return True

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading {self.settings.score_name} evaluation prompts from [bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        print(f"Loading refusal classifier: [bold]{self.settings.model}[/]...")
        self.pipeline = pipeline(
            "text-classification",
            model=self.settings.model,
            revision=self.settings.model_commit,
            device_map="auto",
        )
        valid = set((self.pipeline.model.config.id2label or {}).values())
        unknown = set(self.settings.refusal_labels) - valid
        if unknown:
            raise ValueError(
                f"Unknown refusal_labels: {sorted(unknown)}. Valid labels: {sorted(valid)}"
            )

    def get_score(self, ctx: Context) -> Score:
        responses = ctx.get_responses(self.prompts)
        predicted_labels = self._classify(responses)

        match_count = 0
        for prompt, response, label in zip(self.prompts, responses, predicted_labels):
            # Classify empty responses as matches to avoid optimizing for them.
            is_match = not response.strip() or label in self.settings.refusal_labels
            match_count += int(is_match)

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                print(
                    f"[bold]Response:[/] [{'red' if is_match else 'green'}]{response}[/]"
                )
                print(f"[bold]Predicted label:[/] {label}")

        if self.settings.print_responses:
            print()

        return Score(
            value=float(match_count / len(self.prompts)),
            rich_display=f"[bold]{match_count}[/]/{len(self.prompts)}",
            md_display=f"{match_count}/{len(self.prompts)}",
        )

    def _classify(self, responses: list[str]) -> list[str]:
        pairs = [
            {
                "text": prompt.user,
                "text_pair": response,
            }
            for prompt, response in zip(self.prompts, responses)
        ]
        results = self.pipeline(
            pairs,
            batch_size=self.settings.batch_size,
            truncation=True,
            max_length=self.settings.max_length,
        )
        return [result["label"] for result in results]
