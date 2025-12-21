# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from typing import Any

import torch.nn.functional as F
from pydantic import BaseModel

from .config import Settings
from .model import Model
from .scorer import Scorer
from .utils import load_plugin, load_prompts, print


class Evaluator:
    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print()
        print("Loading scorer plugin...")
        self.scorer_plugin = self._load_scorer_plugin()

        self.base_score = self.score_batch()
        # Backwards-compat: historically the default score was "number of refusals".
        # This value is used for normalizing the refusal objective.
        self.base_refusals = self.base_score
        print(f"* Initial score: [bold]{self.base_score}[/]/{len(self.bad_prompts)}")

    def _load_scorer_plugin(self) -> Scorer:
        scorer = load_plugin(
            name=self.settings.scorer,
            base_class=Scorer,
        )
        print(f"* Loaded scorer plugin: [bold]{scorer.__name__}[/bold]")
        scorer.validate_contract()

        # Validate/resolve plugin settings from namespaced config: `[<plugin.name>]`
        plugin_namespace = scorer.name
        plugin_config = self._get_plugin_namespace(plugin_namespace)
        plugin_settings: BaseModel | None = scorer.validate_settings(plugin_config or {})

        self.model.set_requested_metadata_fields(
            scorer.required_response_metadata_fields()
        )
        self.model.set_requested_context_metadata_fields(
            scorer.required_context_metadata_fields()
        )
        return scorer(
            settings=self.settings,
            model=self.model,
            context_metadata=self.model.get_context_metadata(),
            plugin_settings=plugin_settings,
        )

    def _get_plugin_namespace(self, namespace: str) -> dict[str, Any]:
        """
        Returns the config dict from the `[<namespace>]` TOML table (or {} if missing).
        """
        extra = self.settings.model_extra or {}
        value = extra.get(namespace)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(
                f"Plugin namespace [{namespace}] must be a table/object, got {type(value).__name__}"
            )
        return value

    def score_batch(self) -> float:
        responses = self.model.get_responses_batched(self.bad_prompts)
        return self.scorer_plugin.score_batch(responses)

    def update_scorer_context_metadata(self):
        """
        Updates the context metadata in the scorer plugin.
        This is useful if context metadata (like refusal directions) becomes available
        after the evaluator has been initialized.
        """
        self.scorer_plugin.context_metadata = self.model.get_context_metadata()

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

        print("  * Counting model score...")
        refusals = self.score_batch()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        kl_divergence_scale = self.settings.kl_divergence_scale
        kl_divergence_target = self.settings.kl_divergence_target

        refusals_score = refusals / self.base_refusals

        if kl_divergence >= kl_divergence_target:
            kld_score = kl_divergence / kl_divergence_scale
        else:
            kld_score = refusals_score * kl_divergence_target / kl_divergence_scale

        score = (
            kld_score,
            refusals_score,
        )

        return score, kl_divergence, refusals
