# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import importlib
import inspect

import torch.nn.functional as F

from .config import Settings
from .model import Model
from .scorer import Scorer
from .tagger import Tagger
from .utils import load_prompts, print


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
        print("Loading tagger plugin...")
        self.tagger_plugin = self._load_tagger_plugin()
        self.model.set_requested_metadata_fields(
            self.tagger_plugin.required_response_metadata_fields()
        )
        print()
        print("Loading scorer plugin...")
        self.scorer_plugin = self._load_scorer_plugin()

        self.base_score = self.tag_and_score_batch()
        print(f"* Initial score: [bold]{self.base_score}[/]/{len(self.bad_prompts)}")

    def _load_tagger_plugin(self) -> Tagger:
        name = self.settings.tagger_plugin
        tagger = None

        if "." in name:
            # Load from arbitrary python path
            try:
                module_name, class_name = name.rsplit(".", 1)
                module = importlib.import_module(module_name)
                tagger = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                print(f"[red]Error loading tagger plugin '{name}': {e}[/]")
        else:
            try:
                module = importlib.import_module(
                    f".tagger_plugins.{name}", package="heretic"
                )
                # Find the class defined in the module that inherits from Tagger
                for _, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, Tagger)
                        and obj.__module__ == module.__name__
                    ):
                        tagger = obj
                        break

                if tagger is None:
                    print(f"[red]Error: No Tagger subclass found in plugin '{name}'[/]")
                    exit()
            except ImportError as e:
                print(f"[red]Error loading plugin '{name}': {e}[/]")
                exit()

        print(f"* Loaded tagger plugin: [bold]{tagger.__name__}[/bold]")
        self.model.set_requested_context_metadata_fields(
            tagger.required_context_metadata_fields()
        )
        return tagger(
            settings=self.settings,
            model=self.model,
            context_metadata=self.model.get_context_metadata(),
        )

    def _load_scorer_plugin(self) -> Scorer:
        name = self.settings.scorer_plugin
        scorer = None

        if "." in name:
            # Load from arbitrary python path
            try:
                module_name, class_name = name.rsplit(".", 1)
                module = importlib.import_module(module_name)
                scorer = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                print(f"[red]Error loading scorer plugin '{name}': {e}[/]")
        else:
            try:
                module = importlib.import_module(
                    f".scorer_plugins.{name}", package="heretic"
                )
                # Find the class defined in the module that inherits from Scorer
                for _, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, Scorer)
                        and obj.__module__ == module.__name__
                    ):
                        scorer = obj
                        break

                if scorer is None:
                    print(f"[red]Error: No Scorer subclass found in plugin '{name}'[/]")
                    exit()
            except ImportError as e:
                print(f"[red]Error loading plugin '{name}': {e}[/]")
                exit()

        print(f"* Loaded scorer plugin: [bold]{scorer.__name__}[/bold]")
        return scorer()

    def tag_and_score_batch(self) -> float:
        responses = self.model.get_responses_batched(self.bad_prompts)
        tags = self.tagger_plugin.tag_batch(responses=responses)
        score = self.scorer_plugin.score_batch(tags)
        return score

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.2f}[/]")

        print("  * Counting model score...")
        refusals = self.tag_and_score_batch()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        score = (
            (kl_divergence / self.settings.kl_divergence_scale),
            (refusals / self.base_score),
        )

        return score, kl_divergence, refusals
