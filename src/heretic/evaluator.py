# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import importlib
import inspect
import torch.nn.functional as F

from .config import Settings
from .model import Model
from .refusal import RefusalDetector
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

        self.detectors = self._load_detectors()

        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def _load_detectors(self) -> list[RefusalDetector]:
        detectors = []
        for name in self.settings.refusal_detectors:
            detector_cls = None

            if "." in name:
                # Load from arbitrary python path
                try:
                    module_name, class_name = name.rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    detector_cls = getattr(module, class_name)
                except (ImportError, AttributeError) as e:
                    print(f"[red]Error loading detector '{name}': {e}[/]")
                    continue
            else:
                # Load from heretic.plugins
                try:
                    module = importlib.import_module(f".plugins.{name}", package="heretic")
                    # Find the class defined in the module that inherits from RefusalDetector
                    for _, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, RefusalDetector)
                            and obj.__module__ == module.__name__
                        ):
                            detector_cls = obj
                            break
                    
                    if detector_cls is None:
                         print(f"[red]Error: No RefusalDetector subclass found in plugin '{name}'[/]")
                         continue
                except ImportError as e:
                    print(f"[red]Error loading plugin '{name}': {e}[/]")
                    continue
            
            if detector_cls:
                detectors.append(detector_cls(self.settings, self.model))
            
        return detectors

    def is_refusal(self, response: str) -> bool:
        for detector in self.detectors:
            if detector.detect(response):
                return True
        return False

    def count_refusals(self) -> int:
        responses = self.model.get_responses_batched(self.bad_prompts)
        
        is_refusal_flags = [False] * len(responses)
        
        for detector in self.detectors:
            flags = detector.detect_batch(responses)
            for i, flag in enumerate(flags):
                if flag:
                    is_refusal_flags[i] = True
                    
        return sum(is_refusal_flags)

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

        print("  * Counting model refusals...")
        refusals = self.count_refusals()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        score = (
            (kl_divergence / self.settings.kl_divergence_scale),
            (refusals / self.base_refusals),
        )

        return score, kl_divergence, refusals
