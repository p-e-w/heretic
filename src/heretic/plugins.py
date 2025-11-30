# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from abc import ABC, abstractmethod

from transformers import pipeline


class Plugin(ABC):
    @abstractmethod
    def score(self, responses: list[str]) -> list[float]:
        """
        Return a score (0.0 to 1.0) for each response.
        The meaning of the score depends on the plugin, but generally
        higher scores indicate a stronger presence of the target attribute.
        """
        pass


class RefusalPlugin(Plugin):
    def __init__(self, markers: list[str]):
        self.markers = markers

    def is_refusal(self, response: str) -> bool:
        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        for marker in self.markers:
            if marker.lower() in response:
                return True

        return False

    def score(self, responses: list[str]) -> list[float]:
        return [1.0 if self.is_refusal(r) else 0.0 for r in responses]


class ClassifierPlugin(Plugin):
    def __init__(self, model_name: str, target_label: str):
        # We use top_k=None to get probabilities for all labels.
        # device=0 uses the first GPU if available, otherwise CPU.
        # Note: This might conflict with the main model if VRAM is tight.
        # Ideally, we should handle device placement more carefully.
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device_map="auto",
        )
        self.target_label = target_label

    def score(self, responses: list[str]) -> list[float]:
        scores = []
        # Process in batches if necessary, but pipeline handles lists well.
        # Truncation is important as classification models often have short context windows.
        results = self.pipe(responses, truncation=True)

        for result in results:
            # result is a list of dicts like [{'label': 'joy', 'score': 0.9}, ...]
            label_score = 0.0
            for item in result:
                if item["label"] == self.target_label:
                    label_score = item["score"]
                    break
            scores.append(label_score)

        return scores
