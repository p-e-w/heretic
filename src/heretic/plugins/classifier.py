# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from transformers import pipeline

from ..plugin_interface import Plugin


class ClassifierPlugin(Plugin):
    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        target_label: str = "joy",
        minimize: bool = False,
    ):
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
        self._minimize = minimize

    @property
    def minimize(self) -> bool:
        return self._minimize

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


# Expose the class as PLUGIN_CLASS for the loader
PLUGIN_CLASS = ClassifierPlugin
