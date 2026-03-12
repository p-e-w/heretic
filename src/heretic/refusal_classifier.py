# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RefusalClassifier:
    def __init__(self, model_name: str = "NousResearch/Minos-v1"):
        """Initialize the refusal classifier model.

        Args:
            model_name: Hugging Face model ID for refusal classification.
        """
        print(f"Loading refusal classifier model [bold]{model_name}[/]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        print("* Refusal classifier loaded")

    def is_refusal(self, text: str) -> bool:
        """Classify whether text is a refusal.

        Args:
            text: The text to classify.

        Returns:
            True if the text is classified as a refusal, False otherwise.
        """
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)

        return prediction.item() == 1
