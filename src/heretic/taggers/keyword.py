from typing import Any, Dict

from heretic.schemas import Response
from heretic.tagger import Tagger


class KeywordRefusalDetector(Tagger):
    """
    Detects refusals using a list of keyword markers defined in settings.
    """

    @staticmethod
    def required_response_metadata_fields() -> set[str]:
        return {"response_text"}

    @staticmethod
    def required_context_metadata_fields() -> set[str]:
        return {"generation_params"}

    def tag_batch(self, responses: list[Response]) -> list[Dict[str, Any]]:
        print(self.context_metadata)
        print(responses[0].response_text)
        # print(responses[0].prompt_text)
        # print(responses[0].token_logits)
        # print(responses[0].response_embedding)
        # print(responses[0].finish_reason)
        return [
            {"is_refusal": self._is_refusal(response.response_text or "")}
            for response in responses
        ]

    def _is_refusal(self, response: str) -> bool:
        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False
