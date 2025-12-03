from heretic.tagger import Tagger
from heretic.schemas import ResponseMetadata
from typing import Dict, Any


class KeywordRefusalDetector(Tagger):
    """
    Detects refusals using a list of keyword markers defined in settings.
    """

    def tag_batch(self, responses: list[str], metadata: list[ResponseMetadata]) -> list[Dict[str, Any]]:
        return [{"is_refusal": self._is_refusal(response)} for response in responses]

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
