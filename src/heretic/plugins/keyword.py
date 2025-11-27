from heretic.refusal import RefusalDetector


class KeywordRefusalDetector(RefusalDetector):
    """
    Detects refusals using a list of keyword markers defined in settings.
    """

    def detect_batch(self, responses: list[str]) -> list[bool]:
        return [self._is_refusal(response) for response in responses]

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
