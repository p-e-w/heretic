from pydantic import Field

from heretic.schemas import Response
from heretic.scorer import Scorer


class CountRefusals(Scorer):
    """
    Counts refusals using a list of keyword markers defined in settings.

    This combines the old KeywordRefusalDetector (tagger) + CountRefusals (scorer)
    into a single scorer plugin.
    """

    name = "CountRefusals"

    class Settings(Scorer.Settings):
        refusal_markers: list[str] = Field()

    @staticmethod
    def required_response_metadata_fields() -> set[str]:
        return {"response_text"}

    def score_batch(self, responses: list[Response]) -> float:
        return sum(self._is_refusal((r.response_text or "")) for r in responses)

    def _is_refusal(self, response: str) -> bool:
        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        markers = None
        if self.plugin_settings is not None and hasattr(self.plugin_settings, "refusal_markers"):
            markers = getattr(self.plugin_settings, "refusal_markers")

        for marker in (markers or self.settings.refusal_markers):
            if marker.lower() in response:
                return True

        return False
