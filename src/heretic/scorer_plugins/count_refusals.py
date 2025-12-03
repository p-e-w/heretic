from heretic.scorer import Scorer
from typing import Dict, Any


class CountRefusals(Scorer):
    def score_batch(self, tagged_responses: list[Dict[str, Any]]) -> float:
        return sum(
            tagged_response["is_refusal"] for tagged_response in tagged_responses
        )
