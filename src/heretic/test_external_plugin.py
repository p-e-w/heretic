from pydantic import BaseModel, Field
from heretic.plugin import Plugin
from heretic.scorer import Scorer, Context, Score
import sys

class Settings(BaseModel):
    test: list[str] = Field(description="test")
class MyExternalScorer(Scorer):


    settings: Settings

    def start(self, ctx: Context) -> None:
        print("hai :3")

    def get_score(self, ctx: Context) -> Score:
        print(self.settings.test)
        return self.make_result(1, "blorg")
