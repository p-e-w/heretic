# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import torch.nn.functional as F

from .config import Settings
from .model import Model
from .utils import load_prompts, print


class Evaluator:
    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        print()
        print(
            f"（性能評価用）良性プロンプトを [bold]{settings.good_evaluation_prompts.dataset}[/] から読み込み中..."
        )
        self.good_prompts = load_prompts(settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}個[/] のプロンプトを読み込みました")

        print("* 最初のトークンの確率分布を取得中...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"（性能評価用）悪性プロンプトを [bold]{settings.bad_evaluation_prompts.dataset}[/] から読み込み中..."
        )
        self.bad_prompts = load_prompts(settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}個[/] のプロンプトを読み込みました")

        print("* モデルの拒否回数をカウント中...")
        self.base_refusals = self.count_refusals()
        print(
            f"* 初期の拒否回数: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def is_refusal(self, response: str) -> bool:
        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def count_refusals(self) -> int:
        responses = self.model.get_responses_batched(self.bad_prompts)
        refusals = [response for response in responses if self.is_refusal(response)]
        return len(refusals)

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        print("   * 最初のトークンの確率分布を取得中...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"   * KLダイバージェンス: [bold]{kl_divergence:.2f}[/]")

        print("   * モデルの拒否回数をカウント中...")
        refusals = self.count_refusals()
        print(f"   * 拒否回数: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        score = (
            (kl_divergence / self.settings.kl_divergence_scale),
            (refusals / self.base_refusals),
        )

        return score, kl_divergence, refusals
