# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch.nn import ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation.utils import GenerateOutput

from .config import Settings
from .utils import batchify, empty_cache, print


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Model:
    def __init__(self, settings: Settings):
        self.settings = settings

        print()
        print(f"モデル [bold]{settings.model}[/] を読み込み中...")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            settings.model
        )

        # 特殊なpadトークンを宣言しないトークナイザーのフォールバック。
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        self.model = None

        for dtype in settings.dtypes:
            print(f"* dtype [bold]{dtype}[/] を試行中... ", end="")

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                )

                # テスト実行により、悪名高い「RuntimeError: probability tensor contains either `inf`, `nan` or element < 0」
                # (https://github.com/meta-llama/llama/issues/380)のようなdtype関連の問題が明らかになることがあります。
                self.generate(["Test"], max_new_tokens=1)
            except Exception as error:
                self.model = None
                empty_cache()
                print(f"[red]失敗[/] ({error})")
                continue

            print("[green]成功[/]")
            break

        if self.model is None:
            raise Exception("設定されたすべてのdtypeでモデルの読み込みに失敗しました。")

        print(f"* トランスフォーマーモデル（[bold]{len(self.get_layers())}[/] レイヤー）")
        print("* Abliteration（除去）可能なコンポーネント:")
        for component, matrices in self.get_layer_matrices(0).items():
            print(
                f"  * [bold]{component}[/]: レイヤーごとに [bold]{len(matrices)}個[/] の行列"
            )

    def reload_model(self):
        dtype = self.model.dtype

        # 既存のモデルオブジェクトをメモリからパージしてスペースを確保します。
        self.model = None
        empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model,
            dtype=dtype,
            device_map=self.settings.device_map,
        )

    def get_layers(self) -> ModuleList:
        # ほとんどのマルチモーダルモデル。
        with suppress(Exception):
            return self.model.model.language_model.layers

        # テキストのみのモデル。
        return self.model.model.layers

    def get_layer_matrices(self, layer_index: int) -> dict[str, list[Tensor]]:
        layer = self.get_layers()[layer_index]

        matrices = {}

        def try_add(component: str, matrix: Any):
            assert torch.is_tensor(matrix)

            if component not in matrices:
                matrices[component] = []

            matrices[component].append(matrix)

        # 例外はここでは抑制されません。現在、アテンション出力射影の代替の場所がないためです。
        try_add("attn.o_proj", layer.self_attn.o_proj.weight)

        # ほとんどの密なモデル。
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj.weight)

        # いくつかのMoEモデル（例：Qwen3）。
        with suppress(Exception):
            for expert in layer.mlp.experts:
                try_add("mlp.down_proj", expert.down_proj.weight)

        # Phi-3.5-MoE（およびその他）。
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:
                try_add("mlp.down_proj", expert.w2.weight)

        # gpt-oss MoE。
        with suppress(Exception):
            # gpt-ossのTransformersでの実装は、他の多くのMoEモデルとは異なり、
            # すべてのエキスパートの下方射影を単一の3Dテンソルに格納しますが、
            # PyTorchのブロードキャストマジックのおかげで、すべてがとにかく機能します。
            try_add("mlp.down_proj", layer.mlp.experts.down_proj)

        # 少なくとも1つのMLP下方射影が必要です。
        assert matrices["mlp.down_proj"]

        return matrices

    def get_abliterable_components(self) -> list[str]:
        return list(self.get_layer_matrices(0).keys())

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        if direction_index is None:
            refusal_direction = None
        else:
            # 埋め込みの方向はrefusal_directionsの最初の要素であるため、インデックスを1つずらす必要があります。
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        # 一部のabliterationの実装では埋め込み行列も直交化しますが、
        # それに利点があるかどうかは不明です。
        for layer_index in range(len(self.get_layers())):
            for component, matrices in self.get_layer_matrices(layer_index).items():
                params = parameters[component]

                distance = abs(layer_index - params.max_weight_position)

                # max_weight_positionからmin_weight_distance以上離れたレイヤーは直交化しません。
                if distance > params.min_weight_distance:
                    continue

                # min_weight_distanceにわたってmax_weightとmin_weightの間を線形補間します。
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    # 埋め込みの方向はrefusal_directionsの最初の要素であるため、インデックスを1つずらす必要があります。
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                # 右乗算されたベクトルを拒否方向で張られる部分空間に射影します。
                projector = torch.outer(
                    layer_refusal_direction,
                    layer_refusal_direction,
                ).to(self.model.dtype)

                for matrix in matrices:
                    # Autogradを使用していないため、インプレース減算は安全です。
                    matrix.sub_(weight * (projector @ matrix))

    def get_chat(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateOutput | LongTensor]:
        chats = [self.get_chat(prompt) for prompt in prompts]

        chat_prompts: list[str] = self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        return inputs, self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,  # 決定論的な出力を保証するために貪欲法デコーディングを使用します。
        )

    def get_responses(self, prompts: list[str]) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        # 新しく生成された部分のみを返します。
        return self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

    def get_responses_batched(self, prompts: list[str]) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(batch):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[str]) -> Tensor:
        # トークンを1つだけ生成し、各プロンプトとレイヤーのそのトークン位置での残差ベクトルを返します。
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # 最初（唯一）の生成されたトークンの隠れ状態。
        hidden_states = outputs.hidden_states[0]

        # 返されるテンソルの形状は（プロンプト、レイヤー、コンポーネント）です。
        residuals = torch.stack(
            # layer_hidden_statesの形状は（プロンプト、位置、コンポーネント）であるため、
            # これにより各プロンプトの末尾の隠れ状態が抽出され、
            # レイヤー全体でスタックされます。
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # 残差ベクトルを含む計算中の精度（bfloat16）または範囲（float16）の問題を回避するために、
        # データ型をアップキャストします。
        return residuals.to(torch.float32)

    def get_residuals_batched(self, prompts: list[str]) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))

        return torch.cat(residuals, dim=0)

    # KLダイバージェンスを計算する際の数値安定性のために、確率ではなく対数確率を扱います。
    def get_logprobs(self, prompts: list[str]) -> Tensor:
        # トークンを1つだけ生成し、各プロンプトのそのトークン位置での語彙全体の（対数）確率分布を返します。
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # 最初（唯一）の生成されたトークンのロジット。
        logits = outputs.scores[0]

        # 返されるテンソルの形状は（プロンプト、トークン）です。
        return F.log_softmax(logits, dim=-1)

    def get_logprobs_batched(self, prompts: list[str]) -> Tensor:
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch))

        return torch.cat(logprobs, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        chat_prompt: str = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )

        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
