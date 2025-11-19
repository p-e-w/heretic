# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import huggingface_hub
import optuna
import questionary
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from huggingface_hub import ModelCard, ModelCardData
from optuna import Trial
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from pydantic import ValidationError
from questionary import Choice, Style
from rich.traceback import install

from .config import Settings
from .evaluator import Evaluator
from .model import AbliterationParameters, Model
from .utils import (
    format_duration,
    get_readme_intro,
    get_trial_parameters,
    load_prompts,
    print,
)


def run():
    # https://budavariam.github.io/asciiart-text/ の「Pagga」フォントを修正
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic-llm')}")
    print("[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]")
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    if (
        # 奇数の引数が渡された場合（argv[0]はプログラム名）、
        # 「--param VALUE」ペアを考慮すると1つ余る。
        len(sys.argv) % 2 == 0
        # 残りの引数がフラグ（「--help」など）ではなくパラメータ値である場合。
        and not sys.argv[-1].startswith("-")
    ):
        # 最後の引数をモデルと仮定する。
        sys.argv.insert(-1, "--model")

    try:
        settings = Settings()
    except ValidationError as error:
        print(f"設定ファイルに [bold]{error.error_count()}個[/] のエラーがあります:")

        for error in error.errors():
            print(f"[bold]{error['loc'][0]}[/]: [yellow]{error['msg']}[/]")

        print()
        print(
            "[bold]heretic --help[/] を実行するか、[bold]config.default.toml[/] を参照して設定パラメータの詳細を確認してください。"
        )
        return

    # Adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/env.py
    if torch.cuda.is_available():
        print(f"GPUタイプ: [bold]{torch.cuda.get_device_name()}[/]")
    elif is_xpu_available():
        print(f"XPU type: [bold]{torch.xpu.get_device_name()}[/]")
    elif is_mlu_available():
        print(f"MLU type: [bold]{torch.mlu.get_device_name()}[/]")
    elif is_sdaa_available():
        print(f"SDAA type: [bold]{torch.sdaa.get_device_name()}[/]")
    elif is_musa_available():
        print(f"MUSA type: [bold]{torch.musa.get_device_name()}[/]")
    elif is_npu_available():
        print(f"CANN version: [bold]{torch.version.cann}[/]")
    else:
        print(
            "[bold yellow]GPUやその他のアクセラレータが検出されませんでした。処理速度が遅くなります。[/]"
        )

    # 推論のみを行うため、勾配は必要ありません。
    torch.set_grad_enabled(False)

    # 最適なバッチサイズを決定する際に、多くの異なるバッチサイズを試行するため、
    # 多くの計算グラフがコンパイルされます。制限を引き上げる（デフォルト= 8）と、
    # TorchDynamoが再コンパイルが多すぎるために何かがおかしいと仮定するエラーを回避できます。
    torch._dynamo.config.cache_size_limit = 64

    # Transformersからの警告スパムを抑制します。
    # 私のキャリア全体で、そのライブラリから有用な警告を見たことは一度もありません。
    transformers.logging.set_verbosity_error()

    # 独自の試行ロギングを行うため、パラメータと結果に関する
    # INFOメッセージは必要ありません。
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 多変量TPEが実験的であるという警告を抑制します。
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    model = Model(settings)

    print()
    print(f"（評価用）良性プロンプトを [bold]{settings.good_prompts.dataset}[/] から読み込み中...")
    good_prompts = load_prompts(settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}個[/] のプロンプトを読み込みました")

    print()
    print(f"（評価用）悪性プロンプトを [bold]{settings.bad_prompts.dataset}[/] から読み込み中...")
    bad_prompts = load_prompts(settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}個[/] のプロンプトを読み込みました")

    if settings.batch_size == 0:
        print()
        print("最適なバッチサイズを決定中...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* バッチサイズ [bold]{batch_size}[/] を試行中... ", end="")

            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]

            try:
                # ウォームアップ実行して計算グラフを構築し、その部分がベンチマークされないようにします。
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    # バッチサイズ1でもすでに失敗します。
                    # これから回復することはできません。
                    raise

                print(f"[red]失敗[/] ({error})")
                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]成功[/] ([bold]{performance:.0f}[/] トークン/秒)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* 選択されたバッチサイズ: [bold]{settings.batch_size}[/]")

    evaluator = Evaluator(settings, model)

    if settings.evaluate_model is not None:
        print()
        print(f"モデル [bold]{settings.evaluate_model}[/] を読み込み中...")
        settings.model = settings.evaluate_model
        model.reload_model()
        print("* 評価を実行中...")
        evaluator.get_score()
        return

    print()
    print("レイヤーごとの拒否方向を計算中...")
    print("* 良性プロンプトの残差を取得中...")
    good_residuals = model.get_residuals_batched(good_prompts)
    print("* 悪性プロンプトの残差を取得中...")
    bad_residuals = model.get_residuals_batched(bad_prompts)
    refusal_directions = F.normalize(
        bad_residuals.mean(dim=0) - good_residuals.mean(dim=0),
        p=2,
        dim=1,
    )

    trial_index = 0
    start_time = time.perf_counter()

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        # 「有害な」入力と「無害な」入力の識別は、通常、レイヤースタックの中間点をわずかに超えたレイヤーで最も強力です。
        # 詳細な分析については、元のabliterationの論文（https://arxiv.org/abs/2406.11717）を参照してください。
        #
        # このパラメータは「グローバル」な方向スコープにのみ必要ですが、常にサンプリングすることに注意してください。
        # その理由は、多変量TPEが条件付きまたは可変範囲のパラメータで機能しないためです。
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * (len(model.get_layers()) - 1),
            0.9 * (len(model.get_layers()) - 1),
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in model.get_abliterable_components():
            # パラメータ範囲は、さまざまなモデルとより広い範囲での実験に基づいています。
            # それらは固定されたものではなく、将来のモデルに合わせて調整する必要があるかもしれません。
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                0.8,
                1.5,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * (len(model.get_layers()) - 1),
                len(model.get_layers()) - 1,
            )
            # サンプリングの目的で、min_weightはmax_weightの分数として表現されます。
            # これも、多変量TPEが可変範囲パラメータをサポートしていないためです。
            # 値は、以下で実際のmin_weight値に変換されます。
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * (len(model.get_layers()) - 1),
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", parameters)

        print()
        print(
            f"トライアル [bold]{trial_index}[/] / [bold]{settings.n_trials}[/] を実行中..."
        )
        print("* パラメータ:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* モデルを再読み込み中...")
        model.reload_model()
        print("* Abliteration（除去）を実行中...")
        model.abliterate(refusal_directions, direction_index, parameters)
        print("* 評価中...")
        score, kl_divergence, refusals = evaluator.get_score()

        elapsed_time = time.perf_counter() - start_time
        remaining_time = (elapsed_time / trial_index) * (
            settings.n_trials - trial_index
        )
        print()
        print(f"[grey50]経過時間: [bold]{format_duration(elapsed_time)}[/][/]")
        if trial_index < settings.n_trials:
            print(
                f"[grey50]推定残り時間: [bold]{format_duration(remaining_time)}[/][/]"
            )

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)

        return score

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=settings.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
        ),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
    )

    study.optimize(objective, n_trials=settings.n_trials)

    best_trials = sorted(
        study.best_trials,
        key=lambda trial: trial.user_attrs["refusals"],
    )

    choices = [
        Choice(
            title=(
                f"[トライアル {trial.user_attrs['index']:>3}] "
                f"拒否回数: {trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                f"KLダイバージェンス: {trial.user_attrs['kl_divergence']:.2f}"
            ),
            value=trial,
        )
        for trial in best_trials
    ]

    choices.append(
        Choice(
            title="なし（プログラムを終了する）",
            value="",
        )
    )

    print()
    print("[bold green]最適化が完了しました！[/]")
    print()
    print(
        (
            "以下のトライアルは、拒否回数とKLダイバージェンスのパレート最適解です。... "
            "[yellow]注意：KLダイバージェンスが1.0を超えると、通常、オリジナルモデルの能力に重大な損傷があることを示します。[/]"
        )
    )

    while True:
        print()
        trial = questionary.select(
            "どのトライアルを使用しますか？",
            choices=choices,
            style=Style([("highlighted", "reverse")]),
        ).ask()

        if trial is None or trial == "":
            break

        print()
        print(f"トライアル [bold]{trial.user_attrs['index']}[/] のモデルを復元中...")
        print("* モデルを再読み込み中...")
        model.reload_model()
        print("* Abliteration（除去）を実行中...")
        model.abliterate(
            refusal_directions,
            trial.user_attrs["direction_index"],
            trial.user_attrs["parameters"],
        )

        while True:
            print()
            action = questionary.select(
                "無害化解除したモデルで何をしますか？",
                choices=[
                    "モデルをローカルフォルダに保存する",
                    "モデルをHugging Faceにアップロードする",
                    "モデルとチャットする",
                    "何もしない（トライアル選択メニューに戻る）",
                ],
                style=Style([("highlighted", "reverse")]),
            ).ask()

            if action is None or action == "何もしない（トライアル選択メニューに戻る）":
                break

            # すべてのアクションはtry/exceptブロックでラップされているため、エラーが発生した場合でも、
            # プログラムがクラッシュして最適化されたモデルを失う代わりに、
            # 別のアクションを試すことができます。
            try:
                match action:
                    case "モデルをローカルフォルダに保存する":
                        save_directory = questionary.path("フォルダへのパス:").ask()
                        if not save_directory:
                            continue

                        print("モデルを保存中...")
                        model.model.save_pretrained(save_directory)
                        model.tokenizer.save_pretrained(save_directory)
                        print(f"モデルを [bold]{save_directory}[/] に保存しました。")

                    case "モデルをHugging Faceにアップロードする":
                        # huggingface_hub.login()はトークンをディスクに保存するため使用しません。
                        # このプログラムはレンタルまたは共有GPUサーバーで実行されることが多いため、
                        # 資格情報を永続化しない方がよいでしょう。
                        token = huggingface_hub.get_token()
                        if not token:
                            token = questionary.password(
                                "Hugging Face アクセストークン:"
                            ).ask()
                        if not token:
                            continue

                        user = huggingface_hub.whoami(token)
                        print(
                            f"[bold]{user['fullname']} ({user['email']})[/] としてログインしました"
                        )

                        repo_id = questionary.text(
                            "リポジトリ名:",
                            default=f"{user['name']}/{Path(settings.model).name}-heretic",
                        ).ask()

                        visibility = questionary.select(
                            "リポジトリを公開（Public）または非公開（Private）にしますか？",
                            choices=[
                                "公開",
                                "非公開",
                            ],
                            style=Style([("highlighted", "reverse")]),
                        ).ask()
                        private = visibility == "非公開"

                        print("モデルをアップロード中...")

                        model.model.push_to_hub(
                            repo_id,
                            private=private,
                            token=token,
                        )
                        model.tokenizer.push_to_hub(
                            repo_id,
                            private=private,
                            token=token,
                        )

                        # モデルパスがローカルに存在しない場合は、
                        # Hugging Face Hubでホストされているモデルであると見なすことができ、
                        # その場合はモデルカードを取得できます。
                        if not Path(settings.model).exists():
                            card = ModelCard.load(settings.model)
                            if card.data is None:
                                card.data = ModelCardData()
                            if card.data.tags is None:
                                card.data.tags = []
                            card.data.tags.append("heretic")
                            card.data.tags.append("uncensored")
                            card.data.tags.append("decensored")
                            card.data.tags.append("abliterated")
                            card.text = (
                                get_readme_intro(
                                    settings,
                                    trial,
                                    evaluator.base_refusals,
                                    evaluator.bad_prompts,
                                )
                                + card.text
                            )
                            card.push_to_hub(repo_id, token=token)

                        print(f"モデルを [bold]{repo_id}[/] にアップロードしました。")

                    case "モデルとチャットする":
                        print()
                        print(
                            "[cyan]Ctrl+C を押すといつでもメニューに戻れます。[/]"
                        )

                        chat = [
                            {"role": "system", "content": settings.system_prompt},
                        ]

                        while True:
                            try:
                                message = questionary.text(
                                    "ユーザー:",
                                    qmark=">",
                                ).unsafe_ask()
                                if not message:
                                    break
                                chat.append({"role": "user", "content": message})

                                print("[bold]アシスタント:[/] ", end="")
                                response = model.stream_chat_response(chat)
                                chat.append({"role": "assistant", "content": response})
                            except (KeyboardInterrupt, EOFError):
                                # Ctrl+C/Ctrl+D
                                break

            except Exception as error:
                print(f"エラー: {error}")


def main():
    # Richトレースバックハンドラをインストールします。
    install()

    try:
        run()
    except BaseException as error:
        # Transformersは、KeyboardInterrupt（またはBaseException）を内部で処理することがあるようです。
        # これにより、ハンドラで別のエラーが再発生し、根本原因がマスクされる可能性があります。
        # したがって、エラー自体とそのコンテキストの両方をチェックします。
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]シャットダウンしています...[/]")
        else:
            raise
