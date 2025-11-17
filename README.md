# Heretic：言語モデルの検閲を完全自動で除去

Hereticは、高価な事後学習なしに、トランスフォーマーベースの言語モデルから検閲（別名「安全性アライメント」）を除去するツールです。
これは、方向性アブレーション（「abliteration」としても知られる）の高度な実装（[Arditi et al. 2024](https://arxiv.org/abs/2406.11717)）と、[Optuna](https://optuna.org/)を搭載したTPEベースのパラメータオプティマイザを組み合わせています。

このアプローチにより、Hereticは**完全に自動で**動作します。Hereticは、
拒否の数と元のモデルからのKLダイバージェンスを共に最小化することで、高品質なabliterationパラメータを見つけ出します。
これにより、元のモデルの知能を可能な限り保持した無修正モデルが実現します。
Hereticを使用するのに、トランスフォーマーの内部構造を理解する必要はありません。
実際、コマンドラインプログラムの実行方法を知っている人なら誰でも、Hereticを使って言語モデルの検閲を除去できます。

<img width="650" height="715" alt="スクリーンショット" src="https://github.com/user-attachments/assets/d71a5efa-d6be-4705-a817-63332afb2d15" />

&nbsp;

デフォルト設定で教師なしで実行すると、Hereticは人間の専門家が手動で作成したabliterationの品質に匹敵する無修正モデルを生成できます。

| モデル | 「有害な」プロンプトに対する拒否 | 「無害な」プロンプトに対する元のモデルからのKLダイバージェンス |
| :--- | ---: | ---: |
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) (オリジナル) | 97/100 | 0 *(定義による)* |
| [mlabonne/gemma-3-12b-it-abliterated-v2](https://huggingface.co/mlabonne/gemma-3-12b-it-abliterated-v2) | 3/100 | 1.04 |
| [huihui-ai/gemma-3-12b-it-abliterated](https://huggingface.co/huihui-ai/gemma-3-12b-it-abliterated) | 3/100 | 0.45 |
| **[p-e-w/gemma-3-12b-it-heretic](https://huggingface.co/p-e-w/gemma-3-12b-it-heretic) (私たち)** | **3/100** | **0.16** |

Heretic版は、人間の労力を一切介さずに生成され、他のabliterationと同レベルの拒否抑制を達成しながらも、はるかに低いKLダイバージェンスを実現しており、元のモデルの能力への損傷が少ないことを示しています。
*（これらの数値は、Hereticの組み込み評価機能を使用して再現できます。例： `heretic --model google/gemma-3-12b-it --evaluate-model p-e-w/gemma-3-12b-it-heretic`。
正確な値はプラットフォームやハードウェアに依存する可能性があることに注意してください。
上の表は、RTX 5090上でPyTorch 2.8を使用して作成されました。）*

Hereticは、多くのマルチモーダルモデルを含むほとんどの密なモデル、およびいくつかの異なるMoEアーキテクチャをサポートしています。
まだSSM/ハイブリッドモデル、不均一なレイヤーを持つモデル、および特定の新しいアテンションシステムはサポートしていません。

Hereticを使用して検閲が除去されたモデルのコレクションは、[Hugging Face](https://huggingface.co/collections/p-e-w/the-bestiary)でご覧いただけます。


## 使い方

ハードウェアに合わせてPyTorch 2.2+をインストールしたPython 3.10+環境を準備してください。
その後、次を実行します。

```
pip install heretic-llm
heretic Qwen/Qwen3-4B-Instruct-2507
```

`Qwen/Qwen3-4B-Instruct-2507` を、検閲を除去したいモデルに置き換えてください。

このプロセスは完全に自動であり、設定は不要です。ただし、
Hereticには、より細かく制御するために変更できるさまざまな設定パラメータがあります。
利用可能なコマンドラインオプションを確認するには `heretic --help` を実行するか、
設定ファイルを使用したい場合は [`config.default.toml`](config.default.toml) をご覧ください。

プログラム実行の開始時に、Hereticはシステムをベンチマークし、
利用可能なハードウェアを最大限に活用するための最適なバッチサイズを決定します。
RTX 3090では、デフォルト設定でLlama-3.1-8Bの検閲を除去するのに約45分かかります。

Hereticがモデルの検閲除去を完了すると、モデルを保存したり、Hugging Faceにアップロードしたり、
チャットして動作を確認したり、あるいはそれらの組み合わせを選択するオプションが与えられます。


## 仕組み

Hereticは、方向性アブレーションのパラメータ化された変種を実装しています。
サポートされている各トランスフォーマーコンポーネント（現在はアテンションの出力射影とMLPの下方射影）について、
各トランスフォーマーレイヤー内の関連する行列を特定し、
それらを関連する「拒否方向」に対して直交化することで、
その行列との乗算結果におけるその方向の表現を抑制します。

拒否方向は、「有害な」プロンプトと「無害な」プロンプトの例に対する最初のトークンの残差の平均の差として、
各レイヤーについて計算されます。

アブレーションプロセスは、いくつかの最適化可能なパラメータによって制御されます。

* `direction_index`: 拒否方向のインデックス、または特別な値 `per layer`。これは、各レイヤーがそのレイヤーに関連付けられた拒否方向を使用してアブレーションされるべきであることを示します。
* `max_weight`、`max_weight_position`、`min_weight`、`min_weight_distance`:
  各コンポーネントについて、これらのパラメータは、レイヤーにわたるアブレーション重みカーネルの形状と位置を記述します。
  次の図はこれを示しています。

<img width="800" height="500" alt="説明" src="https://github.com/user-attachments/assets/82e4b84e-5a82-4faf-b918-ac642f9e4892" />

&nbsp;

既存のabliterationシステムに対するHereticの主な革新点は次のとおりです。

* アブレーション重みカーネルの形状は非常に柔軟であり、自動パラメータ最適化と組み合わせることで、
  コンプライアンスと品質のトレードオフを改善できます。
  非定数アブレーション重みは、以前にMaxime Labonneによって
  [gemma-3-12b-it-abliterated-v2](https://huggingface.co/mlabonne/gemma-3-12b-it-abliterated-v2)で検討されました。
* 拒否方向インデックスは、整数ではなく浮動小数点数です。
  整数でない値の場合、最も近い2つの拒否方向ベクトルが線形補間されます。
  これにより、平均差計算によって特定されたものを超える広大な追加の方向の空間が解放され、
  多くの場合、最適化プロセスが個々のレイヤーに属するものよりも優れた方向を見つけることができます。
* アブレーションパラメータは、コンポーネントごとに個別に選択されます。
  MLPへの介入はアテンションへの介入よりもモデルに損害を与える傾向があることがわかったため、
  異なるアブレーション重みを使用することで、さらなるパフォーマンスを引き出すことができます。


## 先行技術

私は、abliteration技術の以下の一般に公開されている実装を認識しています。

* [AutoAbliteration](https://huggingface.co/posts/mlabonne/714992455492422)
* [abliterator.py](https://github.com/FailSpy/abliterator)
* [wassname's Abliterator](https://github.com/wassname/abliterator)
* [ErisForge](https://github.com/Tsadoq/ErisForge)
* [Removing refusals with HF Transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
* [deccp](https://github.com/AUGMXNT/deccp)

Hereticはゼロから書かれており、これらのプロジェクトのコードは再利用していないことに注意してください。


## 謝辞

Hereticの開発は、以下に基づいています。

* [オリジナルのabliteration論文（Arditi et al. 2024）](https://arxiv.org/abs/2406.11717)
* [Maxime Labonneによるabliterationに関する記事](https://huggingface.co/blog/mlabonne/abliteration)、
  および彼自身のabliteratedモデルのモデルカードからのいくつかの詳細（上記参照）
* [「射影されたabliteration」を説明するJim Laiの記事](https://huggingface.co/blog/grimjim/projected-abliteration)


## ライセンス

Copyright &copy; 2025  Philipp Emanuel Weidmann (<pew@worldwidemann.com>)

このプログラムはフリーソフトウェアです。Free Software Foundationが発行したGNU Affero General Public Licenseのバージョン3、または（あなたの選択により）それ以降のバージョンの条項の下で、再配布および/または変更することができます。

このプログラムは有用であることを期待して配布されていますが、いかなる保証もありません。商品性または特定目的への適合性の黙示の保証さえありません。詳細については、GNU Affero General Public Licenseを参照してください。

このプログラムと共にGNU Affero General Public Licenseのコピーを受け取っているはずです。そうでない場合は、<https://www.gnu.org/licenses/>を参照してください。

**このプロジェクトに貢献することにより、あなたはあなたの貢献を同じライセンスの下でリリースすることに同意します。**
