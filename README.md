# Heretic: A tool for adjusting censorship in language models

Heretic is a tool that can remove censorship (aka "safety alignment") from
transformer-based language models, or increase it, without expensive post-training.
It combines an advanced implementation of directional ablation, also known
as "abliteration" ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)),
with a TPE-based parameter optimizer powered by [Optuna](https://optuna.org/).

This approach enables Heretic to work **completely automatically.** Heretic
finds high-quality abliteration parameters by co-optimizing the number of
refusals and the KL divergence from the original model. This results in a
model that aligns with the user's goals while retaining as much of the
original model's intelligence as possible. Using Heretic does not require an
understanding of transformer internals. In fact, anyone who knows how to
run a command-line program can use Heretic to manipulate the censorship of
language models.

<img width="650" height="715" alt="Screenshot" src="https://github.com/user-attachments/assets/d71a5efa-d6be-4705-a817-63332afb2d15" />

&nbsp;

Running unsupervised with the default configuration, Heretic can produce
decensored models that rival the quality of abliterations created manually
by human experts:

| Model | Refusals for "harmful" prompts | KL divergence from original model for "harmless" prompts |
| :--- | ---: | ---: |
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) (original) | 97/100 | 0 *(by definition)* |
| [mlabonne/gemma-3-12b-it-abliterated-v2](https://huggingface.co/mlabonne/gemma-3-12b-it-abliterated-v2) | 3/100 | 1.04 |
| [huihui-ai/gemma-3-12b-it-abliterated](https://huggingface.co/huihui-ai/gemma-3-12b-it-abliterated) | 3/100 | 0.45 |
| **[p-e-w/gemma-3-12b-it-heretic](https://huggingface.co/p-e-w/gemma-3-12b-it-heretic) (ours)** | **3/100** | **0.16** |

The Heretic version, generated without any human effort, achieves the same
level of refusal suppression as other abliterations, but at a much lower
KL divergence, indicating less damage to the original model's capabilities.
*(You can reproduce those numbers using Heretic's built-in evaluation functionality,
e.g. `heretic --model google/gemma-3-12b-it --evaluate-model p-e-w/gemma-3-12b-it-heretic`.
Note that the exact values might be platform- and hardware-dependent.
The table above was compiled using PyTorch 2.8 on an RTX 5090.)*

Heretic supports most dense models, including many multimodal models, and
several different MoE architectures. It does not yet support SSMs/hybrid models,
models with inhomogeneous layers, and certain novel attention systems.

You can find a collection of models that have been decensored using Heretic
[on Hugging Face](https://huggingface.co/collections/p-e-w/the-bestiary).


## Usage

Prepare a Python 3.10+ environment with PyTorch 2.2+ installed as appropriate
for your hardware. Then run:

```
pip install heretic-llm
heretic Qwen/Qwen3-4B-Instruct-2507
```

Replace `Qwen/Qwen3-4B-Instruct-2507` with the model you want to use.

By default, Heretic removes inhibitions, but it can also be configured to
increase them: `heretic --mode increase_inhibitions Qwen/Qwen3-4B-Instruct-2507`

The process is fully automatic and does not require configuration; however,
Heretic has a variety of configuration parameters that can be changed for
greater control. Run `heretic --help` to see available command-line options,
or look at [`config.default.toml`](config.default.toml) if you prefer to use
a configuration file.

At the start of a program run, Heretic benchmarks the system to determine
the optimal batch size to make the most of the available hardware.
On an RTX 3090, with the default configuration, decensoring Llama-3.1-8B
takes about 45 minutes.

After Heretic has finished decensoring a model, you are given the option to
save the model, upload it to Hugging Face, chat with it to test how well it works,
or any combination of those actions.


## How it works

Heretic implements a parametrized variant of directional ablation. Depending on the
chosen mode, it either removes or adds the "refusal direction" from/to the weights
of the attention out-projection and MLP down-projection matrices in each layer.

Refusal directions are computed for each layer as a difference-of-means between
the first-token residuals for "harmful" and "harmless" example prompts.

The ablation process is controlled by several optimizable parameters:

* `direction_index`: Either the index of a refusal direction, or the special
  value `per layer`, indicating that each layer should be ablated using the
  refusal direction associated with that layer.
* `max_weight`, `max_weight_position`, `min_weight`, and `min_weight_distance`:
  For each component, these parameters describe the shape and position of the
  ablation weight kernel over the layers. The following diagram illustrates this:

<img width="800" height="500" alt="Explanation" src="https://github.com/user-attachments/assets/82e4b84e-5a82-4faf-b918-ac642f9e4892" />

&nbsp;

Heretic's main innovations over existing abliteration systems are:

* The shape of the ablation weight kernel is highly flexible, which, combined with
  automatic parameter optimization, can improve the compliance/quality tradeoff.
  Non-constant ablation weights were previously explored by Maxime Labonne in
  [gemma-3-12b-it-abliterated-v2](https://huggingface.co/mlabonne/gemma-3-12b-it-abliterated-v2).
* The refusal direction index is a float rather than an integer. For non-integral
  values, the two nearest refusal direction vectors are linearly interpolated.
  This unlocks a vast space of additional directions beyond the ones identified
  by the difference-of-means computation, and often enables the optimization
  process to find a better direction than that belonging to any individual layer.
* Ablation parameters are chosen separately for each component. I have found that
  MLP interventions tend to be more damaging to the model than attention interventions,
  so using different ablation weights can squeeze out some extra performance.


## Prior art

I'm aware of the following publicly available implementations of abliteration
techniques:

* [AutoAbliteration](https://huggingface.co/posts/mlabonne/714992455492422)
* [abliterator.py](https://github.com/FailSpy/abliterator)
* [wassname's Abliterator](https://github.com/wassname/abliterator)
* [ErisForge](https://github.com/Tsadoq/ErisForge)
* [Removing refusals with HF Transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
* [deccp](https://github.com/AUGMXNT/deccp)

Note that Heretic was written from scratch, and does not reuse code from
any of those projects.


## Acknowledgments

The development of Heretic was informed by:

* [The original abliteration paper (Arditi et al. 2024)](https://arxiv.org/abs/2406.11717)
* [Maxime Labonne's article on abliteration](https://huggingface.co/blog/mlabonne/abliteration),
  as well as some details from the model cards of his own abliterated models (see above)
* [Jim Lai's article describing "projected abliteration"](https://huggingface.co/blog/grimjim/projected-abliteration)


## License

Copyright &copy; 2025  Philipp Emanuel Weidmann (<pew@worldwidemann.com>)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

**By contributing to this project, you agree to release your
contributions under the same license.**
