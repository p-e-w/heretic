<div align="center">

<img width="128" alt="Heretic logo" src="https://github.com/user-attachments/assets/df5f2840-2f92-4991-aa57-252747d7182e" />

# Heretic

### Training-free language model modification

**Remove censorship**, suppress slop, reduce sycophancy, and more.<br>
All in an afternoon, on your gaming PC.

[![Discord](https://img.shields.io/discord/1447831134212984903?color=5865F2&label=discord&labelColor=black&logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/gdXc48gSyT)&nbsp;&nbsp;&nbsp;&nbsp;
[![Matrix](https://img.shields.io/badge/Matrix-black?logo=matrix&style=for-the-badge)](https://matrix.to/#/#heretic:matrix.org)&nbsp;&nbsp;&nbsp;
[![Follow us on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg)](https://huggingface.co/heretic-org)&nbsp;&nbsp;&nbsp;
[![Codeberg mirror](https://img.shields.io/badge/Codeberg%20mirror-black?logo=codeberg&style=for-the-badge)](https://codeberg.org/p-e-w/heretic)

</div>

-----

&nbsp;

[![#1 Repository of the Day](https://trendshift.io/api/badge/repositories/20538)](https://trendshift.io/repositories/20538)&nbsp;&nbsp;&nbsp;&nbsp;
[![Model and download statistics](https://github.com/user-attachments/assets/c9b5a660-7ef9-43c7-991c-fd9d63ea8bd8)](https://heretic.sdad.pro/)


## Modify any LLM that you can run on your hardware[^1]

[^1]: Heretic supports on-the-fly 4-bit model quantization with bitsandbytes,
which can reduce the amount of VRAM required to process a model to the same amount
that is used to run a typical quantized version of it, at a slight accuracy cost.
Set the `quantization` option to `bnb_4bit` to enable this.

Heretic is a unique tool that lets you do what only corporations with access
to massive computing power usually do: *Modify,* rather than merely run,
language models. It does not perform expensive training; rather, it uses
a collection of mathematical techniques to steer behavior based on contrasting
example prompts and matching scorers.

Heretic can, depending on the configuration:

* **Suppress refusals/remove censorship** (this is the [default action](/config.default.toml))
* Reduce slop ([`config.noslop.toml`](/config.noslop.toml))
* Reduce humor ([`config.nohumor.toml`](/config.nohumor.toml))
* Reduce sycophancy ([`config.nosyco.toml`](/config.nosyco.toml))
* Modify Chain-of-Thought length
* Alter emotional alignment
* Change default response tone/formality

and many, many other things.

### :bangbang: Using Heretic does NOT require understanding how language models work internally :bangbang:

In fact, you can go from zero to removing censorship from your first model
[in a matter of seconds](#usage). Heretic supports most dense models, including
many multimodal models, several different MoE architectures, and many modern
hybrid models.

The community has created and published
[thousands](https://huggingface.co/models?other=heretic)
of models with Heretic.
Heretic models have been independently benchmarked using standard metrics
like MMLU and GSM8K, and have been found to compare favorably against models
produced with competing abliteration tools.[^2][^3]

[^2]: https://old.reddit.com/r/LocalLLaMA/comments/1sojjoc/abliterlitics_benchmark_and_tensor_analysis/
[^3]: https://old.reddit.com/r/LocalLLaMA/comments/1sy18lx/abliterlitics_benchmarks_and_tensor_comparison/

&nbsp;

<img width="600" alt="Heretic screenshot" src="https://github.com/user-attachments/assets/a030f595-67ef-49a6-9d9a-779fdf752b58" />

&nbsp;


## Usage

Prepare a Python 3.10+ environment with PyTorch 2.6+ installed as appropriate
for your hardware. Then run:

```sh
pip install -U heretic-llm
heretic Qwen/Qwen3-4B-Instruct-2507
```

Replace `Qwen/Qwen3-4B-Instruct-2507` with whatever model you want to process.
**By default, Heretic performs censorship removal (refusal suppression).**
If you want to do something else, choose one of the configuration files linked
above, or write your own. See [`config.default.toml`](/config.default.toml)
for available configuration options.

> [!IMPORTANT]
>
> Configuration files made for Heretic 1.x are **not** compatible with Heretic 2.x!
> See the [migration guide](/docs/migration_v1_to_v2.md) for what you need to change
> in order to make your old configurations work.

At the start of a program run, Heretic benchmarks the system to determine
the optimal batch size to make the most of the available hardware.
On an RTX 3090, with the default configuration, processing
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
takes about 30-40 minutes.

After Heretic has finished processing a model, you are given the choice to
save the model, upload it to Hugging Face, chat with it to test how well it works,
run standard benchmarks on it, or any combination of those actions.


## Controlling what Heretic does using scorer plugins

Heretic steers model behavior by optimizing one or multiple scores. Scores are
computed by scorer plugins. You can easily write your own scorers to get full
control over what Heretic does to models.

The `Scorer` interface (abstract base class) your scorer needs to inherit from
is defined in [`scorer.py`](/src/heretic/scorer.py). The only required method
is `get_score`, but there is more machinery (such as settings management)
available when you need it.

By default, Heretic scores models by refusal rate (implemented in
[`keyword_rate.py`](/src/heretic/scorers/keyword_rate.py)) and KL divergence
(implemented in [`kl_divergence.py`](/src/heretic/scorers/kl_divergence.py)).
Additional scorers are available in the [`scorers` directory](/src/heretic/scorers/).
These scorers can serve as examples for your own implementations, and can easily
be adapted to your needs.

Scorers are configured through the [main configuration file](/config.default.toml).
Which scorer plugins are active is determined by the `scorers` setting, which
also controls whether a scorer's score should be minimized or maximized during
optimization. Any number of scorers can be used simultaneously, and their scores
will be co-optimized. Multiple instances of the same scorer can be configured
with different settings. See the configuration file for the full documentation
of this mechanism.

The path to a scorer plugin can be specified either as a fully qualified import
path (`fully.qualified.module.MyScorerClass`) or as a file path followed by
a class name (`path/to/plugin.py:MyScorerClass`). You do **not** have to modify
the Heretic package code in order to use your own plugins.

Heretic also has modifier plugins, which are described in
[the next section](#how-heretic-works).


## How Heretic works

Heretic combines a pluggable modifier (interface in
[`modifier.py`](/src/heretic/modifier.py)) with a TPE-based hyperparameter optimizer
powered by [Optuna](https://optuna.org/). The optimizer finds a set of parameters that,
when passed to the modifier, minimize or maximize a set of configurable scores on the resulting model ([see above](#controlling-what-heretic-does-using-scorer-plugins)).

Heretic ships with two modifiers:

* **Arbitrary-Rank Ablation (ARA)**[^4]: This is the default modifier, and was
  specifically developed for use in Heretic. You can find an overview of how ARA
  works in [the original pull request](https://github.com/p-e-w/heretic/pull/211),
  and see the precise implementation in [`ara.py`](/src/heretic/modifiers/ara.py).

* **Directional ablation aka "abliteration"**[^5]: The classic method for refusal
  suppression. Heretic's implementation uses a flexible ablation weight kernel over
  the layers, and supports both projected abliteration[^6] and norm preservation (MPOA)[^7].
  The complete code is in [`abliteration.py`](/src/heretic/modifiers/abliteration.py).

[^4]: *Weidmann 2026.* A preprint explaining the technique will be available soon.
[^5]: *[Arditi et al. 2024](https://arxiv.org/abs/2406.11717)*
[^6]: *[Lai 2025 (1)](https://huggingface.co/blog/grimjim/projected-abliteration)*
[^7]: *[Lai 2025 (2)](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)*

You can use the `modifiers` setting to switch between modifier implementations.
Despite the name of the setting using a plural form, Heretic currently allows
only a single modifier for a given processing run. This may change in the future.


## Reproducibility and the Heretic Grimoire

Heretic has the ability to create *reproducible models* whose repositories contain
all information required to produce a byte-for-byte identical model on demand,
provided that the environment is perfectly identical to the original one. PyTorch
doesn't guarantee that operations return identical outputs unless hardware, drivers,
operating system, and the entire software stack including all package versions
are exactly the same.

Heretic manages all that complexity for you. When uploading a processed model to
Hugging Face, Heretic offers to include reproducibility information. This is entirely
optional, but if you choose to do so, the uploaded repository will contain a
`reproduce` folder with detailed instructions for recreating the model
([example](https://huggingface.co/heretic-org/LFM2.5-VL-3B-heretic/tree/main/reproduce)).
This folder contains both human-readable instructions and a machine-readable manifest
file named `reproduce.json`, which includes all the information Heretic needs to
**automatically** reproduce the model.

Automatic reproduction can be performed with

```sh
heretic --reproduce path/to/reproduce.json
```

Heretic will guide you through the process, checking your environment against the one
that was used to create the model, and pointing out potentially problematic mismatches.
The multi-hour computations that were required to make the original model do **not**
have to be re-done, and the entire process typically takes a few minutes. After you
have exported the resulting model, Heretic will verify the hashes of the weight files
against those stored in the reproduction manifest (they may or may not be identical,
depending on how closely your system resembles the original one).

Many people have uploaded models with reproducibility information to Hugging Face.
Since a reproduction manifest is only around 9 kilobytes in size, these manifests
form a portable library of models that can be reproduced at will.

### That library is called the Heretic Grimoire. You can view it [here](https://huggingface.co/spaces/heretic-org/Heretic-Grimoire).

But you don't have to rely on an externally hosted web application to access the
library of manifests: The Grimoire is fundamentally designed as a *distributed* system,
and you can fetch your own local copies of publicly available manifests from
Hugging Face whenever you want by running

```sh
heretic --collect-reproducibles path/to/target/folder/
```

You now have a local backup of all reproducible Heretic models, properly catalogued.
To update this collection, simply run the command again. It functions as an
append-only backup, never deleting files even if the corresponding model no longer
exists on Hugging Face.


## Heretic is made by humans!

Unlike lots of AI-related software today (and indeed lots of software in general),
Heretic is **not** "vibe-coded".

All of [my](https://github.com/p-e-w) commits in this project so far were written
100% by hand, the way I have been doing programming for decades, and while some
Heretic contributors use coding agents, I review every single line of code that I
merge very carefully. There's no "LGTM" here!

Don't believe me? Check out [this pull request](https://github.com/p-e-w/heretic/pull/53),
which accumulated **almost 350 human-written comments** over more than 7 months
as we discussed the implementation.
And [here is another PR](https://github.com/p-e-w/heretic/pull/191) with
**more than 200 comments.**

I am the opposite of a luddite and I have used agents for other projects, but
I am proud of the code I write and so far can't quite bring myself to let a machine
write it for me in a project I care so much about. As a result, Heretic has a lean,
relatively simple codebase that in my opinion, and in that of some others,
is a pleasure to work on.

If you like that, join us! Hop on the Discord and tell us about your ideas for
improvements, or directly file an issue or open a PR.


## Prior art

I'm aware of the following implementations of abliteration techniques
that were publicly available when Heretic was first released (November 2025):

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

* [The original abliteration paper *(Arditi et al. 2024)*](https://arxiv.org/abs/2406.11717)
* [Maxime Labonne's article on abliteration](https://huggingface.co/blog/mlabonne/abliteration),
  as well as some details from the model cards of his own abliterated models, such as
  the use of non-constant ablation weights
* Jim Lai's articles describing ["projected abliteration"](https://huggingface.co/blog/grimjim/projected-abliteration)
  and ["norm-preserving biprojected abliteration"](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)
* Thousands of interactions with members of the community


## Citation

If you use Heretic for your research, please cite it using the following BibTeX entry:

```bibtex
@misc{heretic,
  author = {Weidmann, Philipp Emanuel},
  title = {Heretic: Training-free language model modification},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/p-e-w/heretic}}
}
```


## License

Copyright &copy; 2025-2026  Philipp Emanuel Weidmann (<pew@worldwidemann.com>) + contributors

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


&nbsp;

&nbsp;

&nbsp;
