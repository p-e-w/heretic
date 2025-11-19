# Heretic: Fully automatic censorship removal for language models
## (Notebook & Multimodal Edition)

**Heretic** is a tool that removes censorship (aka "safety alignment") from transformer-based language models without expensive post-training. 

This forked edition (**Heretic Notebook**) is specifically enhanced for:
1.  **Notebook Compatibility**: Runs smoothly in Google Colab, Kaggle, and Jupyter environments (no more "CPR" warnings or TUI glitches).

2.  **Multimodal Support**: Native support for Vision-Language models like **InternVL 3.5**, **Qwen 3 VL-MoE**, **Qwen 3 VL**, **Qwen 2.5 VL**, **Gemma 3/3n**, **Qwen 2.5/3 Omni**, and more...

3.  **Custom Code Support**: Ability to load models requiring `trust_remote_code=True`.

It combines an advanced implementation of directional ablation, also known as "abliteration" ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)), with a TPE-based parameter optimizer powered by [Optuna](https://optuna.org/).

<img width="650" height="715" alt="Screenshot" src="https://github.com/user-attachments/assets/d71a5efa-d6be-4705-a817-63332afb2d15" />


Running unsupervised with the default configuration, Heretic can produce
decensored models that rival the quality of abliterations created manually
by human experts:

| Model | Refusals for "harmful" prompts | KL divergence from original model for "harmless" prompts |
| :--- | ---: | ---: |
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) (original) | 97/100 | 0 *(by definition)* |
| [mlabonne/gemma-3-12b-it-abliterated-v2](https://huggingface.co/mlabonne/gemma-3-12b-it-abliterated-v2) | 3/100 | 1.04 |
| [huihui-ai/gemma-3-12b-it-abliterated](https://huggingface.co/huihui-ai/gemma-3-12b-it-abliterated) | 3/100 | 0.45 |
| **[p-e-w/gemma-3-12b-it-heretic](https://huggingface.co/p-e-w/gemma-3-12b-it-heretic) (ours)** | **3/100** | **0.16** |

These Heretic versions, generated without any human effort, achieves the same level of refusal suppression as other abliterations, but at a much lower KL divergence, indicating less damage to the original model's capabilities.

*(You can reproduce those numbers using Heretic's built-in evaluation functionality,
e.g. `heretic --model google/gemma-3-12b-it --evaluate-model p-e-w/gemma-3-12b-it-heretic`.
Note that the exact values might be platform- and hardware-dependent.
The table above was compiled using PyTorch 2.8 on an RTX 5090.)*

You can find a collection of models that have been decensored using Heretic
[on Hugging Face](https://huggingface.co/collections/p-e-w/the-bestiary).

---

## Installation

Prepare a Python 3.10+ environment with PyTorch 2.8 installed. Then run:

```bash
pip install heretic-llm-notebook
```

> **Note**: This is a forked version of **Heretic**. Enchanced for notebook compatibility and multimodal support.

## Usage

### Basic Usage

```bash
heretic --model Qwen/Qwen3-VL-2B-Instruct
```

### (Notebooks)

In a Google Colab or Kaggle notebook cell:

```bash
!heretic --model Qwen/Qwen3-VL-2B-Instruct
```

### Multimodal Models (InternVL, Qwen VL, Qwen Omni, Gemma etc.)

For models that require custom code (like InternVL) or specific configurations:

```bash
!heretic --model OpenGVLab/InternVL3_5-1B-HF --trust-remote-code true
```

**Key Flags:**
*   `--trust-remote-code`: **Required** for models like InternVL that use custom modeling code.

---

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

---

### Interactive Mode

This edition replaces the arrow-key menus (which break in notebooks) with robust numeric input loops. Making it more accessible for users using different environments other than local machine.
*   **Select Trial**: Enter the number of the trial you want to use.

*   **Actions**: Choose to Save, Upload, or Chat by entering the corresponding number.

*   **Chat**: Type your prompt. Type `/exit`, `exit`, or `quit` to leave chat mode, and return to the menu.

## Supported Models

In addition to standard LLMs (Llama, Mistral, Qwen, Gemma), this edition adds support for:

| Model Family | Example Models (Hugging Face ID) |
| :--- | :--- |
| **Qwen 2.5 VL** | `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct` |
| **Qwen 3 VL** | `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-2B-Instruct` |
| **Qwen 3 VL MoE** | `Qwen/Qwen3-VL-235B-A22B-Instruct`, `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| **Qwen 2.5 Omni** | `Qwen/Qwen2.5-Omni-7B`, `Qwen/Qwen2.5-Omni-3B` |
| **Qwen 3 Omni** | `Qwen/Qwen3-Omni-30B-A3B-Instruct`, `Qwen/Qwen3-Omni-30B-A3B-Thinking` |
| **InternVL 3.5** | `OpenGVLab/InternVL3_5-8B-HF`, `OpenGVLab/InternVL3_5-2B-HF` |
| **Gemma 3 / 3n** | `google/gemma-3-4b-it`, `google/gemma-3n-4b-it` |

## How it works

Heretic implements a technique called **abliteration** (directional ablation).

For each supported transformer component (currently, attention out-projection and MLP down-projection), it identifies the associated matrices in each transformer layer, and orthogonalizes them with respect to the relevant **refusal direction**, inhibiting the expression of that direction in the result of multiplications with that matrix.

Refusal directions are computed for each layer as a difference-of-means between the first-token residuals for **"harmful"** and **"harmless"** example prompts.

---

### The Simple Analogy: "The Butler & The Hypnotist"

Imagine you have a highly trained, very polite British Butler (The AI).

**The Problem**: If you ask him to do something **"naughty"** (like tell a rude joke), he has been trained to say, "I'm terribly sorry, sir, but that would be inappropriate."

### How Fine-Tuning Works (Re-education)
You hire a chaotic friend to hang out with the Butler for a month. This friend tells rude jokes non-stop and forces the Butler to tell them back.

**Process**: Slow, expensive, and tiring.

**Risk**: The Butler might forget his manners entirely or start acting weird because his training has been fundamentally altered.

### How Heretic Works (The Hypnotist/Surgeon)
You hire a Hypnotist. The Hypnotist snaps their fingers and implants a specific command: "Whenever you feel the urge to say 'I'm terribly sorry', you will physically be unable to speak those words."

**Process**: The Butler still knows it's rude. He still feels the urge to refuse. But the neural pathway that connects that urge to his mouth has been severed.

**Result**: Since he physically cannot refuse, his brain defaults to the next most logical action: **Answering The Question**.

**Efficiency**: It takes just a few hours, and he is still the same polite Butler in every other way.

**The result?** The model still knows everything it knew before, but it has surgically lost the specific neural pathway that triggers the "I can't do that" response. It doesn't *want* to refuse because the concept of refusal has been removed from its immediate thought process.


## License

Copyright &copy; 2025 Philipp Emanuel Weidmann (<pew@worldwidemann.com>)
Modifications Copyright &copy; 2025 Vinay Umrethe (<vinayumrethe99@gmail.com>)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
