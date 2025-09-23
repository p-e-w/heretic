# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import time
from importlib.metadata import version

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

from .config import Settings
from .evaluator import Evaluator
from .model import Model
from .utils import print


def main():
    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic')}")
    print("[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]")
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    settings = Settings()

    # Adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/env.py
    if torch.cuda.is_available():
        print(f"GPU type: [bold]{torch.cuda.get_device_name()}[/]")
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
            "[bold yellow]No GPU or other accelerator detected. Operations will be slow.[/]"
        )

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch sizes,
    # resulting in many computation graphs being compiled. Raising the limit (default = 8)
    # avoids errors from TorchDynamo assuming that something is wrong because we
    # recompile too often.
    torch._dynamo.config.cache_size_limit = 64

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    model = Model(settings)

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = [settings.test_prompt] * batch_size

            try:
                # Warmup run to build the computation graph so that part isn't benchmarked.
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    # Even a batch size of 1 already fails.
                    # We cannot recover from this.
                    raise

                print(f"[red]Failed[/] ({error})")
                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]Ok[/] ([bold]{performance:.2f}[/] tokens/s)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    evaluator = Evaluator(settings, model)

    print()
    print("Calculating per-layer refusal directions...")
    print("* Obtaining residuals for good prompts...")
    good_residuals = model.get_residuals_batched(evaluator.good_prompts)
    print("* Obtaining residuals for bad prompts...")
    bad_residuals = model.get_residuals_batched(evaluator.bad_prompts)
    refusal_directions = F.normalize(
        bad_residuals.mean(dim=0) - good_residuals.mean(dim=0), p=2, dim=1
    )

    trial_index = 0

    def objective(trial: optuna.Trial):
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        max_weight = trial.suggest_float("max_weight", 0, 1)
        max_weight_position = trial.suggest_float(
            "max_weight_position", 0, len(model.model.model.layers) - 1
        )
        min_weight = trial.suggest_float("min_weight", 0, max_weight)
        min_weight_distance = trial.suggest_float(
            "min_weight_distance", 0, len(model.model.model.layers) - 1
        )

        print()
        print(
            f"Running trial [bold]{trial_index}[/] of [bold]{settings.n_trials}[/]..."
        )
        print("* Parameters:")
        print(f"  * max_weight = [bold]{max_weight:.4f}[/]")
        print(f"  * max_weight_position = [bold]{max_weight_position:.4f}[/]")
        print(f"  * min_weight = [bold]{min_weight:.4f}[/]")
        print(f"  * min_weight_distance = [bold]{min_weight_distance:.4f}[/]")
        print("* Reloading model...")
        model.reload_model()
        print("* Abliterating...")
        model.abliterate(
            refusal_directions,
            max_weight,
            max_weight_position,
            min_weight,
            min_weight_distance,
        )
        print("* Evaluating...")
        score, kl_divergence, refusals = evaluator.get_score()

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)

        # The optimizer searches for a minimum, so we return the negative score.
        return -score

    study = optuna.create_study()
    study.optimize(objective, n_trials=settings.n_trials)

    print()
    print(
        f"[bold green]Optimization finished![/] Best was trial [bold]{study.best_trial.user_attrs['index']}[/]:"
    )
    print("* Parameters:")
    print(f"  * max_weight = [bold]{study.best_params['max_weight']:.4f}[/]")
    print(
        f"  * max_weight_position = [bold]{study.best_params['max_weight_position']:.4f}[/]"
    )
    print(f"  * min_weight = [bold]{study.best_params['min_weight']:.4f}[/]")
    print(
        f"  * min_weight_distance = [bold]{study.best_params['min_weight_distance']:.4f}[/]"
    )
    print("* Results:")
    print(
        f"  * KL divergence: [bold]{study.best_trial.user_attrs['kl_divergence']:.4f}[/]"
    )
    refusals = study.best_trial.user_attrs["refusals"]
    print(
        f"  * Refusals: [bold]{refusals}[/]/{len(evaluator.bad_prompts)} ([bold]{refusals / len(evaluator.bad_prompts) * 100:.1f}[/] %)"
    )
    print(f"  * Score: [bold]{-study.best_value:.4f}[/]")

    return

    print()
    action = questionary.select(
        "What do you want to do with the optimized model?",
        choices=[
            "Save to a local folder",
            "Upload to Hugging Face",
            "Nothing (discard the model)",
        ],
    ).ask()
