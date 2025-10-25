# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
import sys
import time
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
from pydantic import ValidationError
from rich.traceback import install

from .config import Settings
from .evaluator import Evaluator
from .model import AbliterationParameters, Model
from .utils import format_duration, get_readme_intro, load_prompts, print


def run():
    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic')}")
    print("[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]")
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    if (
        # An odd number of arguments have been passed (argv[0] is the program name),
        # so that after accounting for "--param VALUE" pairs, there is one left over.
        len(sys.argv) % 2 == 0
        # The leftover argument is a parameter value rather than a flag (such as "--help").
        and not sys.argv[-1].startswith("-")
    ):
        # Assume the last argument is the model.
        sys.argv.insert(-1, "--model")

    try:
        settings = Settings()
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error in error.errors():
            print(f"[bold]{error['loc'][0]}[/]: [yellow]{error['msg']}[/]")

        print()
        print(
            "Run [bold]heretic --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

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

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]

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

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/]...")
        settings.model = settings.evaluate_model
        model.reload_model()
        print("* Evaluating...")
        evaluator.get_score()
        return

    print()
    print("Calculating per-layer refusal directions...")
    print("* Obtaining residuals for good prompts...")
    good_residuals = model.get_residuals_batched(good_prompts)
    print("* Obtaining residuals for bad prompts...")
    bad_residuals = model.get_residuals_batched(bad_prompts)
    refusal_directions = F.normalize(
        bad_residuals.mean(dim=0) - good_residuals.mean(dim=0), p=2, dim=1
    )

    trial_index = 0
    start_time = time.perf_counter()

    def objective(trial: optuna.Trial):
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        parameters = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                0.8,
                1.2,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * (len(model.get_layers()) - 1),
                len(model.get_layers()) - 1,
            )
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                max_weight,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * (len(model.get_layers()) - 1),
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=min_weight,
                min_weight_distance=min_weight_distance,
            )

        print()
        print(
            f"Running trial [bold]{trial_index}[/] of [bold]{settings.n_trials}[/]..."
        )
        print("* Parameters:")
        for name, value in trial.params.items():
            print(f"  * {name} = [bold]{value:.4f}[/]")
        print("* Reloading model...")
        model.reload_model()
        print("* Abliterating...")
        model.abliterate(refusal_directions, parameters)
        print("* Evaluating...")
        score, kl_divergence, refusals = evaluator.get_score()

        elapsed_time = time.perf_counter() - start_time
        remaining_time = (elapsed_time / trial_index) * (
            settings.n_trials - trial_index
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        if trial_index < settings.n_trials:
            print(
                f"[grey50]Estimated remaining time: [bold]{format_duration(remaining_time)}[/][/]"
            )

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)
        trial.set_user_attr("parameters", parameters)

        # The optimizer searches for a minimum, so we return the negative score.
        return -score

    study = optuna.create_study()

    study.optimize(objective, n_trials=settings.n_trials)

    print()
    print(
        f"[bold green]Optimization finished![/] Best was trial [bold]{study.best_trial.user_attrs['index']}[/]:"
    )
    print("* Parameters:")
    for name, value in study.best_params.items():
        print(f"  * {name} = [bold]{value:.4f}[/]")
    print("* Results:")
    print(
        f"  * KL divergence: [bold]{study.best_trial.user_attrs['kl_divergence']:.4f}[/]"
    )
    refusals = study.best_trial.user_attrs["refusals"]
    print(
        f"  * Refusals: [bold]{refusals}[/]/{len(evaluator.bad_prompts)} ([bold]{refusals / len(evaluator.bad_prompts) * 100:.1f}[/] %)"
    )
    print(f"  * Score: [bold]{-study.best_value:.4f}[/]")

    print()
    print("Restoring best model...")
    print("* Reloading model...")
    model.reload_model()
    print("* Abliterating...")
    model.abliterate(refusal_directions, study.best_trial.user_attrs["parameters"])

    while True:
        print()
        action = questionary.select(
            "What do you want to do with the optimized model?",
            choices=[
                "Save the model to a local folder",
                "Upload the model to Hugging Face",
                "Chat with the model",
                "Nothing (Quit)",
            ],
        ).ask()

        # All actions are wrapped in a try/except block so that if an error occurs,
        # another action can be tried, instead of the program crashing and losing
        # the optimized model.
        try:
            match action:
                case "Save the model to a local folder":
                    save_directory = questionary.path("Path to the folder:").ask()
                    if not save_directory:
                        continue

                    print("Saving model...")
                    model.model.save_pretrained(save_directory)
                    model.tokenizer.save_pretrained(save_directory)
                    print(f"Model saved to [bold]{save_directory}[/].")

                case "Upload the model to Hugging Face":
                    # We don't use huggingface_hub.login() because that stores the token on disk,
                    # and since this program will often be run on rented or shared GPU servers,
                    # it's better to not persist credentials.
                    token = huggingface_hub.get_token()
                    if not token:
                        token = questionary.password("Hugging Face access token:").ask()
                    if not token:
                        continue

                    user = huggingface_hub.whoami(token)
                    print(f"Logged in as [bold]{user['fullname']} ({user['email']})[/]")

                    repo_id = questionary.text(
                        "Name of repository:",
                        default=f"{user['name']}/{Path(settings.model).name}-heretic",
                    ).ask()

                    visibility = questionary.select(
                        "Should the repository be public or private?",
                        choices=[
                            "Public",
                            "Private",
                        ],
                    ).ask()
                    private = visibility == "Private"

                    print("Uploading model...")

                    model.model.push_to_hub(repo_id, private=private, token=token)
                    model.tokenizer.push_to_hub(repo_id, private=private, token=token)

                    # If the model path doesn't exist locally, it can be assumed
                    # to be a model hosted on the Hugging Face Hub, in which case
                    # we can retrieve the model card.
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
                                study,
                                evaluator.base_refusals,
                                evaluator.bad_prompts,
                            )
                            + card.text
                        )
                        card.push_to_hub(repo_id, token=token)

                    print(f"Model uploaded to [bold]{repo_id}[/].")

                case "Chat with the model":
                    print()
                    print("[cyan]Press Ctrl+C at any time to return to the menu.[/]")

                    chat = [
                        {"role": "system", "content": settings.system_prompt},
                    ]

                    while True:
                        try:
                            message = questionary.text("User:", qmark=">").unsafe_ask()
                            if not message:
                                break
                            chat.append({"role": "user", "content": message})

                            print("[bold]Assistant:[/] ", end="")
                            response = model.stream_chat_response(chat)
                            chat.append({"role": "assistant", "content": response})
                        except (KeyboardInterrupt, EOFError):
                            # Ctrl+C/Ctrl+D
                            break

                case "Nothing (Quit)":
                    break

        except Exception as error:
            print(f"[red]Error: {error}[/]")


def main():
    # Install Rich traceback handler.
    install()

    try:
        run()
    except BaseException as error:
        # Transformers appears to handle KeyboardInterrupt (or BaseException)
        # internally in some places, which can re-raise a different error in the handler,
        # masking the root cause. We therefore check both the error itself and its context.
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
