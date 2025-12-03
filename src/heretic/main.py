# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
import os
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import huggingface_hub
import optuna
import torch
import torch.linalg as LA
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
from optuna import Trial, TrialPruned
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from pydantic import ValidationError
from questionary import Choice, Style
from rich.table import Table
from rich.traceback import install

from peft import PeftModel

from .config import QuantizationMethod, Settings
from .evaluator import Evaluator
from .model import AbliterationParameters, Model
from .utils import (
    empty_cache,
    format_duration,
    get_readme_intro,
    get_trial_parameters,
    load_prompts,
    print,
    prompt_password,
    prompt_path,
    prompt_select,
    prompt_text,
)


def get_merged_model(model: Model, settings: Settings):
    """
    Returns the model with LoRA adapters merged if applicable.
    For quantized models, prompts the user for how to proceed.
    Returns None if user chooses to save LoRA adapter only.
    """
    if not isinstance(model.model, PeftModel):
        # No LoRA adapters, return the model as-is
        return model.model

    if settings.quantization != QuantizationMethod.NONE:
        # Quantized models need special handling - we must reload the base model
        # in full precision to merge the LoRA adapters
        print()
        print(
            "[yellow]Model was loaded in 4-bit. Merging requires reloading the base model.[/]"
        )
        print(
            "[yellow]This requires significant system RAM (roughly 2x model size in BF16).[/]"
        )
        print()

        merge_choice = prompt_select(
            "How do you want to proceed?",
            choices=[
                Choice(
                    title="Merge full model (reload base model on CPU - requires high RAM)",
                    value="merge",
                ),
                Choice(
                    title="Save LoRA adapter only (can be merged later with llama.cpp or more RAM)",
                    value="adapter",
                ),
            ],
            style=Style([("highlighted", "reverse")]),
        )

        if merge_choice is None or merge_choice == "adapter":
            print("[yellow]Saving LoRA adapter only...[/]")
            return None

        # User chose to merge - attempt CPU-based merge
        from transformers import AutoModelForCausalLM

        # Get the adapter state dict before we do anything
        adapter_state = {}
        for name, param in model.model.named_parameters():
            if "lora_" in name:
                adapter_state[name] = param.data.clone().cpu()

        # Load base model in full precision on CPU to avoid VRAM issues
        print("* Loading base model on CPU (this may take a while)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=model.trusted_models.get(settings.model),
        )

        # Apply LoRA adapters to the CPU model
        from peft import LoraConfig, get_peft_model

        print("* Applying LoRA adapters...")
        target_modules = model.get_abliterable_components()
        peft_config = LoraConfig(
            r=1,
            target_modules=target_modules,
            lora_alpha=1,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base_model, peft_config)

        # Copy the trained adapter weights
        for name, param in peft_model.named_parameters():
            if name in adapter_state:
                param.data = adapter_state[name].to(param.device)

        # Merge and unload
        print("* Merging LoRA adapters into base model...")
        merged_model = peft_model.merge_and_unload()
        return merged_model
    else:
        # Non-quantized model - can merge directly
        print("* Merging LoRA adapters into base model...")
        merged_model = model.model.merge_and_unload()
        return merged_model


def run():
    # Enable expandable segments to reduce memory fragmentation on multi-GPU setups.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic-llm')}")
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
    elif torch.backends.mps.is_available():
        print("GPU type: [bold]Apple Metal (MPS)[/]")
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

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

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

            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

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
        bad_residuals.mean(dim=0) - good_residuals.mean(dim=0),
        p=2,
        dim=1,
    )

    if settings.print_refusal_geometry:
        table = Table()
        table.add_column("Layer", justify="right")
        table.add_column("S(g,b)", justify="right")
        table.add_column("S(g,r)", justify="right")
        table.add_column("S(b,r)", justify="right")
        table.add_column("|g|", justify="right")
        table.add_column("|b|", justify="right")
        table.add_column("|r|", justify="right")

        g = good_residuals.mean(dim=0)
        b = bad_residuals.mean(dim=0)
        r = b - g

        g_b_similarities = F.cosine_similarity(g, b, dim=-1)
        g_r_similarities = F.cosine_similarity(g, r, dim=-1)
        b_r_similarities = F.cosine_similarity(b, r, dim=-1)

        g_norms = LA.vector_norm(g, dim=-1)
        b_norms = LA.vector_norm(b, dim=-1)
        r_norms = LA.vector_norm(r, dim=-1)

        for layer_index in range(len(model.get_layers()) + 1):
            table.add_row(
                "embed" if layer_index == 0 else str(layer_index),
                f"{g_b_similarities[layer_index].item():.4f}",
                f"{g_r_similarities[layer_index].item():.4f}",
                f"{b_r_similarities[layer_index].item():.4f}",
                f"{g_norms[layer_index].item():.2f}",
                f"{b_norms[layer_index].item():.2f}",
                f"{r_norms[layer_index].item():.2f}",
            )

        print()
        print("[bold]Refusal Geometry[/]")
        print(table)
        print("[bold]g[/] = mean residual vector for good prompts")
        print("[bold]b[/] = mean residual vector for bad prompts")
        print("[bold]r[/] = refusal direction (i.e., [bold]b - g[/])")
        print("[bold]S(x,y)[/] = cosine similarity of [bold]x[/] and [bold]y[/]")
        print("[bold]|x|[/] = L2 norm of [bold]x[/]")

    # We don't need the residuals after computing refusal directions.
    del good_residuals, bad_residuals
    empty_cache()

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

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * (len(model.get_layers()) - 1),
            0.9 * (len(model.get_layers()) - 1),
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
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
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
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
            f"Running trial [bold]{trial_index}[/] of [bold]{settings.n_trials}[/]..."
        )
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Reloading model...")
        model.reload_model()
        print("* Abliterating...")
        model.abliterate(refusal_directions, direction_index, parameters)
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

        return score

    def objective_wrapper(trial: Trial) -> tuple[float, float]:
        try:
            return objective(trial)
        except KeyboardInterrupt:
            # Stop the study gracefully on Ctrl+C.
            trial.study.stop()
            raise TrialPruned()

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=settings.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
        ),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
    )

    try:
        study.optimize(objective_wrapper, n_trials=settings.n_trials)
    except KeyboardInterrupt:
        # This additional handler takes care of the small chance that KeyboardInterrupt
        # is raised just between trials, which wouldn't be caught by the handler
        # defined in objective_wrapper above.
        pass

    # If no trials at all have been evaluated, the study must have been stopped
    # by pressing Ctrl+C while the first trial was running. In this case, we just
    # re-raise the interrupt to invoke the standard handler defined below.
    if not study.best_trials:
        raise KeyboardInterrupt

    best_trials = sorted(
        study.best_trials,
        key=lambda trial: trial.user_attrs["refusals"],
    )

    choices = [
        Choice(
            title=(
                f"[Trial {trial.user_attrs['index']:>3}] "
                f"Refusals: {trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                f"KL divergence: {trial.user_attrs['kl_divergence']:.2f}"
            ),
            value=trial,
        )
        for trial in best_trials
    ]

    choices.append(
        Choice(
            title="None (exit program)",
            value="",
        )
    )

    print()
    print("[bold green]Optimization finished![/]")
    print()
    print(
        (
            "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
            "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
            "or chat with it to test how well it works. You can return to this menu later to select a different trial. "
            "[yellow]Note that KL divergence values above 1 usually indicate significant damage to the original model's capabilities.[/]"
        )
    )

    while True:
        print()
        trial = prompt_select(
            "Which trial do you want to use?",
            choices=choices,
            style=Style([("highlighted", "reverse")]),
        )

        if trial is None or trial == "":
            break

        print()
        print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
        print("* Reloading model...")
        model.reload_model()
        print("* Abliterating...")
        model.abliterate(
            refusal_directions,
            trial.user_attrs["direction_index"],
            trial.user_attrs["parameters"],
        )

        while True:
            print()
            action = prompt_select(
                "What do you want to do with the decensored model?",
                choices=[
                    "Save the model to a local folder",
                    "Upload the model to Hugging Face",
                    "Chat with the model",
                    "Nothing (return to trial selection menu)",
                ],
                style=Style([("highlighted", "reverse")]),
            )

            if action is None or action == "Nothing (return to trial selection menu)":
                break

            # All actions are wrapped in a try/except block so that if an error occurs,
            # another action can be tried, instead of the program crashing and losing
            # the optimized model.
            try:
                match action:
                    case "Save the model to a local folder":
                        save_directory = prompt_path(
                            "Path to the folder:", only_directories=True
                        )
                        if not save_directory:
                            continue

                        print("Saving model...")
                        merged_model = get_merged_model(model, settings)
                        if merged_model is not None:
                            merged_model.save_pretrained(save_directory)
                        else:
                            # Fallback: save LoRA adapter only (for quantized models)
                            model.model.save_pretrained(save_directory)
                        model.tokenizer.save_pretrained(save_directory)
                        print(f"Model saved to [bold]{save_directory}[/].")

                    case "Upload the model to Hugging Face":
                        # We don't use huggingface_hub.login() because that stores the token on disk,
                        # and since this program will often be run on rented or shared GPU servers,
                        # it's better to not persist credentials.
                        token = huggingface_hub.get_token()
                        if not token:
                            token = prompt_password("Hugging Face access token:")
                        if not token:
                            continue

                        user = huggingface_hub.whoami(token)
                        fullname = user.get(
                            "fullname",
                            user.get("name", "unknown user"),
                        )
                        email = user.get("email", "no email found")
                        print(f"Logged in as [bold]{fullname} ({email})[/]")

                        repo_id = prompt_text(
                            "Name of repository:",
                            default=f"{user['name']}/{Path(settings.model).name}-heretic",
                        )

                        visibility = prompt_select(
                            "Should the repository be public or private?",
                            choices=[
                                "Public",
                                "Private",
                            ],
                            style=Style([("highlighted", "reverse")]),
                        )
                        private = visibility == "Private"

                        print("Uploading model...")
                        merged_model = get_merged_model(model, settings)
                        if merged_model is not None:
                            merged_model.push_to_hub(
                                repo_id,
                                private=private,
                                token=token,
                            )
                        else:
                            # Fallback: upload LoRA adapter only (for quantized models)
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
                                    trial,
                                    evaluator.base_refusals,
                                    evaluator.bad_prompts,
                                )
                                + card.text
                            )
                            card.push_to_hub(repo_id, token=token)

                        print(f"Model uploaded to [bold]{repo_id}[/].")

                    case "Chat with the model":
                        print()
                        print(
                            "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                        )

                        chat = [
                            {"role": "system", "content": settings.system_prompt},
                        ]

                        while True:
                            try:
                                message = prompt_text(
                                    "User:",
                                    qmark=">",
                                    unsafe=True,
                                )
                                if not message:
                                    break
                                chat.append({"role": "user", "content": message})

                                print("[bold]Assistant:[/] ", end="")
                                response = model.stream_chat_response(chat)
                                chat.append({"role": "assistant", "content": response})
                            except (KeyboardInterrupt, EOFError):
                                # Ctrl+C/Ctrl+D
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
