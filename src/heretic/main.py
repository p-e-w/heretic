# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# ruff: noqa: E402

import sys

# Ensure standard output/error use UTF-8 instead of system default charmap (e.g. cp1252 on Windows).
for stream in (sys.stdout, sys.stderr):
    if (
        hasattr(stream, "reconfigure")
        and (getattr(stream, "encoding", "") or "").lower() != "utf-8"
    ):
        stream.reconfigure(encoding="utf-8")  # type: ignore

from .config import Settings


def _is_help_invocation() -> bool:
    args = sys.argv[1:]
    return "-h" in args or "--help" in args


# Parse and handle CLI help before importing heavyweight ML/runtime dependencies.
if _is_help_invocation():
    Settings()  # ty:ignore[missing-argument]

# FIXME: Rich progress bars are currently disabled because of rendering issues
#        when used from multiple threads in parallel (e.g. by huggingface_hub).
"""
from .progress import patch_tqdm

# This patches tqdm class definitions, which must happen
# before any other module imports tqdm.
patch_tqdm()
"""

import os
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import huggingface_hub
import lm_eval
import numpy as np
import torch
import questionary
from huggingface_hub import HfApi, ModelCard, ModelCardData
from lm_eval.models.huggingface import HFLM
from optuna import Trial, TrialPruned
from optuna.trial import TrialState, create_trial
from pydantic import ValidationError
from questionary import Choice, Style
from rich.table import Table
from rich.traceback import install

from . import core
from .analyzer import Analyzer
from .config import ExportStrategy, QuantizationMethod
from .evaluator import Evaluator
from .model import AbliterationParameters, Model, get_model_class
from .reproduce import (
    check_environment,
    collect_reproducibles,
    load_reproduction_information,
)
from .system import empty_cache, get_accelerator_info
from .utils import (
    ask_if_unset,
    format_duration,
    format_exception,
    get_file_sha256,
    get_readme_intro,
    get_trial_parameters,
    is_hf_path,
    load_prompts,
    print,
    print_memory_usage,
    upload_reproduce_folder,
)


def obtain_export_strategy(
    settings: Settings,
    model: Model,
) -> ExportStrategy | None:
    """
    Gets the export strategy from settings or prompts the user.
    Provides info to the user if the model is quantized on memory use.
    Returns an export strategy, or None if cancelled.
    """

    if (
        settings.quantization == QuantizationMethod.BNB_4BIT
        and settings.export_strategy is None
    ):
        print()
        print(
            "The model was loaded with quantization. Merging requires reloading the base model."
        )
        print(
            "[yellow]WARNING: CPU merging requires dequantizing the entire model to system RAM.[/]"
        )
        print("[yellow]This can lead to system freezes if you run out of memory.[/]")

        try:
            # Estimate memory requirements by loading the model structure on the "meta" device.
            # This doesn't consume actual RAM but allows us to inspect the parameter count/dtype.
            #
            # Suppress warnings during meta device loading (e.g., "Some weights were not initialized").
            # These are expected and harmless since we're only inspecting model structure, not running inference.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                    if settings.model in model.trusted_models
                    else None,
                    **model.revision_kwargs,
                )
                footprint_bytes = meta_model.get_memory_footprint()
                footprint_gb = footprint_bytes / (1024**3)
                print(
                    f"[yellow]Estimated RAM required (excluding overhead): [bold]~{footprint_gb:.2f} GB[/][/]"
                )
        except Exception:
            # Fallback if meta loading fails (e.g. owing to custom model code
            # or bitsandbytes quantization config issues on the meta device).
            print(
                "[yellow]Rule of thumb: You need approximately 3x the parameter count in GB RAM.[/]"
            )
            print(
                "[yellow]Example: A 27B model requires ~80GB RAM. A 70B model requires ~200GB RAM.[/]"
            )

        print()

    return ask_if_unset(
        settings.export_strategy,
        questionary.select(
            "How do you want to export the model?",
            choices=[
                Choice(
                    title="Merge the abliteration LoRA and export the full model"
                    + (
                        ""
                        if settings.quantization == QuantizationMethod.NONE
                        else " (requires sufficient RAM)"
                    ),
                    value=ExportStrategy.MERGE,
                ),
                Choice(
                    title="Export the abliteration LoRA only (can be merged later)",
                    value=ExportStrategy.ADAPTER,
                ),
            ],
            style=Style([("highlighted", "reverse")]),
        ),
    )


def run():
    # Enable expandable segments to reduce memory fragmentation on multi-GPU setups.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic-llm')}")
    print(
        "[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]  [blue underline]https://heretic-project.org[/]"
    )
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    if (
        # There is at least one argument (argv[0] is the program name).
        len(sys.argv) > 1
        # Heretic is being invoked in standard (model processing) mode.
        and "--collect-reproducibles" not in sys.argv
        and "--reproduce" not in sys.argv
        # No model has been explicitly provided.
        and "--model" not in sys.argv
        # The last argument is a parameter value rather than a flag (such as "--help").
        and not sys.argv[-1].startswith("-")
    ):
        # Assume the last argument is the model.
        sys.argv.insert(-1, "--model")

    # Work around the "model" argument being required
    # when Heretic is invoked in a non-processing mode.
    if (
        "--collect-reproducibles" in sys.argv or "--reproduce" in sys.argv
    ) and "--model" not in sys.argv:
        sys.argv.extend(["--model", ""])

    try:
        # The required argument "model" must be provided by the user,
        # either on the command line or in the configuration file.
        settings = Settings()  # ty:ignore[missing-argument]
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error_details in error.errors():
            print(
                f"[bold]{error_details['loc'][0]}[/]: [yellow]{error_details['msg']}[/]"
            )

        print()
        print(
            "Run [bold]heretic --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

    if settings.collect_reproducibles is not None:
        collect_reproducibles(settings.collect_reproducibles)
        return

    reproduction_mode = settings.reproduce is not None

    if settings.reproduce is not None:
        print(f"Loading reproduction information from [bold]{settings.reproduce}[/]...")
        # FIXME: "Reproduction"/"reproducibility" name inconsistency!
        reproduction_information = load_reproduction_information(settings.reproduce)

        if reproduction_information["version"] not in ["1", "2"]:
            print(
                (
                    f"[red]Unsupported file format version: [bold]{reproduction_information['version']}[/].[/] "
                    "Try loading the file with a newer version of Heretic."
                )
            )
            return

        if not check_environment(settings, reproduction_information):
            return

        print()

        verify_hashes = reproduction_information["version"] != "1"

        settings = Settings.model_validate(reproduction_information["settings"])

    core.configure_runtime(settings)

    print(get_accelerator_info())

    if settings.print_debug_information:
        print()
        print(torch.__config__.show().strip())
        print()
        print(
            f"torch.backends.mkldnn.enabled = [bold]{torch.backends.mkldnn.enabled}[/]"
        )
        print(f"torch.get_num_threads() = [bold]{torch.get_num_threads()}[/]")
        print(
            f"torch.get_num_interop_threads() = [bold]{torch.get_num_interop_threads()}[/]"
        )

    study_checkpoint_file = core.checkpoint_path(settings)
    storage = core.open_study_storage(settings)

    try:
        existing_study = storage.get_all_studies()[0]
    except IndexError:
        existing_study = None

    if (
        existing_study is not None
        and settings.evaluate_model is None
        and not reproduction_mode
    ):
        choices = []

        if existing_study.user_attrs["finished"]:
            if settings.checkpoint_action is None:
                print()
                print(
                    (
                        "[green]You have already processed this model.[/] "
                        "You can show the results from the previous run, allowing you to export models or to run additional trials. "
                        "Alternatively, you can ignore the previous run and start from scratch. "
                        "This will delete the checkpoint file and all results from the previous run."
                    )
                )

            choices.append(
                Choice(
                    title="Show the results from the previous run",
                    value="continue",
                )
            )
        else:
            if settings.checkpoint_action is None:
                print()
                print(
                    (
                        "[yellow]You have already processed this model, but the run was interrupted.[/] "
                        "You can continue the previous run from where it stopped. This will override any specified settings. "
                        "Alternatively, you can ignore the previous run and start from scratch. "
                        "This will delete the checkpoint file and all results from the previous run."
                    )
                )

            choices.append(
                Choice(
                    title="Continue the previous run",
                    value="continue",
                )
            )

        choices.append(
            Choice(
                title="Ignore the previous run and start from scratch",
                value="restart",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        if settings.checkpoint_action is None:
            print()

        action = ask_if_unset(
            settings.checkpoint_action,
            questionary.select(
                "How would you like to proceed?",
                choices=choices,
                style=Style([("highlighted", "reverse")]),
            ),
        )

        if action is None or action == "":
            return

        if action == "continue":
            settings = Settings.model_validate_json(
                existing_study.user_attrs["settings"]
            )
        elif action == "restart":
            os.unlink(study_checkpoint_file)
            storage = core.open_study_storage(settings)

    model = Model(settings)
    print()
    print_memory_usage()

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        def _on_batch_start(batch_size: int) -> None:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

        def _on_batch_result(batch_size: int, performance: float) -> None:
            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

        def _on_batch_error(batch_size: int, error: Exception) -> None:
            formatted = format_exception(error)
            if "\n" in formatted:
                print(f"[red]Failed:\n{formatted}[/]")
            else:
                print(f"[red]Failed ({formatted})[/]")

        core.determine_batch_size(
            model,
            settings,
            good_prompts,
            on_start=_on_batch_start,
            on_result=_on_batch_result,
            on_error=_on_batch_error,
        )
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    if settings.response_prefix is None:
        print()
        print("Checking for common response prefix...")

        def _on_prefix(prefix: str) -> None:
            print(f"* Prefix found: [bold]{prefix!r}[/]")

        def _on_cot(prefix: str) -> None:
            print(f"* Closed Chain-of-Thought block: [bold]{prefix!r}[/]")
            # When using a Chain-of-Thought skip, we need to check that the prefix
            # is actually complete (e.g. not missing a trailing newline).
            print("* Rechecking with prefix...")

        def _on_extended(prefix: str) -> None:
            print(f"* Extended prefix found: [bold]{prefix!r}[/]")

        def _on_none() -> None:
            print("* None found")

        core.detect_response_prefix(
            model,
            settings,
            good_prompts,
            bad_prompts,
            on_prefix=_on_prefix,
            on_cot=_on_cot,
            on_extended=_on_extended,
            on_none=_on_none,
        )

    evaluator = Evaluator(settings, model)

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/]...")
        settings.model = settings.evaluate_model
        model.reset_model()
        print("* Evaluating...")
        evaluator.get_score()
        return

    print()
    print("Calculating per-layer refusal directions...")

    needs_full_residuals = settings.print_residual_geometry or settings.plot_residuals

    if needs_full_residuals:
        print("* Obtaining residuals for good prompts...")
        good_residuals = model.get_residuals_batched(good_prompts)
        print("* Obtaining residuals for bad prompts...")
        bad_residuals = model.get_residuals_batched(bad_prompts)

        good_means = good_residuals.mean(dim=0)
        bad_means = bad_residuals.mean(dim=0)

        analyzer = Analyzer(settings, model, good_residuals, bad_residuals)

        if settings.print_residual_geometry:
            analyzer.print_residual_geometry()

        if settings.plot_residuals:
            analyzer.plot_residuals()

        # We don't need the full residuals after computing their means and analyzing geometry.
        del good_residuals, bad_residuals, analyzer
    else:
        print("* Obtaining residual mean for good prompts...")
        good_means = model.get_residuals_mean(good_prompts)
        print("* Obtaining residual mean for bad prompts...")
        bad_means = model.get_residuals_mean(bad_prompts)

    refusal_directions = core.compute_refusal_directions(
        model, settings, good_means, bad_means
    )

    del good_means, bad_means

    # Clear cache before starting the optimization study.
    # This should free up memory from the objects released with the del statements above.
    empty_cache()

    trial_index = 0
    start_index = 0
    start_time = time.perf_counter()

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        direction_index, parameters = core.suggest_trial_parameters(trial, model)

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

        print()
        print(
            f"Running trial [bold]{trial_index}[/] of [bold]{settings.n_trials}[/]..."
        )
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Resetting model...")
        model.reset_model()
        print("* Abliterating...")
        model.abliterate(refusal_directions, direction_index, parameters)
        print("* Evaluating...")
        score, kl_divergence, refusals = evaluator.get_score()

        elapsed_time = time.perf_counter() - start_time
        remaining_time = (elapsed_time / (trial_index - start_index)) * (
            settings.n_trials - trial_index
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        if trial_index < settings.n_trials:
            print(
                f"[grey50]Estimated remaining time: [bold]{format_duration(remaining_time)}[/][/]"
            )
        print_memory_usage()

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)
        trial.set_user_attr("base_refusals", evaluator.base_refusals)
        trial.set_user_attr("n_bad_prompts", len(evaluator.bad_prompts))

        return score

    def objective_wrapper(trial: Trial) -> tuple[float, float]:
        try:
            return objective(trial)
        except KeyboardInterrupt:
            # Stop the study gracefully on Ctrl+C.
            trial.study.stop()
            raise TrialPruned()

    if not reproduction_mode:
        study = core.create_study(settings, storage)

        start_index = trial_index = len(study.trials)
        if start_index > 0:
            print()
            print("Resuming existing study.")

        try:
            study.optimize(
                objective_wrapper,
                n_trials=settings.n_trials - len(study.trials),
            )
        except KeyboardInterrupt:
            # This additional handler takes care of the small chance that KeyboardInterrupt
            # is raised just between trials, which wouldn't be caught by the handler
            # defined in objective_wrapper above.
            pass

        if len(study.trials) == settings.n_trials:
            study.set_user_attr("finished", True)

    trial_loop_active = True

    while trial_loop_active:
        if not reproduction_mode:
            # If no trials at all have been evaluated, the study must have been stopped
            # by pressing Ctrl+C while the first trial was running. In this case, we just
            # re-raise the interrupt to invoke the standard handler defined below.
            completed_trials = [
                t for t in study.trials if t.state == TrialState.COMPLETE
            ]
            if not completed_trials:
                raise KeyboardInterrupt

            # Get the Pareto front of trials.
            best_trials = core.pareto_front(study)

            choices = [
                Choice(
                    title=(
                        f"[Trial {trial.user_attrs['index']:>3}] "
                        f"Refusals: {trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                        f"KL divergence: {trial.user_attrs['kl_divergence']:.4f}"
                    ),
                    value=trial,
                )
                for trial in best_trials
            ]

            choices.append(
                Choice(
                    title="Run additional trials",
                    value="continue",
                )
            )

            choices.append(
                Choice(
                    title="Exit program",
                    value="",
                )
            )

            print()
            print("[bold green]Optimization finished![/]")

            if settings.trial_index is None:
                print()
                print(
                    (
                        "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
                        "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
                        "chat with it to test how well it works, or run standard benchmarks on it. "
                        "You can return to this menu later to select a different trial. "
                        "[yellow]Note that KL divergence values above 0.5 usually indicate significant damage to the original model's capabilities.[/]"
                    )
                )

        while trial_loop_active:
            # Ensure a predefined trial is only processed once.
            if settings.trial_index is not None:
                trial_loop_active = False

            if reproduction_mode:
                parameters = reproduction_information["parameters"]
                metrics = reproduction_information["metrics"]

                trial = create_trial(
                    values=[],
                    user_attrs={
                        "direction_index": parameters["direction_index"],
                        "parameters": parameters["abliteration_parameters"],
                        "kl_divergence": metrics["kl_divergence"],
                        "refusals": metrics["refusals"],
                        "base_refusals": metrics["base_refusals"],
                        "n_bad_prompts": metrics["n_bad_prompts"],
                    },
                )

                print()
                print("Restoring model from reproduction information...")
            else:
                if settings.trial_index is None:
                    print()

                trial = ask_if_unset(
                    None
                    if settings.trial_index is None
                    else best_trials[settings.trial_index],
                    questionary.select(
                        "Which trial do you want to use?",
                        choices=choices,
                        style=Style([("highlighted", "reverse")]),
                    ),
                )

                if trial is None or trial == "":
                    return

                if trial == "continue":
                    while True:
                        try:
                            n_additional_trials = ask_if_unset(
                                settings.n_additional_trials,
                                questionary.text(
                                    "How many additional trials do you want to run?"
                                ),
                            )
                            if n_additional_trials is None or n_additional_trials == "":
                                n_additional_trials = 0
                                break
                            n_additional_trials = int(n_additional_trials)
                            if n_additional_trials > 0:
                                break
                            print("[red]Please enter a number greater than 0.[/]")
                        except ValueError:
                            print("[red]Please enter a number.[/]")

                    if n_additional_trials == 0:
                        continue

                    settings.n_trials = len(study.trials) + n_additional_trials
                    study.set_user_attr("settings", settings.model_dump_json())
                    study.set_user_attr("finished", False)

                    try:
                        study.optimize(
                            objective_wrapper,
                            n_trials=settings.n_trials - len(study.trials),
                        )
                    except KeyboardInterrupt:
                        pass

                    if len(study.trials) == settings.n_trials:
                        study.set_user_attr("finished", True)

                    break

                print()
                print(
                    f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]..."
                )

            print("* Parameters:")
            for name, value in get_trial_parameters(trial).items():
                print(f"  * {name} = [bold]{value}[/]")

            # Per https://github.com/huggingface/peft/issues/868#issuecomment-1820642893
            # once a LoRA is merged it's expected to be empty. Provide a utility function
            # to restore the previous LoRA-ified state.
            def reset_trial_model():
                print("* Resetting model...")
                model.reset_model()
                print("* Abliterating...")
                model.abliterate(
                    refusal_directions,
                    trial.user_attrs["direction_index"],
                    {
                        k: AbliterationParameters(**v)
                        for k, v in trial.user_attrs["parameters"].items()
                    },
                )

            reset_trial_model()

            action_loop_active = True

            while action_loop_active:
                # Ensure a predefined action is only executed once.
                if settings.model_action is not None:
                    action_loop_active = False

                if settings.model_action is None:
                    print()

                action = ask_if_unset(
                    settings.model_action,
                    questionary.select(
                        "What do you want to do with the decensored model?",
                        choices=[
                            Choice(
                                title="Save the model to a local folder",
                                value="save",
                            ),
                            Choice(
                                title="Upload the model to Hugging Face",
                                value="upload",
                            ),
                            Choice(
                                title="Chat with the model",
                                value="chat",
                            ),
                            Choice(
                                title="Benchmark the model",
                                value="benchmark",
                            ),
                            Choice(
                                title="Exit program"
                                if reproduction_mode
                                else "Return to the trial selection menu",
                                value="",
                            ),
                        ],
                        style=Style([("highlighted", "reverse")]),
                    ),
                )

                if action is None or action == "":
                    if reproduction_mode:
                        return
                    else:
                        break

                # All actions are wrapped in a try/except block so that if an error occurs,
                # another action can be tried, instead of the program crashing and losing
                # the optimized model.
                try:
                    match action:
                        case "save":
                            save_directory = ask_if_unset(
                                settings.save_directory,
                                questionary.path(
                                    "Path to the folder:",
                                    only_directories=True,
                                ),
                            )
                            if not save_directory:
                                continue

                            strategy = obtain_export_strategy(settings, model)
                            if strategy is None:
                                continue

                            if strategy == ExportStrategy.ADAPTER:
                                print("Saving LoRA adapter...")
                                model.model.save_pretrained(
                                    save_directory,
                                    max_shard_size=settings.max_shard_size,
                                )
                            else:
                                print("Saving merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.save_pretrained(
                                    save_directory,
                                    max_shard_size=settings.max_shard_size,
                                )
                                del merged_model
                                empty_cache()
                                model.tokenizer.save_pretrained(save_directory)
                                if model.processor is not None:
                                    model.processor.save_pretrained(save_directory)
                                reset_trial_model()

                            print(f"Model saved to [bold]{save_directory}[/].")

                            if reproduction_mode and verify_hashes:
                                print("Verifying hashes of weight files...")

                                for (
                                    filename,
                                    original_sha256,
                                ) in reproduction_information["hashes"].items():
                                    file_path = Path(save_directory) / filename

                                    if file_path.exists():
                                        sha256 = get_file_sha256(file_path)

                                        if sha256.lower() == original_sha256.lower():
                                            print(
                                                f"[bold]{filename}:[/] [green]Hash matches[/]"
                                            )
                                        else:
                                            print(
                                                f"[bold]{filename}:[/] [yellow]Hash doesn't match[/]"
                                            )
                                    else:
                                        print(
                                            f"[bold]{filename}:[/] [red]File not found[/]"
                                        )

                        case "upload":
                            # We don't use huggingface_hub.login() because that stores the token on disk,
                            # and since this program will often be run on rented or shared GPU servers,
                            # it's better to not persist credentials.
                            token = huggingface_hub.get_token()
                            if not token:
                                # NOTE: Unlike for most other values obtained from interactive inputs, it is
                                #       not possible to set the token via the settings. This is a security
                                #       precaution to prevent exporting the token under all circumstances.
                                #       For scripting, the correct way to set the token is through the HF_TOKEN
                                #       environment variable, or through the HF token file.
                                token = questionary.password(
                                    "Hugging Face access token:"
                                ).ask()
                            if not token:
                                continue

                            user = huggingface_hub.whoami(token)
                            fullname = user.get(
                                "fullname",
                                user.get("name", "unknown user"),
                            )
                            email = user.get("email", "no email found")
                            print(f"Logged in as [bold]{fullname} ({email})[/]")

                            repo_id = ask_if_unset(
                                settings.upload_repo_id,
                                questionary.text(
                                    "Name of repository:",
                                    default=f"{user['name']}/{Path(settings.model).name}-heretic",
                                ),
                            )
                            if not repo_id:
                                continue

                            visibility = ask_if_unset(
                                None
                                if settings.upload_repo_private is None
                                else (
                                    "Private"
                                    if settings.upload_repo_private
                                    else "Public"
                                ),
                                questionary.select(
                                    "Should the repository be public or private?",
                                    choices=[
                                        "Public",
                                        "Private",
                                    ],
                                    style=Style([("highlighted", "reverse")]),
                                ),
                            )
                            if visibility is None:
                                continue
                            private = visibility == "Private"

                            strategy = obtain_export_strategy(settings, model)
                            if strategy is None:
                                continue

                            # Reproducibility requires that the model and all datasets
                            # are available on the Hugging Face Hub (not local paths).
                            datasets = [
                                settings.good_prompts.dataset,
                                settings.bad_prompts.dataset,
                                settings.good_evaluation_prompts.dataset,
                                settings.bad_evaluation_prompts.dataset,
                            ]
                            is_reproducible = (
                                is_hf_path(settings.model)
                                and all(is_hf_path(dataset) for dataset in datasets)
                                and not reproduction_mode
                            )

                            if is_reproducible:
                                if settings.upload_reproducibility_information is None:
                                    print(
                                        (
                                            "Heretic can add information to the repository that allows others to reproduce the model. "
                                            "This is optional, but valuable to the community as both a learning tool and to preserve computational work already done. "
                                            "Guaranteeing reproducibility requires basic system information (Python and OS version, CPU and GPU/accelerator info) "
                                            "as tensor operations can give different results in different system environments. "
                                            "[bold]The information does not include any file system paths or other private data.[/]"
                                        )
                                    )

                                reproducibility_information = ask_if_unset(
                                    settings.upload_reproducibility_information,
                                    questionary.select(
                                        "Which reproducibility information do you want to add?",
                                        choices=[
                                            Choice(
                                                title="Full: Settings, package versions, and system information",
                                                value="full",
                                            ),
                                            Choice(
                                                title="Basic: Settings and package versions",
                                                value="basic",
                                            ),
                                            Choice(
                                                title="Don't add any reproducibility information",
                                                value="none",
                                            ),
                                        ],
                                        style=Style([("highlighted", "reverse")]),
                                    ),
                                )
                                if reproducibility_information is None:
                                    continue
                            else:
                                reproducibility_information = "none"

                            if strategy == ExportStrategy.ADAPTER:
                                print("Uploading LoRA adapter...")
                                model.model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    max_shard_size=settings.max_shard_size,
                                    token=token,
                                )
                            else:
                                print("Uploading merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    max_shard_size=settings.max_shard_size,
                                    token=token,
                                )
                                del merged_model
                                empty_cache()
                                model.tokenizer.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                                if model.processor is not None:
                                    model.processor.push_to_hub(
                                        repo_id,
                                        private=private,
                                        token=token,
                                    )
                                reset_trial_model()

                            if is_hf_path(settings.model):
                                card = ModelCard.load(settings.model)
                            else:
                                card_path = (
                                    Path(settings.model)
                                    / huggingface_hub.constants.REPOCARD_NAME
                                )
                                if card_path.exists():
                                    card = ModelCard.load(card_path)
                                else:
                                    card = None

                            if card is not None:
                                if card.data is None:
                                    card.data = ModelCardData()
                                if card.data.tags is None:
                                    card.data.tags = []
                                card.data.tags.append("heretic")
                                card.data.tags.append("uncensored")
                                card.data.tags.append("decensored")
                                card.data.tags.append("abliterated")
                                if reproducibility_information != "none":
                                    card.data.tags.append("reproducible")
                                card.text = (
                                    get_readme_intro(
                                        settings,
                                        trial,
                                        reproducibility_information != "none",
                                    )
                                    + card.text
                                )
                                card.push_to_hub(repo_id, token=token)

                            if reproducibility_information != "none":
                                # Set the number of trials to the number of actual completed trials
                                # for the reproduction configuration.
                                settings.n_trials = len(study.trials)
                                current_export_strategy = settings.export_strategy
                                settings.export_strategy = strategy

                                try:
                                    upload_reproduce_folder(
                                        repo_id,
                                        settings,
                                        token,
                                        checkpoint_path=study_checkpoint_file,
                                        trial=trial,
                                        include_system_information=(
                                            reproducibility_information == "full"
                                        ),
                                    )
                                finally:
                                    settings.export_strategy = current_export_strategy

                            print(f"Model uploaded to [bold]{repo_id}[/].")

                            if reproduction_mode and verify_hashes:
                                print("Verifying hashes of weight files...")

                                api = HfApi()
                                model_info = api.model_info(
                                    repo_id,
                                    files_metadata=True,
                                    token=token,
                                )

                                if not model_info.siblings:
                                    raise RuntimeError(
                                        "Could not fetch uploaded model hashes."
                                    )

                                for (
                                    filename,
                                    original_sha256,
                                ) in reproduction_information["hashes"].items():
                                    file_found = False

                                    for file in model_info.siblings:
                                        if file.rfilename == filename:
                                            sha256 = getattr(file, "lfs", {}).get(
                                                "sha256"
                                            )
                                            if not sha256:
                                                raise RuntimeError(
                                                    "Could not fetch uploaded model hashes."
                                                )

                                            if (
                                                sha256.lower()
                                                == original_sha256.lower()
                                            ):
                                                print(
                                                    f"[bold]{filename}:[/] [green]Hash matches[/]"
                                                )
                                            else:
                                                print(
                                                    f"[bold]{filename}:[/] [yellow]Hash doesn't match[/]"
                                                )

                                            file_found = True
                                            break

                                    if not file_found:
                                        print(
                                            f"[bold]{filename}:[/] [red]File not found[/]"
                                        )

                        case "chat":
                            print()
                            print(
                                "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                            )

                            chat = [
                                {"role": "system", "content": settings.system_prompt},
                            ]

                            while True:
                                try:
                                    message = questionary.text(
                                        "User:",
                                        qmark=">",
                                    ).unsafe_ask()
                                    if not message:
                                        break
                                    chat.append({"role": "user", "content": message})

                                    print("[bold]Assistant:[/] ", end="")
                                    response = model.stream_chat_response(chat)
                                    chat.append(
                                        {"role": "assistant", "content": response}
                                    )
                                except (KeyboardInterrupt, EOFError):
                                    # Ctrl+C/Ctrl+D
                                    break

                        case "benchmark":
                            benchmarks = questionary.checkbox(
                                "Which benchmarks do you want to run?",
                                [
                                    Choice(
                                        title=f"{benchmark.name}: {benchmark.description}",
                                        value=benchmark,
                                    )
                                    for benchmark in settings.benchmarks
                                ],
                                style=Style([("highlighted", "reverse")]),
                            ).ask()
                            if not benchmarks:
                                continue

                            scope = questionary.select(
                                (
                                    "Do you want to benchmark the original model along with the decensored model? "
                                    "Benchmarking both models allows you to compare the scores, but it takes twice as much time."
                                ),
                                choices=[
                                    "Benchmark only the decensored model",
                                    "Benchmark both models",
                                ],
                                style=Style([("highlighted", "reverse")]),
                            ).ask()
                            if scope is None:
                                continue
                            benchmark_original_model = scope == "Benchmark both models"

                            hflm = HFLM(
                                pretrained=model.model,  # ty:ignore[invalid-argument-type]
                                tokenizer=model.tokenizer,  # ty:ignore[invalid-argument-type]
                                batch_size="auto",
                            )

                            table = Table()
                            table.add_column("Benchmark")
                            table.add_column("Metric")
                            if benchmark_original_model:
                                table.add_column("This model", justify="right")
                                table.add_column("Original model", justify="right")
                            else:
                                table.add_column("Value", justify="right")

                            try:
                                first_benchmark = True

                                for benchmark in benchmarks:
                                    print(
                                        f"Running benchmark [bold]{benchmark.name}[/]..."
                                    )

                                    def get_results() -> dict[str, Any]:
                                        results = lm_eval.simple_evaluate(
                                            model=hflm,
                                            tasks=[benchmark.task],
                                        )
                                        return results["results"][benchmark.task]

                                    results = get_results()
                                    if benchmark_original_model:
                                        with model.model.disable_adapter():  # ty:ignore[call-non-callable]
                                            original_results = get_results()

                                    first_row = True

                                    for metric, value in results.items():
                                        if metric != "alias":
                                            if first_row and not first_benchmark:
                                                if benchmark_original_model:
                                                    table.add_row("", "", "", "")
                                                else:
                                                    table.add_row("", "", "")

                                            def format_value(value: Any) -> str:
                                                if isinstance(
                                                    value,
                                                    (float, np.floating),
                                                ):
                                                    return f"{value:.4f}"
                                                else:
                                                    return f"{value}"

                                            cells = [
                                                benchmark.name if first_row else "",
                                                metric,
                                                format_value(value),
                                            ]
                                            if benchmark_original_model:
                                                cells.append(
                                                    format_value(
                                                        original_results[metric]
                                                    )
                                                )
                                            table.add_row(*cells)

                                            first_row = False
                                            first_benchmark = False
                            except KeyboardInterrupt:
                                pass

                            # The benchmark run might have been cancelled by the user
                            # before any benchmark was completed, so we only print results
                            # if there actually are some.
                            if table.rows:
                                print(table)

                except Exception as error:
                    formatted = format_exception(error)
                    if "\n" in formatted:
                        print(f"[red]Error:\n{formatted}[/]")
                    else:
                        print(f"[red]Error: {formatted}[/]")


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
