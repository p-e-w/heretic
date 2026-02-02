#!/usr/bin/env python3
"""
Restore and save a model from a completed heretic checkpoint.
Skips all optimization - just loads checkpoint, applies abliteration, saves.
"""

import math
import os
import sys
import warnings

import optuna
import torch
import torch.nn.functional as F
import transformers
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from questionary import Choice

from .main import save_model
from .model import AbliterationParameters, Model
from .utils import get_trial_parameters, load_prompts, print, prompt_select


def restore():
    if len(sys.argv) < 2:
        print("[red]Usage: heretic-restore <checkpoint.jsonl>[/]")
        print("Example: heretic-restore checkpoints/Qwen--Qwen3-Coder-30B-A3B-Instruct.jsonl")
        return

    checkpoint_file = sys.argv[1]
    if not os.path.exists(checkpoint_file):
        print(f"[red]Checkpoint file not found: {checkpoint_file}[/]")
        return

    # Load checkpoint
    print(f"Loading checkpoint from [bold]{checkpoint_file}[/]...")
    backend = JournalFileBackend(checkpoint_file)
    storage = JournalStorage(backend)

    try:
        existing_study = storage.get_all_studies()[0]
    except IndexError:
        print("[red]No study found in checkpoint file.[/]")
        return

    # Load settings from checkpoint - must clear sys.argv to prevent pydantic-settings
    # from trying to parse the checkpoint path as CLI arguments
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]

    from .config import Settings
    settings = Settings.model_validate_json(existing_study.user_attrs["settings"])

    sys.argv = original_argv
    print(f"Model: [bold]{settings.model}[/]")

    # Load study using the study name from the existing study
    study_name = existing_study.study_name
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Found [bold]{len(completed_trials)}[/] completed trials")

    if not completed_trials:
        print("[red]No completed trials found.[/]")
        return

    # Build Pareto front
    sorted_trials = sorted(
        completed_trials,
        key=lambda trial: (
            trial.user_attrs["refusals"],
            trial.user_attrs["kl_divergence"],
        ),
    )

    min_divergence = math.inf
    best_trials = []
    for trial in sorted_trials:
        kl_divergence = trial.user_attrs["kl_divergence"]
        if kl_divergence < min_divergence:
            min_divergence = kl_divergence
            best_trials.append(trial)

    # Show Pareto front
    choices = [
        Choice(
            title=(
                f"[Trial {trial.user_attrs['index']:>3}] "
                f"Refusals: {trial.user_attrs['refusals']:>2}/100, "
                f"KL divergence: {trial.user_attrs['kl_divergence']:.4f}"
            ),
            value=trial,
        )
        for trial in best_trials
    ]
    choices.append(Choice(title="Exit", value=None))

    print()
    trial = prompt_select("Select trial to restore:", choices)

    if trial is None:
        return

    print()
    print(f"Selected trial [bold]{trial.user_attrs['index']}[/]")
    print("Parameters:")
    for name, value in get_trial_parameters(trial).items():
        print(f"  * {name} = [bold]{value}[/]")

    # Now load model and compute refusal directions
    print()
    print("[yellow]Loading model and computing refusal directions...[/]")
    print("[yellow](This is required to apply abliteration)[/]")

    torch.set_grad_enabled(False)
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore")

    model = Model(settings)

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

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
    del good_residuals, bad_residuals

    # Apply abliteration
    print()
    print("Applying abliteration...")
    model.abliterate(
        refusal_directions,
        trial.user_attrs["direction_index"],
        {
            k: AbliterationParameters(**v)
            for k, v in trial.user_attrs["parameters"].items()
        },
    )

    # Save
    print()
    save_directory = input("Save path (e.g., /workspace/heretic/output): ").strip()
    if save_directory:
        save_model(model, save_directory, settings)
    else:
        print("[yellow]No path provided, skipping save.[/]")


def main():
    from rich.traceback import install
    install()

    try:
        restore()
    except KeyboardInterrupt:
        print()
        print("[red]Cancelled.[/]")


if __name__ == "__main__":
    main()
