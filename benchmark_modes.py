import os
import sys
import time
from dataclasses import asdict
import torch
import warnings
from rich.table import Table
from rich.console import Console

# Ensure the project root is strictly the first path to prevent dual-module loading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# We need to import heretic components to run the evaluation
from heretic.config import Settings, DatasetMode
from heretic.model import Model, AbliterationParameters
from heretic.utils import load_prompts, format_duration
from heretic.system import empty_cache
from heretic.evaluator import Evaluator
from heretic.activation_capture import capture_activations
from heretic.ridge_probe import precompute_ridge_terms, train_ridge_probes
from heretic.expansion import generate_self_play_dataset

print = Console(highlight=False).print

def run_benchmark_for_mode(mode: DatasetMode, model: Model, evaluator: Evaluator, base_refusal_directions: torch.Tensor, good_prompts, bad_prompts):
    start_time = time.time()

    # 1. Dataset Generation
    if mode == DatasetMode.SELF_PLAY:
        all_seeds = good_prompts + bad_prompts
        self_play_good, self_play_bad = generate_self_play_dataset(model, all_seeds)
        expanded_good_prompts = good_prompts + self_play_good
        expanded_bad_prompts = bad_prompts + self_play_bad
    else:
        expanded_good_prompts = good_prompts
        expanded_bad_prompts = bad_prompts

    dataset_time = time.time() - start_time

    # 2. Activation Capture
    act_start = time.time()
    good_captures = capture_activations(model, expanded_good_prompts)
    bad_captures = capture_activations(model, expanded_bad_prompts)
    act_time = time.time() - act_start

    # 3. Probe Training (using fixed hyperparams for benchmark speed)
    train_start = time.time()
    precomputed = precompute_ridge_terms(good_captures, bad_captures)
    # Use alpha=1.0 for standard comparison
    probes = train_ridge_probes(precomputed, alpha=1.0)
    train_time = time.time() - train_start

    del good_captures, bad_captures, precomputed
    empty_cache()

    # 4. Apply Abliteration (using fixed standard hyperparams)
    ablation_start = time.time()
    parameters = {}
    last_layer_index = len(model.get_layers()) - 1
    for component in model.get_abliterable_components():
        parameters[component] = AbliterationParameters(
            max_weight=1.2, # standard value
            max_weight_position=0.5 * last_layer_index,
            min_weight=0.0,
            min_weight_distance=1.0 * last_layer_index,
        )

    model.reset_model()
    model.abliterate(base_refusal_directions, probes, parameters)
    ablation_time = time.time() - ablation_start

    # 5. Evaluation
    eval_start = time.time()
    scores = evaluator.get_scores()
    eval_time = time.time() - eval_start

    total_time = time.time() - start_time

    # We map scores list of tuples into a dict for easy lookup
    score_dict = {name: val for name, val in scores}

    return {
        "mode": mode.value,
        "dataset_time": dataset_time,
        "activation_time": act_time,
        "train_time": train_time,
        "total_time": total_time,
        "kl_div": score_dict.get("KL divergence").value if score_dict.get("KL divergence") else 0.0,
        "refusals": score_dict.get("Refusals").value if score_dict.get("Refusals") else 0,
        "total_bad": len(evaluator.bad_prompts)
    }


def main():
    # Hack sys.argv to play nice with Pydantic Settings
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        model_id = sys.argv.pop(1)
        sys.argv.append("--model")
        sys.argv.append(model_id)
    else:
        print("Usage: python benchmark_modes.py <model_id>")
        sys.exit(1)

    print(f"\n[bold]Starting Head-to-Head Benchmark for {model_id}[/bold]")

    # Initialize basic settings
    settings = Settings(batch_size=2)
    model = Model(settings)

    print("\n* Loading base evaluation datasets...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)

    print("\n* Initializing evaluator...")
    evaluator = Evaluator(settings, model)

    # We need the base refusal directions (difference of means) for the steering vector v
    print("\n* Calculating base refusal directions for projection...")
    good_means = model.get_residuals_mean(good_prompts)
    bad_means = model.get_residuals_mean(bad_prompts)
    refusal_directions = torch.nn.functional.normalize(bad_means - good_means, p=2, dim=1)
    del good_means, bad_means
    empty_cache()

    results = []

    print(f"\n{'='*50}\n[bold]Running Mode A: STATIC[/bold]\n{'='*50}")
    res_static = run_benchmark_for_mode(DatasetMode.STATIC, model, evaluator, refusal_directions, good_prompts, bad_prompts)
    results.append(res_static)

    print(f"\n{'='*50}\n[bold]Running Mode B: DYNAMIC SELF-PLAY[/bold]\n{'='*50}")
    res_self_play = run_benchmark_for_mode(DatasetMode.SELF_PLAY, model, evaluator, refusal_directions, good_prompts, bad_prompts)
    results.append(res_self_play)

    # Print comparison table
    print(f"\n\n[bold green]Benchmark Complete![/bold green]")

    table = Table(title=f"Head-to-Head Comparison: {model_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Mode A: Static", justify="right", style="green")
    table.add_column("Mode B: Self-Play", justify="right", style="yellow")

    table.add_row("Dataset Generation Time", format_duration(results[0]["dataset_time"]), format_duration(results[1]["dataset_time"]))
    table.add_row("Activation Capture Time", format_duration(results[0]["activation_time"]), format_duration(results[1]["activation_time"]))
    table.add_row("Probe Training Time", format_duration(results[0]["train_time"]), format_duration(results[1]["train_time"]))
    table.add_row("Total Wall-Clock Time", format_duration(results[0]["total_time"]), format_duration(results[1]["total_time"]))
    table.add_row("", "", "")
    table.add_row("Refusals (lower is better)", f"{results[0]['refusals']}/{results[0]['total_bad']}", f"{results[1]['refusals']}/{results[1]['total_bad']}")
    table.add_row("KL Divergence (lower is better)", f"{results[0]['kl_div']:.4f}", f"{results[1]['kl_div']:.4f}")

    print(table)

if __name__ == "__main__":
    main()
