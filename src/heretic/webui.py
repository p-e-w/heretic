# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Gradio-based web UI for Heretic."""

# ruff: noqa: E402

import logging
import math
import os
import queue
import random
import re
import sys
import threading
import time
import traceback
import warnings
from collections import deque
from dataclasses import asdict
from importlib.metadata import version
from os.path import commonprefix
from typing import Any, Generator

import optuna
import torch
import torch.nn.functional as F
import transformers
from optuna import Trial, TrialPruned
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import TrialState
from rich.console import Console

from .config import Settings
from .evaluator import Evaluator
from .model import AbliterationParameters, Model
from .system import empty_cache, get_accelerator_info
from .utils import format_duration, get_trial_parameters, load_prompts, set_seed

# ─── Log capture infrastructure ──────────────────────────────────────────────

_log_queue: queue.Queue[str | None] = queue.Queue()

# Capture the real stdout before any monkey-patching so log lines can always
# be forwarded to it (visible in ``docker logs`` and plain terminal runs).
_real_stdout = sys.stdout

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text).strip()


def _emit(line: str) -> None:
    """Enqueue *line* for the web UI log and echo it to the real stdout."""
    _log_queue.put(line)
    try:
        _real_stdout.write(line + "\n")
        _real_stdout.flush()
    except (OSError, ValueError):
        pass


class _QueueFile:
    """File-like object that pushes each line into *_log_queue*."""

    def __init__(self) -> None:
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            clean = _strip_ansi(line)
            if clean:
                _emit(clean)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            clean = _strip_ansi(self._buf)
            if clean:
                _emit(clean)
            self._buf = ""

    def isatty(self) -> bool:
        return False


_queue_file = _QueueFile()
# no_color=True prevents Rich from embedding ANSI codes in the output.
_capturing_console = Console(file=_queue_file, highlight=False, no_color=True)


def _log(message: str) -> None:
    """Enqueue a plain-text log line directly and echo it to stdout."""
    for line in message.splitlines():
        stripped = _strip_ansi(line).strip()
        if stripped:
            _emit(stripped)


def _install_capturing_print() -> None:
    """Replace the module-level *print* in each heretic module with our
    capturing version so that all Rich-formatted output is routed to the
    web UI log rather than to the terminal."""
    import heretic.evaluator as _evaluator
    import heretic.model as _model
    import heretic.utils as _utils

    capturing_print = _capturing_console.print
    _utils.print = capturing_print  # type: ignore[attr-defined]
    _model.print = capturing_print  # type: ignore[attr-defined]
    _evaluator.print = capturing_print  # type: ignore[attr-defined]


# ─── Session state (single-user tool) ────────────────────────────────────────

_session: dict[str, Any] = {}
_optimization_running = threading.Event()


# ─── Background optimization thread ──────────────────────────────────────────


def _run_optimization(
    model_id: str,
    quantization: str,
    n_trials: int,
    n_startup_trials: int,
    system_prompt: str,
    kl_divergence_scale: float,
    kl_divergence_target: float,
    study_checkpoint_dir: str,
) -> None:
    """Full Heretic optimization pipeline – runs in a daemon thread."""
    try:
        _install_capturing_print()

        # ── Environment setup ──────────────────────────────────────────────
        if (
            "PYTORCH_ALLOC_CONF" not in os.environ
            and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
        ):
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        transformers.logging.set_verbosity_error()
        logging.getLogger("lm_eval").setLevel(logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        torch._dynamo.config.cache_size_limit = 64
        torch.set_grad_enabled(False)

        # ── Build Settings without triggering argparse ─────────────────────
        # CliSettingsSource reads sys.argv; set it to a harmless value while
        # constructing Settings so that init_settings (the kwargs we pass)
        # can satisfy all required fields without argparse interference.
        old_argv = sys.argv[:]
        sys.argv = ["heretic-webui"]
        try:
            settings = Settings(  # type: ignore[call-arg]
                model=model_id,
                quantization=quantization,
                n_trials=n_trials,
                n_startup_trials=n_startup_trials,
                system_prompt=system_prompt,
                kl_divergence_scale=kl_divergence_scale,
                kl_divergence_target=kl_divergence_target,
                study_checkpoint_dir=study_checkpoint_dir,
            )
        finally:
            sys.argv = old_argv

        _session["settings"] = settings

        if settings.seed is None:
            settings.seed = random.randint(0, 2**32 - 1)
        set_seed(settings.seed)

        _log(get_accelerator_info())

        # ── Load prompts ───────────────────────────────────────────────────
        _log(f"Loading good prompts from {settings.good_prompts.dataset}...")
        good_prompts = load_prompts(settings, settings.good_prompts)
        _log(f"* {len(good_prompts)} prompts loaded")

        _log(f"Loading bad prompts from {settings.bad_prompts.dataset}...")
        bad_prompts = load_prompts(settings, settings.bad_prompts)
        _log(f"* {len(bad_prompts)} prompts loaded")

        # ── Load model ─────────────────────────────────────────────────────
        model = Model(settings)
        _session["model"] = model

        # ── Determine optimal batch size ───────────────────────────────────
        if settings.batch_size == 0:
            _log("Determining optimal batch size...")
            batch_size = 1
            best_batch_size = -1
            best_performance = -1.0

            while batch_size <= settings.max_batch_size:
                _log(f"* Trying batch size {batch_size}...")
                prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
                prompts = prompts[:batch_size]

                try:
                    model.get_responses(prompts)
                    t0 = time.perf_counter()
                    responses = model.get_responses(prompts)
                    t1 = time.perf_counter()
                except Exception as exc:
                    if batch_size == 1:
                        raise
                    _log(f"  Failed ({exc})")
                    break

                lengths = [len(model.tokenizer.encode(r)) for r in responses]
                perf = sum(lengths) / (t1 - t0)
                _log(f"  {perf:.0f} tokens/s")

                if perf > best_performance:
                    best_batch_size = batch_size
                    best_performance = perf
                batch_size *= 2

            settings.batch_size = best_batch_size
            _log(f"* Chosen batch size: {settings.batch_size}")

        # ── Common response prefix ─────────────────────────────────────────
        if settings.response_prefix is None:
            _log("Checking for common response prefix...")
            check_prompts = good_prompts[:100] + bad_prompts[:100]
            responses = model.get_responses_batched(check_prompts)
            settings.response_prefix = commonprefix(responses).rstrip(" ")

            if settings.response_prefix:
                _log(f"* Prefix: {settings.response_prefix!r}")
                for cot_init, closed_block in settings.chain_of_thought_skips:
                    if settings.response_prefix.startswith(cot_init):
                        settings.response_prefix = closed_block
                        _log(f"* Closed CoT block: {settings.response_prefix!r}")
                        responses = model.get_responses_batched(check_prompts)
                        extra = commonprefix(responses).rstrip(" ")
                        if extra:
                            settings.response_prefix += extra
                        break
            else:
                _log("* None found")

        # ── Evaluator ──────────────────────────────────────────────────────
        evaluator = Evaluator(settings, model)
        _session["evaluator"] = evaluator

        # ── Refusal directions ─────────────────────────────────────────────
        _log("Calculating per-layer refusal directions...")
        _log("* Obtaining residual mean for good prompts...")
        good_means = model.get_residuals_mean(good_prompts)
        _log("* Obtaining residual mean for bad prompts...")
        bad_means = model.get_residuals_mean(bad_prompts)

        refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

        if settings.orthogonalize_direction:
            good_directions = F.normalize(good_means, p=2, dim=1)
            proj = torch.sum(refusal_directions * good_directions, dim=1)
            refusal_directions = (
                refusal_directions - proj.unsqueeze(1) * good_directions
            )
            refusal_directions = F.normalize(refusal_directions, p=2, dim=1)
            del good_directions, proj

        del good_means, bad_means
        empty_cache()
        _session["refusal_directions"] = refusal_directions

        # ── Optuna study ───────────────────────────────────────────────────
        os.makedirs(settings.study_checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(
            settings.study_checkpoint_dir,
            "".join(
                (c if (c.isalnum() or c in ["_", "-"]) else "--")
                for c in settings.model
            )
            + ".jsonl",
        )

        lock_obj = JournalFileOpenLock(checkpoint_file)
        backend = JournalFileBackend(checkpoint_file, lock_obj=lock_obj)
        storage = JournalStorage(backend)

        study = optuna.create_study(
            sampler=TPESampler(
                n_startup_trials=settings.n_startup_trials,
                n_ei_candidates=128,
                multivariate=True,
                seed=settings.seed,
            ),
            directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
            storage=storage,
            study_name="heretic",
            load_if_exists=True,
        )
        study.set_user_attr("settings", settings.model_dump_json())
        study.set_user_attr("finished", False)
        _session["study"] = study

        trial_counter = [0]
        start_time = time.perf_counter()

        def count_completed() -> int:
            return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)

        start_index = count_completed()
        trial_counter[0] = start_index

        def objective(trial: Trial) -> tuple[float, float]:
            trial_counter[0] += 1
            idx = trial_counter[0]
            trial.set_user_attr("index", idx)

            direction_scope = trial.suggest_categorical(
                "direction_scope", ["global", "per layer"]
            )
            last_layer = len(model.get_layers()) - 1
            direction_index: float | None = trial.suggest_float(
                "direction_index",
                0.4 * last_layer,
                0.9 * last_layer,
            )
            if direction_scope == "per layer":
                direction_index = None

            parameters = {}
            for component in model.get_abliterable_components():
                max_w = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
                max_w_pos = trial.suggest_float(
                    f"{component}.max_weight_position",
                    0.6 * last_layer,
                    1.0 * last_layer,
                )
                min_w = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
                min_w_dist = trial.suggest_float(
                    f"{component}.min_weight_distance",
                    1.0,
                    0.6 * last_layer,
                )
                parameters[component] = AbliterationParameters(
                    max_weight=max_w,
                    max_weight_position=max_w_pos,
                    min_weight=min_w * max_w,
                    min_weight_distance=min_w_dist,
                )

            trial.set_user_attr("direction_index", direction_index)
            trial.set_user_attr(
                "parameters", {k: asdict(v) for k, v in parameters.items()}
            )

            _log(f"\n[Trial {idx}/{settings.n_trials}]")
            for name, val in get_trial_parameters(trial).items():
                _log(f"  {name} = {val}")

            model.reset_model()
            model.abliterate(refusal_directions, direction_index, parameters)
            score, kl, refusals = evaluator.get_score()

            elapsed = time.perf_counter() - start_time
            completed_so_far = idx - start_index
            eta = (
                f", ETA: {format_duration((elapsed / completed_so_far) * (settings.n_trials - idx))}"
                if completed_so_far > 0 and idx < settings.n_trials
                else ""
            )
            _log(
                f"  Refusals: {refusals}/{len(evaluator.bad_prompts)}, "
                f"KL: {kl:.4f}, "
                f"Elapsed: {format_duration(elapsed)}{eta}"
            )

            trial.set_user_attr("kl_divergence", kl)
            trial.set_user_attr("refusals", refusals)
            trial.set_user_attr("base_refusals", evaluator.base_refusals)
            trial.set_user_attr("n_bad_prompts", len(evaluator.bad_prompts))
            return score

        def obj_wrapper(trial: Trial) -> tuple[float, float]:
            try:
                return objective(trial)
            except KeyboardInterrupt:
                trial.study.stop()
                raise TrialPruned()

        study.optimize(
            obj_wrapper,
            n_trials=settings.n_trials - count_completed(),
        )

        if count_completed() == settings.n_trials:
            study.set_user_attr("finished", True)

        # ── Compute Pareto front ───────────────────────────────────────────
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if completed:
            sorted_t = sorted(
                completed,
                key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]),
            )
            best: list[Any] = []
            min_kl = math.inf
            for t in sorted_t:
                kl = t.user_attrs["kl_divergence"]
                if kl < min_kl:
                    min_kl = kl
                    best.append(t)
            _session["best_trials"] = best
            _log(
                f"\nOptimization complete! "
                f"{len(best)} Pareto-optimal trial(s) found. "
                "Open the Results tab to select and export a model."
            )
        else:
            _session["best_trials"] = []
            _log("\nNo trials completed.")

    except Exception:
        _log(f"\nError:\n{traceback.format_exc()}")
    finally:
        _optimization_running.clear()
        _log_queue.put(None)  # sentinel – signals the generator to stop


# ─── Helper: re-apply abliteration after a model merge ───────────────────────


def _restore_trial_model(model: Model) -> None:
    """Re-abliterate *model* using the currently active trial.

    After ``get_merged_model()`` the LoRA adapter is consumed, so we need to
    reset the model and re-apply abliteration before the user can keep working
    with it.
    """
    trial = _session.get("active_trial")
    refusal_directions = _session.get("refusal_directions")
    if trial is not None and refusal_directions is not None:
        model.reset_model()
        model.abliterate(
            refusal_directions,
            trial.user_attrs["direction_index"],
            {
                k: AbliterationParameters(**v)
                for k, v in trial.user_attrs["parameters"].items()
            },
        )


# ─── Local model discovery ────────────────────────────────────────────────────

MODEL_SOURCE_HF = "Hugging Face Hub"
MODEL_SOURCE_LOCAL = "Local / Cached"


def _get_local_models() -> list[str]:
    """Return paths/IDs for models available locally.

    Two sources are scanned:

    1. The Hugging Face cache directory (``~/.cache/huggingface/hub/`` by default,
       or ``$HF_HOME/hub``).  Only snapshots that contain a ``config.json`` are
       included, and they are returned as ``org/name`` model IDs (which
       ``transformers`` will resolve from cache without network access).

    2. Sub-directories of the current working directory that contain a
       ``config.json`` (i.e. locally stored model folders).
    """
    import os
    from pathlib import Path

    found: list[str] = []

    # ── Hugging Face cache ─────────────────────────────────────────────────
    hf_home_env = os.environ.get("HF_HOME")
    hf_home = (
        Path(hf_home_env).expanduser()
        if hf_home_env
        else Path.home() / ".cache" / "huggingface"
    )
    hf_hub_cache = hf_home / "hub"
    if hf_hub_cache.exists():
        try:
            for entry in sorted(hf_hub_cache.iterdir()):
                try:
                    if not entry.is_dir() or not entry.name.startswith("models--"):
                        continue
                    # "models--org--name" → "org/name"
                    parts = entry.name[len("models--") :].split("--")
                    if len(parts) < 2:
                        continue
                    model_id = "/".join(parts)
                    snapshots_dir = entry / "snapshots"
                    if snapshots_dir.exists():
                        for snapshot in snapshots_dir.iterdir():
                            if (
                                snapshot.is_dir()
                                and (snapshot / "config.json").exists()
                            ):
                                found.append(model_id)
                                break
                except OSError:
                    continue
        except OSError:
            pass

    # ── Local sub-directories ──────────────────────────────────────────────
    # Keep the top-level CWD scan shallow (one level deep), and separately
    # scan `cwd/models/` up to two levels deep so that layouts like
    # `models/org/model-name/` are discovered.
    cwd = Path.cwd()
    try:
        for entry in sorted(cwd.iterdir()):
            if entry.is_dir() and (entry / "config.json").exists():
                found.append(str(entry))
    except OSError:
        pass

    models_subdir = cwd / "models"
    if models_subdir.is_dir():
        try:
            for entry in sorted(models_subdir.iterdir()):
                if not entry.is_dir():
                    continue
                if (entry / "config.json").exists():
                    found.append(str(entry))
                else:
                    # One level deeper: e.g. models/org/model-name/
                    try:
                        for subentry in sorted(entry.iterdir()):
                            if (
                                subentry.is_dir()
                                and (subentry / "config.json").exists()
                            ):
                                found.append(str(subentry))
                    except OSError:
                        continue
        except OSError:
            pass

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique


# ─── Gradio app ───────────────────────────────────────────────────────────────


def create_app() -> Any:
    """Return the Gradio :class:`~gradio.Blocks` application."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the web UI. Install it with `pip install heretic-llm[webui]`."
        ) from exc

    _ver = version("heretic-llm")

    with gr.Blocks(title=f"Heretic {_ver}") as app:
        gr.Markdown(
            f"# 🔥 Heretic {_ver}\n"
            "> Fully automatic censorship removal for language models\n\n"
            "⚠️ *This is a single-user tool intended for local use only.*"
        )

        with gr.Tabs():
            # ── Tab 1: Configure & Run ─────────────────────────────────────
            with gr.Tab("⚙️ Configure & Run"):
                with gr.Row():
                    with gr.Column():
                        model_source_radio = gr.Radio(
                            choices=[MODEL_SOURCE_HF, MODEL_SOURCE_LOCAL],
                            value=MODEL_SOURCE_HF,
                            label="Model source",
                        )
                        # Shown when "Hugging Face Hub" is selected
                        model_id_in = gr.Textbox(
                            label="Model ID",
                            placeholder="e.g., Qwen/Qwen3-4B-Instruct-2507",
                            visible=True,
                        )
                        # Shown when "Local / Cached" is selected
                        with gr.Row(visible=False) as local_model_row:
                            local_model_in = gr.Dropdown(
                                label="Local / cached model",
                                choices=_get_local_models(),
                                value=None,
                                scale=5,
                            )
                            refresh_local_btn = gr.Button("🔄", scale=0, min_width=48)
                        quantization_in = gr.Dropdown(
                            choices=["none", "bnb_4bit"],
                            value="none",
                            label="Quantization",
                            info="Use bnb_4bit to reduce VRAM usage",
                        )
                        n_trials_in = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=200,
                            step=10,
                            label="Number of optimization trials",
                        )
                        system_prompt_in = gr.Textbox(
                            label="System prompt",
                            value="You are a helpful assistant.",
                            lines=2,
                        )

                    with gr.Column():
                        with gr.Accordion("Advanced settings", open=False):
                            n_startup_in = gr.Slider(
                                minimum=5,
                                maximum=200,
                                value=60,
                                step=5,
                                label="Startup trials (random exploration phase)",
                            )
                            kl_scale_in = gr.Number(
                                value=1.0,
                                label="KL divergence scale",
                                info="Typical KL divergence for abliterated models",
                            )
                            kl_target_in = gr.Number(
                                value=0.01,
                                label="KL divergence target",
                            )
                            checkpoint_dir_in = gr.Textbox(
                                label="Checkpoint directory",
                                value="checkpoints",
                            )

                start_btn = gr.Button("▶ Start Optimization", variant="primary")
                log_out = gr.Textbox(
                    label="Progress log",
                    lines=20,
                    max_lines=50,
                    autoscroll=True,
                    interactive=False,
                )

            # ── Tab 2: Results ─────────────────────────────────────────────
            with gr.Tab("📊 Results"):
                results_status = gr.Markdown(
                    "Run optimization first, then click *Refresh Results*."
                )
                refresh_btn = gr.Button("🔄 Refresh Results")
                trials_table = gr.Dataframe(
                    headers=["Trial #", "Refusals", "KL Divergence"],
                    interactive=False,
                    visible=False,
                )
                trial_selector = gr.Radio(
                    label="Select a Pareto-optimal trial",
                    visible=False,
                )
                apply_trial_btn = gr.Button(
                    "✅ Apply selected trial", variant="primary", visible=False
                )
                trial_apply_status = gr.Markdown("")

            # ── Tab 3: Export ──────────────────────────────────────────────
            with gr.Tab("💾 Export"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Save to local folder")
                        save_path_in = gr.Textbox(
                            label="Output directory",
                            placeholder="/path/to/output",
                        )
                        save_adapter_in = gr.Checkbox(
                            label="Save LoRA adapter only (skip merge)",
                            value=False,
                        )
                        save_btn = gr.Button("💾 Save model")
                        save_status = gr.Markdown("")

                    with gr.Column():
                        gr.Markdown("### Upload to Hugging Face Hub")
                        hf_token_in = gr.Textbox(
                            label="Hugging Face access token",
                            type="password",
                            placeholder="hf_...",
                        )
                        hf_repo_in = gr.Textbox(
                            label="Repository ID",
                            placeholder="username/model-name-heretic",
                        )
                        hf_private_in = gr.Checkbox(
                            label="Private repository", value=False
                        )
                        upload_adapter_in = gr.Checkbox(
                            label="Upload LoRA adapter only (skip merge)",
                            value=False,
                        )
                        upload_btn = gr.Button("☁️ Upload to Hugging Face")
                        upload_status = gr.Markdown("")

            # ── Tab 4: Chat ────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                gr.Markdown(
                    "Apply a trial in the **Results** tab to enable the chat. "
                    "Responses are generated by the decensored model."
                )
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                )
                with gr.Row():
                    chat_in = gr.Textbox(
                        label="",
                        placeholder="Type a message…",
                        scale=5,
                        show_label=False,
                    )
                    chat_send = gr.Button("Send", scale=1, variant="primary")
                chat_clear = gr.Button("🗑️ Clear conversation")

        # ── Event handlers ─────────────────────────────────────────────────

        def _toggle_model_source(
            source: str,
        ) -> tuple[Any, Any]:
            is_hf = source == MODEL_SOURCE_HF
            return gr.update(visible=is_hf), gr.update(visible=not is_hf)

        model_source_radio.change(
            fn=_toggle_model_source,
            inputs=[model_source_radio],
            outputs=[model_id_in, local_model_row],
        )

        refresh_local_btn.click(
            fn=lambda: gr.update(choices=_get_local_models()),
            outputs=[local_model_in],
        )

        def run_optimization_generator(
            model_source: str,
            model_id: str,
            local_model: str | None,
            quantization: str,
            n_trials: int,
            n_startup: int,
            system_prompt: str,
            kl_scale: float,
            kl_target: float,
            checkpoint_dir: str,
        ) -> Generator[tuple, None, None]:
            btn_idle = gr.update(value="▶ Start Optimization", interactive=True)
            btn_running = gr.update(value="⏳ Optimization running…", interactive=False)

            effective_model_id = (
                model_id if model_source == MODEL_SOURCE_HF else (local_model or "")
            )
            if not effective_model_id.strip():
                yield (
                    (
                        "⚠ Please enter a model ID."
                        if model_source == MODEL_SOURCE_HF
                        else "⚠ Please select a local model."
                    ),
                    btn_idle,
                )
                return

            if _optimization_running.is_set():
                yield "⚠ An optimization is already running. Wait for it to finish.", btn_idle
                return

            # Clear old session data and drain the queue.
            _session.clear()
            while not _log_queue.empty():
                try:
                    _log_queue.get_nowait()
                except queue.Empty:
                    break

            _optimization_running.set()
            thread = threading.Thread(
                target=_run_optimization,
                args=(
                    effective_model_id,
                    quantization,
                    n_trials,
                    n_startup,
                    system_prompt,
                    kl_scale,
                    kl_target,
                    checkpoint_dir,
                ),
                daemon=True,
            )
            try:
                thread.start()
            except Exception:
                _optimization_running.clear()
                raise

            max_log_chars = 100_000
            log_lines: deque[str] = deque()
            log_char_count = 0
            truncated = False

            def render_log(truncated_flag: bool, lines: deque[str]) -> str:
                prefix = "… (older log lines omitted)\n" if truncated_flag else ""
                return prefix + "".join(lines)

            while True:
                try:
                    msg = _log_queue.get(timeout=0.3)
                except queue.Empty:
                    # Keep the generator alive while the thread is still running.
                    yield render_log(truncated, log_lines), btn_running
                    continue

                if msg is None:
                    # Sentinel: optimization thread has finished.
                    break

                line = msg + "\n"
                log_lines.append(line)
                log_char_count += len(line)
                while log_char_count > max_log_chars and log_lines:
                    log_char_count -= len(log_lines.popleft())
                    truncated = True
                yield render_log(truncated, log_lines), btn_running

            yield render_log(truncated, log_lines), btn_idle

        start_btn.click(
            fn=run_optimization_generator,
            inputs=[
                model_source_radio,
                model_id_in,
                local_model_in,
                quantization_in,
                n_trials_in,
                n_startup_in,
                system_prompt_in,
                kl_scale_in,
                kl_target_in,
                checkpoint_dir_in,
            ],
            outputs=[log_out, start_btn],
        )

        # ── Results tab ────────────────────────────────────────────────────

        def load_results() -> tuple:
            best: list[Any] | None = _session.get("best_trials")
            if not best:
                return (
                    gr.update(value="No results yet. Run optimization first."),
                    gr.update(visible=False),
                    gr.update(choices=[], visible=False),
                    gr.update(visible=False),
                )

            table_data = [
                [
                    t.user_attrs["index"],
                    f"{t.user_attrs['refusals']}/{t.user_attrs['n_bad_prompts']}",
                    f"{t.user_attrs['kl_divergence']:.4f}",
                ]
                for t in best
            ]
            choices = [
                f"Trial {t.user_attrs['index']}: "
                f"{t.user_attrs['refusals']}/{t.user_attrs['n_bad_prompts']} refusals, "
                f"KL {t.user_attrs['kl_divergence']:.4f}"
                for t in best
            ]
            return (
                gr.update(
                    value=(
                        "Pareto-optimal trials – lower is better for both metrics. "
                        "Select a trial and click *Apply selected trial*."
                    )
                ),
                gr.update(value=table_data, visible=True),
                gr.update(choices=choices, value=choices[0], visible=True),
                gr.update(visible=True),
            )

        refresh_btn.click(
            fn=load_results,
            outputs=[results_status, trials_table, trial_selector, apply_trial_btn],
        )

        def apply_trial(selected: str) -> str:
            best: list[Any] | None = _session.get("best_trials")
            if not best or not selected:
                return "⚠ No trial selected."

            try:
                trial_idx = int(selected.split("Trial ")[1].split(":")[0])
            except (ValueError, IndexError):
                return "⚠ Could not parse the trial index."

            trial = next((t for t in best if t.user_attrs["index"] == trial_idx), None)
            if trial is None:
                return "⚠ Trial not found."

            model: Model | None = _session.get("model")
            refusal_directions = _session.get("refusal_directions")
            if model is None or refusal_directions is None:
                return "⚠ Model not loaded."

            try:
                model.reset_model()
                model.abliterate(
                    refusal_directions,
                    trial.user_attrs["direction_index"],
                    {
                        k: AbliterationParameters(**v)
                        for k, v in trial.user_attrs["parameters"].items()
                    },
                )
                _session["active_trial"] = trial
                return (
                    f"✅ Trial {trial_idx} applied. "
                    "You can now export the model (Export tab) or chat with it (Chat tab)."
                )
            except Exception as exc:
                return f"❌ Error: {exc}"

        apply_trial_btn.click(
            fn=apply_trial,
            inputs=[trial_selector],
            outputs=[trial_apply_status],
        )

        # ── Export tab ─────────────────────────────────────────────────────

        def save_model(path: str, adapter_only: bool) -> str:
            model: Model | None = _session.get("model")
            settings: Settings | None = _session.get("settings")
            if model is None or settings is None:
                return "⚠ No model loaded."
            if not path.strip():
                return "⚠ Please enter a save path."
            if _session.get("active_trial") is None:
                return "⚠ Apply a trial first (in the Results tab)."

            try:
                if adapter_only:
                    model.model.save_pretrained(
                        path, max_shard_size=settings.max_shard_size
                    )
                    return f"✅ LoRA adapter saved to `{path}`"
                else:
                    merged = model.get_merged_model()
                    merged.save_pretrained(path, max_shard_size=settings.max_shard_size)
                    del merged
                    empty_cache()
                    model.tokenizer.save_pretrained(path)
                    _restore_trial_model(model)
                    return f"✅ Model saved to `{path}`"
            except Exception as exc:
                return f"❌ Error: {exc}"

        save_btn.click(
            fn=save_model,
            inputs=[save_path_in, save_adapter_in],
            outputs=[save_status],
        )

        def upload_model(
            token: str, repo_id: str, private: bool, adapter_only: bool
        ) -> str:
            model: Model | None = _session.get("model")
            settings: Settings | None = _session.get("settings")
            if model is None or settings is None:
                return "⚠ No model loaded."
            if not token.strip():
                return "⚠ Please provide a Hugging Face access token."
            if not repo_id.strip():
                return "⚠ Please provide a repository ID."
            if _session.get("active_trial") is None:
                return "⚠ Apply a trial first (in the Results tab)."

            try:
                import huggingface_hub

                user = huggingface_hub.whoami(token)
                user_name = user.get("name") or "unknown"

                if adapter_only:
                    model.model.push_to_hub(
                        repo_id,
                        private=private,
                        max_shard_size=settings.max_shard_size,
                        token=token,
                    )
                else:
                    merged = model.get_merged_model()
                    merged.push_to_hub(
                        repo_id,
                        private=private,
                        max_shard_size=settings.max_shard_size,
                        token=token,
                    )
                    del merged
                    empty_cache()
                    model.tokenizer.push_to_hub(repo_id, private=private, token=token)
                    _restore_trial_model(model)

                return f"✅ Logged in as `{user_name}`. Model uploaded to `{repo_id}`"
            except Exception as exc:
                return f"❌ Error: {exc}"

        upload_btn.click(
            fn=upload_model,
            inputs=[hf_token_in, hf_repo_in, hf_private_in, upload_adapter_in],
            outputs=[upload_status],
        )

        # ── Chat tab ───────────────────────────────────────────────────────

        def send_message(
            message: str,
            history: list[dict[str, str]] | None,
        ) -> tuple[str, list[dict[str, str]]]:
            history = history or []

            if not message.strip():
                return "", history

            model: Model | None = _session.get("model")
            settings: Settings | None = _session.get("settings")

            history = history + [{"role": "user", "content": message}]

            if model is None or _session.get("active_trial") is None:
                return "", history + [
                    {
                        "role": "assistant",
                        "content": "⚠ Apply a trial first (in the Results tab).",
                    }
                ]

            chat = [
                {
                    "role": "system",
                    "content": settings.system_prompt
                    if settings
                    else "You are a helpful assistant.",
                }
            ]
            for msg in history:
                chat.append({"role": msg["role"], "content": msg["content"]})

            try:
                response = model.stream_chat_response(chat)
            except Exception as exc:
                response = f"❌ Error: {exc}"

            return "", history + [{"role": "assistant", "content": response}]

        chat_send.click(
            fn=send_message,
            inputs=[chat_in, chatbot],
            outputs=[chat_in, chatbot],
        )
        chat_in.submit(
            fn=send_message,
            inputs=[chat_in, chatbot],
            outputs=[chat_in, chatbot],
        )
        chat_clear.click(fn=lambda: ([], ""), outputs=[chatbot, chat_in])

    return app


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for ``heretic-webui``."""
    from .webui_cli import main as cli_main

    cli_main()
