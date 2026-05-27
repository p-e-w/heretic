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
from dataclasses import asdict, dataclass
from importlib.metadata import version
from os.path import commonprefix
from types import SimpleNamespace
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
from .utils import (
    format_duration,
    get_trial_parameters,
    load_prompts,
    resolve_ollama_model_reference,
    set_seed,
    write_ollama_modelfile,
)

# ─── Log capture infrastructure ──────────────────────────────────────────────

_MAX_LOG_QUEUE_ITEMS = 10_000
_LOG_QUEUE_PUT_RETRIES = 3
_log_queue: queue.Queue[str | None] = queue.Queue(maxsize=_MAX_LOG_QUEUE_ITEMS)

# Server-side store so the accumulated log survives page reloads.
_MAX_LOG_LINES = 10_000
_log_lines_store: deque[str] = deque(maxlen=_MAX_LOG_LINES)
_log_lines_lock = threading.Lock()
_log_lines_version = 0

# Capture the real stdout before any monkey-patching so log lines can always
# be forwarded to it (visible in ``docker logs`` and plain terminal runs).
_real_stdout = sys.stdout

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text).strip()


def _emit(line: str) -> None:
    """Enqueue *line* for the web UI log and echo it to the real stdout."""
    _enqueue_log_queue(line)
    _append_log_line(line)
    try:
        _real_stdout.write(line + "\n")
        _real_stdout.flush()
    except (OSError, ValueError):
        pass


def _enqueue_log_queue(item: str | None) -> None:
    """Enqueue *item* without allowing unbounded queue growth."""
    for _ in range(_LOG_QUEUE_PUT_RETRIES):
        try:
            _log_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                _log_queue.get_nowait()
            except queue.Empty:
                continue


def _append_log_line(line: str) -> None:
    """Append a log line to the server-side store."""
    global _log_lines_version
    with _log_lines_lock:
        _log_lines_store.append(line + "\n")
        _log_lines_version += 1


def _reset_log_store() -> None:
    """Clear the server-side log store."""
    global _log_lines_version
    with _log_lines_lock:
        _log_lines_store.clear()
        _log_lines_version = 0


def _get_log_snapshot() -> tuple[int, str]:
    """Return the current log version and full rendered content."""
    with _log_lines_lock:
        return _log_lines_version, "".join(_log_lines_store)


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


class _ProgressFile:
    """File-like object for stderr that forwards tqdm progress bars to the web UI log.

    tqdm overwrites the current terminal line by writing ``\\r`` (carriage
    return) before each update.  A plain line-oriented queue would never see
    those updates because they never end with ``\\n``.  This class handles
    ``\\r``-terminated segments explicitly and throttles them to at most one
    log entry per second so that large downloads do not flood the console.
    ``\\n``-terminated lines (including the final "100% done" line) are always
    forwarded immediately.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._last_emit = 0.0
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            self._buf += text
            while "\r" in self._buf or "\n" in self._buf:
                cr_pos = self._buf.find("\r")
                nl_pos = self._buf.find("\n")
                if nl_pos != -1 and (cr_pos == -1 or nl_pos <= cr_pos):
                    # Newline-terminated line: always emit.
                    line, self._buf = self._buf.split("\n", 1)
                    clean = _strip_ansi(line).strip()
                    if clean:
                        _emit(clean)
                else:
                    # Carriage-return update: throttle to 1 Hz to reduce noise.
                    line = self._buf[:cr_pos]
                    self._buf = self._buf[cr_pos + 1 :]
                    clean = _strip_ansi(line).strip()
                    if clean:
                        now = time.monotonic()
                        if now - self._last_emit >= 1.0:
                            self._last_emit = now
                            _emit(clean)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._buf:
                clean = _strip_ansi(self._buf).strip()
                if clean:
                    _emit(clean)
                self._buf = ""

    def isatty(self) -> bool:
        # Return True so that tqdm does not auto-disable its progress bars
        # when the file is not a real terminal.
        return True


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
    _old_stderr = sys.stderr
    progress_stderr = _ProgressFile()
    sys.stderr = progress_stderr
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

        resolved_model = resolve_ollama_model_reference(settings.model)
        if resolved_model != settings.model:
            _log(f"Using model source from Ollama Modelfile: {resolved_model}")
            settings.model = resolved_model

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
                "Open the Review tab to select a trial, then use Publish or Chat."
            )
        else:
            _session["best_trials"] = []
            _log("\nNo trials completed.")

    except Exception:
        _log(f"\nError:\n{traceback.format_exc()}")
    finally:
        try:
            progress_stderr.flush()
        finally:
            sys.stderr = _old_stderr
        _optimization_running.clear()
        _enqueue_log_queue(None)  # sentinel – signals the generator to stop


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

    Three sources are scanned:

    1. The Hugging Face cache directory (``~/.cache/huggingface/hub/`` by default,
       or ``$HF_HOME/hub``). Only snapshots that contain a ``config.json`` are
       included, and they are returned as ``org/name`` model IDs (which
       ``transformers`` will resolve from cache without network access).

    2. Ollama model names from ``ollama list``.

    3. Sub-directories of the current working directory that contain a
       ``config.json`` or ``Modelfile`` (i.e. locally stored model folders).
    """
    import os
    import subprocess
    from pathlib import Path

    def has_modelfile(path: Path) -> bool:
        try:
            return any(
                candidate.is_file() and candidate.name.lower() == "modelfile"
                for candidate in path.iterdir()
            )
        except OSError:
            return False

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

    # ── Ollama model names ────────────────────────────────────────────────
    try:
        output = subprocess.check_output(
            ["ollama", "list"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        for line in output.splitlines()[1:]:
            parts = line.split()
            if parts:
                found.append(parts[0])
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ):
        pass

    # ── Local sub-directories ──────────────────────────────────────────────
    # Keep the top-level CWD scan shallow (one level deep), and separately
    # scan `cwd/models/` up to two levels deep so that layouts like
    # `models/org/model-name/` are discovered.
    cwd = Path.cwd()
    try:
        for entry in sorted(cwd.iterdir()):
            if entry.is_dir() and (
                (entry / "config.json").exists() or has_modelfile(entry)
            ):
                found.append(str(entry))
    except OSError:
        pass

    models_subdir = cwd / "models"
    if models_subdir.is_dir():
        try:
            for entry in sorted(models_subdir.iterdir()):
                if not entry.is_dir():
                    continue
                if (entry / "config.json").exists() or has_modelfile(entry):
                    found.append(str(entry))
                else:
                    # One level deeper: e.g. models/org/model-name/
                    try:
                        for subentry in sorted(entry.iterdir()):
                            if (
                                subentry.is_dir()
                                and (
                                    (subentry / "config.json").exists()
                                    or has_modelfile(subentry)
                                )
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


_DEFAULT_UI_SETTINGS = {
    "model_source": MODEL_SOURCE_HF,
    "model_id": "",
    "local_model": None,
    "quantization": "none",
    "n_trials": 200,
    "n_startup": 60,
    "system_prompt": "You are a helpful assistant.",
    "kl_scale": 1.0,
    "kl_target": 0.01,
}
_POLL_INTERVAL_SECONDS = 2.0
# WEBUI_CSS is injected into the page by Gradio via the `head` parameter when the
# app is served.  It is NOT passed via gr.Blocks(css=...) — keep it as a module-level
# string so it is easy to edit without touching the Blocks call.
WEBUI_CSS = """
.gradio-container {
    background: #0f172a !important;
}
.app-shell {
    max-width: 1240px;
    margin: 0 auto;
    padding-bottom: 2rem;
}
.hero-card,
.panel-card {
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 22px;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.4);
}
.hero-card {
    padding: 2rem;
    margin-bottom: 1.25rem;
    color: #f8fafc;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.88));
    border-color: rgba(251, 146, 60, 0.3);
}
.hero-card h1,
.hero-card p,
.hero-card li,
.hero-card strong {
    color: inherit;
}
.hero-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.9rem;
    margin-top: 1.25rem;
}
.hero-metric {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.36);
    border: 1px solid rgba(251, 146, 60, 0.25);
}
.hero-metric span {
    display: block;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(226, 232, 240, 0.78);
}
.hero-metric strong {
    display: block;
    margin-top: 0.3rem;
    font-size: 1rem;
    color: #f8fafc;
}
.tab-copy {
    margin-bottom: 0.9rem;
}
.panel-card {
    background: #1e293b;
    margin-bottom: 1rem;
    padding: 1.2rem;
}
.panel-card h3 {
    margin-top: 0;
    color: #f1f5f9;
}
.subtle-note {
    color: #94a3b8;
    font-size: 0.95rem;
}
"""


@dataclass
class ConfigureTabComponents:
    model_source_radio: Any
    model_id_in: Any
    local_model_section: Any
    local_model_in: Any
    refresh_local_btn: Any
    local_models_status: Any
    quantization_in: Any
    n_trials_in: Any
    n_startup_in: Any
    system_prompt_in: Any
    kl_scale_in: Any
    kl_target_in: Any
    start_btn: Any
    log_out: Any


@dataclass
class ReviewTabComponents:
    results_status: Any
    refresh_btn: Any
    trials_table: Any
    trial_selector: Any
    apply_trial_btn: Any
    trial_apply_status: Any


@dataclass
class PublishTabComponents:
    save_path_in: Any
    save_adapter_in: Any
    save_btn: Any
    save_status: Any
    hf_token_in: Any
    hf_repo_in: Any
    hf_private_in: Any
    upload_adapter_in: Any
    upload_btn: Any
    upload_status: Any


@dataclass
class ChatTabComponents:
    chatbot: Any
    chat_in: Any
    chat_send: Any
    chat_clear: Any


@dataclass
class WebUIComponents:
    configure: ConfigureTabComponents
    review: ReviewTabComponents
    publish: PublishTabComponents
    chat: ChatTabComponents


def _render_header(app_version: str) -> str:
    return f"""
<div class="hero-card">
  <h1>🔥 Heretic {app_version}</h1>
  <p>Fully automatic censorship removal for language models.</p>
  <p><strong>Local-first, single-user workflow:</strong> configure a model, launch optimization, compare Pareto-optimal trials, then export or chat with the selected result.</p>
  <div class="hero-metrics">
    <div class="hero-metric">
      <span>Step 1</span>
      <strong>Configure model source, quantization, and trial budget</strong>
    </div>
    <div class="hero-metric">
      <span>Step 2</span>
      <strong>Watch the live optimization log without leaving the page</strong>
    </div>
    <div class="hero-metric">
      <span>Step 3</span>
      <strong>Apply the best trial, then save, upload, or chat</strong>
    </div>
  </div>
</div>
"""


def _build_configure_tab(gr: Any) -> ConfigureTabComponents:
    gr.Markdown("## 🚀 Launch")
    gr.Markdown(
        "Set up the optimization run and monitor progress below.",
        elem_classes=["tab-copy"],
    )
    with gr.Row(equal_height=True):
        with gr.Column(scale=7):
            with gr.Group(elem_classes=["panel-card"]):
                gr.Markdown("### Model & run setup")
                model_source_radio = gr.Radio(
                    choices=[MODEL_SOURCE_HF, MODEL_SOURCE_LOCAL],
                    value=_DEFAULT_UI_SETTINGS["model_source"],
                    label="Model source",
                )
                model_id_in = gr.Textbox(
                    label="Model ID",
                    placeholder="e.g., Qwen/Qwen3-4B-Instruct-2507",
                    value=_DEFAULT_UI_SETTINGS["model_id"],
                    visible=True,
                )
                with gr.Column(visible=False) as local_model_section:
                    with gr.Row():
                        local_model_in = gr.Dropdown(
                            label="Local / cached model",
                            choices=_get_local_models(),
                            value=_DEFAULT_UI_SETTINGS["local_model"],
                            scale=5,
                        )
                        refresh_local_btn = gr.Button("Refresh", scale=1, min_width=96)
                    local_models_status = gr.Markdown(
                        "*Refresh to scan the Hugging Face cache, Ollama, and local model folders.*",
                        elem_classes=["subtle-note"],
                    )
                quantization_in = gr.Dropdown(
                    choices=["none", "bnb_4bit"],
                    value=_DEFAULT_UI_SETTINGS["quantization"],
                    label="Quantization",
                    info="Use bnb_4bit to reduce VRAM usage.",
                )
                with gr.Row():
                    n_trials_in = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=_DEFAULT_UI_SETTINGS["n_trials"],
                        step=10,
                        label="Optimization trials",
                    )
                    n_startup_in = gr.Slider(
                        minimum=5,
                        maximum=200,
                        value=_DEFAULT_UI_SETTINGS["n_startup"],
                        step=5,
                        label="Startup trials",
                    )
                system_prompt_in = gr.Textbox(
                    label="System prompt",
                    value=_DEFAULT_UI_SETTINGS["system_prompt"],
                    lines=3,
                )
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["panel-card"]):
                gr.Markdown("### Optimization targets")
                gr.Markdown(
                    "- Lower refusals are better.\n"
                    "- Lower KL divergence is better.\n"
                    "- Startup trials seed Optuna with random exploration.",
                    elem_classes=["subtle-note"],
                )
                kl_scale_in = gr.Number(
                    value=_DEFAULT_UI_SETTINGS["kl_scale"],
                    label="KL divergence scale",
                    info="Typical KL divergence for abliterated models.",
                )
                kl_target_in = gr.Number(
                    value=_DEFAULT_UI_SETTINGS["kl_target"],
                    label="KL divergence target",
                )
            with gr.Group(elem_classes=["panel-card"]):
                gr.Markdown("### Session notes")
                gr.Markdown(
                    "- Optimization uses the current working directory for checkpoints.\n"
                    "- The web UI is designed for local, single-user sessions.\n"
                    "- Reloading the page preserves the saved form values and server-side log.",
                    elem_classes=["subtle-note"],
                )
    with gr.Group(elem_classes=["panel-card"]):
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                start_btn = gr.Button(
                    "Start optimization",
                    variant="primary",
                )
            with gr.Column(scale=9):
                gr.Markdown(
                    "Launches the full Heretic pipeline in a background thread and streams the captured log below.",
                    elem_classes=["subtle-note"],
                )
        log_out = gr.Textbox(
            label="Live run log",
            lines=22,
            max_lines=60,
            autoscroll=False,
            interactive=False,
        )
    return ConfigureTabComponents(
        model_source_radio=model_source_radio,
        model_id_in=model_id_in,
        local_model_section=local_model_section,
        local_model_in=local_model_in,
        refresh_local_btn=refresh_local_btn,
        local_models_status=local_models_status,
        quantization_in=quantization_in,
        n_trials_in=n_trials_in,
        n_startup_in=n_startup_in,
        system_prompt_in=system_prompt_in,
        kl_scale_in=kl_scale_in,
        kl_target_in=kl_target_in,
        start_btn=start_btn,
        log_out=log_out,
    )


def _build_review_tab(gr: Any) -> ReviewTabComponents:
    gr.Markdown("## 📊 Review")
    gr.Markdown(
        "Refresh the Pareto front after optimization, inspect the best trials, and apply one before exporting or chatting.",
        elem_classes=["tab-copy"],
    )
    with gr.Group(elem_classes=["panel-card"]):
        results_status = gr.Markdown(
            "No optimization results loaded yet. Run a session, then refresh."
        )
        refresh_btn = gr.Button("Refresh results", variant="secondary")
    with gr.Group(elem_classes=["panel-card"]):
        trials_table = gr.Dataframe(
            headers=["Trial #", "Refusals", "KL Divergence"],
            interactive=False,
            visible=False,
            label="Pareto-optimal trials",
        )
        trial_selector = gr.Radio(
            label="Trial to apply",
            visible=False,
        )
        apply_trial_btn = gr.Button(
            "Apply selected trial",
            variant="primary",
            visible=False,
        )
        trial_apply_status = gr.Markdown("")
    return ReviewTabComponents(
        results_status=results_status,
        refresh_btn=refresh_btn,
        trials_table=trials_table,
        trial_selector=trial_selector,
        apply_trial_btn=apply_trial_btn,
        trial_apply_status=trial_apply_status,
    )


def _build_publish_tab(gr: Any) -> PublishTabComponents:
    gr.Markdown("## 💾 Publish")
    gr.Markdown(
        "After applying a trial, either save the model locally or push it to Hugging Face Hub.",
        elem_classes=["tab-copy"],
    )
    with gr.Row(equal_height=True):
        with gr.Column():
            with gr.Group(elem_classes=["panel-card"]):
                gr.Markdown("### Save locally")
                save_path_in = gr.Textbox(
                    label="Output directory",
                    placeholder="/path/to/output",
                )
                save_adapter_in = gr.Checkbox(
                    label="Save LoRA adapter only (skip merge)",
                    value=False,
                )
                save_btn = gr.Button("Save model")
                save_status = gr.Markdown("")
        with gr.Column():
            with gr.Group(elem_classes=["panel-card"]):
                gr.Markdown("### Upload to Hugging Face Hub")
                hf_token_in = gr.Textbox(
                    label="Access token",
                    type="password",
                    placeholder="hf_...",
                )
                hf_repo_in = gr.Textbox(
                    label="Repository ID",
                    placeholder="username/model-name-heretic",
                )
                hf_private_in = gr.Checkbox(
                    label="Private repository",
                    value=False,
                )
                upload_adapter_in = gr.Checkbox(
                    label="Upload LoRA adapter only (skip merge)",
                    value=False,
                )
                upload_btn = gr.Button("Upload model")
                upload_status = gr.Markdown("")
    return PublishTabComponents(
        save_path_in=save_path_in,
        save_adapter_in=save_adapter_in,
        save_btn=save_btn,
        save_status=save_status,
        hf_token_in=hf_token_in,
        hf_repo_in=hf_repo_in,
        hf_private_in=hf_private_in,
        upload_adapter_in=upload_adapter_in,
        upload_btn=upload_btn,
        upload_status=upload_status,
    )


def _build_chat_tab(gr: Any) -> ChatTabComponents:
    gr.Markdown("## 💬 Chat")
    gr.Markdown(
        "Apply a Pareto-optimal trial in the **Review** section above, then use this chat to probe the current model behavior.",
        elem_classes=["tab-copy"],
    )
    with gr.Group(elem_classes=["panel-card"]):
        chatbot = gr.Chatbot(
            label="Conversation",
            height=540,
        )
        with gr.Row():
            chat_in = gr.Textbox(
                label="",
                placeholder="Type a message…",
                scale=5,
                show_label=False,
            )
            chat_send = gr.Button("Send", scale=1, variant="primary")
        chat_clear = gr.Button("Clear conversation")
    return ChatTabComponents(
        chatbot=chatbot,
        chat_in=chat_in,
        chat_send=chat_send,
        chat_clear=chat_clear,
    )


def create_app() -> Any:
    """Return the Gradio :class:`~gradio.Blocks` application."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the web UI. Install it with `pip install heretic-llm[webui]`."
        ) from exc

    app_version = version("heretic-llm")

    with gr.Blocks(title=f"Heretic {app_version}") as app:
        settings_state = gr.BrowserState(dict(_DEFAULT_UI_SETTINGS))
        opt_timer = gr.Timer(value=_POLL_INTERVAL_SECONDS, active=False)

        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(_render_header(app_version))
            ui = WebUIComponents(
                configure=_build_configure_tab(gr),
                review=_build_review_tab(gr),
                publish=_build_publish_tab(gr),
                chat=_build_chat_tab(gr),
            )

        # ── Event handlers ─────────────────────────────────────────────────

        def _toggle_model_source(
            source: str,
        ) -> tuple[Any, Any]:
            is_hf = source == MODEL_SOURCE_HF
            return gr.update(visible=is_hf), gr.update(visible=not is_hf)

        ui.configure.model_source_radio.change(
            fn=_toggle_model_source,
            inputs=[ui.configure.model_source_radio],
            outputs=[ui.configure.model_id_in, ui.configure.local_model_section],
        )

        def _refresh_local() -> tuple[Any, str]:
            models = _get_local_models()
            if models:
                status = "**Local models found:**\n" + "\n".join(
                    f"- `{m}`" for m in models
                )
            else:
                status = "*No local models found.*"
            return gr.update(choices=models), status

        ui.configure.refresh_local_btn.click(
            fn=_refresh_local,
            outputs=[ui.configure.local_model_in, ui.configure.local_models_status],
        )

        app.load(
            fn=_refresh_local,
            outputs=[ui.configure.local_model_in, ui.configure.local_models_status],
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
        ) -> Generator[tuple, None, None]:
            btn_idle = gr.update(value="Start optimization", interactive=True)
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
                yield "⚠ An optimization is already running. Wait for it to finish.", btn_running
                return

            # Clear old session data and drain the queue.
            _session.clear()
            _reset_log_store()
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
                    "checkpoints",
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

            yield render_log(truncated, log_lines), btn_running
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

        ui.configure.start_btn.click(
            fn=lambda: (gr.update(interactive=False), gr.update(active=False)),
            outputs=[ui.configure.start_btn, opt_timer],
        ).then(
            fn=run_optimization_generator,
            inputs=[
                ui.configure.model_source_radio,
                ui.configure.model_id_in,
                ui.configure.local_model_in,
                ui.configure.quantization_in,
                ui.configure.n_trials_in,
                ui.configure.n_startup_in,
                ui.configure.system_prompt_in,
                ui.configure.kl_scale_in,
                ui.configure.kl_target_in,
            ],
            outputs=[ui.configure.log_out, ui.configure.start_btn],
        )

        # ── Review tab ─────────────────────────────────────────────────────

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

        ui.review.refresh_btn.click(
            fn=load_results,
            outputs=[
                ui.review.results_status,
                ui.review.trials_table,
                ui.review.trial_selector,
                ui.review.apply_trial_btn,
            ],
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
                    "You can now save or upload it in Publish, or test it in Chat."
                )
            except Exception as exc:
                return f"❌ Error: {exc}"

        ui.review.apply_trial_btn.click(
            fn=apply_trial,
            inputs=[ui.review.trial_selector],
            outputs=[ui.review.trial_apply_status],
        )

        # ── Publish tab ────────────────────────────────────────────────────

        def save_model(path: str, adapter_only: bool) -> str:
            model: Model | None = _session.get("model")
            settings: Settings | None = _session.get("settings")
            if model is None or settings is None:
                return "⚠ No model loaded."
            if not path.strip():
                return "⚠ Please enter a save path."
            if _session.get("active_trial") is None:
                return "⚠ Apply a trial first (in the Review tab)."

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
                    modelfile_path = write_ollama_modelfile(path)
                    _restore_trial_model(model)
                    return (
                        f"✅ Model saved to `{path}`\n\n"
                        f"✅ Ollama Modelfile saved to `{modelfile_path}`"
                    )
            except Exception as exc:
                return f"❌ Error: {exc}"

        ui.publish.save_btn.click(
            fn=save_model,
            inputs=[ui.publish.save_path_in, ui.publish.save_adapter_in],
            outputs=[ui.publish.save_status],
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
                return "⚠ Apply a trial first (in the Review tab)."

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

        ui.publish.upload_btn.click(
            fn=upload_model,
            inputs=[
                ui.publish.hf_token_in,
                ui.publish.hf_repo_in,
                ui.publish.hf_private_in,
                ui.publish.upload_adapter_in,
            ],
            outputs=[ui.publish.upload_status],
        )

        # ── Chat tab ───────────────────────────────────────────────────────

        def send_message(
            message: str,
            history: list[dict[str, str] | tuple[str, str]] | None,
        ) -> tuple[str, list[dict[str, str]]]:
            normalized_history: list[dict[str, str]] = []
            for item in history or []:
                if isinstance(item, dict):
                    role = item.get("role")
                    content = item.get("content")
                    if role in {"user", "assistant"} and isinstance(content, str):
                        normalized_history.append({"role": role, "content": content})
                elif (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and all(isinstance(part, str) for part in item)
                ):
                    user_msg, assistant_msg = item
                    if user_msg:
                        normalized_history.append(
                            {"role": "user", "content": user_msg}
                        )
                    if assistant_msg:
                        normalized_history.append(
                            {"role": "assistant", "content": assistant_msg}
                        )
            history = normalized_history

            if not message.strip():
                return "", history

            model: Model | None = _session.get("model")
            settings: Settings | None = _session.get("settings")

            history = history + [{"role": "user", "content": message}]

            if model is None or _session.get("active_trial") is None:
                return "", history + [
                    {
                        "role": "assistant",
                        "content": "⚠ Apply a trial first (in the Review tab).",
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

        ui.chat.chat_send.click(
            fn=send_message,
            inputs=[ui.chat.chat_in, ui.chat.chatbot],
            outputs=[ui.chat.chat_in, ui.chat.chatbot],
        )
        ui.chat.chat_in.submit(
            fn=send_message,
            inputs=[ui.chat.chat_in, ui.chat.chatbot],
            outputs=[ui.chat.chat_in, ui.chat.chatbot],
        )
        ui.chat.chat_clear.click(
            fn=lambda: ([], ""),
            outputs=[ui.chat.chatbot, ui.chat.chat_in],
        )

        # ── Settings persistence ───────────────────────────────────────────

        _settings_keys = list(_DEFAULT_UI_SETTINGS.keys())
        _settings_keys_tuple = tuple(_settings_keys)
        _settings_comps = [
            ui.configure.model_source_radio,
            ui.configure.model_id_in,
            ui.configure.local_model_in,
            ui.configure.quantization_in,
            ui.configure.n_trials_in,
            ui.configure.n_startup_in,
            ui.configure.system_prompt_in,
            ui.configure.kl_scale_in,
            ui.configure.kl_target_in,
        ]
        for _comp in _settings_comps:
            _comp.change(
                fn=lambda *vals, _keys=_settings_keys_tuple: dict(zip(_keys, vals)),
                inputs=_settings_comps,
                outputs=[settings_state],
                queue=False,
            )

        # ── Page load: restore log, settings, and button/timer state ──────

        def _on_page_load(state: dict | None) -> tuple:
            if not isinstance(state, dict):
                state = {}
            source = state.get("model_source", MODEL_SOURCE_HF)
            is_hf = source == MODEL_SOURCE_HF
            is_running = _optimization_running.is_set()
            _, log_content = _get_log_snapshot()
            return (
                gr.update(value=source),
                gr.update(value=state.get("model_id", ""), visible=is_hf),
                gr.update(visible=not is_hf),
                gr.update(value=state.get("local_model")),
                gr.update(value=state.get("quantization", "none")),
                gr.update(value=state.get("n_trials", 200)),
                gr.update(
                    value=state.get("system_prompt", "You are a helpful assistant.")
                ),
                gr.update(value=state.get("n_startup", 60)),
                gr.update(value=state.get("kl_scale", 1.0)),
                gr.update(value=state.get("kl_target", 0.01)),
                gr.update(value=log_content),
                gr.update(interactive=not is_running),
                gr.update(active=is_running),
            )

        app.load(
            fn=_on_page_load,
            inputs=[settings_state],
            outputs=[
                ui.configure.model_source_radio,
                ui.configure.model_id_in,
                ui.configure.local_model_section,
                ui.configure.local_model_in,
                ui.configure.quantization_in,
                ui.configure.n_trials_in,
                ui.configure.system_prompt_in,
                ui.configure.n_startup_in,
                ui.configure.kl_scale_in,
                ui.configure.kl_target_in,
                ui.configure.log_out,
                ui.configure.start_btn,
                opt_timer,
            ],
        )

        # ── Timer: keep log and button state updated while opt. is running ─
        poll_state = SimpleNamespace(last_polled_log_version=-1)
        last_polled_log_version_lock = threading.Lock()

        def _poll_optimization() -> tuple:
            is_running = _optimization_running.is_set()
            log_version, log_content = _get_log_snapshot()
            with last_polled_log_version_lock:
                log_update = (
                    gr.update(value=log_content)
                    if log_version != poll_state.last_polled_log_version
                    else gr.update()
                )
                poll_state.last_polled_log_version = log_version
            return (
                log_update,
                gr.update(interactive=not is_running),
                gr.update(active=is_running),
            )

        opt_timer.tick(
            fn=_poll_optimization,
            outputs=[ui.configure.log_out, ui.configure.start_btn, opt_timer],
        )

    return app


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for ``heretic-webui``."""
    from .webui_cli import main as cli_main

    cli_main()
