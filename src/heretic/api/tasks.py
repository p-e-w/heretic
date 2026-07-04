# SPDX-License-Identifier: AGPL-3.0-or-later

"""Background task management for API-driven abliteration runs.

The heavy, blocking abliteration work runs in a thread pool executor so that
the FastAPI event loop stays responsive. Only one abliteration task may be
active at a time because a single GPU-resident model is shared across the
process.
"""

import asyncio
import os
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import torch
from optuna import Trial, TrialPruned
from optuna.storages import JournalStorage
from optuna.study import Study
from optuna.trial import FrozenTrial

from .. import core
from ..config import Settings
from ..evaluator import Evaluator
from ..model import AbliterationParameters, Model
from ..system import empty_cache
from ..utils import (
    format_duration,
    format_exception,
    load_prompts,
)
from .models import TaskStatus, TrialInfo

# The user_attrs the API reads back from each trial. A resumed study may contain
# trials from an older schema, so Pareto selection filters on these.
_REQUIRED_TRIAL_ATTRS = (
    "index",
    "refusals",
    "kl_divergence",
    "direction_index",
    "parameters",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pareto_front(study: Study) -> list[FrozenTrial]:
    return core.pareto_front(study, require_attrs=_REQUIRED_TRIAL_ATTRS)


def _trial_info(trial: FrozenTrial) -> TrialInfo:
    return TrialInfo(
        index=trial.user_attrs["index"],
        refusals=trial.user_attrs["refusals"],
        kl_divergence=trial.user_attrs["kl_divergence"],
        direction_index=trial.user_attrs["direction_index"],
        parameters=trial.user_attrs["parameters"],
    )


class ExportJob:
    """Tracks a single asynchronous export (local save or Hugging Face upload)."""

    def __init__(
        self,
        task_id: str,
        destination: str,
        strategy: str,
    ):
        self.export_id = str(uuid.uuid4())
        self.task_id = task_id
        self.destination = destination
        self.strategy = strategy
        self.status = TaskStatus.PENDING
        self.location: str | None = None
        self.created_at = _now()
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.error: str | None = None

    @property
    def status_response(self) -> dict[str, object]:
        return {
            "export_id": self.export_id,
            "task_id": self.task_id,
            "status": self.status,
            "destination": self.destination,
            "strategy": self.strategy,
            "location": self.location,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


class AblationTask:
    """Tracks the lifecycle and results of a single abliteration run."""

    def __init__(self, settings: Settings):
        self.task_id = str(uuid.uuid4())
        self.settings = settings
        self.status = TaskStatus.PENDING
        self.created_at = _now()
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.error: str | None = None

        # ``progress`` is written from the executor thread and read from the
        # event loop thread, so access is guarded by a lock.
        self._progress: dict[str, object] = {}
        self._progress_lock = threading.Lock()

        # The most recent export job for this task, if any. Only one export
        # may run at a time per task (they share the GPU-resident model).
        self._export_job: ExportJob | None = None

        self._model: Model | None = None
        self._evaluator: Evaluator | None = None
        self._study: Study | None = None
        self._refusal_directions: torch.Tensor | None = None
        self._pareto_trials: list[TrialInfo] = []
        self._selected_trial: TrialInfo | None = None
        self._base_refusals: int | None = None
        self._n_bad_prompts: int | None = None
        self._cancel_event = threading.Event()

    # --- Public state -----------------------------------------------------

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def progress(self) -> dict[str, object]:
        """Returns a thread-safe snapshot of the current progress."""

        with self._progress_lock:
            return dict(self._progress)

    def _set_progress(self, **updates: object) -> None:
        with self._progress_lock:
            self._progress.update(updates)

    @property
    def has_model(self) -> bool:
        return self._model is not None

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def result(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "pareto_trials": [t.model_dump() for t in self._pareto_trials],
            "selected_trial": self._selected_trial.model_dump()
            if self._selected_trial
            else None,
            "base_refusals": self._base_refusals,
            "n_bad_prompts": self._n_bad_prompts,
            "error": self.error,
        }

    @property
    def status_response(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,  # snapshot via property
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    # --- Execution --------------------------------------------------------

    async def run(self) -> None:
        self.status = TaskStatus.RUNNING
        self.started_at = _now()

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._run_sync)
            if self.status == TaskStatus.RUNNING:
                self.status = TaskStatus.COMPLETED
        except Exception as error:
            self.status = TaskStatus.FAILED
            self.error = format_exception(error)
        finally:
            self.completed_at = _now()
            self._set_progress(phase=self.status.value)

    def _run_sync(self) -> None:
        settings = self.settings
        core.configure_runtime(settings)

        storage = core.open_study_storage(settings)

        self._set_progress(phase="loading_model")
        model = Model(settings)
        self._model = model

        self._set_progress(phase="loading_prompts")
        good_prompts = load_prompts(settings, settings.good_prompts)
        bad_prompts = load_prompts(settings, settings.bad_prompts)

        if settings.batch_size == 0:
            self._determine_batch_size(model, settings, good_prompts)

        if settings.response_prefix is None:
            self._detect_response_prefix(model, settings, good_prompts, bad_prompts)

        self._set_progress(phase="initializing_evaluator")
        evaluator = Evaluator(settings, model)
        self._evaluator = evaluator
        self._base_refusals = evaluator.base_refusals
        self._n_bad_prompts = len(evaluator.bad_prompts)

        self._set_progress(phase="computing_refusal_directions")
        self._refusal_directions = self._compute_refusal_directions(
            model, settings, good_prompts, bad_prompts
        )

        self._set_progress(phase="abliteration")
        study = self._run_study(model, evaluator, settings, storage)
        self._study = study

        self._pareto_trials = [_trial_info(t) for t in _pareto_front(study)]

    # --- Pipeline steps ---------------------------------------------------

    def _determine_batch_size(
        self, model: Model, settings: Settings, good_prompts: list
    ) -> None:
        self._set_progress(phase="determining_batch_size")
        best_batch_size = core.determine_batch_size(model, settings, good_prompts)
        self._set_progress(batch_size=best_batch_size)

    def _detect_response_prefix(
        self,
        model: Model,
        settings: Settings,
        good_prompts: list,
        bad_prompts: list,
    ) -> None:
        self._set_progress(phase="detecting_prefix")
        core.detect_response_prefix(model, settings, good_prompts, bad_prompts)

    @staticmethod
    def _compute_refusal_directions(
        model: Model,
        settings: Settings,
        good_prompts: list,
        bad_prompts: list,
    ) -> torch.Tensor:
        good_means = model.get_residuals_mean(good_prompts)
        bad_means = model.get_residuals_mean(bad_prompts)
        refusal_directions = core.compute_refusal_directions(
            model, settings, good_means, bad_means
        )
        del good_means, bad_means
        empty_cache()
        return refusal_directions

    def _run_study(
        self,
        model: Model,
        evaluator: Evaluator,
        settings: Settings,
        storage: JournalStorage,
    ) -> Study:
        study = core.create_study(settings, storage)

        refusal_directions = self._refusal_directions
        assert refusal_directions is not None

        start_index = len(study.trials)
        trial_index = start_index
        start_time = time.perf_counter()

        def objective(trial: Trial) -> tuple[float, float]:
            nonlocal trial_index
            trial_index += 1
            trial.set_user_attr("index", trial_index)

            direction_index, parameters = core.suggest_trial_parameters(trial, model)
            trial.set_user_attr("direction_index", direction_index)
            trial.set_user_attr(
                "parameters", {k: asdict(v) for k, v in parameters.items()}
            )

            model.reset_model()
            model.abliterate(refusal_directions, direction_index, parameters)
            score, kl_divergence, refusals = evaluator.get_score()

            trial.set_user_attr("kl_divergence", kl_divergence)
            trial.set_user_attr("refusals", refusals)
            trial.set_user_attr("base_refusals", evaluator.base_refusals)
            trial.set_user_attr("n_bad_prompts", len(evaluator.bad_prompts))

            self._update_progress(
                trial_index, start_index, start_time, settings, kl_divergence, refusals
            )
            return score

        def objective_wrapper(trial: Trial) -> tuple[float, float]:
            if self.cancelled:
                trial.study.stop()
                raise TrialPruned()
            try:
                return objective(trial)
            except KeyboardInterrupt:
                trial.study.stop()
                raise TrialPruned()

        remaining = settings.n_trials - len(study.trials)
        if remaining > 0:
            study.optimize(objective_wrapper, n_trials=remaining)

        if len(study.trials) >= settings.n_trials:
            study.set_user_attr("finished", True)

        if self.cancelled:
            self.status = TaskStatus.CANCELLED

        return study

    def _update_progress(
        self,
        trial_index: int,
        start_index: int,
        start_time: float,
        settings: Settings,
        kl_divergence: float,
        refusals: int,
    ) -> None:
        elapsed = time.perf_counter() - start_time
        completed = max(trial_index - start_index, 1)
        remaining = (elapsed / completed) * (settings.n_trials - trial_index)

        self._set_progress(
            current_trial=trial_index,
            total_trials=settings.n_trials,
            kl_divergence=kl_divergence,
            refusals=refusals,
            elapsed=format_duration(elapsed),
            remaining=format_duration(remaining)
            if trial_index < settings.n_trials
            else "0s",
        )

    # --- Post-run actions -------------------------------------------------

    def pareto_indices(self) -> list[int]:
        """Returns the ``index`` values of the Pareto-optimal trials."""

        return [t.index for t in self._pareto_trials]

    def select_trial(self, trial_index: int) -> TrialInfo | None:
        """Applies the Pareto-optimal trial whose ``index`` matches ``trial_index``.

        ``trial_index`` refers to the ``index`` field reported for each entry
        in the ``pareto_trials`` list (i.e. the trial number), not its position
        within that list.
        """

        if (
            self._study is None
            or self._refusal_directions is None
            or self._model is None
        ):
            return None

        best_trials = _pareto_front(self._study)
        trial = next(
            (t for t in best_trials if t.user_attrs["index"] == trial_index), None
        )
        if trial is None:
            return None

        self._model.reset_model()
        self._model.abliterate(
            self._refusal_directions,
            trial.user_attrs["direction_index"],
            {
                k: AbliterationParameters(**v)
                for k, v in trial.user_attrs["parameters"].items()
            },
        )

        self._selected_trial = _trial_info(trial)
        return self._selected_trial

    def chat(
        self, message: str, system_prompt: str | None = None, max_tokens: int = 4096
    ) -> str:
        if self._model is None:
            raise RuntimeError("No model loaded. Run abliteration first.")

        conversation = [
            {"role": "system", "content": system_prompt or self.settings.system_prompt},
            {"role": "user", "content": message},
        ]
        return self._model.stream_chat_response(conversation, max_new_tokens=max_tokens)

    # --- Export -----------------------------------------------------------

    @property
    def has_selected_trial(self) -> bool:
        return self._selected_trial is not None

    def _reapply_selected_trial(self) -> None:
        """Restores the selected trial's LoRA after a merge destroys it.

        ``get_merged_model()`` unloads/merges the LoRA in place for
        non-quantized models, so the abliteration must be re-applied before the
        model can be used again (e.g. for chat or a subsequent export).
        """

        if self._selected_trial is None or self._refusal_directions is None:
            return
        assert self._model is not None

        self._model.reset_model()
        self._model.abliterate(
            self._refusal_directions,
            self._selected_trial.direction_index,
            {
                k: AbliterationParameters(**v)
                for k, v in self._selected_trial.parameters.items()
            },
        )

    @property
    def export_job(self) -> ExportJob | None:
        return self._export_job

    @property
    def export_in_progress(self) -> bool:
        return self._export_job is not None and self._export_job.status in (
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
        )

    def start_export(self, destination: str, strategy: str) -> ExportJob:
        """Creates and records a new export job for this task."""

        job = ExportJob(self.task_id, destination, strategy)
        self._export_job = job
        return job

    async def run_export(
        self,
        job: ExportJob,
        *,
        save_directory: str | None = None,
        repo_id: str | None = None,
        token: str | None = None,
        private: bool = True,
        max_shard_size: str = "5GB",
    ) -> None:
        """Runs an export job in the executor and records its outcome."""

        job.status = TaskStatus.RUNNING
        job.started_at = _now()

        loop = asyncio.get_running_loop()
        try:
            if job.destination == "local":
                assert save_directory is not None
                location = await loop.run_in_executor(
                    None,
                    self._do_export_local,
                    save_directory,
                    job.strategy,
                    max_shard_size,
                )
            else:
                assert repo_id is not None and token is not None
                location = await loop.run_in_executor(
                    None,
                    self._do_export_huggingface,
                    repo_id,
                    token,
                    private,
                    job.strategy,
                    max_shard_size,
                )
            job.location = location
            job.status = TaskStatus.COMPLETED
        except Exception as error:
            job.status = TaskStatus.FAILED
            job.error = format_exception(error)
        finally:
            job.completed_at = _now()

    def _do_export_local(
        self, save_directory: str, strategy: str, max_shard_size: str
    ) -> str:
        model = self._require_export_ready()
        os.makedirs(save_directory, exist_ok=True)

        if strategy == "adapter":
            model.model.save_pretrained(save_directory, max_shard_size=max_shard_size)
        else:
            merged_model = model.get_merged_model()
            merged_model.save_pretrained(save_directory, max_shard_size=max_shard_size)
            del merged_model
            empty_cache()
            model.tokenizer.save_pretrained(save_directory)
            if model.processor is not None:
                model.processor.save_pretrained(save_directory)
            self._reapply_selected_trial()

        return save_directory

    def _do_export_huggingface(
        self,
        repo_id: str,
        token: str,
        private: bool,
        strategy: str,
        max_shard_size: str,
    ) -> str:
        model = self._require_export_ready()

        if strategy == "adapter":
            model.model.push_to_hub(
                repo_id,
                private=private,
                max_shard_size=max_shard_size,
                token=token,
            )
        else:
            merged_model = model.get_merged_model()
            merged_model.push_to_hub(
                repo_id,
                private=private,
                max_shard_size=max_shard_size,
                token=token,
            )
            del merged_model
            empty_cache()
            model.tokenizer.push_to_hub(repo_id, private=private, token=token)
            if model.processor is not None:
                model.processor.push_to_hub(repo_id, private=private, token=token)
            self._reapply_selected_trial()

        return repo_id

    def _require_export_ready(self) -> Model:
        if self._model is None:
            raise RuntimeError("No model loaded. Run abliteration first.")
        if self._selected_trial is None:
            raise RuntimeError("No trial selected. Call select-trial before exporting.")
        return self._model

    def release(self) -> None:
        """Releases the GPU-resident model and associated tensors."""

        self._model = None
        self._evaluator = None
        self._refusal_directions = None
        empty_cache()


class TaskManager:
    """Thread-safe registry of abliteration tasks.

    A single shared GPU model means only one active task is permitted at a
    time, and only one completed task retains the loaded model for chatting.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, AblationTask] = {}
        self._lock = threading.Lock()

    def create_task(self, settings: Settings) -> AblationTask:
        task = AblationTask(settings)
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> AblationTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def remove_task(self, task_id: str) -> AblationTask | None:
        with self._lock:
            task = self._tasks.pop(task_id, None)
        if task is not None:
            task.release()
        return task

    @property
    def active_task(self) -> AblationTask | None:
        with self._lock:
            for task in self._tasks.values():
                if task.status in (TaskStatus.RUNNING, TaskStatus.PENDING):
                    return task
        return None

    def get_task_with_model(self) -> AblationTask | None:
        with self._lock:
            for task in self._tasks.values():
                if task.status == TaskStatus.COMPLETED and task.has_model:
                    return task
        return None

    def get_export_job(self, export_id: str) -> tuple[AblationTask, ExportJob] | None:
        with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            job = task.export_job
            if job is not None and job.export_id == export_id:
                return task, job
        return None

    @property
    def export_in_progress(self) -> bool:
        with self._lock:
            tasks = list(self._tasks.values())
        return any(task.export_in_progress for task in tasks)

    @property
    def has_active_model(self) -> bool:
        return self.get_task_with_model() is not None


task_manager = TaskManager()
