# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""UI-agnostic core of the abliteration pipeline.

This module holds the pieces of the abliteration workflow that are shared by
both the interactive CLI (``heretic.main``) and the REST API
(``heretic.api.tasks``). Keeping them in one place ensures the two front ends
stay behaviorally identical: the parameter search space and Pareto selection in
particular must not drift between them, or the same request would produce
different models depending on the entry point.

Everything here is deliberately free of any presentation concerns (no
``rich``/``questionary``/progress-dict access). Where a step benefits from
progress reporting, an optional callback is accepted so each front end can plug
in its own reporting without the core depending on it.
"""

import logging
import math
import os
import random
import time
import warnings
from os.path import commonprefix

import optuna
import torch
import torch.nn.functional as F
import transformers
from optuna import Study, Trial
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

from .config import Settings
from .model import AbliterationParameters, Model

STUDY_NAME = "heretic"


def configure_runtime(settings: Settings) -> None:
    """Applies the process-wide runtime configuration used by every run.

    Seeds RNGs, disables gradients (inference only), raises the TorchDynamo
    recompilation limit for batch-size sweeps, and silences library warning
    spam. Mutates ``settings.seed`` in place if it was unset.
    """

    if settings.seed is None:
        settings.seed = random.randint(0, 2**32 - 1)

    transformers.set_seed(settings.seed)

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch
    # sizes, resulting in many computation graphs being compiled. Raising the
    # limit (default = 8) avoids errors from TorchDynamo assuming that something
    # is wrong because we recompile too often.
    torch._dynamo.config.cache_size_limit = 64

    # Silence warning spam from Transformers.
    transformers.logging.set_verbosity_error()

    # Another library that generates warning spam.
    logging.getLogger("lm_eval").setLevel(logging.ERROR)

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    # Enable expandable segments to reduce memory fragmentation on multi-GPU setups.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def checkpoint_path(settings: Settings) -> str:
    """Returns the Optuna journal checkpoint path for ``settings.model``."""

    safe_name = "".join(
        c if (c.isalnum() or c in ("_", "-")) else "--" for c in settings.model
    )
    return os.path.join(settings.study_checkpoint_dir, f"{safe_name}.jsonl")


def open_study_storage(settings: Settings) -> JournalStorage:
    """Opens (creating the directory if needed) the Optuna journal storage."""

    os.makedirs(settings.study_checkpoint_dir, exist_ok=True)
    file = checkpoint_path(settings)
    lock = JournalFileOpenLock(file)
    return JournalStorage(JournalFileBackend(file, lock_obj=lock))


def determine_batch_size(
    model: Model,
    settings: Settings,
    good_prompts: list,
    *,
    on_start=None,
    on_result=None,
    on_error=None,
) -> int:
    """Benchmarks batch sizes and returns the fastest one.

    Tries powers of two up to ``settings.max_batch_size``, measuring throughput
    (tokens/second) with a warmup pass to exclude graph compilation. The chosen
    value is also written back to ``settings.batch_size``.

    The optional callbacks let front ends report progress:
    ``on_start(batch_size)``, ``on_result(batch_size, performance)``, and
    ``on_error(batch_size, error)``. If a batch size of 1 already fails, the
    error is re-raised (unrecoverable).
    """

    batch_size = 1
    best_batch_size = -1
    best_performance = -1.0

    while batch_size <= settings.max_batch_size:
        if on_start is not None:
            on_start(batch_size)

        prompts = (good_prompts * math.ceil(batch_size / len(good_prompts)))[
            :batch_size
        ]

        try:
            # Warmup run to build the computation graph so that part isn't benchmarked.
            model.get_responses(prompts)

            start_time = time.perf_counter()
            responses = model.get_responses(prompts)
            end_time = time.perf_counter()
        except Exception as error:
            if batch_size == 1:
                # Even a batch size of 1 already fails. We cannot recover.
                raise
            if on_error is not None:
                on_error(batch_size, error)
            break

        response_lengths = [
            len(model.tokenizer.encode(response)) for response in responses
        ]
        performance = sum(response_lengths) / (end_time - start_time)

        if on_result is not None:
            on_result(batch_size, performance)

        if performance > best_performance:
            best_batch_size = batch_size
            best_performance = performance

        batch_size *= 2

    settings.batch_size = best_batch_size
    return best_batch_size


def detect_response_prefix(
    model: Model,
    settings: Settings,
    good_prompts: list,
    bad_prompts: list,
    *,
    on_prefix=None,
    on_cot=None,
    on_extended=None,
    on_none=None,
) -> str:
    """Detects a common response prefix and writes it to ``settings.response_prefix``.

    Handles Chain-of-Thought skips: if the detected prefix begins with a
    configured CoT initializer, it is replaced by the corresponding closed CoT
    block and the prompts are re-checked for any additional shared prefix.

    Optional callbacks report findings: ``on_prefix(prefix)``,
    ``on_cot(prefix)``, ``on_extended(prefix)``, ``on_none()``.
    """

    prefix_check_prompts = good_prompts[:100] + bad_prompts[:100]
    responses = model.get_responses_batched(prefix_check_prompts)

    # Despite being located in os.path, commonprefix actually performs a naive
    # string operation without any path-specific logic, which is exactly what we
    # need here. Trailing spaces are removed to avoid issues where multiple
    # different tokens that all start with a space character lead to the common
    # prefix ending with a space, which would result in an uncommon tokenization.
    settings.response_prefix = commonprefix(responses).rstrip(" ")

    if not settings.response_prefix:
        if on_none is not None:
            on_none()
        return settings.response_prefix

    if on_prefix is not None:
        on_prefix(settings.response_prefix)

    for cot_initializer, closed_cot_block in settings.chain_of_thought_skips:
        if settings.response_prefix.startswith(cot_initializer):
            settings.response_prefix = closed_cot_block
            if on_cot is not None:
                on_cot(settings.response_prefix)

            # When using a Chain-of-Thought skip, we need to check that the
            # prefix is actually complete (e.g. not missing a trailing newline).
            responses = model.get_responses_batched(prefix_check_prompts)
            additional_prefix = commonprefix(responses).rstrip(" ")
            if additional_prefix:
                settings.response_prefix += additional_prefix
                if on_extended is not None:
                    on_extended(settings.response_prefix)

            break

    return settings.response_prefix


def compute_refusal_directions(
    model: Model,
    settings: Settings,
    good_means: torch.Tensor,
    bad_means: torch.Tensor,
) -> torch.Tensor:
    """Computes per-layer refusal directions from residual means.

    ``good_means``/``bad_means`` are the per-layer residual means for the good
    and bad prompt sets. When ``settings.orthogonalize_direction`` is set, the
    refusal direction is projected to remove its component along the good
    direction (see https://huggingface.co/blog/grimjim/projected-abliteration).
    """

    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if settings.orthogonalize_direction:
        # Adjust the refusal directions so that only the component that is
        # orthogonal to the good direction is subtracted during abliteration.
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = (
            refusal_directions - projection_vector.unsqueeze(1) * good_directions
        )
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)
        del good_directions, projection_vector

    return refusal_directions


def suggest_trial_parameters(
    trial: Trial, model: Model
) -> tuple[float | None, dict[str, AbliterationParameters]]:
    """Samples the abliteration parameters for a single Optuna trial.

    This defines the search space explored during optimization and MUST be the
    single source of truth for it: the CLI and the API both call this so that
    identical trials produce identical parameters.

    The parameter ranges are based on experiments with various models and much
    wider ranges. They are not set in stone and might have to be adjusted for
    future models.

    Returns ``(direction_index, parameters)`` where ``direction_index`` is
    ``None`` for the "per layer" direction scope.
    """

    direction_scope = trial.suggest_categorical(
        "direction_scope",
        [
            "global",
            "per layer",
        ],
    )

    last_layer_index = len(model.get_layers()) - 1

    # Discrimination between "harmful" and "harmless" inputs is usually strongest
    # in layers slightly past the midpoint of the layer stack. See the original
    # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
    #
    # Note that we always sample this parameter even though we only need it for
    # the "global" direction scope. The reason is that multivariate TPE doesn't
    # work with conditional or variable-range parameters.
    direction_index: float | None = trial.suggest_float(
        "direction_index",
        0.4 * last_layer_index,
        0.9 * last_layer_index,
    )

    if direction_scope == "per layer":
        direction_index = None

    parameters: dict[str, AbliterationParameters] = {}

    for component in model.get_abliterable_components():
        # The MLP gets a negative lower bound that is then clamped to 0, so the
        # optimizer can fully disable its ablation. The clamp puts a positive
        # probability mass on exactly 0 (the continuous sampler would otherwise
        # reach 0 with probability zero). Ablating the MLP is often unnecessary
        # for removing refusals and tends to damage model intelligence more than
        # ablating the attention output, so on many models the optimum is to
        # leave it (mostly) untouched. See issue #202.
        max_weight_lower_bound = -0.25 if component == "mlp.down_proj" else 0.8
        max_weight = max(
            0.0,
            trial.suggest_float(
                f"{component}.max_weight",
                max_weight_lower_bound,
                1.5,
            ),
        )
        max_weight_position = trial.suggest_float(
            f"{component}.max_weight_position",
            0.6 * last_layer_index,
            1.0 * last_layer_index,
        )
        # For sampling purposes, min_weight is expressed as a fraction of
        # max_weight, again because multivariate TPE doesn't support
        # variable-range parameters. The value is transformed into the actual
        # min_weight value below.
        min_weight = trial.suggest_float(
            f"{component}.min_weight",
            0.0,
            1.0,
        )
        min_weight_distance = trial.suggest_float(
            f"{component}.min_weight_distance",
            1.0,
            max(0.6 * last_layer_index, 1.0),
        )

        parameters[component] = AbliterationParameters(
            max_weight=max_weight,
            max_weight_position=max_weight_position,
            min_weight=(min_weight * max_weight),
            min_weight_distance=min_weight_distance,
        )

    return direction_index, parameters


def create_study(settings: Settings, storage: JournalStorage) -> Study:
    """Creates (or loads) the multi-objective Optuna study used for abliteration."""

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=settings.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
            seed=settings.seed,
        ),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
        storage=storage,
        study_name=STUDY_NAME,
        load_if_exists=True,
    )
    study.set_user_attr("settings", settings.model_dump_json())
    study.set_user_attr("finished", False)
    return study


def pareto_front(
    study: Study, *, require_attrs: tuple[str, ...] = ()
) -> list[FrozenTrial]:
    """Returns the Pareto-optimal completed trials.

    Among completed trials (sorted by refusals then KL divergence), keeps those
    that strictly improve KL divergence as the refusal count increases. This is
    the shared selection logic for both front ends.

    We can't use ``study.best_trials`` directly as ``get_score()`` doesn't
    return the pure KL divergence and refusal count. Note: unlike
    ``study.best_trials``, this does not handle objective constraints.

    ``require_attrs`` optionally restricts the result to trials that carry all
    the given ``user_attrs`` keys. A resumed study (``load_if_exists=True``) may
    contain trials written by an older or different schema; callers that read
    extra attributes can use this to skip them rather than raising.
    """

    completed = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and all(key in trial.user_attrs for key in require_attrs)
    ]

    sorted_trials = sorted(
        completed,
        key=lambda trial: (
            trial.user_attrs["refusals"],
            trial.user_attrs["kl_divergence"],
        ),
    )

    best_trials: list[FrozenTrial] = []
    min_divergence = math.inf
    for trial in sorted_trials:
        kl_divergence = trial.user_attrs["kl_divergence"]
        if kl_divergence < min_divergence:
            min_divergence = kl_divergence
            best_trials.append(trial)

    return best_trials
