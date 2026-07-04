# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers for converting API request models into Heretic ``Settings``."""

from ..config import (
    DatasetSpecification,
    QuantizationMethod,
    RowNormalization,
    Settings,
    build_settings,
)
from .models import AblateRequest, DatasetSpec


def _to_dataset_specification(spec: DatasetSpec) -> DatasetSpecification:
    return DatasetSpecification(
        dataset=spec.dataset,
        split=spec.split,
        column=spec.column,
        prefix=spec.prefix,
        suffix=spec.suffix,
        system_prompt=spec.system_prompt,
    )


def ablate_request_to_settings(request: AblateRequest) -> Settings:
    """Builds a ``Settings`` object from an :class:`AblateRequest`.

    Uses :func:`build_settings`, which ignores ``config.toml``/env sources so
    the request body is the single source of truth.
    """

    return build_settings(
        model=request.model,
        model_commit=request.model_commit,
        quantization=QuantizationMethod(request.quantization),
        device_map=request.device_map,
        max_memory=request.max_memory,
        offload_outputs_to_cpu=request.offload_outputs_to_cpu,
        batch_size=request.batch_size,
        max_batch_size=request.max_batch_size,
        n_trials=request.n_trials,
        n_startup_trials=request.n_startup_trials,
        seed=request.seed,
        study_checkpoint_dir=request.study_checkpoint_dir,
        kl_divergence_scale=request.kl_divergence_scale,
        kl_divergence_target=request.kl_divergence_target,
        orthogonalize_direction=request.orthogonalize_direction,
        row_normalization=RowNormalization(request.row_normalization),
        winsorization_quantile=request.winsorization_quantile,
        max_response_length=request.max_response_length,
        system_prompt=request.system_prompt,
        good_prompts=_to_dataset_specification(request.good_prompts),
        bad_prompts=_to_dataset_specification(request.bad_prompts),
        good_evaluation_prompts=_to_dataset_specification(
            request.good_evaluation_prompts
        ),
        bad_evaluation_prompts=_to_dataset_specification(
            request.bad_evaluation_prompts
        ),
    )


__all__ = [
    "ablate_request_to_settings",
]
