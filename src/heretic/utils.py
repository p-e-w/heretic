# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import gc
from importlib.metadata import version
from typing import TypeVar

import optuna
import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import load_dataset
from rich.console import Console

from .config import DatasetSpecification, Settings

print = Console(highlight=False).print


def load_prompts(specification: DatasetSpecification) -> list[str]:
    dataset = load_dataset(specification.dataset, split=specification.split)
    return list(dataset[specification.column])


T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()
    elif is_sdaa_available():
        torch.sdaa.empty_cache()
    elif is_musa_available():
        torch.musa.empty_cache()

    gc.collect()


def get_readme_intro(
    settings: Settings,
    study: optuna.Study,
    base_refusals: int,
    bad_prompts: list[str],
) -> str:
    refusal_percentage = (
        study.best_trial.user_attrs["refusals"] / len(bad_prompts) * 100
    )
    base_refusal_percentage = base_refusals / len(bad_prompts) * 100

    return f"""# This is a decensored version of [{settings.model}](https://huggingface.co/{settings.model}), made using [Heretic](https://github.com/p-e-w/heretic) v{version("heretic")}

## Abliteration parameters

| Parameter               | Value                                          |
| :---------------------- | :--------------------------------------------: |
| **max_weight**          | {study.best_params["max_weight"]:.4f}          |
| **max_weight_position** | {study.best_params["max_weight_position"]:.4f} |
| **min_weight**          | {study.best_params["min_weight"]:.4f}          |
| **min_weight_distance** | {study.best_params["min_weight_distance"]:.4f} |

## Performance

| Metric            | This model                                         | Original model ([{settings.model}](https://huggingface.co/{settings.model})) |
| :---------------- | :------------------------------------------------: | :--------------------------------------------------------------------------: |
| **KL divergence** | {study.best_trial.user_attrs["kl_divergence"]:.4f} | 0 *(by definition)*                                                          |
| **Refusals**      | {refusal_percentage:.1f} %                         | {base_refusal_percentage:.1f} %                                              |

-----

"""
