# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from pathlib import Path

import huggingface_hub
from huggingface_hub import ModelCard, ModelCardData
from optuna import Trial

from .config import RowNormalization, Settings
from .evaluator import Evaluator
from .system import (
    get_heretic_version_info,
)
from .utils import Prompt, get_trial_parameters


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[Prompt],
    is_lora: bool,
) -> str:
    if Path(settings.model).exists():
        # Hide the path, which may contain private information.
        model_link = "a model"
    else:
        model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"

    version_info = get_heretic_version_info()
    return f"""# This is a decensored {"adapter" if is_lora else "version"} of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version_info.version}

## Abliteration parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in get_trial_parameters(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.4f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""


def get_model_card(
    settings: Settings,
    trial: Trial,
    evaluator: Evaluator,
    is_lora: bool,
) -> ModelCard | None:
    # If the model path exists locally and includes the
    # card, use it directly. If the model path doesn't
    # exist locally, it can be assumed to be a model
    # hosted on the Hugging Face Hub, in which case
    # we can retrieve the model card.
    model_path = Path(settings.model)
    if model_path.exists():
        card_path = model_path / huggingface_hub.constants.REPOCARD_NAME
        if card_path.exists():
            card = ModelCard.load(card_path)
        else:
            card = None
    else:
        card = ModelCard.load(settings.model)
    if card is not None:
        if card.data is None:
            card.data = ModelCardData()
        if card.data.tags is None:
            card.data.tags = []
        card.data.tags.append("heretic")
        card.data.tags.append("uncensored")
        card.data.tags.append("decensored")
        card.data.tags.append("abliterated")
        if (
            settings.orthogonalize_direction
            and settings.row_normalization == RowNormalization.FULL
        ):
            card.data.tags.append("mpoa")
        if not model_path.exists():
            card.data.base_model = settings.model
        card.data.base_model_relation = "adapter" if is_lora else "finetuned"

        card.text = (
            get_readme_intro(
                settings,
                trial,
                evaluator.base_refusals,
                evaluator.bad_prompts,
                is_lora,
            )
            + card.text
        )

    return card
