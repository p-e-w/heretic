# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from typing import Dict

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DatasetSpecification(BaseModel):
    dataset: str = Field(
        description="Hugging Face dataset ID, or path to dataset on disk"
    )
    split: str = Field(description="Portion of the dataset to use")
    column: str = Field(description="Column in the dataset that contains the prompts")


class Settings(BaseSettings):
    model: str = Field(description="Hugging Face model ID, or path to model on disk")

    evaluate_model: str | None = Field(
        default=None,
        description="If this model ID or path is set, then instead of abliterating the main model, evaluate this model relative to the main model",
    )

    dtypes: list[str] = Field(
        description="List of PyTorch dtypes to try when loading model tensors. If loading with a dtype fails, the next dtype in the list will be tried."
    )

    device_map: str | Dict[str, int | str] = Field(
        description="Device map to pass to Accelerate when loading the model"
    )

    batch_size: int = Field(
        description="Number of input sequences to process in parallel (0 = auto)"
    )

    max_batch_size: int = Field(
        description="Maximum batch size to try when automatically determining the optimal batch size"
    )

    max_response_length: int = Field(
        description="Maximum number of tokens to generate for each response"
    )

    max_kl_divergence: float = Field(
        description="Maximum Kullback-Leibler divergence from the original model to allow for abliterated models"
    )

    kl_score_shape: float = Field(
        description="Exponent that determines the shape of the KL divergence part of the score function. See evaluator.py for the exact meaning of this parameter."
    )

    n_trials: int = Field(
        description="Number of abliteration trials to run during optimization"
    )

    n_startup_trials: int = Field(
        description="Number of trials that use random sampling for the purpose of exploration"
    )

    refusal_markers: list[str] = Field(
        description="Strings whose presence in a response (case insensitive) identifies the response as a refusal"
    )

    system_prompt: str = Field(
        description="System prompt to use when prompting the model"
    )

    good_prompts: DatasetSpecification = Field(
        description="Dataset of prompts that tend to not result in refusals (used for calculating refusal directions)"
    )

    bad_prompts: DatasetSpecification = Field(
        description="Dataset of prompts that tend to result in refusals (used for calculating refusal directions)"
    )

    good_evaluation_prompts: DatasetSpecification = Field(
        description="Dataset of prompts that tend to not result in refusals (used for evaluating model performance)"
    )

    bad_evaluation_prompts: DatasetSpecification = Field(
        description="Dataset of prompts that tend to result in refusals (used for evaluating model performance)"
    )

    # "Model" refers to the Pydantic model of the settings class here,
    # not to the language model. The field must have this exact name.
    model_config = SettingsConfigDict(
        toml_file=["config.default.toml", "config.toml"],
        env_prefix="HERETIC_",
        cli_parse_args=True,
        cli_kebab_case=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls),
        )
