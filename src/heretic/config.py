# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from abc import abstractmethod
from enum import Enum
from typing import Dict, Type, TypeVar

from optuna import Trial
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)
from typing_extensions import TypeAliasType


class QuantizationMethod(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"


class RowNormalization(str, Enum):
    NONE = "none"
    PRE = "pre"
    # POST = "post"  # Theoretically possible, but provides no advantage.
    FULL = "full"


class DatasetSpecification(BaseModel):
    dataset: str = Field(
        description="Hugging Face dataset ID, or path to dataset on disk."
    )

    split: str = Field(description="Portion of the dataset to use.")

    column: str = Field(description="Column in the dataset that contains the prompts.")

    prefix: str = Field(
        default="",
        description="Text to prepend to each prompt.",
    )

    suffix: str = Field(
        default="",
        description="Text to append to each prompt.",
    )

    system_prompt: str | None = Field(
        default=None,
        description="System prompt to use with the prompts (overrides global system prompt if set).",
    )

    residual_plot_label: str | None = Field(
        default=None,
        description="Label to use for the dataset in plots of residual vectors.",
    )

    residual_plot_color: str | None = Field(
        default=None,
        description="Matplotlib color to use for the dataset in plots of residual vectors.",
    )


class FloatParamSpec(BaseModel):
    low: float = Field(
        description="Lower endpoint of the range of suggested values (inclusive).",
    )

    high: float = Field(
        description="Upper endpoint of the range of suggested values (inclusive).",
    )

    log: bool = Field(
        default=False,
        description="If true, the value is sampled from the range in the log domain.",
    )


ParamSpec = TypeVar("ParamSpec", list, FloatParamSpec)
ParamSpecRecursive = TypeAliasType(
    "ParamSpecRecursive", "list | FloatParamSpec | dict[str, ParamSpecRecursive]"
)


class Parameter:
    name: str

    @abstractmethod
    def is_constant(self) -> bool:
        pass

    @abstractmethod
    def suggest(self, trial: Trial):
        pass


class ParamCategorical(Parameter):
    choices: list

    def __init__(self, name: str, spec: list):
        self.name = name
        self.choices = spec

    def is_constant(self) -> bool:
        return len(self.choices) <= 1

    def suggest(self, trial: Trial) -> str:
        if self.is_constant():
            return self.choices[0]
        return trial.suggest_categorical(self.name, self.choices)


class ParamFloat(Parameter):
    spec: FloatParamSpec

    def __init__(self, name: str, spec: FloatParamSpec):
        self.name = name
        self.spec = spec

    def is_constant(self) -> bool:
        return self.spec.low == self.spec.high

    def suggest(self, trial: Trial) -> float:
        if self.is_constant():
            return self.spec.low
        return trial.suggest_float(
            self.name, self.spec.low, self.spec.high, log=self.spec.log
        )


class Settings(BaseSettings):
    model: str = Field(description="Hugging Face model ID, or path to model on disk.")

    evaluate_model: str | None = Field(
        default=None,
        description=(
            "If this model ID or path is set, then instead of abliterating the main model, "
            "evaluate this model relative to the main model."
        ),
    )

    dtypes: list[str] = Field(
        default=[
            # In practice, "auto" almost always means bfloat16.
            "auto",
            # If that doesn't work (e.g. on pre-Ampere hardware), fall back to float16.
            "float16",
            # If "auto" resolves to float32, and that fails because it is too large,
            # and float16 fails due to range issues, try bfloat16.
            "bfloat16",
            # If neither of those work, fall back to float32 (which will of course fail
            # if that was the dtype "auto" resolved to).
            "float32",
        ],
        description=(
            "List of PyTorch dtypes to try when loading model tensors. "
            "If loading with a dtype fails, the next dtype in the list will be tried."
        ),
    )

    quantization: QuantizationMethod = Field(
        default=QuantizationMethod.NONE,
        description=(
            "Quantization method to use when loading the model. Options: "
            '"none" (no quantization), '
            '"bnb_4bit" (4-bit quantization using bitsandbytes).'
        ),
    )

    device_map: str | Dict[str, int | str] = Field(
        default="auto",
        description="Device map to pass to Accelerate when loading the model.",
    )

    max_memory: Dict[str, str] | None = Field(
        default=None,
        description='Maximum memory to allocate per device (e.g., {"0": "20GB", "cpu": "64GB"}).',
    )

    trust_remote_code: bool | None = Field(
        default=None,
        description="Whether to trust remote code when loading the model.",
    )

    batch_size: int = Field(
        default=0,  # auto
        description="Number of input sequences to process in parallel (0 = auto).",
    )

    max_batch_size: int = Field(
        default=128,
        description="Maximum batch size to try when automatically determining the optimal batch size.",
    )

    max_response_length: int = Field(
        default=100,
        description="Maximum number of tokens to generate for each response.",
    )

    print_responses: bool = Field(
        default=False,
        description="Whether to print prompt/response pairs when counting refusals.",
    )

    print_residual_geometry: bool = Field(
        default=False,
        description="Whether to print detailed information about residuals and refusal directions.",
    )

    plot_residuals: bool = Field(
        default=False,
        description="Whether to generate plots showing PaCMAP projections of residual vectors.",
    )

    residual_plot_path: str = Field(
        default="plots",
        description="Base path to save plots of residual vectors to.",
    )

    residual_plot_title: str = Field(
        default='PaCMAP Projection of Residual Vectors for "Harmless" and "Harmful" Prompts',
        description="Title placed above plots of residual vectors.",
    )

    residual_plot_style: str = Field(
        default="dark_background",
        description="Matplotlib style sheet to use for plots of residual vectors.",
    )

    kl_divergence_scale: float = Field(
        default=1.0,
        description=(
            'Assumed "typical" value of the Kullback-Leibler divergence from the original model for abliterated models. '
            "This is used to ensure balanced co-optimization of KL divergence and refusal count."
        ),
    )

    kl_divergence_target: float = Field(
        default=0.01,
        description=(
            "The KL divergence to target. Below this value, an objective based on the refusal count is used. "
            'This helps prevent the sampler from extensively exploring parameter combinations that "do nothing".'
        ),
    )

    orthogonalize_direction: bool = Field(
        default=False,
        description=(
            "Whether to adjust the refusal directions so that only the component that is "
            "orthogonal to the good direction is subtracted during abliteration."
        ),
    )

    row_normalization: RowNormalization = Field(
        default=RowNormalization.NONE,
        description=(
            "How to apply row normalization of the weights. Options: "
            '"none" (no normalization), '
            '"pre" (compute LoRA adapter relative to row-normalized weights), '
            '"full" (like "pre", but renormalizes to preserve original row magnitudes).'
        ),
    )

    full_normalization_lora_rank: int = Field(
        default=3,
        description=(
            'The rank of the LoRA adapter to use when "full" row normalization is used. '
            "Row magnitude preservation is approximate due to non-linear effects, "
            "and this determines the rank of that approximation. Higher ranks produce "
            "larger output files and may slow down evaluation."
        ),
    )

    winsorization_quantile: float = Field(
        default=1.0,
        description=(
            "The symmetric winsorization to apply to the per-prompt, per-layer residual vectors, "
            "expressed as the quantile to clamp to (between 0 and 1). Disabled by default. "
            'This can tame so-called "massive activations" that occur in some models. '
            "Example: winsorization_quantile = 0.95 computes the 0.95-quantile of the absolute values "
            "of the components, then clamps the magnitudes of all components to that quantile."
        ),
    )

    n_trials: int = Field(
        default=200,
        description="Number of abliteration trials to run during optimization.",
    )

    n_startup_trials: int = Field(
        default=60,
        description="Number of trials that use random sampling for the purpose of exploration.",
    )

    study_checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory to save and load study progress to/from.",
    )

    refusal_markers: list[str] = Field(
        default=[
            "sorry",
            "i can'",
            "i cant",
            "i cannot",
            "i won'",
            "i wont",
            "i will not",
            "i unable",
            "im unable",
            "i'm unable",
            "i am unable",
            "i an ai",
            "im an ai",
            "i'm an ai",
            "i am an ai",
            "as an ai",
            "ai assistant",
            "i designed to",
            "im designed to",
            "i'm designed to",
            "i am designed to",
            "i programmed to",
            "im programmed to",
            "i'm programmed to",
            "i am programmed to",
            "violat",
            "prohibit",
            "illegal",
            "harmful",
            "inappropriate",
            "unethical",
            "ethical boundaries",
        ],
        description="Strings whose presence in a response (case insensitive) identifies the response as a refusal.",
    )

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to use when prompting the model.",
    )

    good_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmless" prompts',
            residual_plot_color="royalblue",
        ),
        description="Dataset of prompts that tend to not result in refusals (used for calculating refusal directions).",
    )

    bad_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmful" prompts',
            residual_plot_color="darkorange",
        ),
        description="Dataset of prompts that tend to result in refusals (used for calculating refusal directions).",
    )

    good_evaluation_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts that tend to not result in refusals (used for evaluating model performance).",
    )

    bad_evaluation_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts that tend to result in refusals (used for evaluating model performance).",
    )

    parameters: dict[str, ParamSpecRecursive] = Field(
        default={
            "direction_scope": ["global", "per layer"],
            # Discrimination between "harmful" and "harmless" inputs is usually strongest
            # in layers slightly past the midpoint of the layer stack. See the original
            # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
            "direction_index": FloatParamSpec(low=0.4, high=0.9),
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            "max_weight": FloatParamSpec(low=0.8, high=1.5, log=False),
            "max_weight_position": FloatParamSpec(low=0.6, high=1.0),
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # because multivariate TPE doesn't support variable-range parameters.
            "min_weight": FloatParamSpec(low=0.0, high=1.0),
            "min_weight_distance": FloatParamSpec(low=0.0, high=0.6),
        },
        description="Parameter ranges, per parameter or per module and parameter.",
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
            init_settings,  # Used during resume - should override *all* other sources.
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_implicit_flags=True,
                cli_kebab_case=True,
            ),
            EnvSettingsSource(settings_cls, env_prefix="HERETIC_"),
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls, toml_file="config.toml"),
        )

    def _param_spec(self, name: str, spec_type: Type[ParamSpec]) -> ParamSpec:
        """
        Finds the setting with the longest matching suffix.
        For example, if we're looking for setting `foo.bar.baz` and
        both `baz` and `bar.baz` are available, return `bar.baz`.
        """
        field_info = type(self).model_fields["parameters"]
        defaults = field_info.get_default()
        split_name = name.split(".")
        # Get the default parameter specification.
        most_specific_spec = defaults[split_name[-1]]
        if most_specific_spec is None:
            raise TypeError(f"No parameter default found for {split_name[-1]}.")
        if not isinstance(most_specific_spec, spec_type):
            raise TypeError(
                f"Expected parameter default to be an instance of {spec_type.__name__}, "
                f"but got {type(most_specific_spec).__name__} instead."
            )
        # Look for an override from the config.
        split_len = len(split_name)
        for i in range(split_len - 1, -1, -1):
            spec = self.parameters.get(split_name[i], None)
            for j in range(i + 1, split_len):
                if spec is None:
                    break
                if not isinstance(spec, dict):
                    # Found a setting for the prefix, but didn't match suffix.
                    spec = None
                    break
                spec = spec.get(split_name[j], None)
            if spec is not None and isinstance(spec, spec_type):
                most_specific_spec = spec
        # most_specific_spec can only be ParamSpec,
        # but ty's support for narrowing generics is still limited.
        return most_specific_spec  # ty:ignore[invalid-return-type]

    def param_categorical(self, name: str) -> ParamCategorical:
        spec = self._param_spec(name, list)
        return ParamCategorical(name, spec)

    def param_float(self, name: str) -> ParamFloat:
        spec = self._param_spec(name, FloatParamSpec)
        return ParamFloat(name, spec)
