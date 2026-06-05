# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar, cast

from annotated_types import Interval, IsFinite, MinLen
from optuna import Trial
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    model_validator,
)
from typing_extensions import Self, TypeAliasType


class ModelComponent(str, Enum):
    ATTN_O_PROJ = "attn.o_proj"
    MLP_DOWN_PROJ = "mlp.down_proj"


class ParamKind(str, Enum):
    SCALAR = "param_kind_scalar"
    LIST = "param_kind_list"
    DICT = "param_kind_dict"
    FLAT = "param_kind_flat"
    NESTED = "param_kind_nested"


Scalar = TypeVar("Scalar", bound=None | bool | int | float | str)
UnitFloat = Annotated[float, Interval(ge=0.0, le=1.0)]


class FloatParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: Annotated[
        IsFinite[float],
        Field(
            description="Lower endpoint of the range of suggested values (inclusive).",
        ),
    ]

    high: Annotated[
        IsFinite[float],
        Field(
            description="Upper endpoint of the range of suggested values (inclusive).",
        ),
    ]

    log: Annotated[
        bool,
        Field(
            description="If true, the value is sampled from the range in the log domain.",
        ),
    ] = False

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be ≤ high ({self.high}).")
        if self.log and self.low <= 0:
            raise ValueError(f"low ({self.low}) must be positive when log = true.")
        return self


class UnitParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: Annotated[
        UnitFloat,
        Field(
            description="Lower endpoint of the range of suggested values (inclusive).",
        ),
    ]

    high: Annotated[
        UnitFloat,
        Field(
            description="Upper endpoint of the range of suggested values (inclusive).",
        ),
    ]

    # Included to make constructor compatible with FloatParamSpec.
    log: Literal[False] = False

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be ≤ high ({self.high}).")
        return self


def discriminate_flat(param: Any) -> ParamKind | None:
    if param is None or isinstance(param, (bool, int, float, str)):
        return ParamKind.SCALAR
    if isinstance(param, (FloatParamSpec, UnitParamSpec)):
        return ParamKind.DICT
    if isinstance(param, (list, tuple)):
        return ParamKind.LIST
    if isinstance(param, dict):
        return ParamKind.DICT
    return None


def discriminate_nested(param: Any) -> ParamKind | None:
    if param is None or isinstance(param, (bool, int, float, str)):
        return ParamKind.FLAT
    if isinstance(param, (FloatParamSpec, UnitParamSpec)):
        return ParamKind.FLAT
    if isinstance(param, (list, tuple)):
        return ParamKind.FLAT
    if isinstance(param, dict):
        if any(key in ModelComponent for key in param):
            return ParamKind.NESTED
        elif any(key == "low" or key == "high" or key == "log" for key in param):
            return ParamKind.FLAT
        else:
            # Assume it's nested for a more helpful error message.
            return ParamKind.NESTED
    return None


FlatCategoricalParamType = TypeAliasType(
    "FlatCategoricalParamType",
    Annotated[
        Annotated[Scalar, Tag(ParamKind.SCALAR)]
        | Annotated[list[Scalar], MinLen(1), Tag(ParamKind.LIST)],
        Discriminator(discriminate_flat),
    ],
    type_params=(Scalar,),
)
NestedCategoricalParamType = TypeAliasType(
    "NestedCategoricalParamType",
    dict[ModelComponent, FlatCategoricalParamType[Scalar]],
    type_params=(Scalar,),
)
CategoricalParamType = TypeAliasType(
    "CategoricalParamType",
    Annotated[
        Annotated[FlatCategoricalParamType[Scalar], Tag(ParamKind.FLAT)]
        | Annotated[NestedCategoricalParamType[Scalar], Tag(ParamKind.NESTED)],
        Discriminator(discriminate_nested),
    ],
    type_params=(Scalar,),
)

FlatFloatParamType = Annotated[
    Annotated[IsFinite[float], Tag(ParamKind.SCALAR)]
    | Annotated[FloatParamSpec, Tag(ParamKind.DICT)],
    Discriminator(discriminate_flat),
]
NestedFloatParamType = dict[ModelComponent, FlatFloatParamType]
FloatParamType = Annotated[
    Annotated[FlatFloatParamType, Tag(ParamKind.FLAT)]
    | Annotated[NestedFloatParamType, Tag(ParamKind.NESTED)],
    Discriminator(discriminate_nested),
]

FlatUnitParamType = Annotated[
    Annotated[UnitFloat, Tag(ParamKind.SCALAR)]
    | Annotated[UnitParamSpec, Tag(ParamKind.DICT)],
    Discriminator(discriminate_flat),
]
NestedUnitParamType = dict[ModelComponent, FlatUnitParamType]
UnitParamType = Annotated[
    Annotated[FlatUnitParamType, Tag(ParamKind.FLAT)]
    | Annotated[NestedUnitParamType, Tag(ParamKind.NESTED)],
    Discriminator(discriminate_nested),
]


class CategoricalParam(Generic[Scalar]):
    name: str
    choices: list[Scalar]

    def __init__(self, name: str, choices: list[Scalar]) -> None:
        self.name = name
        self.choices = choices

    def suggest(self, trial: Trial) -> Scalar:
        if len(self.choices) < 2:
            return self.choices[0]
        return cast(Scalar, trial.suggest_categorical(self.name, self.choices))


class FloatParam:
    name: str
    low: float
    high: float
    log: bool

    def __init__(self, name: str, low: float, high: float, log: bool) -> None:
        self.name = name
        self.low = low
        self.high = high
        self.log = log

    def suggest(self, trial: Trial) -> float:
        if self.low == self.high:
            return self.low
        return trial.suggest_float(self.name, self.low, self.high, log=self.log)


class FlatCategoricalParamProxy(Generic[Scalar]):
    name: str
    current: FlatCategoricalParamType[Scalar]
    default: FlatCategoricalParamType[Scalar]

    def __init__(
        self,
        name: str,
        current: FlatCategoricalParamType[Scalar],
        default: FlatCategoricalParamType[Scalar],
    ) -> None:
        self.name = name
        self.current = current
        self.default = default

    def get(self) -> CategoricalParam[Scalar]:
        choices: list[Scalar]
        if self.current is None or isinstance(self.current, (bool, int, float, str)):
            choices = [self.current]
        else:
            choices = self.current
        return CategoricalParam[Scalar](self.name, choices)

    def suggest(self, trial: Trial) -> Scalar:
        return self.get().suggest(trial)


class CategoricalParamProxy(Generic[Scalar]):
    name: str
    current: CategoricalParamType[Scalar]
    default: CategoricalParamType[Scalar]

    def __init__(
        self,
        name: str,
        current: CategoricalParamType[Scalar],
        default: CategoricalParamType[Scalar],
    ) -> None:
        self.name = name
        self.current = current
        self.default = default

    def get(self, component: ModelComponent) -> CategoricalParam[Scalar]:
        choices: list[Scalar]
        # First, get the choices from the defaults, which should be fully specified.
        if self.default is None or isinstance(self.default, (bool, int, float, str)):
            choices = [self.default]
        elif isinstance(self.default, list):
            choices = self.default
        else:
            param: Scalar | list[Scalar] = self.default[component]
            if param is None or isinstance(param, (bool, int, float, str)):
                choices = [param]
            else:
                choices = param
        # Now check for a user-specified override. In the per-component form, this may not be fully specified.
        if self.current is None or isinstance(self.current, (bool, int, float, str)):
            choices = [self.current]
        elif isinstance(self.current, list):
            choices = self.current
        elif component in self.current:
            param: Scalar | list[Scalar] = self.current.get(component, choices)
            if param is None or isinstance(param, (bool, int, float, str)):
                choices = [param]
            else:
                choices = param
        return CategoricalParam[Scalar](f"{component.value}.{self.name}", choices)

    def suggest(self, trial: Trial, component: ModelComponent) -> Scalar:
        return self.get(component).suggest(trial)


class FlatFloatParamProxy:
    name: str
    current: FlatFloatParamType | FlatUnitParamType
    default: FlatFloatParamType | FlatUnitParamType

    def __init__(
        self,
        name: str,
        current: FlatFloatParamType | FlatUnitParamType,
        default: FlatFloatParamType | FlatUnitParamType,
    ) -> None:
        self.name = name
        self.current = current
        self.default = default

    def get(self) -> FloatParam:
        log = False
        if isinstance(self.current, (int, float)):
            low = high = self.current
        else:
            param = self.current
            low, high, log = param.low, param.high, param.log
        return FloatParam(self.name, low, high, log)

    def suggest(self, trial: Trial) -> float:
        return self.get().suggest(trial)


class FloatParamProxy:
    name: str
    current: FloatParamType | UnitParamType
    default: FloatParamType | UnitParamType

    def __init__(
        self,
        name: str,
        current: FloatParamType | UnitParamType,
        default: FloatParamType | UnitParamType,
    ) -> None:
        self.name = name
        self.current = current
        self.default = default

    def get(self, component: ModelComponent) -> FloatParam:
        log = False
        # First, get the choices from the defaults, which should be fully specified.
        if isinstance(self.default, (int, float)):
            low = high = self.default
        elif isinstance(self.default, (FloatParamSpec, UnitParamSpec)):
            param = self.default
            low, high, log = param.low, param.high, param.log
        else:
            param = self.default[component]
            if isinstance(param, (int, float)):
                low = high = param
            else:
                low, high, log = param.low, param.high, param.log
        # Now check for a user-specified override. In the per-component form, this may not be fully specified.
        if isinstance(self.current, (int, float)):
            low = high = self.current
        elif isinstance(self.current, (FloatParamSpec, UnitParamSpec)):
            param = self.current
            low, high, log = param.low, param.high, param.log
        elif component in self.current:
            param = self.current[component]
            if isinstance(param, (int, float)):
                low = high = param
            else:
                low, high, log = param.low, param.high, param.log
        return FloatParam(f"{component.value}.{self.name}", low, high, log)

    def suggest(self, trial: Trial, component: ModelComponent) -> float:
        return self.get(component).suggest(trial)


class DirectionScope(str, Enum):
    GLOBAL = "global"
    PER_LAYER = "per layer"


class ParameterSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction_scope: FlatCategoricalParamType[DirectionScope] = Field(
        default=[DirectionScope.GLOBAL, DirectionScope.PER_LAYER],
        description=(
            "The different refusal direction scopes that can be applied to each trial. "
            '"global": Choose a refusal direction by interpolating between 2 layers and apply it globally. '
            '"per layer": For each layer within range, apply the layer\'s own refusal direction to itself.'
        ),
    )

    # Discrimination between "harmful" and "harmless" inputs is usually strongest in layers slightly past the midpoint
    # of the layer stack. See the original abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
    direction_fraction: FlatUnitParamType = Field(
        default=UnitParamSpec(low=0.4, high=0.9),
        description="For the global direction scope, the layer from which to choose the refusal direction.",
    )

    # The parameter ranges are based on experiments with various models and much wider ranges.
    # They are not set in stone and might have to be adjusted for future models.
    max_weight: FloatParamType = Field(
        default=FloatParamSpec(low=0.8, high=1.5),
        description=(
            "The maximum weight with which to apply the abliteration. Set log = true to "
            "sample from the log space, which will select lower values more frequently. "
            "Note that low must be greater than 0 when log = true."
        ),
    )

    max_weight_position_fraction: UnitParamType = Field(
        default=UnitParamSpec(low=0.6, high=1.0),
        description="The position (layer) at which the maximum weight should be applied.",
    )

    # For sampling purposes, min_weight is expressed as a fraction of max_weight,
    # because multivariate TPE doesn't support variable-range parameters.
    min_weight_relative: UnitParamType = Field(
        default=UnitParamSpec(low=0.0, high=1.0),
        description="The minimum weight as a fraction of the maximum weight.",
    )

    min_weight_distance_fraction: UnitParamType = Field(
        default=UnitParamSpec(low=0.0, high=0.6),
        description=(
            "The distance from max_weight_position across which the weight drops from "
            "max_weight to min_weight. Beyond this distance, the weight is set to 0."
        ),
    )


class Parameters:
    direction_scope: FlatCategoricalParamProxy[DirectionScope]
    direction_fraction: FlatFloatParamProxy
    max_weight: FloatParamProxy
    max_weight_position_fraction: FloatParamProxy
    min_weight_relative: FloatParamProxy
    min_weight_distance_fraction: FloatParamProxy

    def __init__(self, current_params: ParameterSpecification) -> None:
        default_params = ParameterSpecification()

        def _create_proxy(name: str, proxy_cls: type) -> None:
            current = getattr(current_params, name)
            default = getattr(default_params, name)
            proxy = proxy_cls(name, current, default)
            setattr(self, name, proxy)

        _create_proxy("direction_scope", FlatCategoricalParamProxy[DirectionScope])
        _create_proxy("direction_fraction", FlatFloatParamProxy)
        _create_proxy("max_weight", FloatParamProxy)
        _create_proxy("max_weight_position_fraction", FloatParamProxy)
        _create_proxy("min_weight_relative", FloatParamProxy)
        _create_proxy("min_weight_distance_fraction", FloatParamProxy)
