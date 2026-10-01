# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, get_args

from optuna import Trial
from optuna.trial import FrozenTrial
from pydantic import BaseModel

from .config import (
    ModifierConfig,
)
from .config import (
    Settings as HereticSettings,
)
from .model import Model
from .plugin import Context, Plugin, load_plugin
from .utils import print


class Serializable(Protocol):
    def to_dict(self) -> dict[str, Any]: ...

    def to_presentation_dict(self) -> dict[str, str]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable": ...


Parameters = TypeVar("Parameters", bound=Serializable)


class Modifier(Generic[Parameters], Plugin, ABC):
    """
    Abstract base class for modifier plugins.

    Modifiers modify models based on an implementation-dependent set of optimizable parameters.

    Examples: Standard abliteration, ARA, SOMA, etc.
    """

    @property
    def modifier_name(self) -> str:
        """
        The name of the modifier.
        This is what shows up in the CLI and Markdown on HF.
        """
        return self.__class__.__name__

    @property
    def parameters_class(self) -> type[Parameters]:
        """
        The class of the modifier's parameters type.
        """
        base_class = self.__class__.__orig_bases__[0]  # ty:ignore[unresolved-attribute]
        generic_type = get_args(base_class)[0]
        return generic_type

    def __init__(
        self,
        heretic_settings: HereticSettings,
        settings: BaseModel | None = None,
    ) -> None:
        super().__init__(heretic_settings=heretic_settings, settings=settings)

    @abstractmethod
    def suggest_parameters(self, ctx: Context, trial: Trial) -> Parameters:
        """
        Sample parameters for a trial using the trial's `suggest_*` methods,
        collect them in an implementation-dependent parameters object, and
        return that object.
        """

    @abstractmethod
    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        """
        Modify the model (obtainable via `ctx.get_model()`)
        according to the provided parameters.
        """

    @abstractmethod
    def reset_model(self, ctx: Context) -> None:
        """
        Reset the model (obtainable via `ctx.get_model()`),
        undoing any changes made by `modify_model`.
        """

    def render_trial_parameters(self, trial: Trial | FrozenTrial) -> dict[str, str]:
        """
        Transform the names and values of the modifier's parameters
        that are contained in the trial's user attributes into a form
        suitable for presentation.
        """
        return self.parameters_class.from_dict(
            trial.user_attrs["parameters"]
        ).to_presentation_dict()


@dataclass
class ModifierEntry:
    modifier: Modifier[Any]
    name: str
    config: ModifierConfig


def load_and_init_modifiers(
    settings: HereticSettings,
    model: Model,
) -> list[ModifierEntry]:
    """
    Load and instantiate all configured modifier plugins,
    then runs their initialization hooks.
    """
    modifier_configs = settings.modifiers
    if not modifier_configs:
        raise ValueError("No modifiers configured. Set 'modifiers' in config.toml")
    if len(modifier_configs) > 1:
        raise ValueError("Using multiple modifiers is not yet supported")

    modifier_keys: set[str] = set()

    modifier_entries: list[ModifierEntry] = []

    # Resolve plugin classes from names and validate.
    for config in modifier_configs:
        modifier_cls = load_plugin(name=config.plugin, base_class=Modifier)
        modifier_cls.validate_contract()

        print(
            f"* Loaded: [bold]{modifier_cls.__name__}{' - ' + config.instance_name if config.instance_name else ''}[/bold]"
        )

        # Instantiate modifiers.
        instance_name = config.instance_name or None

        raw_settings = modifier_cls.get_settings_raw(
            settings.model_extra,
            "modifier",
            instance_name,
        )
        modifier_settings: BaseModel | None = modifier_cls.validate_settings(
            raw_settings
        )

        modifier = modifier_cls(
            heretic_settings=settings,
            settings=modifier_settings,
        )

        # External labeling key: ensures multiple instances can coexist.
        # Uses underscore to match the TOML namespace format (`modifier.<Class>_<instance>`).
        modifier_key = (
            modifier_cls.__name__
            if not instance_name
            else f"{modifier_cls.__name__}_{instance_name}"
        )
        if modifier_key in modifier_keys:
            raise ValueError(
                f"Duplicate modifier instance name: {modifier_key}. "
                "Give each instance a unique `instance_name`."
            )
        modifier_keys.add(modifier_key)

        modifier_instance_name = (
            f"{modifier.modifier_name} - {instance_name}"
            if instance_name
            else modifier.modifier_name
        )
        modifier_entries.append(
            ModifierEntry(
                modifier=modifier,
                config=config,
                name=modifier_instance_name,
            )
        )

    # Run modifier init hooks.
    ctx = Context(settings=settings, model=model)

    for entry in modifier_entries:
        entry.modifier.init(ctx)

    return modifier_entries
