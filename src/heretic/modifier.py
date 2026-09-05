# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from optuna import Trial
from pydantic import BaseModel

from heretic.plugin import Context, Plugin

from .config import Settings as HereticSettings

Parameters = TypeVar("Parameters")


class Modifier(Generic[Parameters], Plugin, ABC):
    """
    Abstract base class for modifier plugins.

    Modifiers modify models based on an implementation-dependent set of optimizable parameters.

    Examples: Standard abliteration, ARA, SOMA, etc.
    """

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
