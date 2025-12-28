# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel


class Plugin:
    """
    Base class for Heretic plugins.

    Plugins must define:
    - `name`: stable identifier for namespaced config. TOML table: `[<name>]`

    Plugins may define:
    - nested `Settings` model (subclass of pydantic.BaseModel)
      Heretic will validate `[<name>]` against it and pass an instance as `plugin_settings`.
    """

    name: ClassVar[str] = ""

    class Settings(BaseModel):
        """Base settings class for plugins. Subclasses can extend this."""
        pass

    def __init__(self, *, plugin_settings: BaseModel | None = None):
        self.plugin_settings = plugin_settings

    @classmethod
    def validate_contract(cls) -> None:
        if not isinstance(getattr(cls, "name", None), str) or not cls.name.strip():
            raise ValueError(
                f"{cls.__name__} must define a non-empty class attribute `name`"
            )

        settings_model = getattr(cls, "Settings", None)
        if settings_model is None:
            return

        if not isinstance(settings_model, type) or not issubclass(
            settings_model, BaseModel
        ):
            raise TypeError(
                f"{cls.__name__}.Settings must be a subclass of pydantic.BaseModel"
            )

    @classmethod
    def validate_settings(cls, raw_namespace: dict[str, Any] | None) -> BaseModel | None:
        """
        Validates plugin settings for this plugin class.

        - If `Settings` is present: returns an instance of that model
        - Otherwise returns None
        """
        settings_model = getattr(cls, "Settings", None)
        if settings_model is None:
            return None
        return settings_model.model_validate(raw_namespace or {})
