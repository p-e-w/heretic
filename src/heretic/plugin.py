# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import importlib
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


def load_plugin(
    name: str,
    base_class: type[T],
) -> type[T]:
    """
    Load a plugin class from either a filesystem `.py` file or a fully-qualified
    Python import path.

    Accepted forms:
    - `path/to/plugin.py:MyPluginClass` (relative or absolute): load `MyPluginClass`
      from that file.
    - `fully.qualified.module.MyPluginClass`: import the module and load the class.
    """

    def import_module(module_name: str) -> ModuleType:
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Error loading plugin '{name}': {e}") from e

    # file path with explicit class name, e.g "C:\\path\\plugin.py:MyPlugin"
    if ":" in name:
        file_path, class_name = name.rsplit(":", 1)
        if not file_path.endswith(".py") or not class_name:
            raise ValueError(
                "File-based plugin must use the form 'path/to/plugin.py:ClassName'"
            )

        plugin_path = Path(file_path)
        if not plugin_path.is_absolute():
            plugin_path = Path.cwd() / plugin_path
        plugin_path = plugin_path.resolve()

        if not plugin_path.is_file():
            raise ImportError(f"Plugin file '{plugin_path}' does not exist")

        # Use a unique, stable module name to avoid clashes on repeated loads.
        module_name = f"_heretic_plugin_{plugin_path.stem}_{abs(hash(plugin_path))}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load plugin '{name}' (invalid module spec)")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        obj = getattr(module, class_name, None)
        if not inspect.isclass(obj):
            raise ValueError(
                f"Plugin '{name}' does not export a class named '{class_name}'"
            )
        plugin_cls = obj
    # Fully-qualified import path, e.g "heretic.scorers.refusal_rate.RefusalRate"
    else:
        if "." not in name:
            raise ValueError(
                "Import-based plugin must use the form 'fully.qualified.module.ClassName'"
            )
        module_name, class_name = name.rsplit(".", 1)
        module = import_module(module_name)
        obj = getattr(module, class_name, None)
        if not inspect.isclass(obj):
            raise ValueError(
                f"Plugin '{name}' does not export a class named '{class_name}'"
            )
        plugin_cls = obj

    if not issubclass(plugin_cls, base_class):
        raise TypeError(f"Plugin '{name}' must subclass {base_class.__name__}")

    return plugin_cls


class Plugin:
    """
    Base class for Heretic plugins.

    Plugins may define:
    - nested `Settings` model (subclass of pydantic.BaseModel)
      Heretic will validate `[<name>]` against it and pass an instance as `plugin_settings`.
    """

    def __init__(self, *, plugin_settings: BaseModel | None = None):
        self.plugin_settings = plugin_settings

    @classmethod
    def validate_contract(cls) -> None:
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
    def validate_settings(
        cls, raw_namespace: dict[str, Any] | None
    ) -> BaseModel | None:
        """
        Validates plugin settings for this plugin class.

        - If `Settings` is present: returns an instance of that model
        - Otherwise returns None
        """
        settings_model = getattr(cls, "Settings", None)
        if settings_model is None:
            return None
        return settings_model.model_validate(raw_namespace or {})
