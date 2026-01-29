# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, TypeVar, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

T = TypeVar("T")


def get_plugin_namespace(model_extra, namespace: str) -> dict[str, Any]:
    """
    Returns the config dict from the `[<namespace>]` TOML table.
    """
    cur: Any = model_extra
    for part in namespace.split("."):
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(part)

    if cur is None:
        return {}
    if not isinstance(cur, dict):
        raise TypeError(
            f"Plugin namespace [{namespace}] must be a table/object, got {type(cur).__name__}"
        )
    return cur


def load_plugin(
    name: str,
    base_class: type[T],
) -> type[T]:
    """
    Load a plugin class from either a filesystem `.py` file or a fully-qualified Python import path.
    Also checks that the class exists in the module and that it
    subclasses the correct Plugin subclass (e.g Scorer).

    Accepted forms:
    - `path/to/plugin.py:MyPluginClass` (relative or absolute): load `MyPluginClass`
      from that file.
    - `fully.qualified.module.MyPluginClass`: import the module and load the class.
    """ 

    def validate_class(module: ModuleType, class_name: str) -> type[Any]:
        """
        Checks that the module actually exports the class as claimed and returns the class.
        """
        obj = getattr(module, class_name, None)
        if not inspect.isclass(obj):
            raise ValueError(
                f"Plugin '{name}' does not export a class named '{class_name}'"
            )
        return obj

    # common user trap with filepath imports
    if name.endswith(".py"):
        raise ValueError(
            "You must append the plugin class name to the filepath like this: path/to/plugin.py:ClassName"
        )

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

        # We're writing directly to the sys.modules dict,
        # so the typical restrictions on module names
        # (no dots, slashes, etc.) don't apply
        module_name = f"heretic_plugin_{plugin_path}"

        # Reuse already-loaded modules to avoid re-executing the plugin on repeated loads.
        module = sys.modules.get(plugin_path)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Could not load plugin '{name}' (invalid module spec)"
                )

            module = importlib.util.module_from_spec(spec)

            # Cache before executing to match normal import semantics and allow
            # circular imports. If execution fails, remove the entry.
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            
        plugin_cls = validate_class(module, class_name)
    # Fully-qualified import path, e.g "heretic.scorers.refusal_rate.RefusalRate"
    else:
        if "." not in name:
            raise ValueError(
                "Import-based plugin must use the form 'fully.qualified.module.ClassName'"
            )
        module_name, class_name = name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Error loading plugin '{name}': {e}") from e
        plugin_cls = validate_class(module, class_name)

    if not issubclass(plugin_cls, base_class):
        raise TypeError(f"Plugin '{name}' must subclass {base_class.__name__}")

    return plugin_cls


class Plugin:
    """
    Base class for Heretic plugins.

    Plugins may define:
    - `settings: <BaseModelSubclass>` type annotation (recommended)
      Heretic will validate the corresponding config table against it and pass
      an instance as `settings`.
    """

    def __init__(self, *, settings: BaseModel | None = None):
        self.settings = settings

    @classmethod
    def get_settings_model(cls) -> type[BaseModel] | None:
        """
        Return the plugin settings model, if present.
        - If the plugin has a `settings: <BaseModelSubclass>` type annotation,
          that type is used as the settings schema.
        - Otherwise: no settings schema.
        """

        def unwrap_settings_type(tp: Any) -> Any:
            """Unwrap `Annotated[T, ...]`."""
            while True:
                origin = get_origin(tp)
                if origin is Annotated:
                    tp = get_args(tp)[0]
                    continue
                return tp

        hints = get_type_hints(cls, include_extras=True)
        annotated = hints.get("settings")
        if annotated is None:
            return None

        model = unwrap_settings_type(annotated)
        origin = get_origin(model)
        if origin in (Union, types.UnionType) and type(None) in get_args(model):
            raise TypeError(
                f"{cls.__name__}.settings must not be Optional; "
                "use a non-optional pydantic.BaseModel subclass (e.g. `settings: Settings`)."
            )
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError(
                f"{cls.__name__}.settings must be annotated with a pydantic.BaseModel subclass"
            )
        return model

    @classmethod
    def validate_settings(
        cls, raw_namespace: dict[str, Any] | None
    ) -> BaseModel | None:
        """
        Validates plugin settings for this plugin class.

        - If a settings model is present: returns an instance of that model
        - Otherwise returns None
        """
        settings_model = cls.get_settings_model()
        if settings_model is None:
            return None
        return settings_model.model_validate(raw_namespace or {})
