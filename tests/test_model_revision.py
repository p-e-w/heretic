# SPDX-License-Identifier: AGPL-3.0-or-later
"""Architecture detection must use the same selected revision as weight loading."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from heretic import model as model_module
from heretic.config import InitOnlySettings


class ModelClassRevisionTests(unittest.TestCase):
    def test_selected_revision_controls_architecture(self):
        revision = "a" * 40

        def config_for_ref(model, **kwargs):
            return (
                {"vision_config": {}} if kwargs.get("revision") == revision else {},
                {},
            )

        with patch.object(
            model_module.PretrainedConfig, "get_config_dict", side_effect=config_for_ref
        ) as lookup:
            selected = model_module.get_model_class("org/base", revision=revision)
        self.assertIs(selected, model_module.AutoModelForImageTextToText)
        lookup.assert_called_once_with("org/base", revision=revision)

    def test_default_lookup_remains_unchanged(self):
        with patch.object(
            model_module.PretrainedConfig, "get_config_dict", return_value=({}, {})
        ) as lookup:
            selected = model_module.get_model_class("org/base")
        self.assertIs(selected, model_module.AutoModelForCausalLM)
        lookup.assert_called_once_with("org/base")

    def test_reload_uses_same_captured_revision_for_config_and_weights(self):
        revision = "a" * 40
        model = model_module.Model.__new__(model_module.Model)
        model.settings = InitOnlySettings(
            model="org/base", model_commit=revision, device_map="cpu"
        )
        model.revision_kwargs = {"revision": revision}
        model.model = None
        model.needs_reload = True
        model.dtype = model_module.torch.float32
        model.max_memory = None
        model.trusted_models = set()
        loaded = SimpleNamespace()
        with (
            patch.object(model_module, "empty_cache"),
            patch.object(
                model_module.PretrainedConfig,
                "get_config_dict",
                return_value=({"vision_config": {}}, {}),
            ) as lookup,
            patch.object(
                model_module.AutoModelForImageTextToText,
                "from_pretrained",
                return_value=loaded,
            ) as weights,
        ):
            self.assertFalse(model.reset_model())
        lookup.assert_called_once_with("org/base", revision=revision)
        self.assertEqual(weights.call_args.args, ("org/base",))
        self.assertEqual(weights.call_args.kwargs["revision"], revision)
        self.assertIs(model.model, loaded)
        self.assertFalse(model.needs_reload)


if __name__ == "__main__":
    unittest.main()
