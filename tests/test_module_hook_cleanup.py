# SPDX-License-Identifier: AGPL-3.0-or-later
"""I/O capture hooks must not survive failed inference or registration."""

import unittest
from types import MethodType
from unittest.mock import patch

import torch

from heretic.model import Model


class ModuleHookCleanupTests(unittest.TestCase):
    def setUp(self):
        self.modules = [torch.nn.Linear(2, 2, bias=False) for _ in range(2)]
        for module in self.modules:
            with torch.no_grad():
                module.weight.copy_(torch.eye(2))
        self.inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        self.model = Model.__new__(Model)
        self.model.get_layers = lambda: [self.modules]
        self.model.get_layer_modules = lambda index: {"mlp": self.modules}
        self.fail = False
        self.interrupt = False

        def generate(instance, prompts, **kwargs):
            result = self.modules[0](self.inputs)
            if self.interrupt:
                raise KeyboardInterrupt
            if self.fail:
                raise RuntimeError("inference failed")
            return self.modules[1](result)

        self.model.generate = MethodType(generate, self.model)

    def assert_no_capture_hooks(self):
        self.assertTrue(all(not module._forward_hooks for module in self.modules))

    def test_failed_inference_does_not_poison_a_retry(self):
        self.fail = True
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.model.get_module_io([])
        self.fail = False
        captured = self.model.get_module_io([])
        torch.testing.assert_close(captured[0]["mlp"][0][0], self.inputs[:, -1, :])
        torch.testing.assert_close(captured[0]["mlp"][1][1], self.inputs[:, -1, :])
        self.assert_no_capture_hooks()

    def test_partial_registration_failure_removes_preceding_hooks(self):
        with patch.object(
            self.modules[1],
            "register_forward_hook",
            side_effect=RuntimeError("registration failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                self.model.get_module_io([])
        self.assert_no_capture_hooks()

    def test_interrupt_removes_capture_hooks(self):
        self.interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            self.model.get_module_io([])
        self.assert_no_capture_hooks()

    def test_success_preserves_existing_user_hooks(self):
        invocations = []
        handle = self.modules[0].register_forward_hook(
            lambda *args: invocations.append(True)
        )
        try:
            captured = self.model.get_module_io([])
            self.assertEqual(invocations, [True])
            self.assertEqual(list(self.modules[0]._forward_hooks), [handle.id])
            self.assertFalse(self.modules[1]._forward_hooks)
            self.assertEqual(set(captured[0]["mlp"]), {0, 1})
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
