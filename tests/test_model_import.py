# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import subprocess
import sys
import textwrap
import unittest


class ModelImportTests(unittest.TestCase):
    def test_model_import_does_not_require_bitsandbytes(self) -> None:
        script = textwrap.dedent(
            """
            import builtins

            original_import = builtins.__import__

            def reject_bitsandbytes(name, *args, **kwargs):
                if name == "bitsandbytes" or name.startswith("bitsandbytes."):
                    raise AssertionError("bitsandbytes imported while loading heretic.model")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = reject_bitsandbytes

            import heretic.model
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
