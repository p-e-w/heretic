# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import subprocess
import sys
import unittest
from io import StringIO
from unittest.mock import patch

from rich.console import Console
from rich.progress import BarColumn

from heretic import progress


class ProgressRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = StringIO()
        self.console = Console(
            file=self.output,
            width=120,
            color_system=None,
            force_terminal=True,
            force_interactive=True,
        )
        self.display = progress._new_progress(console=self.console, auto_refresh=False)
        self.patch = patch.object(progress, "_progress", self.display)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.display.stop)

    def bar(self, description: str, **kwargs):
        bar = progress.TqdmShim(
            desc=description,
            file=self.output,
            disable=False,
            mininterval=0,
            miniters=1,
            **kwargs,
        )
        self.addCleanup(bar.close)
        return bar

    def render(self) -> str:
        return "".join(
            segment.text for segment in self.console.render(self.display.get_renderable())
        )

    def test_custom_format_uses_a_rich_bar(self) -> None:
        bar = self.bar(
            "Downloading bytes",
            total=8192,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            bar_format="{desc}: {bar}| {n_fmt}B / {total_fmt}B, {rate_fmt}",
        )
        bar.last_print_t = bar.start_t
        with patch.object(bar, "_time", return_value=bar.start_t + 1):
            bar.update(1024)

        output = self.render()
        self.assertIn("Downloading bytes", output)
        self.assertIn("1.00kB / 8.00kB", output)
        self.assertIn("1.00kB/s", output)
        self.assertIsInstance(self.display.columns[1], BarColumn)
        self.assertNotIn("|", output)

    def test_simultaneous_bars_keep_their_own_data(self) -> None:
        fetching = self.bar("Fetching 2 files", total=2)
        reconstructing = self.bar(
            "Reconstructing", total=4096, unit="B", unit_scale=True
        )
        downloading = self.bar("Downloading bytes", total=None, unit="B", unit_scale=True)
        fetching.update(1)
        reconstructing.update(1024)
        downloading.update(1024)
        reconstructing.set_postfix_str("2.00kB/s  ")

        output = self.render()
        self.assertEqual(output.count("Fetching 2 files"), 1)
        self.assertEqual(output.count("Reconstructing"), 1)
        self.assertEqual(output.count("Downloading bytes"), 1)
        self.assertIn("1.02kB / 4.10kB", output)
        self.assertIn("2.00kB/s", output)
        reconstructing_row = next(line for line in output.splitlines() if "Reconstructing" in line)
        self.assertEqual(reconstructing_row.count("B/s"), 1)
        self.assertIn("1.02kB", output)

    def test_closing_one_bar_leaves_the_other_visible(self) -> None:
        first = self.bar("first.bin", total=2)
        second = self.bar("second.bin", total=2)
        first.close()
        second.update(1)

        output = self.render()
        self.assertNotIn("first.bin", output)
        self.assertIn("second.bin", output)

    def test_completed_bar_is_removed_before_later_output(self) -> None:
        bar = self.bar("finished.bin", total=2)
        bar.update(2)

        self.assertFalse(self.display.task_ids)
        self.assertFalse(self.display.live.is_started)
        bar.reset(total=2)
        bar.update(1)
        self.assertTrue(self.display.task_ids)
        bar.close()
        self.console.print("AFTER")
        self.assertNotIn("finished.bin", self.output.getvalue().split("AFTER", 1)[1])

    def test_hugging_face_imports_the_shim_in_a_clean_process(self) -> None:
        code = """
from heretic.progress import TqdmShim, patch_tqdm
patch_tqdm()
import tqdm
import tqdm.auto
import tqdm.std
import importlib
import huggingface_hub._snapshot_download as snapshot
hub_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")
assert tqdm.tqdm is TqdmShim
assert tqdm.auto.tqdm is TqdmShim
assert tqdm.std.tqdm is TqdmShim
assert issubclass(snapshot.base_tqdm, TqdmShim)
assert issubclass(hub_tqdm.tqdm, TqdmShim)
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
