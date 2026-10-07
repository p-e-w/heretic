# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
import unittest
from contextlib import ExitStack
from io import StringIO
from unittest.mock import patch

from rich.console import Console
from rich.progress import Progress

from heretic import progress


class ProgressRenderingTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.output = StringIO()
        self.patches.enter_context(patch.dict(os.environ, {"TERM": "xterm"}))
        self.patch(self.output, "isatty", return_value=True)
        self.console = Console(
            file=self.output,
            width=120,
            color_system=None,
            force_terminal=True,
            force_interactive=True,
            force_jupyter=False,
            legacy_windows=False,
        )
        self.displays = []

        def make_progress(*columns, **options):
            options.update(console=self.console, auto_refresh=False)
            display = getattr(progress, "_Progress", Progress)(*columns, **options)
            self.displays.append(display)
            self.addCleanup(display.stop)
            return display

        self.patch(progress, "Progress", side_effect=make_progress)
        shared = getattr(progress, "_progress", None)
        if shared is not None:
            self.patch(
                progress,
                "_progress",
                new=make_progress(*shared.columns, transient=True),
            )

    def patch(self, target, attribute, **options):
        self.patches.enter_context(patch.object(target, attribute, **options))

    def bar(self, description, **options):
        options = {"total": 100, "mininterval": 0, "miniters": 1, **options}
        bar = progress.TqdmShim(
            desc=description, file=self.output, disable=None, **options
        )
        self.addCleanup(bar.close)
        return bar

    def refresh(self):
        for display in self.displays:
            display.refresh()

    def output_after(self, message):
        self.console.print(message)
        self.refresh()
        return self.output.getvalue().split(message, 1)[1]

    def test_completed_download_does_not_redraw_over_later_messages(self):
        bar = self.bar("download.bin", mininterval=3600)
        bar.update(50)
        bar.refresh()
        self.refresh()
        self.assertIn("50%", self.output.getvalue())
        bar.update(50)
        self.assertNotIn("download.bin", self.output_after("AFTER"))

    def test_closing_earlier_bar_keeps_later_bar_rendering(self):
        first = self.bar("first.bin")
        second = self.bar("second.bin")
        first.close()
        second.update(50)
        after = self.output_after("AFTER")
        self.assertRegex(after, r"second\.bin[^\n]*50%")
        self.assertNotIn("first.bin", after)

    def test_overlapping_bars_keep_counts_units_and_literal_postfix(self):
        download = self.bar(
            "bytes.bin", total=4096, unit="B", unit_scale=True, unit_divisor=1024
        )
        training = self.bar(
            "training",
            total=8,
            unit="batch",
            bar_format="STEP {n}/{total} {unit}{postfix}",
        )
        self.patch(download, "_time", return_value=download.start_t + 1)
        download.update(1024)
        download.set_postfix_str("file=[ok]")
        training.update(2)
        training.set_postfix(loss=0.42)
        self.refresh()
        output = self.output.getvalue()
        self.assertRegex(
            output, r"bytes\.bin[^\n]*1\.00k/4\.00k[^\n]*1\.02kB/s, file=\[ok\]"
        )
        self.assertIn("STEP 2/8 batch, loss=0.42", output)

    def test_custom_formats_render_one_complete_row(self):
        # Hugging Face's Xet formats contain their own description and bar.
        formats = (
            "{desc}: {bar}| {n_fmt:>5}B{postfix:>12}",
            "{l_bar}{bar}| {n_fmt:>5}B / {total_fmt:>5}B{postfix:>12}",
        )
        self.bar("default", initial=25)
        for index, bar_format in enumerate(formats):
            self.bar(
                f"custom-{index}",
                total=4096,
                initial=1024,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                bar_format=bar_format,
                postfix="1.02kB/s [ok]",
            )
        output = "\n".join(
            "".join(segment.text for segment in line)
            for line in self.console.render_lines(
                self.displays[0].get_renderable(), pad=False
            )
        )
        self.assertRegex(output, r"default[^\n]*25%[^\n]*25/100")
        for index in range(2):
            self.assertEqual(output.count(f"custom-{index}"), 1)
        self.assertEqual(output.count("1.00kB"), 2)
        self.assertEqual(output.count("1.02kB/s [ok]"), 2)
        self.assertIn("4.00kB", output)
        # Each custom format includes one text bar, never an extra Rich bar.
        for line in output.splitlines():
            if "custom-" in line:
                self.assertNotIn("━", line)

    def test_context_manager_cancellation_stops_display(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.bar("cancelled.bin") as bar:
                bar.update(25)
                raise KeyboardInterrupt
        self.assertTrue(all(not display.live.is_started for display in self.displays))
        self.assertNotIn("cancelled.bin", self.output_after("RECOVERED"))

    def test_close_recovers_when_task_removal_is_interrupted(self):
        bar = self.bar("interrupted.bin", total=1)
        display = self.displays[0]
        remove_task = display.remove_task

        def interrupted_remove(task_id):
            remove_task(task_id)
            raise KeyboardInterrupt

        with patch.object(display, "remove_task", side_effect=interrupted_remove):
            with self.assertRaises(KeyboardInterrupt):
                bar.update(1)
        bar.close()
        self.assertFalse(display.task_ids)
        self.assertFalse(display.live.is_started)

    def test_initialization_cancellation_stops_unowned_display(self):
        display = self.displays[0]
        for method in ("start", "add_task"):
            with self.subTest(method=method):
                original = getattr(display, method)

                def interrupted(*args, **kwargs):
                    original(*args, **kwargs)
                    raise KeyboardInterrupt

                with patch.object(display, method, side_effect=interrupted):
                    with self.assertRaises(KeyboardInterrupt):
                        self.bar("interrupted.bin")
                self.assertFalse(display.task_ids)
                self.assertFalse(display.live.is_started)


if __name__ == "__main__":
    unittest.main()
