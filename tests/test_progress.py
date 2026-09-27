# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import io
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

from rich.console import Console
from rich.progress import Progress

from heretic import progress


class TqdmShimTest(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.rich_progress = Progress(
            console=Console(file=self.output, force_terminal=False),
            auto_refresh=False,
            transient=True,
        )
        self.progress_patch = patch.object(progress, "_progress", self.rich_progress)
        self.progress_patch.start()
        self.addCleanup(self.progress_patch.stop)

    def test_concurrent_bars_share_one_live_display(self):
        ready = Barrier(3)
        done = Barrier(3)

        def run_bar(label):
            with progress.TqdmShim(total=1, desc=label, mininterval=0) as bar:
                ready.wait(timeout=15)
                done.wait(timeout=15)
                bar.update(1)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(run_bar, "model")
            second = executor.submit(run_bar, "dataset")
            ready.wait(timeout=15)
            self.assertTrue(self.rich_progress.live.is_started)
            self.assertEqual(
                {task.description for task in self.rich_progress.tasks},
                {"model", "dataset"},
            )
            done.wait(timeout=15)
            first.result(timeout=15)
            second.result(timeout=15)

        self.assertFalse(self.rich_progress.live.is_started)
        self.assertEqual(self.rich_progress.tasks, [])
        self.assertEqual(progress._active_tasks, 0)
        self.assertEqual(self.output.getvalue().strip(), "")
        self.assertLessEqual(self.output.getvalue().count("\n"), 1)

    def test_disabled_bar_does_not_start_display(self):
        with progress.TqdmShim(total=1, disable=True):
            pass

        self.assertFalse(self.rich_progress.live.is_started)
        self.assertEqual(progress._active_tasks, 0)

    def test_auto_detected_bar_uses_original_terminal_stream(self):
        class TerminalStream(io.StringIO):
            def isatty(self) -> bool:
                return True

        terminal_progress = Progress(
            console=Console(file=self.output, force_terminal=True),
            auto_refresh=False,
            transient=True,
        )
        with (
            patch.object(progress, "_progress", terminal_progress),
            patch("sys.stderr", TerminalStream()),
            progress.TqdmShim(
                total=0, desc="Downloading (incomplete total...)", disable=None
            ) as bar,
        ):
            self.assertFalse(bar.disable)
            bar.set_description("Download complete")
            self.assertEqual(
                terminal_progress.tasks[0].description, "Download complete: "
            )

        self.assertFalse(terminal_progress.live.is_started)
        self.assertEqual(terminal_progress.tasks, [])
        self.assertEqual(progress._active_tasks, 0)

    def test_sequential_bars_restart_display(self):
        for label in ("model", "dataset"):
            with progress.TqdmShim(total=1, desc=label) as bar:
                self.assertTrue(self.rich_progress.live.is_started)
                bar.update(1)

            self.assertFalse(self.rich_progress.live.is_started)
            self.assertEqual(self.rich_progress.tasks, [])
            self.assertEqual(progress._active_tasks, 0)
