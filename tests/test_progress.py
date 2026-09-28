# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

from rich.console import Console
from rich.progress import Progress

from heretic import progress


class TerminalStream(io.StringIO):
    def isatty(self) -> bool:
        return True


class TqdmShimTest(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"TERM": "xterm"})
        environment.start()
        self.addCleanup(environment.stop)
        self.output = TerminalStream()
        self.console = Console(
            file=self.output,
            force_terminal=True,
            force_interactive=True,
            color_system=None,
        )
        self.displays = []

        def make_progress(*args, **kwargs):
            display = Progress(
                *args, console=self.console, auto_refresh=False, **kwargs
            )
            self.displays.append(display)
            return display

        # Intercept Rich's public constructor before loading either shim version.
        # The fixture does not depend on globals introduced by the fix.
        for target, replacement in (
            ("sys.stdout", self.output),
            ("sys.stderr", self.output),
            ("rich.progress.Progress", make_progress),
        ):
            context = patch(target, replacement)
            context.start()
            self.addCleanup(context.stop)
        spec = importlib.util.spec_from_file_location(
            "progress_under_test", progress.__file__
        )
        assert spec is not None and spec.loader is not None
        self.shim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.shim)
        self.addCleanup(self.stop_displays)

    def stop_displays(self):
        for display in reversed(self.displays):
            display.stop()

    def live_displays(self):
        return [display for display in self.displays if display.live.is_started]

    def live_tasks(self):
        return [task for display in self.live_displays() for task in display.tasks]

    def test_disabled_bar_never_starts_display(self):
        with self.shim.TqdmShim(total=1, disable=True):
            self.assertEqual(self.live_displays(), [])

    def test_auto_detected_bar_updates_in_terminal(self):
        with self.shim.TqdmShim(total=100, disable=None, mininterval=0) as bar:
            self.assertFalse(bar.disable)
            bar.update(50)
            bar.set_description("Half downloaded")
            task = self.live_tasks()[0]
            self.assertEqual(task.completed, 50)
            self.assertEqual(task.total, 100)
            self.assertEqual(task.description, "Half downloaded: ")

    def test_auto_detected_bar_stays_disabled_for_nonterminal_file(self):
        with self.shim.TqdmShim(total=1, file=io.StringIO(), disable=None):
            self.assertEqual(self.live_displays(), [])

    def test_second_auto_detected_bar_updates_while_first_is_open(self):
        with (
            self.shim.TqdmShim(total=2, disable=None) as first,
            self.shim.TqdmShim(total=2, disable=None, mininterval=0) as second,
        ):
            self.assertFalse(first.disable)
            self.assertFalse(second.disable)
            second.update(1)
            self.assertEqual([task.completed for task in self.live_tasks()], [0, 1])

    def test_positional_file_argument_keeps_terminal_detection(self):
        with (
            self.shim.TqdmShim(total=1),
            self.shim.TqdmShim(
                None, "positional", 1, True, sys.stderr, disable=None
            ) as bar,
        ):
            self.assertFalse(bar.disable)

    def test_overlapping_threads_can_close_in_creation_order(self):
        first_ready = Event()
        second_ready = Event()
        first_done = Event()

        def first_worker():
            with self.shim.TqdmShim(total=1, desc="first"):
                first_ready.set()
                self.assertTrue(second_ready.wait(timeout=10))
            first_done.set()

        def second_worker():
            self.assertTrue(first_ready.wait(timeout=10))
            with self.shim.TqdmShim(total=2, desc="second", mininterval=0) as bar:
                second_ready.set()
                self.assertTrue(first_done.wait(timeout=10))
                bar.update(1)
                self.assertEqual(len(self.live_displays()), 1)
                tasks = self.live_tasks()
                self.assertEqual([task.description for task in tasks], ["second"])
                self.assertEqual(tasks[0].completed, 1)
                offset = len(self.output.getvalue())
                self.live_displays()[0].refresh()
                self.assertIn("second", self.output.getvalue()[offset:])

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(first_worker)
            second = executor.submit(second_worker)
            first.result(timeout=15)
            second.result(timeout=15)
        self.assertEqual(self.live_displays(), [])

    def test_sequential_bars_stop_and_restore_streams(self):
        for label in ("model", "dataset"):
            with self.shim.TqdmShim(total=1, desc=label, mininterval=0) as bar:
                self.assertTrue(self.live_displays())
                bar.update(1)
            self.assertEqual(self.live_displays(), [])
            self.assertIs(sys.stderr, self.output)
            self.assertIs(sys.stdout, self.output)
            offset = len(self.output.getvalue())
            self.console.print("After download")
            self.assertEqual(self.output.getvalue()[offset:], "After download\n")

    def test_completed_bar_stops_display_without_being_closed(self):
        with self.shim.TqdmShim(total=100, mininterval=0) as bar:
            bar.update(100)
            self.assertEqual(self.live_displays(), [])
            self.assertFalse(bar.disable)
            self.assertEqual(bar.n, 100)
            self.assertIs(sys.stderr, self.output)
            self.assertIs(sys.stdout, self.output)
            offset = len(self.output.getvalue())
            self.console.print("After download")
            self.assertEqual(self.output.getvalue()[offset:], "After download\n")

    def test_completed_bar_resumes_when_total_grows(self):
        with self.shim.TqdmShim(total=100, mininterval=0, miniters=1) as bar:
            bar.update(100)
            self.assertEqual(self.live_displays(), [])
            bar.total = 200
            bar.refresh()
            self.assertEqual(self.live_tasks()[0].completed, 100)
            self.assertEqual(self.live_tasks()[0].total, 200)
            bar.update(50)
            self.assertEqual(self.live_tasks()[0].completed, 150)
            bar.update(50)
            self.assertEqual(self.live_displays(), [])

    def test_completed_bar_can_be_reset(self):
        with self.shim.TqdmShim(total=100, mininterval=0, miniters=1) as bar:
            bar.update(100)
            self.assertEqual(self.live_displays(), [])
            bar.reset(total=200)
            self.assertEqual(self.live_tasks()[0].completed, 0)
            self.assertEqual(self.live_tasks()[0].total, 200)
            bar.update(50)
            self.assertEqual(self.live_tasks()[0].completed, 50)

    def test_final_update_releases_display_despite_tqdm_throttling(self):
        for mininterval, miniters in ((3600, 1), (0, 101)):
            with (
                self.subTest(mininterval=mininterval, miniters=miniters),
                self.shim.TqdmShim(
                    total=100, mininterval=mininterval, miniters=miniters
                ) as bar,
            ):
                bar.update(100)
                self.assertEqual(self.live_displays(), [])
                self.assertFalse(bar.disable)
                self.assertEqual(bar.n, 100)

    def test_zero_total_can_grow_into_visible_progress(self):
        with self.shim.TqdmShim(total=0, mininterval=0) as bar:
            self.assertEqual(self.live_displays(), [])
            bar.total = 100
            bar.refresh()
            bar.update(50)
            self.assertEqual(self.live_tasks()[0].completed, 50)
            self.assertEqual(self.live_tasks()[0].total, 100)

    def test_unknown_total_stays_visible_until_closed(self):
        with self.shim.TqdmShim(total=None, mininterval=0) as bar:
            bar.update(100)
            self.assertEqual(self.live_tasks()[0].completed, 100)
            self.assertIsNone(self.live_tasks()[0].total)
        self.assertEqual(self.live_displays(), [])

    def test_file_bars_keep_independent_byte_progress(self):
        with (
            self.shim.TqdmShim(total=100, desc="first.bin", mininterval=0) as first,
            self.shim.TqdmShim(total=200, desc="second.bin", mininterval=0) as second,
        ):
            first.update(50)
            second.update(75)
            self.assertEqual(
                [
                    (task.description, task.completed, task.total)
                    for task in self.live_tasks()
                ],
                [("first.bin", 50, 100), ("second.bin", 75, 200)],
            )
            first.update(50)
            self.assertEqual(
                [(task.description, task.completed) for task in self.live_tasks()],
                [("second.bin", 75)],
            )
            self.assertFalse(second.disable)

    def test_clear_and_refresh_preserve_progress(self):
        with self.shim.TqdmShim(total=100, mininterval=0) as bar:
            bar.update(50)
            bar.clear()
            bar.clear()
            self.assertEqual(self.live_tasks(), [])
            bar.refresh()
            self.assertEqual(self.live_tasks()[0].completed, 50)
        bar.close()
        self.assertEqual(self.live_displays(), [])

    def test_tqdm_write_restores_cleared_bar(self):
        with self.shim.TqdmShim(total=100, mininterval=0) as bar:
            bar.update(50)
            bar.write("Download message", file=bar.fp)
            self.assertIn("Download message", self.output.getvalue())
            self.assertEqual(self.live_tasks()[0].completed, 50)

    def test_context_manager_closes_on_exception(self):
        with self.shim.TqdmShim(total=2, desc="other work") as unrelated:
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                with self.shim.TqdmShim(total=100, mininterval=0) as bar:
                    bar.update(50)
                    raise RuntimeError("download failed")
            self.assertEqual(
                [task.description for task in self.live_tasks()], ["other work"]
            )
            self.assertFalse(unrelated.disable)
        self.assertEqual(self.live_displays(), [])

    def check_snapshot(self, case):
        # patch_tqdm must run before Hub imports tqdm. Use a fresh interpreter so
        # the integration checks exercise that real import order independently.
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--snapshot", case],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "TQDM_MININTERVAL": "0"},
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_snapshot_keeps_byte_progress_and_stops_display_without_gc(self):
        self.check_snapshot("success")

    def test_snapshot_respects_disabled_progress(self):
        self.check_snapshot("disabled")

    def test_snapshot_preserves_custom_tqdm_class(self):
        self.check_snapshot("custom")


def check_snapshot(case):
    test = TqdmShimTest()
    test.setUp()
    try:
        test.shim.patch_tqdm()
        import huggingface_hub
        from huggingface_hub import HfApi, ModelInfo, _snapshot_download
        from huggingface_hub.utils import disable_progress_bars
        from huggingface_hub.utils import tqdm as hub_tqdm

        if case == "custom":

            class CustomTqdm(hub_tqdm):
                pass

            with patch.object(
                _snapshot_download, "thread_map", return_value=[]
            ) as workers:
                info = ModelInfo(
                    id="test/model",
                    sha="a" * 40,
                    siblings=[{"rfilename": "weights.bin"}],
                )
                with (
                    patch.object(HfApi, "repo_info", return_value=info),
                    tempfile.TemporaryDirectory() as cache,
                ):
                    huggingface_hub.snapshot_download(
                        "test/model", cache_dir=cache, tqdm_class=CustomTqdm
                    )
                test.assertIs(workers.call_args.kwargs["tqdm_class"], CustomTqdm)
            return

        if case == "disabled":
            disable_progress_bars()

        def download(*args, **kwargs):
            with kwargs["tqdm_class"](total=100) as byte_bar:
                byte_bar.update(50)
                if case == "disabled":
                    test.assertEqual(
                        [task.description for task in test.live_tasks()], ["other work"]
                    )
                else:
                    byte_tasks = [
                        task
                        for task in test.live_tasks()
                        if "Downloading" in task.description
                    ]
                    test.assertEqual(
                        len(byte_tasks), 1, "Snapshot byte progress must remain visible"
                    )
                    test.assertEqual(byte_tasks[0].completed, 50)
                    test.assertEqual(byte_tasks[0].total, 100)
                byte_bar.update(50)
            return "weights.bin"

        # Keep an unrelated bar open to check ownership during snapshot cleanup.
        with test.shim.TqdmShim(total=2, desc="other work") as unrelated:
            info = ModelInfo(
                id="test/model", sha="a" * 40, siblings=[{"rfilename": "weights.bin"}]
            )
            with (
                patch.object(HfApi, "repo_info", return_value=info),
                patch.object(
                    _snapshot_download, "hf_hub_download", side_effect=download
                ),
                tempfile.TemporaryDirectory() as cache,
            ):
                result = huggingface_hub.snapshot_download(
                    "test/model", cache_dir=cache
                )
                test.assertEqual(Path(result).name, "a" * 40)
            test.assertEqual(
                [task.description for task in test.live_tasks()], ["other work"]
            )
            test.assertFalse(unrelated.disable)
        test.assertEqual(test.live_displays(), [])
    finally:
        test.doCleanups()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--snapshot":
        check_snapshot(sys.argv[2])
    else:
        unittest.main()
