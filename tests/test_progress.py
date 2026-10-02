# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import errno
import os
import re
import subprocess
import sys
import unittest


class ProgressRenderingTests(unittest.TestCase):
    def render(self, code):
        env = {**os.environ, "TERM": "xterm", "COLUMNS": "120", "PYTHONUTF8": "1"}
        if sys.platform == "win32":
            from winpty import PtyProcess

            process = PtyProcess.spawn(
                [sys.executable, "-c", code], env=env, dimensions=(24, 120)
            )
            process.fileobj.settimeout(30)
            chunks = []
            try:
                while True:
                    try:
                        chunks.append(process.read())
                    except EOFError:
                        break
            finally:
                process.close(force=True)
            output_text = "".join(chunks)
            self.assertEqual(process.exitstatus, 0, output_text)
            return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output_text).replace("\r", "")
        import pty

        master, slave = pty.openpty()
        try:
            try:
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    stdout=slave,
                    stderr=slave,
                    env=env,
                    timeout=30,
                    check=False,
                )
            finally:
                os.close(slave)
            output = bytearray()
            while True:
                try:
                    chunk = os.read(master, 4096)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
                if not chunk:
                    break
                output.extend(chunk)
        finally:
            os.close(master)
        self.assertEqual(result.returncode, 0, output.decode(errors="replace"))
        return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output.decode()).replace("\r", "")

    def test_terminal_initialization_uses_rich_without_tqdm_printer(self):
        output = self.render("""
import sys
from unittest.mock import patch
from rich import print
from heretic.progress import TqdmShim, _progress

assert sys.stderr.isatty()
with patch.object(sys.stderr, "write", wraps=sys.stderr.write) as terminal_write:
    bars = []
    for description, total, options in [
        ("default", 8, {}),
        ("custom", 8, dict(bar_format="STEP {n}/{total} {unit}", leave=False)),
        ("unknown", None, dict(unit="item")),
    ]:
        bar = TqdmShim(total=total, desc=description, disable=None,
                       mininterval=0, miniters=1, nrows=2, **options)
        assert bar.gui is False and bar.sp is None
        assert bar.start_t == bar.last_print_t and bar.n == 0
        assert bar.rich_task_id in _progress.task_ids
        bars.append(bar)
    print("INITIAL")
    _progress.refresh()
    for bar in bars:
        bar.update(2)
    print("UPDATED")
    _progress.refresh()
    bars[0].close()
    assert _progress.live.is_started
    assert _progress.task_ids == [bar.rich_task_id for bar in bars[1:]]
    bars[1].close()
    assert _progress.live.is_started
    assert _progress.task_ids == [bars[2].rich_task_id]
    bars[2].close()
    assert not _progress.task_ids and not _progress.live.is_started
    assert not any(call.args[0] for call in terminal_write.call_args_list)
print("CLOSED")
""")
        initial, updated = output.split("UPDATED", 1)
        self.assertRegex(initial, r"default[^\n]*0/8 \[00:00<\?, \?it/s\]")
        self.assertRegex(initial, r"custom[^\n]*STEP 0/8 it")
        self.assertRegex(initial, r"unknown[^\n]*0item \[00:00, \?item/s\]")
        updated, closed = updated.split("CLOSED", 1)
        self.assertRegex(updated, r"default[^\n]*2/8")
        self.assertRegex(updated, r"custom[^\n]*STEP 2/8 it")
        self.assertRegex(updated, r"unknown[^\n]*2item")
        for description in ("default", "custom", "unknown"):
            self.assertNotIn(description, closed)

    def test_cancelled_bars_do_not_redraw_over_recovered_prompt(self):
        output = self.render("""
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from unittest.mock import patch
from heretic.progress import TqdmShim, close_progress, _progress

TqdmShim.set_lock(RLock())
# Keep the incomplete bars alive, as an interrupted caller's traceback can do.
first = TqdmShim(total=10, desc="cancelled-first", mininterval=0, miniters=1)
second = TqdmShim(total=20, desc="cancelled-second", mininterval=0, miniters=1)
first.update(2)
second.update(3)
_progress.refresh()
retained_tracebacks = []
# Interrupt before and during tqdm's initialization, retaining each failed bar.
for method in ("__init__", "set_postfix"):
    with patch.object(TqdmShim.__bases__[0], method, side_effect=KeyboardInterrupt):
        try:
            TqdmShim(total=10, postfix=dict(status="starting"))
        except KeyboardInterrupt as error:
            retained_tracebacks.append(error.__traceback__)
            close_progress()
# tqdm registers the instance in __new__, before the shim's __init__ runs.
uninitialized = TqdmShim.__new__(TqdmShim)
close_progress()
original_add_task = _progress.add_task
def interrupted_add_task(*args, **kwargs):
    original_add_task(*args, **kwargs)
    _progress.refresh()
    raise KeyboardInterrupt
_progress.add_task = interrupted_add_task
try:
    TqdmShim(total=10, desc="cancelled-constructor")
except KeyboardInterrupt as error:
    retained_tracebacks.append(error.__traceback__)
    close_progress()
finally:
    _progress.add_task = original_add_task
def acquire_tqdm_lock():
    lock = TqdmShim.get_lock()
    acquired = lock.acquire(blocking=False)
    if acquired:
        lock.release()
    return acquired
with ThreadPoolExecutor(max_workers=1) as worker:
    assert worker.submit(acquire_tqdm_lock).result(timeout=5)
assert first.disable and second.disable
assert not _progress.task_ids and not _progress.live.is_started
print("RECOVERED")
first.update(1)
second.update(1)
with TqdmShim(total=10, desc="next-operation", mininterval=0, miniters=1) as bar:
    bar.update(5)
    _progress.refresh()
assert not _progress.task_ids and not _progress.live.is_started
""")
        before, after = output.split("RECOVERED", 1)
        self.assertIn("cancelled-first", before)
        self.assertIn("cancelled-second", before)
        self.assertIn("cancelled-constructor", before)
        self.assertNotIn("cancelled-first", after)
        self.assertNotIn("cancelled-second", after)
        self.assertNotIn("cancelled-constructor", after)
        self.assertRegex(after, r"next-operation[^\n]*50%")

    def test_completed_download_does_not_redraw_over_later_messages(self):
        output = self.render("""
import os
from rich import print
from heretic.progress import TqdmShim
assert os.isatty(2)
bar = TqdmShim(total=100, desc="download.bin", disable=None, mininterval=3600)
# Force frames at the two checkpoints instead of relying on Rich's refresh thread.
progress = getattr(bar, "rich_progress", None)
if progress is None:
    from heretic.progress import _progress as progress
bar.update(50)
bar.refresh()
progress.refresh()
# mininterval=3600 prevents a final refresh; the caller keeps the bar alive.
bar.update(50)
print("AFTER")
print("Testing batch size")
if progress.live.is_started:
    progress.refresh()
bar.close()
""")
        halfway, after = output.split("AFTER", 1)
        self.assertIn("Testing batch size", after)
        self.assertNotIn("download.bin", after)
        self.assertRegex(halfway, r"download\.bin[^\n]*50%")

    def test_closing_earlier_bar_keeps_later_bar_rendering(self):
        output = self.render("""
from concurrent.futures import ThreadPoolExecutor
from rich import print
from heretic.progress import TqdmShim
with ThreadPoolExecutor(max_workers=1) as worker:
    first = worker.submit(TqdmShim, total=100, desc="first.bin").result()
    second = TqdmShim(total=100, desc="second.bin", disable=None, mininterval=0)
    worker.submit(first.close).result()
    second.update(50)
    print("AFTER")
    progress = getattr(second, "rich_progress", None)
    if progress is None:
        from heretic.progress import _progress as progress
    progress.refresh()
    second.close()
""")
        after = output.split("AFTER", 1)[1]
        self.assertRegex(after, r"second\.bin[^\n]*50%")
        self.assertNotIn("first.bin", after)

    def test_counts_units_rates_and_postfix_are_preserved(self):
        output = self.render("""
from rich import print
from heretic.progress import TqdmShim
for description, total, count, options in [
    ("bytes.bin", 4096, 1024, dict(unit="B", unit_scale=True, unit_divisor=1024)),
    ("training", 8, 2, dict(unit="batch")),
    ("unknown", None, 3, dict(unit="item")),
]:
    bar = TqdmShim(total=total, desc=description, mininterval=0, miniters=1, **options)
    # Use one elapsed second so clock resolution cannot hide the numeric rate.
    bar._time = lambda start_t=bar.start_t: start_t + 1
    bar.update(count)
    if total is None:
        bar.set_postfix_str("loss=0.42, note=[ok]")
    else:
        bar.set_postfix(loss=0.42, note="[ok]")
    print("CHECK " + description)
    progress = getattr(bar, "rich_progress", None)
    if progress is None:
        from heretic.progress import _progress as progress
    progress.refresh()
    bar.close()
    print("END " + description)
""")
        for description, counts, unit in [
            ("bytes.bin", "1.00k/4.00k", "B"),
            ("training", "2/8", "batch"),
            ("unknown", "3item", "item"),
        ]:
            with self.subTest(description=description):
                frame = output.split("CHECK " + description, 1)[1].split(
                    "END " + description, 1
                )[0]
                self.assertIn(description, frame)
                self.assertIn(counts, frame)
                self.assertRegex(
                    frame,
                    r"\d[\d.]*[kMGTPEZYmunpf]?"
                    + rf"(?:{unit}/s|s/{unit})[^\n]*loss=0\.42, note=\[ok\]",
                )
                if description != "unknown":
                    self.assertRegex(frame, r"\[\d+:\d+<\d+:\d+,")

    def test_overlapping_bars_keep_their_own_formats(self):
        output = self.render("""
from rich import print
from heretic.progress import TqdmShim, _progress
download = TqdmShim(total=None, desc="bytes.bin", unit="B", unit_scale=True,
                    unit_divisor=1024, mininterval=0, miniters=1)
training = TqdmShim(total=8, desc="training", unit="batch", mininterval=0,
                    miniters=1, bar_format="STEP {n}/{total} {unit}{postfix}")
print("INITIAL")
_progress.refresh()
download.update(1024)
download.set_postfix_str("file=ok")
download.total = 4096
download.refresh()
training.update(2)
training.set_postfix(loss=0.42)
print("BEFORE")
_progress.refresh()
training.close()
print("AFTER")
_progress.refresh()
download.close()
""")
        before, after = output.split("AFTER", 1)
        initial = before.split("BEFORE", 1)[0]
        self.assertRegex(initial, r"training[^\n]*STEP 0/8 batch")
        self.assertRegex(before, r"bytes\.bin[^\n]*1\.00k/4\.00k[^\n]*B/s, file=ok")
        self.assertRegex(before, r"training[^\n]*STEP 2/8 batch, loss=0\.42")
        self.assertNotIn("training", after)
        self.assertIn("1.00k/4.00k", after)
        self.assertIn("file=ok", after)


if __name__ == "__main__":
    unittest.main()
