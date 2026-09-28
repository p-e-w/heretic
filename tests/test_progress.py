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

    def test_completed_download_does_not_redraw_over_later_messages(self):
        output = self.render("""
import os
from time import sleep
from rich import print
from heretic.progress import TqdmShim
assert os.isatty(2)
bar = TqdmShim(total=100, desc="weights.bin", disable=None, mininterval=0)
bar.update(50)
sleep(0.2)  # Allow Rich's 10 Hz renderer to emit the updated frame.
print("HALFWAY")
bar.update(50)
print("AFTER")
print("Testing batch size")
bar.close()
""")
        halfway, after = output.split("AFTER", 1)
        self.assertRegex(halfway, r"weights\.bin[^\n]*50%")
        self.assertIn("Testing batch size", after)
        self.assertNotIn("weights.bin", after)

    def test_closing_earlier_bar_keeps_later_bar_rendering(self):
        output = self.render("""
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from rich import print
from heretic.progress import TqdmShim
with ThreadPoolExecutor(max_workers=1) as worker:
    first = worker.submit(TqdmShim, total=100, desc="first.bin").result()
    second = TqdmShim(total=100, desc="second.bin", disable=None, mininterval=0)
    worker.submit(first.close).result()
    second.update(50)
    sleep(0.2)
    print("AFTER")
    second.close()
""")
        after = output.split("AFTER", 1)[1]
        self.assertRegex(after, r"second\.bin[^\n]*50%")
        self.assertNotIn("first.bin", after)

    def test_disabled_bar_does_not_render(self):
        output = self.render("""
from rich import print
from heretic.progress import TqdmShim
with TqdmShim(total=100, desc="hidden", disable=True):
    print("MESSAGE")
""")
        self.assertIn("MESSAGE", output)
        self.assertNotIn("hidden", output)

    def test_incomplete_bar_is_removed_after_exception_or_collection(self):
        output = self.render("""
import gc
from rich import print
from heretic.progress import TqdmShim
try:
    with TqdmShim(total=100, desc="interrupted") as bar:
        bar.update(10)
        raise RuntimeError("interrupted")
except RuntimeError:
    pass
bar = TqdmShim(total=100, desc="unfinished")
bar.update(10)
bar.cycle = bar
del bar
gc.collect()
print("AFTER")
print("Next application message")
""")
        self.assertIn("Next application message", output)
        after = output.split("AFTER", 1)[1]
        self.assertNotIn("unfinished", after)
        self.assertNotIn("interrupted", after)


if __name__ == "__main__":
    unittest.main()
