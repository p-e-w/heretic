# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from threading import RLock
from typing import Any

import tqdm
import tqdm.auto
from rich.progress import Progress, TaskID

_progress = Progress(transient=True)
_progress_lock = RLock()
_active_tasks = 0


# A class that provides the same interface as tqdm,
# but displays progress bars using Rich.
class TqdmShim(tqdm.tqdm):
    def __init__(self, *args: Any, **kwargs: Any):
        self.rich_task_id: TaskID | None = None
        kwargs["gui"] = True
        super().__init__(*args, **kwargs)
        if self.disable:
            return

        global _active_tasks
        with _progress_lock:
            if _active_tasks == 0:
                _progress.start()
            self.rich_task_id = _progress.add_task(self.desc or "", total=self.total)
            _active_tasks += 1
        self.display()

    def display(self, *args: Any, **kwargs: Any):
        if self.rich_task_id is not None:
            _progress.update(
                self.rich_task_id,
                description=self.desc or "",
                total=self.total,
                completed=self.n,
            )

    def close(self, *args: Any, **kwargs: Any):
        global _active_tasks
        try:
            if self.rich_task_id is not None:
                self.display()
            super().close()
        finally:
            with _progress_lock:
                if self.rich_task_id is not None:
                    _progress.remove_task(self.rich_task_id)
                    self.rich_task_id = None
                    _active_tasks -= 1
                    if _active_tasks == 0:
                        _progress.stop()


def patch_tqdm():
    tqdm.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
    tqdm.auto.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
