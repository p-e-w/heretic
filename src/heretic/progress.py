# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from threading import RLock
from typing import Any

import tqdm
import tqdm.auto
from rich.progress import Progress, TaskID

# A single live display lets individual bars close in any order.
_progress = Progress(transient=True)
_progress_lock = RLock()


# A class that provides the same interface as tqdm,
# but displays progress bars using Rich.
class TqdmShim(tqdm.tqdm):
    def __init__(self, *args: Any, **kwargs: Any):
        self.rich_task_id: TaskID | None = None
        kwargs["gui"] = True

        # Chain up to the parent constructor to ensure that the internal state of the superclass
        # is correctly initialized, which some methods that we don't override might rely on.
        super().__init__(*args, **kwargs)
        self.display()

    def update(self, *args: Any, **kwargs: Any):
        result = super().update(*args, **kwargs)
        # Remove completed tasks even when tqdm throttles the final refresh.
        if not self.disable and self.total is not None and self.n >= self.total:
            self.clear()
        return result

    def display(self, *args: Any, **kwargs: Any):
        with _progress_lock:
            if self.disable:
                return

            # Completion ends the transient display, not the tqdm object's lifetime.
            # A later reset or larger total can make the same bar visible again.
            if self.total is not None and self.n >= self.total:
                self.clear()
                return

            if self.rich_task_id is None:
                if not _progress.task_ids:
                    _progress.start()
                self.rich_task_id = _progress.add_task(
                    self.desc or "", total=self.total, completed=self.n
                )
            _progress.update(
                self.rich_task_id,
                description=self.desc or "",
                total=self.total,
                completed=self.n,
            )

    def clear(self, *args: Any, **kwargs: Any):
        with _progress_lock:
            if self.rich_task_id is not None:
                _progress.remove_task(self.rich_task_id)
                self.rich_task_id = None
                if not _progress.task_ids:
                    _progress.stop()
                else:
                    _progress.refresh()

    def close(self, *args: Any, **kwargs: Any):
        try:
            super().close()
        finally:
            self.clear()


def patch_tqdm():
    tqdm.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
    tqdm.auto.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
