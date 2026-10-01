# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from threading import RLock
from typing import Any

import tqdm
import tqdm.auto
from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn

# A single live display lets individual bars close in any order.
_progress = Progress(
    TextColumn("{task.description}", style="progress.description", markup=False),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("{task.fields[stats]}", markup=False),
    transient=True,
)
_progress_lock = RLock()


# A class that provides the same interface as tqdm,
# but displays progress bars using Rich.
class TqdmShim(tqdm.tqdm):
    def __init__(self, *args: Any, **kwargs: Any):
        self.rich_task_id: TaskID | None = None
        # Use tqdm's bookkeeping without its terminal printer; Rich handles width.
        kwargs["gui"] = True
        kwargs["dynamic_ncols"] = False

        # Chain up to the parent constructor to ensure that the internal state of the superclass
        # is correctly initialized, which some methods that we don't override might rely on.
        super().__init__(*args, **kwargs)
        self.ncols = None
        # GUI mode skips tqdm's initial refresh, so draw the initial Rich task here.
        self.display()

    def update(self, *args: Any, **kwargs: Any) -> bool | None:
        result = super().update(*args, **kwargs)
        # mininterval/miniters can make tqdm skip display() on the final update.
        if not self.disable:
            self._clear_completed()
        return result

    def display(self, *args: Any, **kwargs: Any):
        with _progress_lock:
            if self.disable:
                return

            # Completion ends the transient display, not the tqdm object's lifetime.
            # A later reset or larger total can make the same bar visible again.
            if self._clear_completed():
                return

            # The total can become known after construction; preserve custom formats.
            format_dict = self.format_dict
            format_dict["bar_format"] = self.bar_format or (
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                if self.total
                else "{n_fmt}{unit} [{elapsed}, {rate_fmt}{postfix}]"
            )
            stats = self.format_meter(**format_dict)

            if self.rich_task_id is None:
                if not _progress.task_ids:
                    _progress.start()
                self.rich_task_id = _progress.add_task(
                    self.desc or "", total=self.total, completed=self.n, stats=stats
                )
            else:
                _progress.update(
                    self.rich_task_id,
                    description=self.desc or "",
                    total=self.total,
                    completed=self.n,
                    stats=stats,
                )

    def _clear_completed(self) -> bool:
        if self.total is not None and self.n >= self.total:
            self.clear()
            return True
        return False

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
        super().close()
        self.clear()


def patch_tqdm():
    tqdm.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
    tqdm.auto.tqdm = TqdmShim  # ty:ignore[invalid-assignment]


def close_progress():
    # An interrupted caller can retain an unfinished bar in its traceback.
    # Snapshot tqdm's registry before closing: close() removes each instance.
    with TqdmShim.get_lock():
        instances = list(TqdmShim._instances)
    for instance in instances:
        if isinstance(instance, TqdmShim):
            instance.close()
    with _progress_lock:
        # A signal can arrive after add_task() but before its ID is stored.
        for task_id in _progress.task_ids:
            _progress.remove_task(task_id)
        _progress.stop()
