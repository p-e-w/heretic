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
        kwargs["dynamic_ncols"] = False

        # Chain up to the parent constructor to ensure that the internal state of the superclass
        # is correctly initialized, which some methods that we don't override might rely on.
        super().__init__(*args, **kwargs)
        self.ncols = None

    @staticmethod
    def status_printer(file: Any) -> None:
        # Rich renders the output; tqdm still initializes and tracks terminal bars.
        return None

    def refresh(self, nolock: bool = False, lock_args: Any = None) -> bool | None:
        if self.disable:
            return None
        if not nolock:
            if lock_args:
                if not self._lock.acquire(*lock_args):
                    return False
            else:
                self._lock.acquire()
        try:
            return super().refresh(nolock=True)
        finally:
            # A cancelled display must not leave tqdm's shared lock held.
            if not nolock:
                self._lock.release()

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
            # The parent can refresh before our constructor finishes; Rich owns width.
            format_dict["ncols"] = None
            format_dict["bar_format"] = self.bar_format or (
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                if self.total
                else "{n_fmt}{unit} [{elapsed}, {rate_fmt}{postfix}]"
            )
            stats = self.format_meter(**format_dict)

            if self.rich_task_id is None:
                previous_tasks = set(_progress.task_ids)
                try:
                    if not previous_tasks:
                        _progress.start()
                    self.rich_task_id = _progress.add_task(
                        self.desc or "", total=self.total, completed=self.n, stats=stats
                    )
                except BaseException:
                    # Task creation can be interrupted before its ID is returned.
                    for task_id in set(_progress.task_ids) - previous_tasks:
                        _progress.remove_task(task_id)
                    if not _progress.task_ids:
                        _progress.stop()
                    raise
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
            task_id = getattr(self, "rich_task_id", None)
            if task_id is not None:
                # Cancellation can interrupt removal before the ID is cleared.
                if task_id in _progress.task_ids:
                    _progress.remove_task(task_id)
                self.rich_task_id = None
                if not _progress.task_ids:
                    _progress.stop()
                else:
                    _progress.refresh()

    def close(self, *args: Any, **kwargs: Any):
        # Rich owns row layout; discard without tqdm rearranging other bars.
        # This also works when cancellation interrupts tqdm's initialization.
        self.disable = True
        with self.get_lock():
            self._instances.discard(self)
        self.clear()


def patch_tqdm():
    tqdm.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
    tqdm.auto.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
