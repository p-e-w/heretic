# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from typing import Any

import tqdm
import tqdm.auto
import tqdm.std
from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn

_Tqdm = tqdm.tqdm


def _new_progress(**kwargs: Any) -> Progress:
    return Progress(
        TextColumn("{task.description}", style="progress.description", markup=False),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[details]}", markup=False),
        transient=True,
        **kwargs,
    )


# A single Live display keeps concurrently-created bars on separate Rich rows.
_progress = _new_progress()


# A class that provides the same interface as tqdm,
# but displays progress bars using Rich.
class TqdmShim(_Tqdm):
    @staticmethod
    def status_printer(file: Any) -> None:
        # Rich owns terminal output; tqdm still performs its normal bookkeeping.
        return None

    def __init__(self, *args: Any, **kwargs: Any):
        self.rich_task_id: TaskID | None = None
        self.rich_total: float | None = None
        super().__init__(*args, **kwargs)

    def _remove_task(self) -> None:
        if self.rich_task_id is None:
            return
        if self.rich_task_id in _progress.task_ids:
            _progress.remove_task(self.rich_task_id)
        self.rich_task_id = None
        if not _progress.task_ids:
            _progress.stop()

    def display(self, *args: Any, **kwargs: Any) -> None:
        if self.disable:
            return
        if self.total and self.n >= self.total:
            self._remove_task()
            return

        format_dict = self.format_dict | {
            "bar_format": (
                "{n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                if self.total
                else "{n_fmt}{unit} [{elapsed}, {rate_fmt}{postfix}]"
            ),
            "ncols": None,
        }
        details = self.format_meter(**format_dict)
        rich_total = self.total or None
        # Rich treats update(total=None) as "leave the total unchanged".
        if rich_total is None and self.rich_total is not None:
            self._remove_task()
        self.rich_total = rich_total
        if self.rich_task_id is None:
            _progress.start()
            self.rich_task_id = _progress.add_task(
                self.desc or "",
                total=rich_total,
                completed=self.n,
                details=details,
            )
        else:
            _progress.update(
                self.rich_task_id,
                description=self.desc or "",
                total=rich_total,
                completed=self.n,
                details=details,
            )

    def update(self, n: float = 1) -> bool | None:
        refreshed = super().update(n)
        if self.total and self.n >= self.total:
            with self.get_lock():
                self._remove_task()
        return refreshed

    def clear(self, *args: Any, **kwargs: Any) -> None:
        with self.get_lock():
            self._remove_task()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.clear()


def patch_tqdm() -> None:
    tqdm.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
    tqdm.auto.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
    tqdm.std.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
