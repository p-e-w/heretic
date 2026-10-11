# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from typing import Any

import tqdm
import tqdm.auto
import tqdm.std
from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn

_Tqdm = tqdm.tqdm


def _format_number(value: float | None, scale: bool, divisor: int) -> str:
    if value is None:
        return "?"
    if scale:
        return _Tqdm.format_sizeof(value, divisor=divisor)
    return f"{value:g}"


def _format_details(format_dict: dict[str, Any]) -> str:
    n = format_dict["n"]
    total = format_dict["total"]
    rate = format_dict["rate"]
    unit = format_dict["unit"]
    divisor = format_dict["unit_divisor"]
    unit_scale = format_dict["unit_scale"]

    # This is the same unit-scaling rule tqdm uses for its text statistics.
    scale = unit_scale is True or unit_scale == 1
    if unit_scale not in (False, True, 0, 1):
        n *= unit_scale
        total = total * unit_scale if total is not None else None
        rate = rate * unit_scale if rate is not None else None

    n_text = _format_number(n, scale, divisor)
    total_text = _format_number(total, scale, divisor)
    if total is None:
        progress = n_text
        if unit != "it":
            progress += unit if scale else f" {unit}"
    elif scale:
        progress = f"{n_text}{unit} / {total_text}{unit}"
    else:
        progress = f"{n_text} / {total_text}"
        if unit != "it":
            progress += f" {unit}"

    postfix = str(format_dict["postfix"] or "").strip().removeprefix(", ").strip()
    # Some callers provide an aggregate throughput as their postfix. It is more
    # useful than the per-bar rate and avoids displaying two speeds for one row.
    has_rate_postfix = postfix.endswith(f"{unit}/s")
    if rate and not has_rate_postfix:
        if rate < 1:
            rate_text = f"{_format_number(1 / rate, scale, divisor)}s/{unit}"
        else:
            rate_text = f"{_format_number(rate, scale, divisor)}{unit}/s"
    elif not has_rate_postfix:
        rate_text = f"?{unit}/s"
    else:
        rate_text = None

    details = progress if rate_text is None else f"{progress} • {rate_text}"
    if postfix:
        details += f" • {postfix}"
    return details


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
    def __init__(self, *args: Any, **kwargs: Any):
        self.rich_task_id: TaskID | None = None
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
        if self.total is not None and self.n >= self.total:
            self._remove_task()
            return

        format_dict = self.format_dict
        details = _format_details(format_dict)
        if self.rich_task_id is None:
            _progress.start()
            self.rich_task_id = _progress.add_task(
                self.desc or "",
                total=self.total,
                completed=self.n,
                details=details,
            )
        else:
            _progress.update(
                self.rich_task_id,
                description=self.desc or "",
                total=self.total,
                completed=self.n,
                details=details,
            )

    def update(self, n: float = 1) -> bool | None:
        refreshed = super().update(n)
        if self.total is not None and self.n >= self.total:
            with self.get_lock():
                self._remove_task()
        return refreshed

    def clear(self, *args: Any, **kwargs: Any) -> None:
        with self.get_lock():
            self._remove_task()

    def close(self) -> None:
        if self.disable:
            return
        super().close()
        with self.get_lock():
            self._remove_task()


def patch_tqdm() -> None:
    tqdm.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
    tqdm.auto.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
    tqdm.std.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
