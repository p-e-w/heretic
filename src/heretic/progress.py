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
    elapsed = format_dict["elapsed"]
    if rate is None and elapsed:
        rate = (n - format_dict["initial"]) / elapsed
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
    if not total:
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
    if postfix.endswith(f"{unit}/s"):
        rate_text, postfix = postfix, ""
    elif rate:
        value, suffix = (1 / rate, f"s/{unit}") if rate < 1 else (rate, f"{unit}/s")
        rate_text = f"{_format_number(value, scale, 1000)}{suffix}"
    else:
        rate_text = f"?{unit}/s"

    timing = _Tqdm.format_interval(elapsed)
    if total:
        remaining = _Tqdm.format_interval(max(0, (total - n) / rate)) if rate else "?"
        timing += f"<{remaining}"
    extra = f", {postfix}" if postfix else ""
    return f"{progress} [{timing}, {rate_text}{extra}]"


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

        format_dict = self.format_dict
        details = _format_details(format_dict)
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
