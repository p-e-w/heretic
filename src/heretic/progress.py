# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from typing import Any

import tqdm
import tqdm.auto
from rich.progress import Progress


# A class that provides the same interface as tqdm,
# but displays progress bars using Rich.
class TqdmShim(tqdm.tqdm):
    def __init__(self, *args: Any, **kwargs: Any):
        self.rich_progress = Progress(transient=True)
        self.rich_progress.start()
        self.rich_task_id = self.rich_progress.add_task(
            kwargs.get("desc", ""),
            total=kwargs.get("total", None),
        )

        # Chain up to the parent constructor to ensure that the internal state of the superclass
        # is correctly initialized, which some methods that we don't override might rely on.
        super().__init__(*args, **kwargs)

    def display(self, *args: Any, **kwargs: Any):
        self.rich_progress.update(
            self.rich_task_id,
            description=self.desc,
            total=self.total,
            completed=self.n,
        )

    def close(self, *args: Any, **kwargs: Any):
        self.rich_progress.stop()


def patch_tqdm():
    tqdm.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
    tqdm.auto.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
