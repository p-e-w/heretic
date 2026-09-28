# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import sys
from contextlib import ExitStack
from functools import wraps
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

        # An existing Rich display may have redirected the stream through FileProxy,
        # whose isatty() does not reflect the terminal behind it.
        file = args[4] if len(args) > 4 else kwargs.get("file")
        file = file if file is not None else sys.stderr
        file = getattr(file, "rich_proxied_file", file)
        if len(args) > 4:
            args = (*args[:4], file, *args[5:])
        else:
            kwargs["file"] = file

        # Chain up to the parent constructor to ensure that the internal state of the superclass
        # is correctly initialized, which some methods that we don't override might rely on.
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

    import huggingface_hub
    from huggingface_hub import _snapshot_download
    from huggingface_hub.utils import tqdm as hub_tqdm

    snapshot_download = _snapshot_download.snapshot_download

    @wraps(snapshot_download)
    def download_with_progress_cleanup(*args: Any, **kwargs: Any):
        if kwargs.get("tqdm_class") is not None:
            return snapshot_download(*args, **kwargs)

        # Hub 1.7.2 does not close its aggregate byte bar. Keep it visible during
        # the download, then close this call's bars even if a worker raises.
        with ExitStack() as bars:

            class SnapshotTqdm(hub_tqdm):
                def __init__(self, *args: Any, **kwargs: Any):
                    super().__init__(*args, **kwargs)
                    bars.callback(self.close)

            kwargs["tqdm_class"] = SnapshotTqdm
            return snapshot_download(*args, **kwargs)

    huggingface_hub.snapshot_download = download_with_progress_cleanup  # ty:ignore[invalid-assignment]
    _snapshot_download.snapshot_download = download_with_progress_cleanup  # ty:ignore[invalid-assignment]
