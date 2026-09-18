# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from heretic.logger import Logger, LoggerEvent

_path_locks: dict[Path, Any] = {}
_path_locks_guard = threading.Lock()


def _get_path_lock(path: Path) -> Any:
    """Return the process-local append lock for a log path.

    Logger instances normally share one process, so this serializes their
    validation-and-append sequence. The canonical path also makes symlinked
    spellings of the same log share one lock; callers should not have multiple
    Heretic processes append to the same JSONL file concurrently.
    """
    key = path.expanduser().resolve(strict=False)
    with _path_locks_guard:
        return _path_locks.setdefault(key, threading.Lock())


def _read_last_nonempty_line(path: Path) -> tuple[bytes, bool]:
    """Return the last non-empty line and whether it was newline-terminated."""
    with path.open("rb") as input_file:
        input_file.seek(0, 2)
        file_size = input_file.tell()
        cursor = file_size

        while cursor > 0:
            line_end = cursor
            search_end = cursor
            line_start = 0

            while search_end > 0:
                chunk_start = max(0, search_end - 8192)
                input_file.seek(chunk_start)
                chunk = input_file.read(search_end - chunk_start)
                newline_index = chunk.rfind(b"\n")
                if newline_index >= 0:
                    line_start = chunk_start + newline_index + 1
                    break
                search_end = chunk_start

            input_file.seek(line_start)
            line = input_file.read(line_end - line_start)
            if line.strip():
                return line, line_end < file_size

            if line_start == 0:
                break
            # Ignore blank lines while searching for the final record.
            cursor = line_start - 1

    return b"", True


def _validate_existing_log(path: Path) -> bool:
    """Validate the final existing record and return whether a separator is needed."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    line, newline_terminated = _read_last_nonempty_line(path)
    if not line:
        return False

    try:
        json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot append to {path}: its final JSONL record is malformed"
        ) from error

    return not newline_terminated


class Settings(BaseModel):
    path: Path = Field(
        description="Path to append one JSON object per generated response."
    )


class JSONL(Logger):
    """Append generated evaluation responses as UTF-8 JSON Lines.

    Each line contains the invocation ``session_id``, the lifecycle ``phase``,
    Optuna's stable ``trial_number`` (or ``null`` for baseline evaluation),
    explicit scorer/dataset/category associations, and the system prompt,
    user prompt, and response text. Responses requested from the same cached
    generation by multiple scorers share one line and retain all associations.
    """

    settings: Settings

    def log(self, event: LoggerEvent) -> None:
        if not event.responses:
            return

        records = []
        for response in event.responses:
            records.append(
                {
                    "session_id": event.session_id,
                    "phase": event.phase,
                    "trial_number": event.trial_number,
                    "scorers": [
                        {
                            "name": source.scorer,
                            "dataset": source.dataset,
                            "category": source.category,
                        }
                        for source in response.sources
                    ],
                    "system": response.prompt.system,
                    "prompt": response.prompt.user,
                    "response": response.response,
                }
            )

        payload = "".join(
            f"{json.dumps(record, ensure_ascii=False)}\n" for record in records
        )
        path = self.settings.path
        with _get_path_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            needs_separator = _validate_existing_log(path)
            with path.open("a", encoding="utf-8", newline="") as output:
                if needs_separator:
                    output.write("\n")
                output.write(payload)
                output.flush()
