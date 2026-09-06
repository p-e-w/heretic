# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from pathlib import Path

import huggingface_hub
from rich.markup import escape

from .utils import print


def tensor_shapes(
    location: str, revision: str | None = None, token: str | None = None
) -> dict[str, list[int]]:
    try:
        if token is not None:
            metadata = huggingface_hub.get_safetensors_metadata(location, token=token)
        else:
            snapshot = location
            if not Path(location).is_dir():
                snapshot = huggingface_hub.snapshot_download(
                    location, revision=revision, local_files_only=True
                )
            metadata = huggingface_hub.get_local_safetensors_metadata(snapshot)
    except Exception as error:
        print(f"* Could not read {location}: {error}", markup=False)
        return {}
    return {
        name: info.shape
        for file in metadata.files_metadata.values()
        for name, info in file.tensors.items()
    }


def check_tensors(
    source: dict[str, list[int]], location: str, token: str | None = None
) -> dict[str, list[list[int] | None]] | None:
    if not source:
        return None
    written = tensor_shapes(location, token=token)
    if not written:
        return None
    names = sorted(set(source) | set(written))
    differences = {
        name: [source.get(name), written.get(name)]
        for name in names
        if source.get(name) != written.get(name)
    }
    if differences:
        print(
            f"[yellow]* Model differs from the source checkpoint at [bold]"
            f"{len(differences)}[/] of {len(names)} names (source -> saved):[/]"
        )
        for name in list(differences)[:10]:
            before, after = source.get(name, "absent"), written.get(name, "absent")
            print(f"[yellow]    {escape(f'{name}: {before} -> {after}')}[/]")
    else:
        print("* Model matches the source checkpoint")
    return differences
