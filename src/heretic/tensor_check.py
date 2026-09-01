# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from pathlib import Path

from huggingface_hub import get_local_safetensors_metadata, snapshot_download
from rich.markup import escape

from .utils import print


def tensor_shapes(location: str) -> dict[str, tuple[int, ...]]:
    try:
        if not Path(location).is_dir():
            location = snapshot_download(location, local_files_only=True)
        metadata = get_local_safetensors_metadata(location)
    except Exception:
        return {}
    return {
        name: tuple(info.shape)
        for file in metadata.files_metadata.values()
        for name, info in file.tensors.items()
    }


def check_tensors(source: dict[str, tuple[int, ...]], directory: str) -> bool:
    if not source:
        return True
    written = tensor_shapes(directory)
    if not written:
        print("[yellow]* Model has no readable safetensors weights[/]")
        return False
    names = sorted(set(source) | set(written))
    differences = [
        f"{name}: {source.get(name, 'absent')} -> {written.get(name, 'absent')}"
        for name in names
        if source.get(name) != written.get(name)
    ]
    if not differences:
        print("* Model matches the source checkpoint")
        return True
    print(
        f"[yellow]* Model differs from the source checkpoint at [bold]"
        f"{len(differences)}[/] of {len(names)} names (source -> saved):[/]"
    )
    for difference in differences[:10]:
        print(f"[yellow]    {escape(difference)}[/]")
    return False
