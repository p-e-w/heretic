# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

from .utils import print


def collect_reproducibles(path: str):
    print(
        f"Collecting [bold]reproduce.json[/] files from Hugging Face and storing them in [bold]{path}[/]..."
    )
    print()

    api = HfApi()

    models = api.list_models(
        filter=["heretic", "reproducible"],
        sort="created_at",
    )

    found = 0
    downloaded = 0

    # We're only downloading tiny files, so the progress bars are just noise.
    disable_progress_bars()

    try:
        for model in models:
            # Ignore repositories containing quantizations.
            if model.tags is not None and "gguf" in model.tags:
                continue

            print(f"[bold]{model.id}[/]...", end="")

            user, repository = model.id.split("/")

            paths_info = api.get_paths_info(
                model.id,
                "reproduce/reproduce.json",
                expand=True,
            )
            # The reproduce.json file might not exist in the repository
            # despite the relevant tags being present.
            if not paths_info:
                print(" [yellow]no reproduce.json found[/]")
                continue

            found += 1

            commit_hash = paths_info[0].last_commit.oid

            file_path = (
                Path(path)
                / "huggingface.co"
                / user
                / f"{repository}-{commit_hash[:7]}.json"
            )
            if file_path.exists():
                print(" already stored")
                continue

            cache_path = hf_hub_download(
                model.id,
                "reproduce/reproduce.json",
            )

            file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache_path, file_path)
            print(" [green]downloaded[/]")

            downloaded += 1
    finally:
        enable_progress_bars()

    print()
    print(f"Found: [bold]{found}[/] files")
    print(f"Downloaded: [bold]{downloaded}[/] files")
    print(f"Already stored: [bold]{found - downloaded}[/] files")
