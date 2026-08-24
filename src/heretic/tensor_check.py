# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

from huggingface_hub import parse_local_safetensors_file_metadata
from huggingface_hub.constants import SAFETENSORS_INDEX_FILE, SAFETENSORS_SINGLE_FILE
from rich.markup import escape

# Name -> (shape, nbytes). nbytes is carried for the preflight sizing in model.py.
# The index deliberately carries no dtype: a correct transformers 5.3.0 save of
# gpt-oss widens 48 expert bias dtypes, so comparing dtype would flag the very
# model this check exists to protect.
CheckpointIndex = dict[str, tuple[tuple[int, ...], int]]


class Verdict(str, Enum):
    CLEAN = "clean"
    DIFFERS = "differs"
    NO_WEIGHTS = "no_weights"
    UNAVAILABLE = "unavailable"


class CheckSite(str, Enum):
    PREFLIGHT = "preflight"
    POST_SAVE = "post_save"
    UPLOAD = "upload"


@dataclass
class CheckResult:
    verdict: Verdict
    reason: str = ""

    source_count: int = 0
    written_count: int = 0

    # Names in the source and not in the artifact, and the reverse, both sorted.
    absent: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=list
    )

    def describe(self, site: CheckSite, written_directory: Path) -> str:
        if self.verdict == Verdict.NO_WEIGHTS:
            lines = [
                f"{self.reason.capitalize()}. Configuration files may exist, "
                "but no weights do.",
            ]
        else:
            lines = [
                "Heretic rewrites tensor values and never renames, so a correct "
                "save writes exactly the source checkpoint's names and shapes; "
                "this save did not.",
            ]

        lines.append(
            f"  source names: {self.source_count}, written names: {self.written_count}"
        )

        shape_samples = [
            f"{name}: {source_shape} -> {written_shape}"
            for name, source_shape, written_shape in self.shape_mismatches
        ]
        for label, entries in (
            ("in the source checkpoint but not in the saved model", self.absent),
            ("in the saved model but not in the source checkpoint", self.extra),
            ("written under the same name with a different shape", shape_samples),
        ):
            if entries:
                lines.append(f"  {label}: {len(entries)}")
                lines.extend(f"    {entry}" for entry in entries[:5])

        lines.append(
            "  This may indicate a serialization bug in transformers, or a benign "
            "difference intended by transformers; either way, it is not anything "
            "abliteration did."
        )

        if site == CheckSite.PREFLIGHT:
            lines.append(
                "  This preflight save cost no optimization time. It is at "
                f"{written_directory}: kept if the run stops here, removed if "
                "you choose to continue."
            )
        elif site == CheckSite.POST_SAVE:
            lines.append(
                f"  The model at {written_directory} was already written when "
                "this was detected. Saving is destructive in place, so if that "
                "folder held an earlier model, it is gone."
            )
        else:
            lines.append(
                f"  Nothing has been uploaded. The model is at {written_directory}."
            )

        # Heretic's printer parses square brackets as markup, so an unescaped
        # tensor-name list would be silently swallowed.
        return escape("\n".join(lines))


def resolve_shards(directory: Path) -> list[Path]:
    # Globbing over-collects: Mistral-7B-Instruct-v0.3 ships a top-level
    # consolidated.safetensors holding 291 tensors in a naming transformers never
    # reads, and a reused output directory can hold a previous run's adapter or shards.
    #
    # save_pretrained only deletes files matching (.*?)-\d{5}-of-\d{5}, so a stale
    # index survives an unsharded save and a stale model.safetensors survives a
    # sharded one. Filtering index entries by existence resolves both directions.
    index_path = directory / SAFETENSORS_INDEX_FILE
    if index_path.is_file():
        try:
            with index_path.open("r", encoding="utf-8") as file:
                weight_map = json.load(file)["weight_map"]
            names = {str(shard) for shard in weight_map.values()}
        except Exception:
            names = set()

        shards = sorted(
            {directory / name for name in names if (directory / name).is_file()},
            key=lambda path: path.name,
        )
        if shards:
            return shards

    single_file = directory / SAFETENSORS_SINGLE_FILE
    return [single_file] if single_file.is_file() else []


def read_checkpoint_index(directory: Path) -> CheckpointIndex:
    return {
        name: (tuple(info.shape), info.data_offsets[1] - info.data_offsets[0])
        for shard in resolve_shards(directory)
        for name, info in parse_local_safetensors_file_metadata(shard).tensors.items()
    }


def compare_checkpoint(
    written_directory: Path,
    source_index: Mapping[str, tuple[tuple[int, ...], int]],
) -> CheckResult:
    if not source_index:
        return CheckResult(
            verdict=Verdict.UNAVAILABLE,
            reason="no source checkpoint index to compare against",
        )

    written_index = read_checkpoint_index(written_directory)
    result = CheckResult(
        verdict=Verdict.CLEAN,
        source_count=len(source_index),
        written_count=len(written_index),
    )

    if not written_index:
        result.verdict = Verdict.NO_WEIGHTS
        result.reason = "no safetensors shards were written"
        return result

    source_names = set(source_index)
    written_names = set(written_index)

    result.absent = sorted(source_names - written_names)
    result.extra = sorted(written_names - source_names)

    result.shape_mismatches = [
        (name, source_index[name][0], written_index[name][0])
        for name in sorted(source_names & written_names)
        if source_index[name][0] != written_index[name][0]
    ]

    if result.absent or result.extra or result.shape_mismatches:
        result.verdict = Verdict.DIFFERS

    return result
