# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from rich.markup import escape
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from .utils import print


class TensorCheckError(Exception):
    pass


@dataclass(frozen=True)
class Entry:
    id: str
    title: str
    mechanism: str


# Every reason a written name may legitimately differ from the source checkpoint is
# one of these, and each claims a specific name by naming the mechanism that entails
# it. A name that no entry claims is unexplained, and unexplained means rejection.
# Widening a comparison to accommodate a new case is never correct here; adding an
# entry that states its mechanism is.
WHITELIST = {
    entry.id: entry
    for entry in [
        Entry(
            "E1",
            "reported unexpected",
            "the loader reported this exact key in unexpected_keys, so the model never "
            "registered it and the writer will not emit it",
        ),
        Entry(
            "E2",
            "suppressed unexpected",
            "this name matches a pattern transformers applies to unexpected keys but "
            "filters out of its own report, and it is not a live tensor",
        ),
        Entry(
            "E3",
            "newly initialized",
            "the loader reported this exact key in missing_keys, so transformers "
            "initialized it and the writer emits it although the source lacked it",
        ),
        Entry(
            "E4",
            "tie group",
            "this name and that one tie to a single weight; safetensors forbids "
            "aliasing, so which of them is written is transformers' choice",
        ),
        Entry(
            "E5",
            "converter fan-in",
            "this absent source name lies under the module path of an un-locatable "
            "unexpected_keys name, so it fuses into that parameter",
        ),
        Entry(
            "E6",
            "namespace un-locatable",
            "this missing_keys name is in neither header, so it is a model-namespace "
            "name that cannot be an expectation about checkpoint-namespace output",
        ),
        Entry(
            "E7",
            "declined reverse transpose",
            "the model declares a converter whose source and target patterns are "
            "identical and which transposes, and on save the reverse operation finds "
            "the tensor already matching and declines to transpose back",
        ),
    ]
}


@dataclass
class Claim:
    name: str
    entry: str
    detail: str = ""


# Name -> (dtype, shape, nbytes). The dtype is carried for the memory estimate in
# model.py and is never compared: a correct transformers 5.3.0 save of gpt-oss widens
# 48 expert bias tensors from BF16 to F32, so comparing dtype would reject the very
# model this check exists to protect.
CheckpointIndex = dict[str, tuple[str, tuple[int, ...], int]]


@dataclass
class CheckResult:
    verdict: str
    reason: str = ""
    message: str = ""

    source_count: int = 0
    written_count: int = 0

    unclaimed_absent: list[str] = field(default_factory=list)
    unclaimed_extra: list[str] = field(default_factory=list)
    unclaimed_shape: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=list
    )
    claims: list[Claim] = field(default_factory=list)

    def claims_by_entry(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for claim in self.claims:
            counts[claim.entry] = counts.get(claim.entry, 0) + 1
        return counts


def resolve_shards(directory: Path) -> list[Path]:
    # Globbing over-collects: Mistral-7B-Instruct-v0.3 ships a top-level
    # consolidated.safetensors holding 291 tensors in a naming transformers never
    # reads, and a reused output directory can hold a previous run's adapter or shards.
    #
    # save_pretrained only deletes files matching (.*?)-\d{5}-of-\d{5}, so a stale
    # index survives an unsharded save and a stale model.safetensors survives a
    # sharded one. Filtering index entries by existence resolves both directions.
    index_path = directory / SAFE_WEIGHTS_INDEX_NAME
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

    single_file = directory / SAFE_WEIGHTS_NAME
    return [single_file] if single_file.is_file() else []


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise TensorCheckError(f"{path}: truncated length prefix")
        (length,) = struct.unpack("<Q", prefix)
        header = file.read(length)
        if len(header) != length:
            raise TensorCheckError(f"{path}: truncated header")

    return json.loads(header.decode("utf-8"))


def read_checkpoint_index(directory: Path) -> CheckpointIndex:
    index: CheckpointIndex = {}

    for shard in resolve_shards(directory):
        for name, entry in read_safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            index[name] = (
                str(entry["dtype"]),
                tuple(int(dimension) for dimension in entry["shape"]),
                int(end) - int(start),
            )

    return index


@dataclass
class Captures:
    missing_keys: set[str] = field(default_factory=set)
    unexpected_keys: set[str] = field(default_factory=set)
    suppression_patterns: list[str] = field(default_factory=list)
    held_names: set[str] = field(default_factory=set)
    tied_groups: list[set[str]] = field(default_factory=list)
    transpose_patterns: list[str] = field(default_factory=list)


def capture_state(
    model: Any,
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
) -> Captures:
    """Collects everything the whitelist needs from the freshly loaded model.

    Must be called on the bare PreTrainedModel, before PEFT wrapping. A PeftModel
    prefixes every state_dict name with "base_model.model.", which leaves zero overlap
    with the checkpoint namespace and makes all three suppression guards vacuous, and
    its class does not declare _keys_to_ignore_on_load_unexpected at all.
    """
    return Captures(
        missing_keys=set(missing_keys),
        unexpected_keys=set(unexpected_keys),
        suppression_patterns=capture_suppression_patterns(model),
        # Capturing this before any writer runs is what makes the second suppression
        # guard meaningful. A set derived at a call site would be downstream of the
        # component under test.
        held_names=set(model.state_dict().keys()),
        tied_groups=capture_tied_groups(model),
        transpose_patterns=capture_transpose_patterns(model),
    )


def capture_suppression_patterns(model: Any) -> list[str]:
    # list() rather than augmented assignment: += would bind and mutate the
    # transformers class attribute, which is a live list on several models.
    patterns = list(
        getattr(type(model), "_keys_to_ignore_on_load_unexpected", None) or []
    )

    if any(name.endswith("rotary_emb.inv_freq") for name, _ in model.named_buffers()):
        patterns.append(r"rotary_emb\.inv_freq")

    return patterns


def capture_tied_groups(model: Any) -> list[set[str]]:
    # Grouped by shared source rather than per pair. Where three or more names tie to
    # one weight, as in ProphetNetForCausalLM, per-pair grouping is non-transitive and
    # leaves the two targets in separate groups.
    mapping = getattr(model, "all_tied_weights_keys", None) or {}

    groups: dict[str, set[str]] = {}
    for target, source in mapping.items():
        groups.setdefault(source, {source}).add(target)

    return [group for group in groups.values() if len(group) > 1]


def capture_transpose_patterns(model: Any) -> list[str]:
    """Returns the target patterns of any converter that transposes without renaming.

    A predicate read of the conversion mapping, not a reimplementation of the reverse
    conversion. Only qwen3_vl_moe and ernie4_5_vl_moe declare a Transpose at all in
    transformers 5.3.0, and any failure yields an empty list, which forgives nothing.
    """
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import Transpose, WeightConverter
    except Exception:
        return []

    patterns: list[str] = []

    try:
        for converter in get_model_conversion_mapping(model):
            if not isinstance(converter, WeightConverter):
                continue
            operations = getattr(converter, "operations", None) or []
            if not any(isinstance(operation, Transpose) for operation in operations):
                continue

            source_patterns = list(getattr(converter, "source_patterns", None) or [])
            target_patterns = list(getattr(converter, "target_patterns", None) or [])
            if source_patterns and source_patterns == target_patterns:
                patterns.extend(str(pattern) for pattern in target_patterns)
    except Exception:
        return []

    return patterns


def build_expectation(
    source_names: set[str],
    written_names: set[str],
    captures: Captures,
) -> tuple[set[str], list[Claim]]:
    claims: list[Claim] = []

    reported_unexpected = source_names & captures.unexpected_keys
    for name in sorted(reported_unexpected):
        claims.append(Claim(name, "E1", "reported in unexpected_keys"))

    # Transformers drops keys matching these patterns from the report it returns, so a
    # formula resting on unexpected_keys alone rejects a correct save of any checkpoint
    # carrying legacy per-layer rotary_emb.inv_freq keys.
    #
    # All three guards are load-bearing and none subsumes the others. Absence from the
    # artifact alone is self-fulfilling, since the writer is the thing under test and a
    # defect that drops a matched name creates the condition that forgives it. Not
    # being held by the model is fixed before the writer runs, but goes vacuous
    # wherever the model holds the tensor under another name. The module path prefix
    # covers that: fusion preserves the path, so a fused parameter's module is a prefix
    # of every name fusing into it, while a ghost layer is not a module the model has.
    # Both cases are live on Glm4MoeForCausalLM, which declares an ignore pattern for
    # layer 46 that is an ordinary layer on the 92-layer GLM-4.5 and GLM-4.6.
    #
    # inv_freq is exempted from the third guard because GptOssAttention registers sinks
    # directly on self_attn, making it a held module path. Those keys are transformers'
    # own unconditional pattern and name non-persistent buffers, so they cannot be a
    # corruption signal in the first place.
    held_modules = {name.rsplit(".", 1)[0] for name in captures.held_names}
    suppressed = {
        name
        for name in source_names
        if name not in written_names
        and name not in captures.held_names
        and (
            name.endswith("rotary_emb.inv_freq")
            or not any(name.startswith(module + ".") for module in held_modules)
        )
        and any(re.search(pattern, name) for pattern in captures.suppression_patterns)
    }
    for name in sorted(suppressed):
        claims.append(Claim(name, "E2", "matches a suppressed-unexpected pattern"))

    # A missing_keys name absent from both headers is a model-namespace name with no
    # checkpoint-namespace counterpart, so it cannot be an expectation about what the
    # writer emits. Injecting it anyway is what made byte-correct saves of fused MoE
    # checkpoints report absences.
    locatable = captures.missing_keys & (source_names | written_names)
    for name in sorted(locatable):
        claims.append(Claim(name, "E3", "reported in missing_keys, newly initialized"))
    for name in sorted(captures.missing_keys - locatable):
        claims.append(Claim(name, "E6", "in missing_keys but in neither header"))

    expectation = (source_names - reported_unexpected - suppressed) | locatable
    return expectation, claims


def canonicalise(names: set[str], tied_groups: Sequence[set[str]]) -> set[str]:
    result = set(names)

    for group in tied_groups:
        if result & group:
            result -= group
            result.add(min(group))

    return result


def claim_absences(
    absent: set[str],
    source_names: set[str],
    written_names: set[str],
    captures: Captures,
) -> tuple[set[str], list[Claim]]:
    # The basis is restricted to unexpected_keys because the two halves of the load
    # report are not symmetric. An unexpected_keys name describes something the
    # checkpoint has and the model does not, so it is absent from the artifact by
    # construction and no writer can move it in or out of the un-locatable set. A
    # missing_keys name is written on a correct save, so a defect that drops it would
    # otherwise promote its own module path into the basis and excuse every absent
    # sibling under it.
    unlocatable = (captures.missing_keys | captures.unexpected_keys) - (
        source_names | written_names
    )
    basis = unlocatable & captures.unexpected_keys
    modules = {name.rsplit(".", 1)[0] for name in basis}

    claims: list[Claim] = []
    unclaimed: set[str] = set()

    for name in sorted(absent):
        # The prefix relation enforces its own precondition. Fusion preserves the
        # module path only where the converter does not also rename, so on a renaming
        # converter such as the Mixtral family's nothing matches, nothing is claimed,
        # and the name is reported.
        module = next(
            (module for module in modules if name.startswith(module + ".")), None
        )
        if module is None:
            unclaimed.add(name)
        else:
            claims.append(Claim(name, "E5", f"fuses into un-locatable {module}"))

    return unclaimed, claims


def claim_shape(
    name: str,
    source_shape: tuple[int, ...],
    written_shape: tuple[int, ...],
    captures: Captures,
) -> Claim | None:
    # Scoped per name rather than per model, so a model declaring such a converter
    # still has every other tensor's permutation treated as a defect. gpt-oss declares
    # none, which matters because its expert scales are (32, 2880, 90) and
    # (32, 5760, 90): unequal in the transposed dimensions, so a permutation there is
    # visible in the header and is rejected.
    if sorted(source_shape) != sorted(written_shape):
        return None

    for pattern in captures.transpose_patterns:
        try:
            matched = re.search(pattern, name) is not None
        except re.error:
            matched = pattern in name

        if matched:
            return Claim(
                name,
                "E7",
                f"permutation {source_shape} -> {written_shape} on a declared "
                f"transpose converter",
            )

    return None


def compose_failure_message(
    result: CheckResult, context: str, written_directory: Path
) -> str:
    lines = [
        "Tensor name verification failed: the saved model does not carry the names "
        "and shapes of the source checkpoint.",
        "Heretic rewrites values and never renames, so a correct save writes exactly "
        "the source's names with the source's shapes.",
        f"  source names: {result.source_count}, written names: {result.written_count}",
    ]

    if result.unclaimed_shape:
        lines.append(f"  unexplained shape differences: {len(result.unclaimed_shape)}")
        for name, source_shape, written_shape in sorted(result.unclaimed_shape)[:5]:
            lines.append(f"    {name}: {source_shape} -> {written_shape}")

    if result.unclaimed_absent:
        lines.append(f"  expected but not written: {len(result.unclaimed_absent)}")
        for name in sorted(result.unclaimed_absent)[:5]:
            lines.append(f"    {name}")

    # What was written in their place is the half that identifies the defect: the
    # reported one is recognizable on sight, by names ending in _blocks_blocks.
    if result.unclaimed_extra:
        lines.append(f"  written but not expected: {len(result.unclaimed_extra)}")
        for name in sorted(result.unclaimed_extra)[:5]:
            lines.append(f"    {name}")

    # Derived from the ledger rather than enumerated. Hand-written lists of the benign
    # causes of a rejection went stale every time the whitelist changed, either naming
    # a cause that had been closed or omitting one that had been opened.
    fired = result.claims_by_entry()
    if fired:
        lines.append("  whitelist entries that did apply:")
        for entry_id in sorted(fired):
            lines.append(
                f"    {entry_id} {WHITELIST[entry_id].title}: {fired[entry_id]} name(s)"
            )
    else:
        lines.append("  no whitelist entry applied.")

    lines.append(
        "  This is usually a serialization bug in transformers rather than anything "
        "abliteration did. Known defects of this shape affect MXFP4 checkpoints on "
        "5.4.0 through 5.5.1 and from 5.5.2 onward, and vision-language checkpoints "
        "on 5.4.0 through 5.5.4."
    )

    if context == "preflight":
        lines.append(
            "  This was a trial save made before any optimization, so no work has "
            "been lost. Pass --no-preflight-save-check to skip the trial; the "
            "post-save and upload checks are not affected by that flag."
        )
        lines.append(f"  The trial save has been kept at {written_directory}.")
    elif context == "upload":
        lines.append(f"  Nothing was uploaded. The model is at {written_directory}.")
    else:
        lines.append(
            f"  The model at {written_directory} was already written when this was "
            "detected, and has been left alone. Saving is destructive in place, so if "
            "that folder held an earlier model, it is gone."
        )

    return escape("\n".join(lines))


def compose_warning_message(result: CheckResult) -> str:
    lines = [
        f"Tensor name check: the saved model carries {len(result.unclaimed_extra)} "
        "name(s) that the source checkpoint does not.",
        "Every expected name is present at the expected shape, so this is not "
        "corruption; the extra tensors are reported as unexpected on load and ignored.",
        f"  source names: {result.source_count}, written names: {result.written_count}",
    ]

    for name in sorted(result.unclaimed_extra)[:5]:
        lines.append(f"    {name}")
    if len(result.unclaimed_extra) > 5:
        lines.append(f"    ... and {len(result.unclaimed_extra) - 5} more")

    return escape("\n".join(lines))


def check_checkpoint(
    written_directory: Path,
    source_index: Mapping[str, tuple[str, tuple[int, ...], int]],
    captures: Captures,
    context: str,
) -> CheckResult:
    result = CheckResult(verdict="pass")

    if not source_index:
        result.verdict = "unavailable"
        result.reason = (
            "the source checkpoint has no safetensors shards to compare against"
        )
        return result

    written_index = read_checkpoint_index(written_directory)
    result.source_count = len(source_index)
    result.written_count = len(written_index)

    if not written_index:
        result.verdict = "fail"
        result.reason = "no safetensors shards were written"
        result.message = result.reason
        return result

    source_names = set(source_index)
    written_names = set(written_index)

    expectation, claims = build_expectation(source_names, written_names, captures)
    result.claims.extend(claims)

    expected_canonical = canonicalise(expectation, captures.tied_groups)
    written_canonical = canonicalise(written_names, captures.tied_groups)
    for group in captures.tied_groups:
        expected_members = expectation & group
        written_members = written_names & group
        if expected_members and written_members and expected_members != written_members:
            result.claims.append(
                Claim(min(group), "E4", f"tie group {sorted(group)} canonicalised")
            )

    unclaimed_absent, absence_claims = claim_absences(
        expected_canonical - written_canonical, source_names, written_names, captures
    )
    result.claims.extend(absence_claims)
    result.unclaimed_absent = sorted(unclaimed_absent)
    result.unclaimed_extra = sorted(written_canonical - expected_canonical)

    for name in sorted(source_names & written_names):
        _, source_shape, _ = source_index[name]
        _, written_shape, _ = written_index[name]
        if source_shape == written_shape:
            continue

        claim = claim_shape(name, source_shape, written_shape, captures)
        if claim is None:
            result.unclaimed_shape.append((name, source_shape, written_shape))
        else:
            result.claims.append(claim)

    # A tensor written under the right name with the wrong geometry is corruption even
    # though nothing is missing, so shapes are decided first. Extras on their own are
    # not: every expected tensor is present at the expected shape, so the model loads
    # and computes correctly and the extra names are ignored on load.
    if result.unclaimed_shape:
        result.verdict = "fail"
        result.reason = f"{len(result.unclaimed_shape)} unexplained shape difference(s)"
        result.message = compose_failure_message(result, context, written_directory)
    elif result.unclaimed_absent:
        result.verdict = "fail"
        result.reason = (
            f"{len(result.unclaimed_absent)} expected tensor name(s) were not written"
        )
        result.message = compose_failure_message(result, context, written_directory)
    elif result.unclaimed_extra:
        result.verdict = "warn"
        result.reason = f"{len(result.unclaimed_extra)} unexpected extra name(s)"
        result.message = compose_warning_message(result)
    else:
        result.reason = "written names and shapes match the source checkpoint"

    return result


def verify_checkpoint(
    written_directory: Path,
    source_index: Mapping[str, tuple[str, tuple[int, ...], int]],
    captures: Captures,
    context: str = "postsave",
    confirm: Callable[[CheckResult], bool] | None = None,
) -> None:
    result = check_checkpoint(written_directory, source_index, captures, context)

    if result.verdict == "unavailable":
        print(f"* Tensor name check skipped: {result.reason}")
        return

    if result.verdict == "warn":
        print(f"[yellow]{result.message}[/]")
        if confirm is not None and not confirm(result):
            raise TensorCheckError(
                f"Upload cancelled. The model is still at {written_directory}."
            )
        return

    if result.verdict == "fail":
        raise TensorCheckError(result.message)
