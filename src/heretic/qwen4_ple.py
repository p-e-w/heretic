# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Disk-backed row lookup for Qwen4-Exp PLE tables.

The Qwen4-Exp checkpoint stores the large n-gram embedding table as a set of
safetensors shards. Loading it as a normal parameter needs ~95 GiB of RAM.
This module keeps the shards memory-mapped and reads only the rows that a
forward pass actually looks up.
"""

import json
import mmap
import shutil
import struct
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}

_model_path: str | None = None
_enabled = False
_installed = False


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as file:
        header_length = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_length))
    return header, header_length


class _ShardedRowTable:
    """Read-only row lookup over one or more safetensors shards."""

    def __init__(
        self,
        model_path: str,
        layer_index: int,
        num_shards: int,
        expected_rows: int,
    ):
        base = Path(model_path)
        if not base.is_dir():
            raise ValueError(
                "Disk-backed PLE requires a local checkpoint directory, "
                f"got {model_path!r}."
            )

        index = json.loads((base / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        prefix = (
            f"model.language_model.layers.{layer_index}"
            ".ple.ple_embedding.ngram_embedding"
        )
        shard_keys = [f"{prefix}.shard_{index}.weight" for index in range(num_shards)]
        if not all(key in index for key in shard_keys):
            single = f"{prefix}.weight"
            if single not in index:
                raise KeyError(f"No PLE embedding tensor found for {prefix!r}.")
            shard_keys = [single]

        self._files = []
        self._maps = []
        self._shards = []
        row_counts = []
        dim = None
        dtype = None
        for key in shard_keys:
            path = base / index[key]
            header, header_length = _read_header(path)
            info = header[key]
            shape = tuple(info["shape"])
            if dim is None:
                dim = shape[1]
                dtype = _DTYPES[info["dtype"]]
            elif shape[1] != dim or _DTYPES[info["dtype"]] != dtype:
                raise ValueError("PLE shards have inconsistent shapes or dtypes.")
            row_counts.append(shape[0])

            file = path.open("rb")
            mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                mapping.madvise(mmap.MADV_RANDOM)
            except (AttributeError, OSError):
                pass

            offset = 8 + header_length + info["data_offsets"][0]
            count = shape[0] * shape[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if info["dtype"] == "BF16":
                    tensor = (
                        torch.frombuffer(
                            mapping,
                            dtype=torch.uint16,
                            count=count,
                            offset=offset,
                        )
                        .view(shape)
                        .view(torch.bfloat16)
                    )
                else:
                    tensor = torch.frombuffer(
                        mapping,
                        dtype=dtype,
                        count=count,
                        offset=offset,
                    ).view(shape)

            self._files.append(file)
            self._maps.append(mapping)
            self._shards.append(tensor)

        if any(count != row_counts[0] for count in row_counts):
            raise ValueError("PLE shards do not have equal row counts.")

        self.rows_per_shard = row_counts[0]
        self.rows = self.rows_per_shard * len(self._shards)
        if self.rows < expected_rows:
            raise ValueError(
                f"PLE table has {self.rows} rows, but the model expects "
                f"{expected_rows}."
            )
        self.dim = dim
        self.dtype = dtype
        # A meta tensor keeps the parent's device check happy without
        # materializing the table.
        self._weight = torch.empty((expected_rows, dim), dtype=dtype, device="meta")

    @property
    def weight(self) -> Tensor:
        return self._weight

    def lookup(self, ids: Tensor) -> Tensor:
        cpu_ids = ids.detach().to("cpu", dtype=torch.long)
        flat = cpu_ids.reshape(-1)
        shard_index = torch.div(flat, self.rows_per_shard, rounding_mode="floor")
        row_index = flat.remainder(self.rows_per_shard)
        output = torch.empty((flat.numel(), self.dim), dtype=self.dtype)
        for shard in torch.unique(shard_index).tolist():
            mask = shard_index == shard
            output[mask] = torch.index_select(self._shards[shard], 0, row_index[mask])
        return output.reshape(*ids.shape, self.dim).to(ids.device)


class _DiskEmbedding(nn.Module):
    def __init__(self, table: _ShardedRowTable):
        super().__init__()
        self._table = table
        # Deliberately not registered as a parameter or buffer: it is never
        # part of the model state dict.
        self._weight = table.weight

    @property
    def weight(self) -> Tensor:
        return self._weight

    def forward(self, input_ids: Tensor) -> Tensor:
        return self._table.lookup(input_ids)


def copy_auxiliary_tensors(model_path: str, save_directory: str) -> None:
    """Copy auxiliary Qwen4 tensors into a saved merged model.

    The PLE embedding and the MTP head are deliberately absent from the
    model's state dict, so a normal ``save_pretrained`` would produce an
    incomplete directory. This copies the original auxiliary shard file(s)
    and adds their keys to the output safetensors index.
    """
    source = Path(model_path)
    output = Path(save_directory)
    source_index_path = source / "model.safetensors.index.json"
    output_index_path = output / "model.safetensors.index.json"
    if not source_index_path.is_file() or not output_index_path.is_file():
        return

    source_map = json.loads(source_index_path.read_text())["weight_map"]
    auxiliary_keys = {
        key: filename
        for key, filename in source_map.items()
        if ".ngram_embedding." in key or key.startswith("mtp.")
    }
    if not auxiliary_keys:
        return

    output_index = json.loads(output_index_path.read_text())
    output_map = output_index.setdefault("weight_map", {})
    new_keys = {
        key: filename
        for key, filename in auxiliary_keys.items()
        if key not in output_map
    }
    if not new_keys:
        return

    destination_names = {}
    for filename in sorted(set(new_keys.values())):
        source_file = source / filename
        if not source_file.is_file():
            raise FileNotFoundError(
                f"Auxiliary tensor file {source_file} does not exist."
            )
        destination_name = f"aux-{source_file.name}"
        destination = output / destination_name
        if not destination.exists():
            shutil.copy2(source_file, destination)
        destination_names[filename] = destination_name

    auxiliary_bytes = 0
    for filename in sorted(set(new_keys.values())):
        header, _ = _read_header(source / filename)
        for key, key_filename in new_keys.items():
            if key_filename == filename:
                start, end = header[key]["data_offsets"]
                auxiliary_bytes += end - start

    for key, filename in new_keys.items():
        output_map[key] = destination_names[filename]

    metadata = output_index.setdefault("metadata", {})
    metadata["total_size"] = int(metadata.get("total_size", 0)) + auxiliary_bytes
    output_index_path.write_text(json.dumps(output_index, indent=2) + "\n")


def install(model_path: str, enabled: bool = False) -> None:
    """Make Qwen4-Exp use a disk-backed PLE embedding.

    Disk-backed loading is opt-in via ``enabled``. Safe to call repeatedly.
    The most recent model path is used for any subsequent model construction.
    """
    global _model_path, _enabled, _installed
    _model_path = model_path

    _enabled = False
    index_path = Path(model_path) / "model.safetensors.index.json"
    if enabled and index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text())["weight_map"]
        except (OSError, KeyError, ValueError):
            weight_map = {}
        _enabled = any("ngram_embedding" in key for key in weight_map)

    if _installed:
        return

    try:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpPreTrainedModel,
            Qwen4ExpTextNGramEmbedding,
        )
    except ImportError:
        # Older Transformers versions don't have Qwen4-Exp.
        return

    original_init = Qwen4ExpTextNGramEmbedding.__init__

    def init(
        self,
        config: Any,
        embedding_dim: int,
        layer_idx: int,
        ple_layer_index: int = 0,
    ):
        if not _enabled:
            original_init(
                self,
                config,
                embedding_dim,
                layer_idx,
                ple_layer_index,
            )
            return

        # The stock embedding would allocate the full table. Build it on the
        # meta device so only its shape survives, then replace it.
        with torch.device("meta"):
            original_init(
                self,
                config,
                embedding_dim,
                layer_idx,
                ple_layer_index,
            )
        rows, _ = self.ngram_embedding.weight.shape
        table = _ShardedRowTable(
            _model_path,
            layer_idx,
            int(getattr(config, "split_ngram_parts", 128)),
            rows,
        )
        self.ngram_embedding = _DiskEmbedding(table)

    Qwen4ExpTextNGramEmbedding.__init__ = init

    ignored = list(
        getattr(Qwen4ExpPreTrainedModel, "_keys_to_ignore_on_load_unexpected", []) or []
    )
    for pattern in (
        r".*ngram_embedding\.shard_\d+\.weight",
        r".*ngram_embedding\.weight",
    ):
        if pattern not in ignored:
            ignored.append(pattern)
    Qwen4ExpPreTrainedModel._keys_to_ignore_on_load_unexpected = ignored

    _installed = True
