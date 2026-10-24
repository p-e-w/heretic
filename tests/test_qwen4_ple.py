# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from heretic import qwen4_ple
from heretic.qwen4_ple import copy_auxiliary_tensors


class Qwen4AuxiliaryTensorTests(unittest.TestCase):
    def test_copies_ple_and_mtp_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "output"
            source.mkdir()
            output.mkdir()

            ple_key = (
                "model.language_model.layers.1.ple.ple_embedding."
                "ngram_embedding.shard_0.weight"
            )
            mtp_key = "mtp.layers.0.mlp.experts.0.down_proj.weight"
            save_file(
                {ple_key: torch.zeros((2, 4), dtype=torch.bfloat16)},
                source / "ple.safetensors",
            )
            save_file(
                {mtp_key: torch.ones((3, 5), dtype=torch.bfloat16)},
                source / "mtp.safetensors",
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            ple_key: "ple.safetensors",
                            mtp_key: "mtp.safetensors",
                        }
                    }
                )
            )
            (output / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {"model.foo": "model-00001.safetensors"},
                        "metadata": {"total_size": 100},
                    }
                )
            )

            copy_auxiliary_tensors(str(source), str(output))

            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(
                index["weight_map"][ple_key],
                "aux-ple.safetensors",
            )
            self.assertEqual(
                index["weight_map"][mtp_key],
                "aux-mtp.safetensors",
            )
            self.assertEqual(index["metadata"]["total_size"], 146)

            with safe_open(
                str(output / "aux-ple.safetensors"), framework="pt"
            ) as tensors:
                self.assertEqual(list(tensors.keys()), [ple_key])
            with safe_open(
                str(output / "aux-mtp.safetensors"), framework="pt"
            ) as tensors:
                self.assertEqual(list(tensors.keys()), [mtp_key])


class Qwen4PleOptInTests(unittest.TestCase):
    def test_disk_backing_requires_opt_in(self) -> None:
        key = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight"
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {key: "ple.safetensors"}})
            )

            qwen4_ple.install(directory)
            self.assertFalse(qwen4_ple._enabled)

            qwen4_ple.install(directory, enabled=True)
            self.assertTrue(qwen4_ple._enabled)

        qwen4_ple.install(directory)


if __name__ == "__main__":
    unittest.main()
