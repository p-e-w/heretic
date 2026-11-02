# SPDX-License-Identifier: AGPL-3.0-or-later
"""Saved datasets must apply the same split instructions as their JSON source."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset, DatasetDict, ReadInstruction, load_dataset

from heretic.config import InitOnlySettings, SingleDatasetSpecification
from heretic.utils import load_prompts


class SavedDatasetSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory(prefix="heretic-splits-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.rows = [f"prompt-{i}" for i in range(6)]
        cls.saved = cls.root / "saved"
        cls.dataset = Dataset.from_dict({"text": cls.rows})
        cls.dataset.save_to_disk(cls.saved)
        cls.source = cls.root / "source.jsonl"
        cls.source.write_text(
            "".join(json.dumps({"text": row}) + "\n" for row in cls.rows),
            encoding="utf-8",
        )
        cls.settings = InitOnlySettings(model="unused", system_prompt="fixture")

    def prompts(self, split):
        return load_prompts(
            self.settings,
            SingleDatasetSpecification(
                dataset=str(self.saved), split=split, column="text"
            ),
        )

    def test_combined_instructions_match_native_json_loading(self):
        for split in [
            "train[:2]+train[4:]",
            "train[4:]+train[:2]",
            "train[:2]+train[1:3]",
            "train+train",
            "train[:50%]+train[83%:]",
            "train[-2:]+train[:1]",
            "train[2:2]+train[4:5]",
            "train",
            "train[:2]",
            "train[-2:]",
        ]:
            with self.subTest(split=split):
                expected = list(
                    load_dataset(
                        "json",
                        data_files={"train": str(self.source)},
                        split=split,
                        cache_dir=str(self.root / "cache"),
                    )["text"]
                )
                actual = self.prompts(split)
                self.assertEqual([prompt.user for prompt in actual], expected)
                self.assertTrue(all(prompt.system == "fixture" for prompt in actual))

    def test_empty_single_selection_remains_empty(self):
        self.assertEqual(self.prompts("train[2:2]"), [])

    def test_invalid_or_unknown_split_raises_native_error(self):
        for split in ["train[not-a-slice]", "train[:1]+test[:1]"]:
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.prompts(split)

    def test_plain_text_single_slice_keeps_existing_behavior(self):
        text = self.root / "prompts.txt"
        text.write_text("\n".join(self.rows), encoding="utf-8")
        actual = load_prompts(
            self.settings, SingleDatasetSpecification(dataset=str(text), split="[1:3]")
        )
        self.assertEqual([prompt.user for prompt in actual], self.rows[1:3])

    def test_accepted_plain_text_instructions_match_native_resolution(self):
        text = self.root / "union-prompts.txt"
        text.write_text("\n".join(self.rows), encoding="utf-8")
        for split in [
            "[:2]+_[4:]",
            "[4:]+_[:2]",
            "[:2]+_[1:3]",
            "[:50%]+_[83%:]",
            "[-2:]+_[:1]",
            "[2:2]",
            "[1:3]",
        ]:
            with self.subTest(split=split):
                native = ReadInstruction.from_spec("_" + split).to_absolute(
                    {"_": len(self.rows)}
                )
                expected = [
                    prompt
                    for span in native
                    for prompt in self.rows[span.from_ : span.to]
                ]
                actual = load_prompts(
                    self.settings,
                    SingleDatasetSpecification(dataset=str(text), split=split),
                )
                self.assertEqual([prompt.user for prompt in actual], expected)

    def test_unsupported_plain_text_syntax_keeps_native_errors(self):
        text = self.root / "invalid-prompts.txt"
        text.write_text("\n".join(self.rows), encoding="utf-8")
        for split in ["[:2]+[4:]", "[:2]+other[4:]"]:
            with self.subTest(split=split), self.assertRaises(ValueError):
                load_prompts(
                    self.settings,
                    SingleDatasetSpecification(dataset=str(text), split=split),
                )

    def test_dataset_dict_directory_keeps_its_existing_generic_loading(self):
        directory = self.root / "dict"
        DatasetDict({"train": self.dataset}).save_to_disk(directory)
        actual = load_prompts(
            self.settings,
            SingleDatasetSpecification(
                dataset=str(directory), split="train", column="text"
            ),
        )
        self.assertEqual([prompt.user for prompt in actual], self.rows)


if __name__ == "__main__":
    unittest.main()
