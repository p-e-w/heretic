# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from datasets import load_dataset
from optuna.trial import create_trial

from heretic.config import (
    DatasetSpecification,
    HuggingFaceDatasetProvenance,
    Settings,
)
from heretic.dataset_provenance import (
    get_dataset_content_sha256,
    get_dataset_reproducibility_error,
    get_prompt_content_sha256,
)
from heretic.utils import generate_reproduce_json, load_prompts


DATASET_ID = "fka/awesome-chatgpt-prompts"
REVISION = "ca0bf873b687e093f27beaddce8421f92d8ea7b4"
INDICES = [3, 104]
CONTENT_SHA256 = "b4dab93108fc06d1c03fafbbf8c000274c6777964124bccc24681f42e0f41558"


@unittest.skipUnless(
    os.environ.get("HERETIC_RUN_NETWORK_TESTS") == "1",
    "set HERETIC_RUN_NETWORK_TESTS=1 to run the public-dataset integration test",
)
class PublicDatasetProvenanceIntegrationTests(unittest.TestCase):
    def test_rematerializes_sanitized_manifest_with_identical_multiline_prompts(
        self,
    ) -> None:
        source = load_dataset(DATASET_ID, revision=REVISION, split="train")
        materialized = source.select(INDICES)
        self.assertEqual(
            get_dataset_content_sha256(materialized, "prompt"),
            CONTENT_SHA256,
        )
        self.assertTrue(all("\n" in prompt for prompt in materialized["prompt"]))

        provenance = HuggingFaceDatasetProvenance(
            dataset=DATASET_ID,
            revision=REVISION,
            split="train",
            indices=INDICES,
            column="prompt",
            content_sha256=CONTENT_SHA256,
        )

        with TemporaryDirectory() as temp_directory:
            dataset_path = Path(temp_directory) / "materialized"
            materialized.save_to_disk(dataset_path)
            local_specification = DatasetSpecification(
                dataset=str(dataset_path),
                split="train[:]",
                column="prompt",
                provenance=provenance,
            )
            settings = Settings.model_construct(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                system_prompt="You are a helpful assistant.",
                good_prompts=local_specification,
            )

            local_prompts = load_prompts(settings, local_specification)
            self.assertIsNone(
                get_dataset_reproducibility_error(local_specification)
            )

            manifest_contents = generate_reproduce_json(
                settings,
                create_trial(
                    values=[],
                    user_attrs={
                        "direction_index": None,
                        "parameters": {},
                        "scores": [],
                    },
                ),
                timestamp="2026-08-24T00:00:00",
                uploaded_model_hashes={},
                include_system_information=False,
            )
            manifest = json.loads(manifest_contents)

            self.assertEqual(manifest["version"], "4")
            self.assertNotIn(str(dataset_path), manifest_contents)
            self.assertEqual(
                manifest["settings"]["good_prompts"]["dataset"],
                DATASET_ID,
            )

        with patch.object(sys, "argv", ["heretic"]):
            reproduced_settings = Settings.model_validate(manifest["settings"])

        reproduced_specification = reproduced_settings.good_prompts
        reproduced_prompts = load_prompts(
            reproduced_settings,
            reproduced_specification,
        )

        local_prompt_values = [prompt.user for prompt in local_prompts]
        reproduced_prompt_values = [prompt.user for prompt in reproduced_prompts]
        self.assertEqual(reproduced_prompt_values, local_prompt_values)
        self.assertEqual(
            get_prompt_content_sha256(reproduced_prompt_values),
            CONTENT_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
