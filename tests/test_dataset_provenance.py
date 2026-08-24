# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from heretic.dataset_provenance import get_prompt_content_sha256


class PromptContentHashTests(unittest.TestCase):
    def test_hashes_multiline_and_empty_prompts_canonically(self) -> None:
        sha256 = get_prompt_content_sha256(["alpha\nbeta", "", "gamma"])

        self.assertEqual(
            sha256,
            "4a2a61a12a4d0af07c3aea6ea2514db765db945b43c6e04d3ec3b8d15aaad77a",
        )

    def test_hash_preserves_record_boundaries(self) -> None:
        self.assertNotEqual(
            get_prompt_content_sha256(["ab", "c"]),
            get_prompt_content_sha256(["a", "bc"]),
        )

    def test_hash_preserves_record_order(self) -> None:
        self.assertNotEqual(
            get_prompt_content_sha256(["first", "second"]),
            get_prompt_content_sha256(["second", "first"]),
        )


if __name__ == "__main__":
    unittest.main()
