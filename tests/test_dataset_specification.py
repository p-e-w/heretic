# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from heretic.config import SingleDatasetSpecification
from heretic.utils import (
    _is_public_dataset,
    is_dataset_specification_reproducible,
    iter_single_datasets,
)


def _make_dataset_specification(
    name: str, commit: str | None = "commit"
) -> SingleDatasetSpecification:
    return SingleDatasetSpecification(dataset=name, commit=commit)


class IterSingleDatasetsTests(unittest.TestCase):
    """
    A `DatasetSpecification` can hold a single dataset (`SingleDatasetSpecification`)
    or a list of several datasets. The `iter_single_datasets` flattens either form into
    one dataset at a time, in order.
    """

    def test_single_dataset_is_yielded_as_is(self) -> None:
        single_dataset = _make_dataset_specification("org/dataset")

        self.assertEqual(list(iter_single_datasets(single_dataset)), [single_dataset])

    def test_list_of_datasets_is_yielded_in_order(self) -> None:
        first_dataset = _make_dataset_specification("org/first")
        second_dataset = _make_dataset_specification("org/second")

        # Order must be preserved because the model card's `datasets`
        # list is built from the ordered deduplication of dataset IDs.
        self.assertEqual(
            list(iter_single_datasets([first_dataset, second_dataset])),
            [first_dataset, second_dataset],
        )

    def test_empty_list_yields_nothing(self) -> None:
        self.assertEqual(list(iter_single_datasets([])), [])


class IsDatasetSpecificationReproducibleTests(unittest.TestCase):
    """
    A dataset is reproducible only if all of these conditions are true:
      - `is_hf_path`: it is a Hugging Face Hub repository ID, not a local path.
      - `_is_public_dataset`: it is not a private repository.
      - `commit is not None`: it is pinned to a commit hash.

    The checks are short-circuited in that same order.
    """

    def test_pinned_public_dataset_is_reproducible(self) -> None:
        specification = _make_dataset_specification("org/dataset", commit="sha123")

        # Available on HF + public + pinned commit = a reproducible dataset.
        with (
            patch("heretic.utils.is_hf_path", return_value=True),
            patch("heretic.utils._is_public_dataset", return_value=True),
        ):
            self.assertTrue(is_dataset_specification_reproducible(specification))

    def test_unpinned_dataset_is_not_reproducible(self) -> None:
        specification = _make_dataset_specification("org/dataset", commit=None)

        # An unpinned dataset is not reproducible. The `_is_public_dataset` check
        # runs before the commit check, so it is patched to isolate the commit
        # condition and avoid network calls to the Hub.
        with (
            patch("heretic.utils.is_hf_path", return_value=True),
            patch("heretic.utils._is_public_dataset", return_value=True),
        ):
            self.assertFalse(is_dataset_specification_reproducible(specification))

    def test_private_dataset_is_not_reproducible(self) -> None:
        specification = _make_dataset_specification("org/dataset", commit="sha123")

        # A pinned, HF dataset is still not reproducible if it is private.
        with (
            patch("heretic.utils.is_hf_path", return_value=True),
            patch("heretic.utils._is_public_dataset", return_value=False),
        ):
            self.assertFalse(is_dataset_specification_reproducible(specification))

    def test_local_path_is_not_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specification = _make_dataset_specification(directory, commit=None)

            # Local paths are not HF datasets and have no commit hash to pin to.
            # `is_hf_path` returns False for a real local directory, short-circuiting
            # `_is_public_dataset`. A bare "local/path" string would be treated
            # as a valid repository ID unless it actually exists on disk.
            self.assertFalse(is_dataset_specification_reproducible(specification))

    def test_list_requires_every_dataset_to_be_reproducible(self) -> None:
        pinned_dataset = _make_dataset_specification("org/good", commit="sha1")
        unpinned_dataset = _make_dataset_specification("org/unpinned", commit=None)
        another_pinned_dataset = _make_dataset_specification("org/good2", commit="sha2")

        # A list of specifications is reproducible only if every dataset in it is reproducible.
        with (
            patch("heretic.utils.is_hf_path", return_value=True),
            patch("heretic.utils._is_public_dataset", return_value=True),
        ):
            self.assertFalse(
                is_dataset_specification_reproducible(
                    [pinned_dataset, unpinned_dataset]
                )
            )
            self.assertTrue(
                is_dataset_specification_reproducible(
                    [pinned_dataset, another_pinned_dataset]
                )
            )


class IsPublicDatasetTests(unittest.TestCase):
    """Tests `_is_public_dataset`, which calls `huggingface_hub.dataset_info`."""

    # The `_is_public_dataset` check is cached, so clear it before each test
    # to avoid mocked results leaking between the tests.
    def setUp(self) -> None:
        _is_public_dataset.cache_clear()

    def test_public_dataset_is_public(self) -> None:
        # The `dataset_info` is mocked and no real network request is made.
        with (
            patch("heretic.utils.is_hf_path", return_value=True),
            patch(
                "heretic.utils.huggingface_hub.dataset_info",
                return_value=SimpleNamespace(private=False),
            ),
        ):
            self.assertTrue(_is_public_dataset("org/dataset"))

    def test_private_dataset_is_not_public(self) -> None:
        with (
            patch("heretic.utils.is_hf_path", return_value=True),
            patch(
                "heretic.utils.huggingface_hub.dataset_info",
                return_value=SimpleNamespace(private=True),
            ),
        ):
            self.assertFalse(_is_public_dataset("org/dataset"))

    def test_dataset_info_failure_is_not_public(self) -> None:
        # If the dataset cannot be inspected, don't claim it is publicly reproducible.
        with (
            patch("heretic.utils.is_hf_path", return_value=True),
            patch(
                "heretic.utils.huggingface_hub.dataset_info",
                side_effect=RuntimeError("boom"),
            ),
        ):
            self.assertFalse(_is_public_dataset("org/dataset"))

    def test_local_path_is_not_public(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("heretic.utils.huggingface_hub.dataset_info") as dataset_info,
        ):
            self.assertFalse(_is_public_dataset(directory))
            dataset_info.assert_not_called()

    def test_result_is_cached_per_dataset(self) -> None:
        # We use `@cache`, so repeated calls for the same dataset
        # don't request the Hub again.
        with (
            patch("heretic.utils.is_hf_path", return_value=True),
            patch(
                "heretic.utils.huggingface_hub.dataset_info",
                return_value=SimpleNamespace(private=False),
            ) as dataset_info,
        ):
            self.assertTrue(_is_public_dataset("org/dataset"))
            self.assertTrue(_is_public_dataset("org/dataset"))
            self.assertEqual(dataset_info.call_count, 1)


if __name__ == "__main__":
    unittest.main()
