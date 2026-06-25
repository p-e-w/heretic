# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import hashlib
import os
import subprocess
import sys
from pathlib import Path


# TODO: Replace this with hashlib.file_digest when we drop support for Python 3.10.
def get_file_sha256(file_path: str | Path) -> str:
    hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        # Read the file in 64 kB blocks.
        for block in iter(lambda: file.read(65536), b""):
            hash.update(block)

    return hash.hexdigest()


script_directory = Path(__file__).resolve().parent

project_directory = script_directory.parent

environment = os.environ | {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

for test_directory in script_directory.iterdir():
    if test_directory.is_dir():
        config_file = test_directory / "config.toml"
        hash_file = test_directory / "SHA256SUMS"

        if config_file.is_file() and hash_file.is_file():
            print("#" * 50)
            print(f"Running test {test_directory.name}")
            print("#" * 50)
            print()

            subprocess.run(
                [
                    "uv",
                    "run",
                    "--project",
                    project_directory,
                    "--directory",
                    test_directory,
                    "heretic",
                ],
                env=environment,
                check=True,
            )

            print()

            # To update the hashes after a logic change, run the tests, then execute
            #
            # cd <test_dir>/model
            # sha256sum * > ../SHA256SUMS

            with open(hash_file, "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        original_sha256, filename = line.split()
                        sha256 = get_file_sha256(test_directory / "model" / filename)

                        if sha256.lower() != original_sha256.lower():
                            sys.exit(
                                (
                                    f"Test {test_directory.name} has FAILED!\n"
                                    f"Output file {filename} doesn't match.\n"
                                    f"Expected hash: {original_sha256}\n"
                                    f"Actual hash:   {sha256}"
                                )
                            )

print("All tests passed!")
