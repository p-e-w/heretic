# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import hashlib
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

tests_failed = False

for test_directory in script_directory.iterdir():
    if test_directory.is_dir():
        config_file = test_directory / "config.toml"
        hash_files = list(test_directory.glob("SHA256SUMS.*"))

        if config_file.is_file() and hash_files:
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
                check=True,
            )

            print()

            valid_hashes: dict[str, list[str]] = {}

            for hash_file in hash_files:
                with open(hash_file, "r", encoding="utf-8") as file:
                    for line in file:
                        if line.strip():
                            sha256, filename = line.split()
                            filename = filename.removeprefix("*")

                            if filename not in valid_hashes:
                                valid_hashes[filename] = []

                            valid_hashes[filename].append(sha256.lower())

            for filename in valid_hashes:
                sha256 = get_file_sha256(test_directory / "model" / filename)

                if sha256.lower() not in valid_hashes[filename]:
                    print(
                        (
                            f"Test {test_directory.name} has FAILED!\n"
                            f"Output file {filename} doesn't match any valid hash.\n\n"
                            f"Valid hashes:\n"
                            f"{chr(10).join(valid_hashes[filename])}\n\n"
                            f"Actual hash:\n"
                            f"{sha256}\n"
                        )
                    )
                    tests_failed = True

if tests_failed:
    sys.exit("Tests failed.")
else:
    print("All tests passed.")
