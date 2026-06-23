# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import hashlib
import subprocess
import sys
from pathlib import Path


# TODO: Replace this with hashlib.file_digest when we drop support for Python 3.10.
def get_file_sha256(file_path: str | Path) -> str:
    file_path = Path(file_path)
    hash_obj = hashlib.sha256()

    # Common text / config file extensions used by Hugging Face models.
    text_extensions = {".json", ".jinja", ".txt", ".py", ".md", ".yaml"}

    if file_path.suffix.lower() in text_extensions:
        # For text files, read as binary but normalize \r\n down to \n in memory.
        with open(file_path, "rb") as file:
            content = file.read()
            normalized_content = content.replace(b"\r\n", b"\n")
            hash_obj.update(normalized_content)
    else:
        # For binary files (like .safetensors), read raw bytes.
        with open(file_path, "rb") as file:
            for block in iter(lambda: file.read(65536), b""):
                hash_obj.update(block)

    return hash_obj.hexdigest()


script_directory = Path(__file__).resolve().parent

project_directory = script_directory.parent

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
                        filename = filename.removeprefix("*")
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
