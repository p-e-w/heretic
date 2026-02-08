# Style guide and coding conventions

* Identifier names should not contain abbreviations unless those abbreviations are very widely used and understood (e.g. "KL divergence").
* Comments should start with a capital letter and end with a period. They should use correct grammar and spelling.
* Function and method signatures **must** be fully type-annotated, including the return type (if any).
* Every Python code file **must** start with an SPDX/Copyright header.
* Settings descriptions should start with a capital letter and end with a period.
* When new settings are added in `config.py`, they should also be added to `config.default.toml`, set to their default value and with their description as a comment. The order of settings in `config.default.toml` should match that in `config.py`.
* Pull requests should implement one change, and one change only.
  * PRs containing multiple semantically independent changes **must** be split into multiple PRs.
  * PRs **must not** change existing code unless the changes are *directly related* to the PR. This includes changes to formatting and comments.
