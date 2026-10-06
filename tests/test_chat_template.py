# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from jinja2.exceptions import TemplateError

from heretic.model import chat_template_supports_system_role, fold_system_message


class FakeTokenizer:
    def __init__(self, supports_system_role: bool):
        self.supports_system_role = supports_system_role

    def apply_chat_template(self, chat, **kwargs):
        if not self.supports_system_role and chat[0]["role"] == "system":
            raise TemplateError("System role not supported")
        return ""


class ChatTemplateTests(unittest.TestCase):
    def test_detects_system_role_support(self) -> None:
        self.assertTrue(chat_template_supports_system_role(FakeTokenizer(True)))
        self.assertFalse(chat_template_supports_system_role(FakeTokenizer(False)))

    def test_folds_system_prompt_into_first_user_message(self) -> None:
        chat = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ]

        self.assertEqual(
            fold_system_message(chat),
            [
                {"role": "user", "content": "Be brief.\n\nHi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Bye"},
            ],
        )

    def test_drops_empty_system_prompt(self) -> None:
        chat = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hi"},
        ]

        self.assertEqual(fold_system_message(chat), [{"role": "user", "content": "Hi"}])

    def test_leaves_chat_without_system_message_unchanged(self) -> None:
        chat = [{"role": "user", "content": "Hi"}]

        self.assertEqual(fold_system_message(chat), chat)


if __name__ == "__main__":
    unittest.main()
