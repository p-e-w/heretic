# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from heretic.model import chat_template_supports_system_role, fold_system_message

PLAIN_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
)

# Gemma 2's template rejects the system role the same way.
NO_SYSTEM_ROLE_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}"
    "{% endif %}" + PLAIN_TEMPLATE
)


def make_tokenizer(chat_template: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        chat_template=chat_template,
    )


class ChatTemplateTests(unittest.TestCase):
    def test_detects_system_role_support(self) -> None:
        self.assertTrue(
            chat_template_supports_system_role(make_tokenizer(PLAIN_TEMPLATE))
        )
        self.assertFalse(
            chat_template_supports_system_role(make_tokenizer(NO_SYSTEM_ROLE_TEMPLATE))
        )

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
