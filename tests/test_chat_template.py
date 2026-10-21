# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from heretic.model import Model, chat_template_supports_system_role

# A template that rejects the system role the way Gemma 2's does.
GEMMA_STYLE_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}"
    "{% endif %}"
    "<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)

# A template that renders the system role normally.
SYSTEM_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>\n{{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def _tokenizer(template: str) -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template=template,
    )


class ChatTemplateSupportTests(unittest.TestCase):
    def test_detects_unsupported_system_role(self) -> None:
        self.assertFalse(
            chat_template_supports_system_role(_tokenizer(GEMMA_STYLE_TEMPLATE))
        )

    def test_detects_supported_system_role(self) -> None:
        self.assertTrue(chat_template_supports_system_role(_tokenizer(SYSTEM_TEMPLATE)))


class ApplySystemRoleTests(unittest.TestCase):
    def _model(self, supports: bool) -> Model:
        model = Model.__new__(Model)
        model.supports_system_role = supports
        return model

    def test_passthrough_when_supported(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        # Same object contents, so the rendered chat is byte-identical to before.
        self.assertEqual(self._model(True)._apply_system_role(messages), messages)

    def test_folds_into_first_user_message(self) -> None:
        result = self._model(False)._apply_system_role(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        )
        self.assertEqual(
            result, [{"role": "user", "content": "You are helpful.\n\nhi"}]
        )

    def test_drops_empty_system_prompt(self) -> None:
        result = self._model(False)._apply_system_role(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "hi"},
            ]
        )
        self.assertEqual(result, [{"role": "user", "content": "hi"}])

    def test_no_system_message_is_unchanged(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(self._model(False)._apply_system_role(messages), messages)

    def test_folds_system_role_appearing_later(self) -> None:
        # A system role anywhere (not just leading) must be removed for templates
        # that reject it.
        result = self._model(False)._apply_system_role(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "and now?"},
            ]
        )
        self.assertFalse(any(m["role"] == "system" for m in result))
        self.assertEqual(
            result,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "Be terse.\n\nand now?"},
            ],
        )

    def test_folded_later_system_role_renders_without_raising(self) -> None:
        # End to end: a later system role would make the Gemma-style template raise;
        # after folding it must render fine.
        model = self._model(False)
        tok = _tokenizer(GEMMA_STYLE_TEMPLATE)
        chat = model._apply_system_role(
            [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "and now?"},
            ]
        )
        rendered = tok.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        self.assertIn("Be terse.", rendered)


if __name__ == "__main__":
    unittest.main()
