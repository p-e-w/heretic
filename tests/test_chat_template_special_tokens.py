# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generation must match native chat-template tokenization, without extra BOS/EOS."""

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from heretic.config import Settings
from heretic.model import Model
from heretic.utils import Prompt


class ChatTemplateSpecialTokensTests(unittest.TestCase):
    templates = (
        "{{ bos_token }}{% for message in messages %}{{ message['content'] }} {{ eos_token }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}",
        "{% for message in messages %}{{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}",
    )

    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory(prefix="heretic-chat-tokens-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.paths = []
        for eos in (False, True):
            directory = Path(cls.directory.name) / str(eos)
            backend = Tokenizer(
                models.WordLevel(
                    {
                        "<pad>": 0,
                        "<bos>": 1,
                        "<eos>": 2,
                        "<unk>": 3,
                        "hello": 4,
                        "assistant": 5,
                        "system": 6,
                    },
                    unk_token="<unk>",
                )
            )
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            backend.post_processor = processors.TemplateProcessing(
                single="<bos> $A <eos>" if eos else "<bos> $A",
                special_tokens=[("<bos>", 1), ("<eos>", 2)],
            )
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                bos_token="<bos>",
                eos_token="<eos>",
                pad_token="<pad>",
                unk_token="<unk>",
            )
            tokenizer.chat_template = cls.templates[0]
            tokenizer.save_pretrained(directory)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7)
                model = LlamaForCausalLM(
                    LlamaConfig(
                        vocab_size=7,
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        max_position_embeddings=128,
                        bos_token_id=1,
                        eos_token_id=2,
                        pad_token_id=0,
                    )
                )
            model.save_pretrained(directory)
            cls.paths.append(directory)

    def load_model(self, path):
        with patch.object(sys, "argv", ["heretic"]):
            settings = Settings(
                model=str(path),
                dtypes=["float32"],
                device_map="cpu",
                batch_size=2,
                response_prefix="",
            )
        return Model(settings)

    def test_generate_matches_native_chat_tokens_and_padding(self):
        prompts = [
            Prompt(system="system", user="hello"),
            Prompt(system="", user="hello <bos> hello <eos>"),
        ]
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]
        for path in self.paths:
            model = self.load_model(path)
            for template in self.templates:
                with self.subTest(postprocessor=path.name, template=template):
                    model.tokenizer.chat_template = template
                    expected = model.tokenizer.apply_chat_template(
                        chats,
                        tokenize=True,
                        add_generation_prompt=True,
                        padding=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                    inputs, outputs = model.generate(prompts, max_new_tokens=1)
                    torch.testing.assert_close(
                        inputs["input_ids"], expected["input_ids"]
                    )
                    torch.testing.assert_close(
                        inputs["attention_mask"], expected["attention_mask"]
                    )
                    torch.testing.assert_close(
                        outputs[:, : inputs["input_ids"].shape[1]], inputs["input_ids"]
                    )

    def test_streaming_uses_the_same_native_chat_tokens(self):
        chat = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "hello <bos> hello <eos>"},
        ]
        for path in self.paths:
            model = self.load_model(path)
            # Keep the actual native streaming generation bounded to one EOS.
            model.model.generation_config.suppress_tokens = [0, 1, 3, 4, 5, 6]
            for template in self.templates:
                with self.subTest(postprocessor=path.name, template=template):
                    model.tokenizer.chat_template = template
                    expected = model.tokenizer.apply_chat_template(
                        chat,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                    captured = []
                    handle = model.model.register_forward_pre_hook(
                        lambda module, args, kwargs: captured.append(
                            kwargs["input_ids"][0].tolist()
                        ),
                        with_kwargs=True,
                    )
                    try:
                        with redirect_stdout(StringIO()):
                            response = model.stream_chat_response(chat)
                    finally:
                        handle.remove()
                    self.assertEqual(response, "")
                    self.assertEqual(captured, [expected])

    def test_response_prefix_is_appended_without_changing_special_tokens(self):
        prompt = Prompt(system="system", user="hello")
        chat = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        for path in self.paths:
            model = self.load_model(path)
            model.settings.response_prefix = "hello <bos>"
            expected = model.tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            expected += model.tokenizer.encode(
                model.settings.response_prefix, add_special_tokens=False
            )
            inputs, _ = model.generate([prompt], max_new_tokens=1)
            self.assertEqual(inputs["input_ids"][0].tolist(), expected)


if __name__ == "__main__":
    unittest.main()
