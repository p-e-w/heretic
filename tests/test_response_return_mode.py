import tempfile
import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2MoeConfig, Qwen2MoeForCausalLM
from transformers.generation.utils import GenerateDecoderOnlyOutput

from heretic.config import InitOnlySettings
from heretic.model import Model
from heretic.utils import Prompt


class ResponseReturnModeTests(unittest.TestCase):
    def load_model(self, return_dict):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        backend = Tokenizer(
            models.WordLevel(
                {
                    "answer": 0,
                    "[BOS]": 1,
                    "[EOS]": 2,
                    "[UNK]": 3,
                    "hello": 4,
                    "assistant": 5,
                },
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            pad_token="[EOS]",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}assistant{% endif %}"
        )
        with torch.random.fork_rng(devices=[]):
            model = Qwen2MoeForCausalLM(
                Qwen2MoeConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    moe_intermediate_size=4,
                    shared_expert_intermediate_size=8,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    num_experts=2,
                    num_experts_per_tok=1,
                    num_hidden_layers=1,
                    vocab_size=6,
                    max_position_embeddings=64,
                    pad_token_id=2,
                    eos_token_id=2,
                    mlp_only_layers=[0],
                    attn_implementation="eager",
                )
            )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        model.generation_config.return_dict_in_generate = return_dict
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        return Model(
            InitOnlySettings(
                model=directory,
                dtypes=["float32"],
                device_map="cpu",
                max_response_length=2,
                response_prefix="",
                system_prompt="",
            )
        )

    def test_structured_default(self):
        model = self.load_model(True)
        _, output = model.generate([Prompt("", "hello")], max_new_tokens=1)
        self.assertIsInstance(output, GenerateDecoderOnlyOutput)
        self.assertEqual(model.get_responses([Prompt("", "hello")]), ["answer answer"])
        self.assertTrue(model.model.generation_config.return_dict_in_generate)

    def test_tensor_default(self):
        model = self.load_model(False)
        self.assertEqual(model.get_responses([Prompt("", "hello")]), ["answer answer"])
        self.assertFalse(model.model.generation_config.return_dict_in_generate)

    def test_explicit_structured_generation(self):
        model = self.load_model(False)
        _, output = model.generate(
            [Prompt("", "hello")],
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_logits=True,
        )
        self.assertIsInstance(output, GenerateDecoderOnlyOutput)
        self.assertEqual(tuple(output.logits[0].shape), (1, 6))
        self.assertFalse(model.model.generation_config.return_dict_in_generate)

    def test_structured_consumers_keep_their_return_mode(self):
        model = self.load_model(True)
        prompts = [Prompt("", "hello")]
        self.assertEqual(tuple(model.get_logits(prompts).shape), (1, 6))
        self.assertEqual(tuple(model.get_residuals(prompts).shape), (1, 2, 8))
        self.assertTrue(model.model.generation_config.return_dict_in_generate)
