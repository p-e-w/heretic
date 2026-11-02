# SPDX-License-Identifier: AGPL-3.0-or-later
"""A flattened shared expert still has one final-token I/O pair per prompt."""

import unittest
from types import MethodType

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from heretic.model import Model
from heretic.utils import Prompt


class SharedExpertLastTokenTests(unittest.TestCase):
    def exercise(self, lora, dense=False):
        torch.manual_seed(42)
        config = Qwen2MoeConfig(
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=4,
            shared_expert_intermediate_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=2,
            num_experts_per_tok=1,
            num_hidden_layers=1,
            vocab_size=32,
            max_position_embeddings=64,
            pad_token_id=0,
            attn_implementation="eager",
            mlp_only_layers=[0] if dense else [],
        )
        tiny = Qwen2MoeForCausalLM(config).eval()
        if lora:
            tiny = get_peft_model(
                tiny, LoraConfig(r=2, target_modules=["o_proj", "down_proj"])
            )
            tiny.eval()
        holder = Model.__new__(Model)
        holder.model = tiny
        # Unequal prompt lengths, padded on the left like Heretic's tokenizer.
        ids = torch.tensor([[0, 3, 4], [5, 6, 7]])
        attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        prompts = [Prompt(system="", user="short"), Prompt(system="", user="longer")]

        def generate(self, prompts, **kwargs):
            return {"input_ids": ids, "attention_mask": attention_mask}, tiny(
                ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        holder.generate = MethodType(generate, holder)
        layer = holder.get_layers()[0]
        shared = layer.mlp.down_proj if dense else layer.mlp.shared_expert.down_proj
        inventory = holder.get_layer_modules(0)
        self.assertIs(inventory["mlp.down_proj"][0], shared)
        expected = {}

        def remember_shared(module, inputs, outputs):
            self.assertEqual(inputs[0].ndim, 3 if dense else 2)
            self.assertEqual(inputs[0].numel(), 2 * 3 * inputs[0].shape[-1])
            # For B=2,T=3, the known final prompt tokens occupy flat rows 2 and 5.
            # Index them directly so the oracle does not repeat the unflattening code.
            last_input = inputs[0][:, -1, :] if dense else inputs[0][[2, 5]]
            last_output = outputs[:, -1, :] if dense else outputs[[2, 5]]
            expected["shared_input"] = last_input.detach().clone()
            expected["shared_output"] = last_output.detach().clone()

        def remember_attention(module, inputs, outputs):
            self.assertEqual(inputs[0].ndim, 3)
            expected["attention_input"] = inputs[0][:, -1, :].detach().clone()
            expected["attention_output"] = outputs[:, -1, :].detach().clone()

        handles = [
            shared.register_forward_hook(remember_shared),
            layer.self_attn.o_proj.register_forward_hook(remember_attention),
        ]
        try:
            with torch.no_grad():
                holder.generate(prompts)
                captured = holder.get_module_io(prompts)
            for component, prefix in [
                ("mlp.down_proj", "shared"),
                ("attn.o_proj", "attention"),
            ]:
                actual_input, actual_output = captured[0][component][0]
                self.assertEqual(actual_input.shape[0], 2)
                torch.testing.assert_close(actual_input, expected[prefix + "_input"])
                torch.testing.assert_close(actual_output, expected[prefix + "_output"])
        finally:
            for handle in handles:
                handle.remove()

    def test_dense_layer_keeps_final_token_per_prompt(self):
        self.exercise(lora=False, dense=True)

    def test_actual_shared_expert_and_dense_attention(self):
        self.exercise(lora=False)

    def test_actual_lora_wrapped_shared_expert_and_attention(self):
        self.exercise(lora=True)


if __name__ == "__main__":
    unittest.main()
