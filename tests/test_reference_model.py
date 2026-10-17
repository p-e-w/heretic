# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from model_fixtures import PROMPTS, TinyModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from heretic.config import Settings as HereticSettings
from heretic.model import Model
from heretic.modifier import Context
from heretic.modifiers.backward_restoration import BackwardRestoration
from heretic.modifiers.backward_restoration import Parameters as RestorationParameters
from heretic.modifiers.backward_restoration import Settings as RestorationSettings
from heretic.scorers.kl_divergence import KLDivergence, Settings


class ReferenceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.original = str(Path(self.directory.name) / "original")
        self.edited = str(Path(self.directory.name) / "edited")
        model = TinyModel().model
        model.save_pretrained(self.original)
        with torch.no_grad():
            model.get_parameter("lm_head.weight")[0].add_(0.2)
            model.get_parameter("model.layers.0.self_attn.o_proj.weight").add_(0.01)
        model.save_pretrained(self.edited)
        vocab = {str(i): i for i in range(1, 30)}
        vocab.update({"<pad>": 0, "<unk>": 30, "<eos>": 31})
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            eos_token="<eos>",
        )
        fast.chat_template = "{{ messages[-1]['content'] }}"
        fast.save_pretrained(self.original)
        fast.save_pretrained(self.edited)
        self.settings = HereticSettings.model_construct(
            model=self.edited,
            dtypes=["float32"],
            device_map="cpu",
            batch_size=2,
            response_prefix="",
            seed=42,
        )

    def test_reference_release_and_student_reload_even_on_failure(self):
        model = Model(self.settings)
        expected = model.get_logits_batched(PROMPTS)
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with model.reference_model(self.original) as reference:
                self.assertIsNone(model.model)
                self.assertIs(reference.tokenizer, model.tokenizer)
                self.assertFalse(
                    torch.equal(reference.get_logits_batched(PROMPTS), expected)
                )
                raise RuntimeError("test failure")
        self.assertIsNone(reference.model)
        torch.testing.assert_close(
            model.get_logits_batched(PROMPTS), expected, atol=0, rtol=0
        )
        with patch("heretic.model.Model", side_effect=RuntimeError("load failed")):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                with model.reference_model(self.original):
                    self.fail("Reference loading should have failed")
        torch.testing.assert_close(
            model.get_logits_batched(PROMPTS), expected, atol=0, rtol=0
        )
        model.apply_lora(2)
        with self.assertRaisesRegex(ValueError, "before applying adapters"):
            with model.reference_model(self.original):
                self.fail("An active adapter must not be discarded")

    def test_external_kl_baseline_measures_the_edited_model(self):
        model = Model(self.settings)
        ctx = Context(self.settings, model)
        scorer = KLDivergence(
            heretic_settings=self.settings,
            settings=Settings(reference_model=self.original),
        )
        with patch.object(ctx, "load_prompts", return_value=PROMPTS):
            scorer.init(ctx)
        baseline = scorer.get_baseline_score(ctx)
        self.assertGreater(baseline.value, 0)
        self.assertEqual(baseline.value, scorer.get_score(ctx).value)
        original_settings = self.settings.model_copy(update={"model": self.original})
        original = Model(original_settings)
        self.assertAlmostEqual(
            scorer.get_score(Context(original_settings, original)).value, 0, places=6
        )

    def test_vocabulary_mismatch_restores_student(self):
        model = Model(self.settings)
        before = model.get_logits_batched(PROMPTS)
        with patch.object(model.tokenizer, "get_vocab", return_value={"different": 0}):
            with self.assertRaisesRegex(ValueError, "same token vocabulary"):
                with model.reference_model(self.original):
                    self.fail(
                        "Incompatible token IDs must not be used as teacher targets"
                    )
        torch.testing.assert_close(
            model.get_logits_batched(PROMPTS), before, atol=0, rtol=0
        )

    def test_restoration_merge_reload_and_reset(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                model = Model(self.settings)
                ctx = Context(self.settings, model)
                baseline = model.get_logits_batched(PROMPTS)
                modifier = BackwardRestoration(
                    heretic_settings=self.settings,
                    settings=RestorationSettings(
                        original_model=self.original,
                        streaming=streaming,
                        lora_rank=2,
                        n_optimization_steps=1,
                        max_iter=4,
                    ),
                )
                with patch.object(
                    ctx, "load_prompts", side_effect=[PROMPTS[:2], PROMPTS[2:]]
                ):
                    modifier.init(ctx)
                params = RestorationParameters(0, 3, 4.0)
                modifier.modify_model(ctx, params)
                trained = model.get_logits_batched(PROMPTS)
                merged = model.get_merged_model()
                destination = str(Path(self.directory.name) / f"restored-{streaming}")
                merged.save_pretrained(destination)
                model.tokenizer.save_pretrained(destination)
                restored = Model(
                    self.settings.model_copy(update={"model": destination})
                )
                torch.testing.assert_close(
                    restored.get_logits_batched(PROMPTS), trained, atol=1e-6, rtol=1e-5
                )
                modifier.reset_model(ctx)
                torch.testing.assert_close(
                    model.get_logits_batched(PROMPTS), baseline, atol=0, rtol=0
                )
                modifier.modify_model(ctx, params)
                torch.testing.assert_close(
                    model.get_logits_batched(PROMPTS), trained, atol=0, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
