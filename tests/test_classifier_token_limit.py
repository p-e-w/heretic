import tempfile
import unittest
from unittest.mock import Mock

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from heretic.config import InitOnlySettings
from heretic.plugin import Context
from heretic.scorers.bert_classifier import BERTClassifier, Settings
from heretic.utils import Prompt


class ClassifierTokenLimitTests(unittest.TestCase):
    def classifier(self, *, tokenizer_limit=8, model_limit=8, **settings):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        backend = Tokenizer(
            models.WordLevel(
                {"[PAD]": 0, "[UNK]": 1, "hello": 2, "answer": 3}, unk_token="[UNK]"
            )
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            unk_token="[UNK]",
            model_max_length=tokenizer_limit,
        )
        with torch.random.fork_rng(devices=[]):
            model = BertForSequenceClassification(
                BertConfig(
                    vocab_size=4,
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                    max_position_embeddings=model_limit,
                    id2label={0: "negative", 1: "positive"},
                )
            )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        scorer = BERTClassifier(
            InitOnlySettings(model="unused"),
            Settings(
                model=directory,
                dtypes=["float32"],
                device_map="cpu",
                match_labels=["negative"],
                use_text_pairs=settings.pop("use_text_pairs", False),
                **settings,
            ),
        )
        context = Mock(spec=Context)
        context.load_prompts.return_value = [Prompt("", "hello " * 6)]
        context.get_responses.return_value = ["answer " * 16]
        scorer.init(context)
        observed_lengths = []

        def capture(_module, _args, kwargs):
            observed_lengths.append(kwargs["input_ids"].shape[-1])

        hook = scorer.pipeline.model.register_forward_pre_hook(
            capture, with_kwargs=True
        )
        self.addCleanup(hook.remove)
        return scorer, context, observed_lengths

    def test_declared_tokenizer_limit_bounds_default_and_larger_user_caps(self):
        for settings in ({}, {"max_length": 16}):
            with self.subTest(settings=settings):
                scorer, context, lengths = self.classifier(**settings)
                self.assertEqual(scorer.get_score(context).value, 1.0)
                self.assertEqual(lengths, [8])

    def test_smaller_explicit_cap_is_preserved(self):
        scorer, context, lengths = self.classifier(max_length=4)
        self.assertEqual(scorer.get_score(context).value, 1.0)
        self.assertEqual(lengths, [4])

    def test_unbounded_tokenizer_keeps_the_explicit_cap(self):
        scorer, context, lengths = self.classifier(
            tokenizer_limit=10**30, model_limit=32, max_length=12
        )
        self.assertEqual(scorer.get_score(context).value, 1.0)
        self.assertEqual(lengths, [12])

    def test_pair_inputs_share_the_declared_limit(self):
        scorer, context, lengths = self.classifier(use_text_pairs=True)
        self.assertEqual(scorer.get_score(context).value, 1.0)
        self.assertEqual(lengths, [8])
