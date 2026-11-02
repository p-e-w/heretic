import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from heretic.config import InitOnlySettings
from heretic.evaluator import Evaluator
from heretic.plugin import Context
from heretic.scorers.bert_classifier import BERTClassifier, Settings
from heretic.utils import Prompt


class ClassifierReproducibilityTests(unittest.TestCase):
    def make_scorer(self, **kwargs):
        return BERTClassifier(InitOnlySettings(model="unused"), Settings(**kwargs))

    def test_local_classifier_runs_but_disables_reproduction_offer(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        backend = Tokenizer(
            models.WordLevel(
                {"[PAD]": 0, "[UNK]": 1, "hello": 2, "answer": 3}, unk_token="[UNK]"
            )
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
        )
        with torch.random.fork_rng(devices=[]):
            tiny = BertForSequenceClassification(
                BertConfig(
                    vocab_size=4,
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                    max_position_embeddings=32,
                    id2label={0: "negative", 1: "positive"},
                )
            )
        with torch.no_grad():
            for parameter in tiny.parameters():
                parameter.zero_()
        tiny.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        scorer = self.make_scorer(
            model=directory,
            model_commit="a" * 40,
            dtypes=["float32"],
            device_map="cpu",
            match_labels=["negative"],
            use_text_pairs=False,
            max_length=8,
        )
        context = Mock(spec=Context)
        context.load_prompts.return_value = [Prompt("", "hello")]
        context.get_responses.return_value = ["answer"]
        scorer.init(context)
        self.assertEqual(scorer.get_score(context).value, 1.0)
        evaluator = Evaluator.__new__(Evaluator)
        evaluator._scorer_entries = [SimpleNamespace(scorer=scorer)]
        self.assertFalse(evaluator.all_scorers_reproducible())
        # Loading already consumed these files; upload eligibility must not
        # reinterpret the vanished local path as a Hub identifier or raise.
        Path(directory).rename(directory + "-moved")
        self.addCleanup(lambda: shutil.rmtree(directory + "-moved"))
        self.assertEqual(scorer.get_score(context).value, 1.0)
        self.assertFalse(evaluator.all_scorers_reproducible())

    def test_local_path_with_explicit_revision_is_not_a_hub_model(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        scorer = self.make_scorer(model=directory, model_commit="a" * 40)
        self.assertFalse(scorer.reproducible)

    def test_unpinned_hub_classifier_is_not_yet_reproducible(self):
        self.assertFalse(self.make_scorer(model="owner/classifier").reproducible)

    def test_hub_classifier_becomes_reproducible_only_after_successful_load(self):
        for supplied_commit in [None, "a" * 40]:
            with self.subTest(supplied_commit=supplied_commit):
                scorer = self.make_scorer(
                    model="owner/classifier",
                    model_commit=supplied_commit,
                    match_labels=["negative"],
                    dtypes=["float32"],
                )
                self.assertFalse(scorer.reproducible)
                context = Mock(spec=Context)
                context.load_prompts.return_value = [Prompt("", "hello")]
                classifier = Mock(
                    model=SimpleNamespace(
                        config=SimpleNamespace(id2label={0: "negative"})
                    )
                )
                with (
                    patch(
                        "heretic.scorers.bert_classifier.pipeline",
                        return_value=classifier,
                    ) as load,
                    patch(
                        "heretic.scorers.bert_classifier.huggingface_hub.model_info",
                        return_value=SimpleNamespace(sha="a" * 40),
                    ) as info,
                ):
                    scorer.init(context)
                    self.assertTrue(scorer.reproducible)
                    self.assertEqual(load.call_args.kwargs["revision"], "a" * 40)
                    self.assertEqual(info.call_count, int(supplied_commit is None))

    def test_failed_reinitialization_revokes_prior_eligibility(self):
        scorer = self.make_scorer(
            model="owner/classifier",
            model_commit="a" * 40,
            match_labels=["negative"],
            dtypes=["float32"],
        )
        context = Mock(spec=Context)
        context.load_prompts.return_value = [Prompt("", "hello")]
        classifier = Mock(
            model=SimpleNamespace(config=SimpleNamespace(id2label={0: "negative"}))
        )
        with patch("heretic.scorers.bert_classifier.pipeline", return_value=classifier):
            scorer.init(context)
        self.assertTrue(scorer.reproducible)
        with patch(
            "heretic.scorers.bert_classifier.pipeline",
            side_effect=RuntimeError("load failed"),
        ):
            with self.assertRaisesRegex(Exception, "Failed to load text classifier"):
                scorer.init(context)
        self.assertFalse(scorer.reproducible)
