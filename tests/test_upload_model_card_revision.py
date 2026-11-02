# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exercise main's upload path with real Settings, trials, and ModelCard parsing.

Model computation and external Hub operations are explicit fixture boundaries;
these tests do not perform inference, authenticate, or upload anything.
"""

import sys
import unittest
import warnings
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub import repocard
from optuna.trial import create_trial

from heretic import main, utils
from heretic.config import ExportStrategy, Settings


class UploadModelCardRevisionTests(unittest.TestCase):
    def run_export(self, *, revision=None, local=False, missing=False):
        downloads, exports, cards, messages = [], [], [], []
        with (
            TemporaryDirectory(prefix="heretic-card-") as directory,
            patch.object(sys, "argv", ["heretic"]),
        ):
            directory = Path(directory)
            head = directory / "head.md"
            pinned = directory / "pinned.md"
            head.write_text(
                "---\nlicense: mit\n---\nHEAD CARD BODY\n", encoding="utf-8"
            )
            pinned.write_text(
                "---\nlicense: apache-2.0\n---\nPINNED CARD BODY\n", encoding="utf-8"
            )
            model_path = directory / "local-model"
            model_path.mkdir()
            if local and not missing:
                (model_path / "README.md").write_bytes(pinned.read_bytes())
            source = str(model_path) if local else "fixture/base"
            settings = Settings(
                _cli_parse_args=False,
                model=source,
                model_commit=revision,
                batch_size=1,
                response_prefix="",
                seed=7,
                n_trials=1,
                trial_index=0,
                model_action="upload",
                upload_repo_id="fixture/output",
                upload_repo_private=True,
                upload_reproducibility_information="none",
                export_strategy=ExportStrategy.MERGE,
                study_checkpoint_dir=str(directory / "journal"),
            )
            score = {"value": 0.5, "rich_display": "0.5", "md_display": "0.5"}
            trial = create_trial(
                values=[0.5],
                user_attrs={
                    "index": 1,
                    "parameters": {},
                    "scores": [{"name": "fixture", "score": score, "baseline": score}],
                },
            )
            study = SimpleNamespace(
                trials=[trial],
                best_trials=[trial],
                set_user_attr=lambda *args: None,
                optimize=lambda *args, **kwargs: None,
            )
            weights = SimpleNamespace(
                push_to_hub=lambda *args, **kwargs: exports.append("weights")
            )
            model = SimpleNamespace(
                revision_kwargs={} if revision is None else {"revision": revision},
                model=weights,
                processor=None,
                tokenizer=SimpleNamespace(
                    push_to_hub=lambda *args, **kwargs: exports.append("tokenizer")
                ),
                get_merged_model=lambda: weights,
            )
            modifier = SimpleNamespace(
                modifier_name="Fixture",
                reproducible=False,
                render_trial_parameters=lambda *args: {},
                reset_model=lambda *args: None,
                modify_model=lambda *args: None,
                parameters_class=SimpleNamespace(from_dict=lambda value: value),
                get_dataset_specifications=list,
            )
            evaluator = SimpleNamespace(
                get_dataset_specifications=list,
                get_objective_names=lambda: ["fixture"],
                get_objective_directions=lambda: ["minimize"],
                all_scorers_reproducible=lambda: False,
            )
            entry = SimpleNamespace(
                modifier=modifier,
                name="Fixture",
                config=SimpleNamespace(plugin="fixture"),
            )

            def download(repo_id, filename, **kwargs):
                self.assertEqual((repo_id, filename), ("fixture/base", "README.md"))
                downloads.append(kwargs)
                if missing:
                    raise OSError("fixture README unavailable")
                return str(
                    pinned if kwargs.get("revision") == revision and revision else head
                )

            def capture_card(card, *args, **kwargs):
                self.assertEqual(args, ("fixture/output",))
                cards.append(
                    {
                        "license": card.data.license,
                        "text": card.text,
                        "base_model": card.data.base_model,
                        "tags": card.data.tags,
                    }
                )

            # Run the actual application control flow while replacing only its
            # runtime, UI-output, and external-operation fixture boundaries.
            with ExitStack() as stack:
                stack.enter_context(warnings.catch_warnings())
                logger = main.logging.getLogger("lm_eval")
                stack.callback(logger.setLevel, logger.level)
                stack.callback(
                    main.optuna.logging.set_verbosity,
                    main.optuna.logging.get_verbosity(),
                )
                stack.callback(
                    main.transformers.logging.set_verbosity,
                    main.transformers.logging.get_verbosity(),
                )
                replacements = [
                    (main, "Settings", lambda: settings),
                    (main, "Model", lambda *args: model),
                    (main, "Evaluator", lambda *args: evaluator),
                    (main, "load_and_init_modifiers", lambda *args: [entry]),
                    (
                        main,
                        "print",
                        lambda *args, **kwargs: messages.extend(map(str, args)),
                    ),
                    (main, "print_memory_usage", lambda: None),
                    (main, "get_accelerator_info", lambda: "fixture CPU"),
                    (main, "empty_cache", lambda: None),
                    (main, "version", lambda *args: "fixture"),
                    (utils, "version", lambda *args: "fixture"),
                    (main.optuna, "create_study", lambda **kwargs: study),
                    (main.transformers, "set_seed", lambda *args: None),
                    (main.torch, "set_grad_enabled", lambda *args: None),
                    (
                        main.huggingface_hub,
                        "get_token",
                        lambda: "synthetic-not-a-credential",
                    ),
                    (main.huggingface_hub, "whoami", lambda *args: {"name": "fixture"}),
                    (main.huggingface_hub, "hf_hub_download", download),
                    (repocard, "hf_hub_download", download),
                    (main.ModelCard, "push_to_hub", capture_card),
                ]
                for module, name, replacement in replacements:
                    stack.enter_context(patch.object(module, name, replacement))
                stack.enter_context(
                    main.torch._dynamo.config.patch(
                        cache_size_limit=main.torch._dynamo.config.cache_size_limit
                    )
                )
                main.run()
            self.assertEqual(exports, ["weights", "tokenizer"])
        return downloads, cards, messages

    def test_pinned_hub_card_uses_the_selected_revision(self):
        revision = "a" * 40
        downloads, cards, _ = self.run_export(revision=revision)
        self.assertEqual(len(downloads), 1)
        self.assertEqual(downloads[0].get("revision"), revision)
        self.assertEqual(cards[0]["license"], "apache-2.0")
        self.assertIn("PINNED CARD BODY", cards[0]["text"])
        self.assertNotIn("HEAD CARD BODY", cards[0]["text"])
        self.assertEqual(cards[0]["base_model"], "fixture/base")
        self.assertIn("heretic", cards[0]["tags"])

    def test_unpinned_hub_card_keeps_default_revision(self):
        downloads, cards, _ = self.run_export()
        self.assertEqual(len(downloads), 1)
        self.assertNotIn("revision", downloads[0])
        self.assertEqual(cards[0]["license"], "mit")
        self.assertIn("HEAD CARD BODY", cards[0]["text"])

    def test_local_card_uses_the_local_readme(self):
        downloads, cards, _ = self.run_export(local=True, revision="a" * 40)
        self.assertEqual(downloads, [])
        self.assertEqual(cards[0]["license"], "apache-2.0")
        self.assertIn("PINNED CARD BODY", cards[0]["text"])

    def test_missing_local_card_remains_optional(self):
        downloads, cards, messages = self.run_export(local=True, missing=True)
        self.assertEqual((downloads, cards), ([], []))
        self.assertFalse(any("Error:" in message for message in messages))

    def test_hub_read_failure_keeps_existing_error_path(self):
        downloads, cards, messages = self.run_export(revision="a" * 40, missing=True)
        self.assertEqual(len(downloads), 1)
        self.assertEqual(cards, [])
        self.assertTrue(
            any("fixture README unavailable" in message for message in messages)
        )


if __name__ == "__main__":
    unittest.main()
