# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from heretic import main
from heretic.config import Settings


class ModelInitialized(Exception):
    """Stop startup before any model loading or inference."""


class ResumeSeedTests(unittest.TestCase):
    def run_setup(
        self,
        launch_seed: int | None,
        *,
        checkpoint: bool = True,
        action: str = "continue",
    ) -> tuple[tuple[int | None, float], MagicMock]:
        initial = Settings.model_construct(
            model="test/model", seed=launch_seed, checkpoint_action=action
        )
        saved = Settings.model_construct(model="test/model", seed=42)
        study = SimpleNamespace(
            user_attrs={"finished": False, "settings": saved.model_dump_json()}
        )
        storage = SimpleNamespace(get_all_studies=lambda: [study] if checkpoint else [])
        settings_factory = MagicMock(return_value=initial)
        settings_factory.model_validate_init_only = Settings.model_validate_init_only
        rng = random.Random()
        set_seed = MagicMock(side_effect=rng.seed)
        observed = []

        def initialize_model(settings: Settings) -> None:
            observed.append((settings.seed, rng.random()))
            raise ModelInitialized

        with (
            patch.multiple(
                main,
                Settings=settings_factory,
                sys=SimpleNamespace(argv=["heretic"]),
                random=SimpleNamespace(randint=MagicMock(return_value=1234)),
                os=SimpleNamespace(
                    path=os.path, makedirs=MagicMock(), unlink=MagicMock()
                ),
                print=MagicMock(),
                version=MagicMock(return_value="test"),
                get_accelerator_info=MagicMock(return_value="test"),
                transformers=SimpleNamespace(set_seed=set_seed, logging=MagicMock()),
                torch=MagicMock(),
                logging=MagicMock(),
                warnings=MagicMock(),
                optuna=MagicMock(),
                JournalFileOpenLock=MagicMock(),
                JournalFileBackend=MagicMock(),
                JournalStorage=MagicMock(return_value=storage),
                Choice=MagicMock(),
                Style=MagicMock(),
                questionary=MagicMock(),
                ask_if_unset=lambda value, question: value,
                Model=MagicMock(side_effect=initialize_model),
            ),
            self.assertRaises(ModelInitialized),
        ):
            main.run()

        self.assertEqual(len(observed), 1)
        return observed[0], set_seed

    def test_continue_applies_the_saved_seed_before_model_initialization(self) -> None:
        result, set_seed = self.run_setup(99)
        self.assertEqual(result, (42, random.Random(42).random()))
        set_seed.assert_called_once_with(42)

    def test_continue_without_a_launch_seed_applies_the_saved_seed(self) -> None:
        result, set_seed = self.run_setup(None)
        self.assertEqual(result, (42, random.Random(42).random()))
        set_seed.assert_called_once_with(42)

    def test_restart_keeps_the_launch_seed(self) -> None:
        result, set_seed = self.run_setup(99, action="restart")
        self.assertEqual(result, (99, random.Random(99).random()))
        set_seed.assert_called_once_with(99)

    def test_fresh_run_keeps_the_launch_seed(self) -> None:
        result, set_seed = self.run_setup(99, checkpoint=False)
        self.assertEqual(result, (99, random.Random(99).random()))
        set_seed.assert_called_once_with(99)


if __name__ == "__main__":
    unittest.main()
