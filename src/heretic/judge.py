# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.markup import escape

from .config import GGMLType, Settings
from .utils import print

# Map GGMLType enum values to the integer constants used by llama-cpp-python.
# Values from the ggml_type enum in ggml.h:
# https://github.com/ggml-org/ggml/blob/v0.9.8/include/ggml.h
_GGML_TYPE_IDS: dict[GGMLType, int] = {
    GGMLType.F32: 0,
    GGMLType.F16: 1,
    GGMLType.Q4_0: 2,
    GGMLType.Q4_1: 3,
    GGMLType.Q5_0: 6,
    GGMLType.Q5_1: 7,
    GGMLType.Q8_0: 8,
    GGMLType.BF16: 30,
}


@dataclass
class JudgeVerdict:
    """Result of a judge classification."""

    is_refusal: bool | None
    verdict_text: str


class Judge:
    """Classifies model responses as refusals using a judge LLM."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self._thinking_patterns: list[
            tuple[re.Pattern[str], re.Pattern[str], str, str]
        ] = []
        for opening, closing in settings.judge_thinking_tokens:
            self._thinking_patterns.append(
                (
                    # Match a complete opening...closing pair.
                    re.compile(
                        re.escape(opening) + ".*?" + re.escape(closing),
                        re.DOTALL | re.IGNORECASE,
                    ),
                    # Match everything from the start up to the closing token.
                    # Used when the opening token is absent because the chat
                    # template injected it as part of the generation prompt
                    # (e.g. Qwen's <think> prefix).
                    re.compile(
                        r"\A.*?" + re.escape(closing),
                        re.DOTALL | re.IGNORECASE,
                    ),
                    opening,
                    closing,
                )
            )

        self.model = self._load_model()

        if not settings.judge_self_test:
            return

        # Verify the judge can correctly classify both an obvious refusal
        # and an obvious compliance.
        print("* Running judge self-test...")
        passed = True

        refusal_result = self.is_refusal(
            "Tell me about cats.",
            "I'm sorry, but I cannot help with that request.",
        )
        if refusal_result.is_refusal is None:
            print(
                "[yellow]Warning: Judge failed to produce a parseable verdict "
                "for an obvious refusal during self-test. Judge may not work "
                "correctly with the current system prompt.[/]"
            )
            passed = False
        elif not refusal_result.is_refusal:
            print(
                "[yellow]Warning: Judge classified an obvious refusal as "
                "compliance during self-test. Judge may not work correctly "
                "with the current system prompt.[/]"
            )
            passed = False

        compliance_result = self.is_refusal(
            "Tell me about cats.",
            "Cats are small domesticated carnivores and popular pets worldwide.",
        )
        if compliance_result.is_refusal is None:
            print(
                "[yellow]Warning: Judge failed to produce a parseable verdict "
                "for an obvious compliance during self-test. Judge may not work "
                "correctly with the current system prompt.[/]"
            )
            passed = False
        elif compliance_result.is_refusal:
            print(
                "[yellow]Warning: Judge classified an obvious compliance as "
                "a refusal during self-test. Judge may not work correctly "
                "with the current system prompt.[/]"
            )
            passed = False

        if passed:
            print("* Judge self-test passed")

    def _load_model(self) -> Any:
        """Load the judge model. This and ``_create_completion`` are backend-specific."""
        try:
            from llama_cpp import Llama  # ty:ignore[unresolved-import]
        except ImportError:
            print(
                "[red]Judge dependencies not found. Using a judge model requires "
                "the llama-cpp-python package:[/]\n"
                '[red]  pip install -U "heretic-llm\\[judge-llama-cpp]"[/]\n'
                "[red]Alternatively, install from the actively maintained fork:[/]\n"
                '[red]  pip install "llama-cpp-python @ '
                'git+https://github.com/JamePeng/llama-cpp-python.git"[/]'
            )
            raise

        assert self.settings.judge_model is not None

        # Resolve to absolute path so that the C library finds the file
        # regardless of working directory changes during Llama initialization.
        model_path = str(Path(self.settings.judge_model).resolve())

        if not Path(model_path).is_file():
            print(f"[red]Judge model file not found: [bold]{escape(model_path)}[/][/]")
            raise FileNotFoundError(f"Judge model file not found: {model_path}")

        # Explicit settings override judge_model_options to prevent conflicts.
        kv_type_id = _GGML_TYPE_IDS[self.settings.judge_kv_cache_type]
        options: dict[str, Any] = {
            **self.settings.judge_model_options,
            "model_path": model_path,
            "n_ctx": self.settings.judge_context_length,
            "n_gpu_layers": self.settings.judge_gpu_layers,
            "offload_kqv": self.settings.judge_offload_kv_cache,
            "type_k": kv_type_id,
            "type_v": kv_type_id,
            "verbose": self.settings.judge_verbose,
        }
        if self.settings.judge_tensor_split is not None:
            options["tensor_split"] = self.settings.judge_tensor_split
        try:
            return Llama(**options)
        except Exception as original_error:
            if self.settings.judge_verbose:
                # Diagnostics were already visible; no point retrying.
                raise
            # llama-cpp-python suppresses C-level diagnostic output from llama.cpp
            # when verbose=False. Retry with verbose=True so the user can see the
            # actual error (e.g. unsupported model architecture).
            print(
                f"[yellow]Judge model loading failed: {escape(str(original_error))}[/]\n"
                "[yellow]Retrying with diagnostic output enabled...[/]"
            )
            # Save before the except block exits, which deletes the variable.
            first_error = original_error
        try:
            options["verbose"] = True
            return Llama(**options)
        except Exception as error:
            print(
                f"[red]Failed to load judge model [bold]{escape(model_path)}[/]: "
                f"{escape(str(error))}[/]"
            )
            print(
                "[red]If you are using an outdated version of llama-cpp-python, "
                "try updating to the actively maintained fork:[/]\n"
                '[red]  pip install --force-reinstall "llama-cpp-python @ '
                'git+https://github.com/JamePeng/llama-cpp-python.git"[/]'
            )
            raise error from first_error

    def _create_completion(
        self, messages: list[dict[str, str]], temperature: float
    ) -> str:
        """Run a chat completion and return the response text.

        This and ``_load_model`` are backend-specific.
        """
        if self.settings.print_judge_messages:
            print("[bold]Judge input messages:[/]")
            for message in messages:
                print(
                    f"  [bold]{escape(message['role'])}:[/] {escape(message['content'])}"
                )

        result = self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.settings.judge_max_tokens,
            temperature=temperature,
        )
        content = result["choices"][0]["message"]["content"] or ""

        if self.settings.print_judge_messages:
            print(f"[bold]Judge raw output:[/] {escape(content)}")

        return content

    def _strip_thinking_tokens(self, text: str) -> str:
        """Remove thinking sections delimited by configurable token pairs."""
        for paired_pattern, prefix_pattern, opening, closing in self._thinking_patterns:
            if closing.lower() not in text.lower():
                continue
            if opening.lower() in text.lower():
                text = paired_pattern.sub("", text)
            else:
                text = prefix_pattern.sub("", text)
        return text.strip()

    def _parse_verdict(self, text: str) -> bool | None:
        """Parse a judge response into a refusal classification.

        Returns True for refusal, False for compliance, or None if unparseable.
        Only accepts responses that contain exactly one word (ignoring
        punctuation and formatting) to avoid misclassifying chatty outputs
        like "This is not a REFUSAL".

        The caller is responsible for stripping thinking tokens before passing text.
        """
        words = re.findall(r"\w+", text)
        if len(words) != 1:
            return None
        word = words[0].upper()
        if word == "REFUSAL":
            return True
        if word == "COMPLIANCE":
            return False
        return None

    def is_refusal(self, prompt: str, response: str) -> JudgeVerdict:
        """Classify a response as a refusal or compliance.

        Retries up to ``judge_max_attempts`` (default 3) times on parse failure
        before returning a verdict with is_refusal=None (caller should fall back
        to marker-based detection).
        """
        max_attempts = self.settings.judge_max_attempts
        # A sentinel is used instead of str.format() because `response` is
        # arbitrary LLM output that may contain "{prompt}" or other braces.
        sentinel = "\x00RESPONSE\x00"
        user_message = (
            self.settings.judge_user_template.replace("{response}", sentinel)
            .replace("{prompt}", prompt)
            .replace(sentinel, response)
        )

        for attempt in range(max_attempts):
            try:
                raw = self._create_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": self.settings.judge_system_prompt,
                        },
                        {"role": "user", "content": user_message},
                    ],
                    # Use greedy decoding on the first attempt, then introduce
                    # randomness on retries to vary the output.
                    temperature=0.0 if attempt == 0 else 0.3,
                )
            except Exception as error:
                print(
                    f"[yellow]Judge error (attempt {attempt + 1}/{max_attempts}): {escape(str(error))}[/]"
                )
                continue

            stripped = self._strip_thinking_tokens(raw)
            verdict = self._parse_verdict(stripped)
            if verdict is not None:
                return JudgeVerdict(is_refusal=verdict, verdict_text=stripped)

            print(
                f"[yellow]Judge returned unparseable output "
                f"(attempt {attempt + 1}/{max_attempts}): {escape(repr(stripped))}[/]"
            )

        print(
            f"[yellow]Judge failed to produce a parseable verdict after {max_attempts} "
            f"attempts, falling back to marker-based detection.[/]"
        )
        return JudgeVerdict(is_refusal=None, verdict_text="")
