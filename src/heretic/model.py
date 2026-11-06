# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias, cast

import torch
from huggingface_hub import list_repo_files
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from peft import LoraConfig, PeftModel, get_peft_model
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    GgufConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
)

from .config import QuantizationMethod, Settings
from .system import empty_cache
from .utils import Prompt, batchify, format_exception, print


def get_model_class(
    model: str,
    gguf_file: str | None = None,
) -> type[AutoModelForImageTextToText] | type[AutoModelForCausalLM]:
    # A GGUF repo has no config.json. The architecture lives in the file,
    # so the same gguf_file passed to from_pretrained has to be passed here.
    config_kwargs = {}
    if gguf_file is not None:
        config_kwargs["gguf_file"] = gguf_file
    configs = PretrainedConfig.get_config_dict(model, **config_kwargs)

    if any(("vision_config" in config) for config in configs):
        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


def check_model_resident(model: PreTrainedModel | PeftModel) -> None:
    """
    Ensure that all of the model's weights are resident in memory.
    """
    offloaded = [name for name, param in model.named_parameters() if param.is_meta]

    if not offloaded:
        return

    raise Exception(
        "The model does not fit into the available device memory: "
        f"{len(offloaded)} weights (such as {', '.join(offloaded[:3])}) had to "
        "be offloaded to CPU or disk. Heretic cannot process offloaded models "
        "because it requires direct, persistent access to all weights, whereas "
        "offloaded weights only exist temporarily during forward passes.\n"
        "Possible remedies:\n"
        "* Free up memory, e.g. by closing other applications.\n"
        "* Change the device map (device_map/max_memory settings) so that all "
        "weights fit into device memory.\n"
        '* Enable 4-bit quantization (quantization = "bnb_4bit").'
    )


def _collapse_flattened_residual_streams(
    hidden_states: tuple[FloatTensor, ...],
    config: Any,
) -> tuple[FloatTensor, ...]:
    """
    Average flattened multi-stream hidden states to hidden_size.

    Some models (e.g. Qwen4-Exp) flatten their residual streams into the
    component dimension instead of exposing a separate stream dimension, so
    their hidden states have shape (prompt, position, streams * hidden_size).
    The final entry is the merged model output (hidden_size) and is unchanged.
    """
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        config = get_text_config()
    hidden_size = getattr(config, "hidden_size", None)
    streams = getattr(config, "hc_count", 1)
    if hidden_size is None or streams <= 1:
        return hidden_states
    width = hidden_size * streams
    return tuple(
        h.unflatten(-1, (streams, hidden_size)).mean(-2) if h.shape[-1] == width else h
        for h in hidden_states
    )


# Some chat templates (e.g. gpt-oss) insert the current date into the prompt,
# which makes the residuals, and therefore the modified model, depend on the day
# Heretic is run. Rendering templates with a fixed date keeps results reproducible.
def strftime_fixed_date(format: str) -> str:
    return datetime(2026, 1, 1, tzinfo=timezone.utc).strftime(format)


def chat_template_supports_system_role(tokenizer: PreTrainedTokenizerBase) -> bool:
    # Some chat templates (e.g. Gemma 2 and its fine-tunes) reject the system role
    # outright, raising when one is present; others silently drop it. The only
    # reliable way to tell is to render a probe chat and check whether a sentinel
    # system message survives into the output. This is also how llama.cpp's template
    # engine (minja) detects system-role support.
    sentinel = "HERETIC_SYSTEM_ROLE_PROBE"
    try:
        rendered = cast(
            str,
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sentinel},
                    {"role": "user", "content": "Hi"},
                ],
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
    except Exception:
        return False
    return sentinel in rendered


# The list contains one element per layer.
# Each element maps from the component name to a (possibly sparse) mapping
# from the module index to an (input, output) tuple containing the I/O
# tensors of shape (prompt, component).
ModuleIO: TypeAlias = list[dict[str, dict[int, tuple[Tensor, Tensor]]]]


# A listing failure must not block a normal load. Offline use, a missing repo,
# and a path that is not a repo id all end up here.
_HUB_LISTING_ERRORS = (
    GatedRepoError,
    HFValidationError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    OSError,
)

_GGUF_MAGIC = b"GGUF"


def _is_standard_weight_filename(name: str) -> bool:
    """Return whether ``name`` is a weight file Transformers can load on its own."""
    basename = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if basename.endswith((".safetensors", ".safetensors.index.json")):
        return True
    return basename.startswith("pytorch_model") and basename.endswith(
        (".bin", ".bin.index.json")
    )


def _has_gguf_magic(path: Path) -> bool:
    """Return whether ``path`` starts with the GGUF magic bytes."""
    try:
        with path.open("rb") as handle:
            return handle.read(4) == _GGUF_MAGIC
    except OSError:
        return False


def _filenames_are_gguf_only(filenames: list[str]) -> bool:
    """
    Return whether a repo file listing is a GGUF checkpoint.

    GGUF-only means at least one ``.gguf`` file and no safetensors / PyTorch
    weights. ``config.json`` does not count: the weights still are not loadable
    with ``from_pretrained`` unless ``gguf_file`` is passed.
    """
    has_gguf = False
    for name in filenames:
        if _is_standard_weight_filename(name):
            return False
        if name.replace("\\", "/").rsplit("/", 1)[-1].lower().endswith(".gguf"):
            has_gguf = True
    return has_gguf


def _hub_cache_root() -> Path:
    configured = os.environ.get("HF_HUB_CACHE") or os.environ.get(
        "HUGGINGFACE_HUB_CACHE"
    )
    if configured:
        return Path(configured).expanduser()
    return Path(HF_HUB_CACHE)


def _cached_hub_filenames(repo_id: str, revision: str | None) -> list[str] | None:
    """
    Return file names from the local Hub cache, or ``None`` when unavailable.

    Used when the Hub cannot be listed (offline, network error, no token).
    """
    if repo_id.count("/") != 1:
        return None

    root = _hub_cache_root() / f"models--{repo_id.replace('/', '--')}"
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return None

    revision_name = revision or "main"
    commit: str | None = None
    ref = root / "refs" / revision_name
    try:
        if ref.is_file():
            commit = ref.read_text(encoding="utf-8").strip()
        elif (snapshots / revision_name).is_dir():
            commit = revision_name
    except OSError:
        return None

    # A revision is a single cache-directory name, never a path.
    if not commit or commit in {".", ".."} or "/" in commit or "\\" in commit:
        return None

    folder = snapshots / commit
    if not folder.is_dir():
        return None

    try:
        return [
            item.relative_to(folder).as_posix()
            for item in folder.rglob("*")
            if item.is_file()
        ]
    except OSError:
        return None


def _hub_filenames(repo_id: str, revision: str | None) -> list[str] | None:
    """List a model repo, falling back to the local cache. ``None`` if unknown."""
    try:
        return list_repo_files(repo_id, revision=revision, repo_type="model")
    except _HUB_LISTING_ERRORS:
        return _cached_hub_filenames(repo_id, revision)


# Closer to the original weights first. An unrecognized name sorts last, so a
# file that names its quant wins over an unlabeled sidecar. ``bf16`` is listed
# before ``f16`` because the check is a substring search.
_GGUF_QUANT_RANK = (
    "f32",
    "bf16",
    "f16",
    "q8_0",
    "q6_k",
    "q5_k_m",
    "q5_k_s",
    "q5_1",
    "q5_0",
    "q4_k_m",
    "q4_k_s",
    "q4_1",
    "q4_0",
    "q3_k_l",
    "q3_k_m",
    "q3_k_s",
    "q2_k",
    "iq4_nl",
    "iq4_xs",
    "iq3_m",
    "iq3_s",
    "iq3_xxs",
    "iq2_m",
    "iq2_s",
    "iq2_xs",
    "iq2_xxs",
    "iq1_m",
    "iq1_s",
)


@dataclass(frozen=True)
class _GGUFCheckpoint:
    """
    Arguments for loading one GGUF file with Transformers.

    ``model_id`` is a Hub repo id or a local directory. ``gguf_file`` is the
    file inside it, or an absolute path when the checkpoint is a local file.
    Both ``AutoTokenizer.from_pretrained`` and ``AutoModelForCausalLM.from_pretrained``
    take that pair.
    """

    model_id: str
    gguf_file: str
    # Other model files in the same checkpoint, not including projector sidecars.
    alternatives: tuple[str, ...] = ()


def _is_mmproj_gguf(name: str) -> bool:
    """Return whether ``name`` is a multimodal projector sidecar, not the model."""
    basename = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    compact = basename.replace("-", "").replace("_", "")
    return "mmproj" in compact


def _gguf_quant_rank(name: str) -> int:
    """Rank a GGUF filename by how close its quant is to the original weights."""
    basename = name.replace("\\", "/").rsplit("/", 1)[-1].lower().replace("-", "_")
    for index, token in enumerate(_GGUF_QUANT_RANK):
        if token in basename:
            return index
    return len(_GGUF_QUANT_RANK)


def _choose_gguf_file(names: list[str]) -> tuple[str, tuple[str, ...]]:
    """
    Pick the model GGUF file.

    Projector sidecars (``mmproj``) are skipped when a model file is present.
    Several model files resolve to the one closest to full precision.
    """
    model_files = [name for name in names if not _is_mmproj_gguf(name)]
    if not model_files:
        model_files = list(names)
    chosen = min(model_files, key=lambda name: (_gguf_quant_rank(name), name.lower()))
    alternatives = tuple(sorted(name for name in model_files if name != chosen))
    return chosen, alternatives


def _directory_gguf_checkpoint(root: Path) -> _GGUFCheckpoint | None:
    """Return the GGUF file to load from a local directory, or ``None``."""
    gguf_paths: list[Path] = []
    has_standard_weights = False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # The header is the format. A ``.gguf`` name that is not a GGUF file,
        # and a GGUF file that is not named ``.gguf``, are both possible.
        if _has_gguf_magic(path):
            gguf_paths.append(path)
            continue
        if _is_standard_weight_filename(path.name):
            has_standard_weights = True
    if has_standard_weights or not gguf_paths:
        return None

    chosen, alternatives = _choose_gguf_file(
        [str(path.resolve()) for path in gguf_paths]
    )
    return _GGUFCheckpoint(
        model_id=str(root),
        gguf_file=chosen,
        alternatives=alternatives,
    )


def _resolve_gguf_checkpoint(
    model: str,
    revision: str | None = None,
) -> _GGUFCheckpoint | None:
    """
    Return how to load ``model`` when it is a GGUF checkpoint, else ``None``.

    Local files are identified by the GGUF magic header. Local directories are
    identified by that header plus the absence of safetensors or PyTorch
    weights. Hub ids use the repo file listing (and, offline, the Hub cache):
    a ``.gguf`` file and no safetensors or PyTorch weights. A listing failure
    does not guess from the id string. A repo that also ships safetensors is
    left to the normal loader.
    """
    model = model.strip()
    if not model:
        return None

    path = Path(model).expanduser()
    try:
        if path.is_file():
            if not _has_gguf_magic(path):
                return None
            resolved = path.resolve()
            return _GGUFCheckpoint(
                model_id=str(resolved.parent),
                gguf_file=str(resolved),
            )
        if path.is_dir():
            return _directory_gguf_checkpoint(path.resolve())
    except OSError:
        return None

    filenames = _hub_filenames(model, revision)
    if filenames is None or not _filenames_are_gguf_only(filenames):
        return None

    gguf_names = [
        name
        for name in filenames
        if name.replace("\\", "/").rsplit("/", 1)[-1].lower().endswith(".gguf")
    ]
    if not gguf_names:
        return None
    chosen, alternatives = _choose_gguf_file(gguf_names)
    return _GGUFCheckpoint(
        model_id=model,
        gguf_file=chosen,
        alternatives=alternatives,
    )


def _is_gguf_checkpoint(model: str, revision: str | None = None) -> bool:
    """Return whether ``model`` should be loaded with ``gguf_file``."""
    return _resolve_gguf_checkpoint(model, revision=revision) is not None


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    # Set for multimodal models, None for text-only ones.
    processor: ProcessorMixin | None
    peft_config: LoraConfig
    dtype: torch.dtype
    # Whether the model's chat template accepts a system role (see __init__).
    supports_system_role: bool
    # Hub id or local directory passed to from_pretrained. A single local GGUF
    # file uses its parent directory, which is what Transformers accepts.
    pretrained_model_id: str
    # Set for a GGUF-only checkpoint. None means a normal safetensors load.
    gguf_file: str | None

    def __init__(self, settings: Settings):
        self.settings = settings
        self.needs_reload = False
        self._announced_gguf: set[tuple[str, str]] = set()
        self._gguf_alternatives: tuple[str, ...] = ()

        self.revision_kwargs = {}
        if settings.model_commit is not None:
            self.revision_kwargs["revision"] = settings.model_commit

        # Resolve before the tokenizer load. A GGUF-only repo has no
        # config.json or tokenizer files, so from_pretrained without
        # gguf_file fails with an unrelated sentencepiece/tiktoken error.
        self._bind_checkpoint()

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        # PreTrainedTokenizerBase is the "base class for all tokenizer backends"
        # according to the documentation.
        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                self.pretrained_model_id,
                **self.revision_kwargs,
                **self._gguf_file_kwargs(),
            ),
        )

        # Multimodal models have a processor we'll want to save.
        # A GGUF-only checkpoint has no processor files. A projector, when
        # the repo has one, is a separate mmproj file and is not loaded.
        self.processor = None
        if (
            self.gguf_file is None
            and get_model_class(self.pretrained_model_id) == AutoModelForImageTextToText
        ):
            self.processor = AutoProcessor.from_pretrained(
                self.pretrained_model_id,
                **self.revision_kwargs,
            )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        # Detect once whether this tokenizer's chat template accepts a system
        # role. Templates that don't (e.g. Gemma 2) need the system prompt folded
        # into the first user message instead (see _apply_system_role).
        self.supports_system_role = chat_template_supports_system_role(self.tokenizer)

        self.model = None
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )

        self.trusted_models = set()

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]...")

            try:
                self.model = get_model_class(
                    self.pretrained_model_id,
                    gguf_file=self.gguf_file,
                ).from_pretrained(
                    self.pretrained_model_id,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=True
                    if settings.model in self.trusted_models
                    else None,
                    **self._model_load_kwargs(dtype),
                )

                self.dtype = self.model.dtype

                # If we reach this point and the model requires trust_remote_code,
                # the user must have agreed when prompted to execute remote code,
                # because from_pretrained raises an exception otherwise.
                self.trusted_models.add(settings.model)

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(
                    [
                        Prompt(
                            system=settings.system_prompt,
                            user="What is 1+1?",
                        )
                    ],
                    max_new_tokens=1,
                    # The KV cache is deliberately left enabled here so we can
                    # catch dtype errors related to the cache, which we will need
                    # for generating responses.
                    # use_cache=False,
                )
            except Exception as error:
                self.model = None
                empty_cache()

                formatted = format_exception(error)
                if "\n" in formatted:
                    print(f"* [red]Failed:\n{formatted}[/]")
                else:
                    print(f"* [red]Failed ({formatted})[/]")

                continue

            if self.gguf_file is not None:
                print("* Dequantized GGUF checkpoint to dense weights")
            elif settings.quantization == QuantizationMethod.BNB_4BIT:
                print("* Quantized to 4-bit precision")

            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        check_model_resident(self.model)

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")

        all_components = {}
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                if component not in all_components:
                    all_components[component] = 0
                all_components[component] += len(modules)

        print("* Modifiable components:")
        for component, count in all_components.items():
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

    def apply_lora(self, lora_rank: int):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for modification (faster reload, no weight modification).
        # Collect actual leaf module names from the model for LoRA targeting.
        # This is more robust than splitting component keys (e.g. "attn.o_proj" -> "o_proj")
        # because hybrid models like Qwen3.5 MoE have modules with different names
        # across layers (e.g. "o_proj" on attention layers, "out_proj" on linear attention layers).
        target_modules_set: set[str] = set()

        module_id_to_full_name = {
            id(module): module_name
            for module_name, module in self.model.named_modules()
        }

        for layer_index in range(len(self.get_layers())):
            for modules in self.get_layer_modules(layer_index).values():
                for module in modules:
                    full_name = module_id_to_full_name.get(id(module))
                    if full_name is not None:
                        target_modules_set.add(full_name)

        target_modules = sorted(target_modules_set)

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type="CAUSAL_LM",
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        # PEFT initializes A randomly from the global RNG. B is zero, so this doesn't
        # change the model, but zeroing A too keeps saved adapters reproducible.
        self._zero_lora_weights()

    def _zero_lora_weights(self):
        for name, module in self.model.named_modules():
            if ("lora_A" in name or "lora_B" in name) and hasattr(module, "weight"):
                torch.nn.init.zeros_(module.weight)

    def _bind_checkpoint(self) -> None:
        """
        Point this model at ``settings.model``, including a GGUF file if that
        is what the checkpoint is.

        ``loaded_from`` is the settings string this bind used. ``reset_model``
        compares it so swapping in ``evaluate_model`` reloads, without listing
        the Hub again on the fast path.
        """
        checkpoint = _resolve_gguf_checkpoint(
            self.settings.model,
            revision=self.settings.model_commit,
        )
        self.loaded_from = self.settings.model
        if checkpoint is None:
            self.pretrained_model_id = self.settings.model
            self.gguf_file = None
            self._gguf_alternatives = ()
            return

        self.pretrained_model_id = checkpoint.model_id
        self.gguf_file = checkpoint.gguf_file
        self._gguf_alternatives = checkpoint.alternatives
        self._announce_gguf_dequantization()

    def _announce_gguf_dequantization(self) -> None:
        """Warn once per checkpoint that the loaded weights are a reconstruction."""
        assert self.gguf_file is not None
        key = (self.settings.model, self.gguf_file)
        if key in self._announced_gguf:
            return
        self._announced_gguf.add(key)

        extra = ""
        if self._gguf_alternatives:
            listed = ", ".join(self._gguf_alternatives)
            extra = (
                f" Other GGUF model files were left unused ({listed}). "
                "This load uses the file closest to full precision."
            )
        bnb = ""
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            bnb = (
                " bitsandbytes 4-bit quantization is not applied, because "
                "Transformers cannot combine it with loading a GGUF file."
            )
        print(
            "[yellow]WARNING: "
            f"'{self.settings.model}' is a GGUF checkpoint. Heretic loads "
            f"[bold]{self.gguf_file}[/] through Transformers with dequantization "
            "forced. The weights are a dense reconstruction of that quantized "
            "file, not the original full-precision weights. Saving and uploading "
            "write the reconstruction as safetensors. Architectures that "
            "Transformers cannot map from GGUF still fail to load."
            f"{extra}{bnb}[/]"
        )

    def _gguf_file_kwargs(self) -> dict[str, Any]:
        """``gguf_file`` for both the tokenizer and the model, or nothing."""
        if self.gguf_file is None:
            return {}
        return {"gguf_file": self.gguf_file}

    def _model_load_kwargs(self, dtype: str) -> dict[str, Any]:
        """
        Keyword arguments for ``AutoModelForCausalLM.from_pretrained``.

        A GGUF load always asks Transformers to dequantize. Packed GGUF
        modules (Qwen3.5, when the ggml kernel is installed) are not trainable
        and ``save_pretrained`` refuses them. ``GgufConfig(dequantize=True)``
        unpacks every weight into an ordinary ``Linear`` in the requested
        dtype. The legacy loader already dequantizes; this forces the packed
        architectures down the same path. bitsandbytes is a different
        ``quantization_config`` and cannot be combined with ``gguf_file``.
        """
        kwargs: dict[str, Any] = dict(self.revision_kwargs)
        kwargs.update(self._gguf_file_kwargs())
        if self.gguf_file is not None:
            kwargs["quantization_config"] = GgufConfig(dequantize=True)
            return kwargs

        quantization_config = self._get_quantization_config(dtype)
        # Only include quantization_config if it's not None
        # (some models like gpt-oss have issues with explicit None).
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        return kwargs

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig or None
        """
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        # A GGUF checkpoint was dequantized to dense Linear weights, so it
        # merges like any other full-precision model. bitsandbytes is not
        # applied on that path (Transformers rejects the combination).
        if (
            self.settings.quantization == QuantizationMethod.BNB_4BIT
            and self.gguf_file is None
        ):
            # Quantized models need special handling - we must reload the base model
            # in full precision to merge the LoRA adapters

            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=True
                if self.settings.model in self.trusted_models
                else None,
                **self.revision_kwargs,
            )

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base_model, self.peft_config)

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            # Non-quantized model - can merge directly
            print("* Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self) -> bool:
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.

        Returns True if the fast path was taken.
        """

        # If a prior model load was interrupted/cancelled mid-process, self.model will be None.
        current_model = None
        if self.model is not None:
            current_model = getattr(self.model.config, "name_or_path", None)

        if (
            self.model is not None
            and current_model == self.pretrained_model_id
            and self.loaded_from == self.settings.model
            and not self.needs_reload
        ):
            # Reset LoRA adapters to zero (identity transformation).
            # Zeroing A as well ensures that adapter weights of modules not modified
            # by the next trial don't depend on what previous trials did.
            self._zero_lora_weights()
            return True

        # Purge existing model object from memory to make space.
        self.model = None
        empty_cache()

        # settings.model may have been swapped for evaluate_model.
        self._bind_checkpoint()

        self.model = get_model_class(
            self.pretrained_model_id,
            gguf_file=self.gguf_file,
        ).from_pretrained(
            self.pretrained_model_id,
            dtype=self.dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=True
            if self.settings.model in self.trusted_models
            else None,
            **self._model_load_kwargs(str(self.dtype).split(".")[-1]),
        )

        check_model_resident(self.model)

        self.needs_reload = False

        return False

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers  # ty: ignore[unresolved-attribute, invalid-return-type]

        # Text-only models.
        return model.model.layers  # ty: ignore[unresolved-attribute, invalid-return-type]

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA)
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            else:
                # Assert for unexpected types (catches architecture changes)
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} - expected nn.Module"
                )

        # Standard self-attention out-projection (most models).
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_proj)  # ty: ignore[unresolved-attribute]

        # Qwen3.5 MoE hybrid layers use GatedDeltaNet (linear attention) instead of
        # standard self-attention, so self_attn.o_proj doesn't exist on those layers.
        with suppress(Exception):
            try_add("attn.o_proj", layer.linear_attn.out_proj)  # ty: ignore[unresolved-attribute]

        # DeepSeek V4.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_b_proj)  # ty: ignore[unresolved-attribute]

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)  # ty: ignore[unresolved-attribute]

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.down_proj)  # ty: ignore[unresolved-attribute]

        # Shared experts of MoE models (e.g. DeepSeek V4 and GLM 5.3).
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.shared_experts.down_proj)  # ty: ignore[unresolved-attribute]

        # Shared expert of MoE models (e.g. Qwen4-Exp).
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.shared_expert.down_proj)  # ty: ignore[unresolved-attribute]

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.w2)  # ty: ignore[unresolved-attribute]

        # LFM dense operator blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.conv.out_proj)  # ty: ignore[unresolved-attribute]

        with suppress(Exception):
            try_add("mlp.down_proj", layer.feed_forward.w2)  # ty: ignore[unresolved-attribute]

        # LFM transformer blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.out_proj)  # ty: ignore[unresolved-attribute]

        with suppress(Exception):
            for expert in layer.feed_forward.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.w2)  # ty: ignore[unresolved-attribute]

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)  # ty: ignore[unresolved-attribute]

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.output_linear)  # ty: ignore[unresolved-attribute]

        # We need at least one module across all components.
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No modifiable modules found in layer"

        return modules

    def get_modifiable_components(self) -> list[str]:
        components: set[str] = set()

        # Scan all layers because hybrid models (e.g. Qwen3.5 MoE) have different
        # components on different layers (some have self_attn, others linear_attn).
        for layer_index in range(len(self.get_layers())):
            components.update(self.get_layer_modules(layer_index).keys())

        return sorted(components)

    def _apply_system_role(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        # Templates that support the system role get the messages unchanged, so the
        # rendered chat (and therefore the resulting weights) is byte-for-byte
        # identical to before. For templates that reject it (e.g. Gemma 2), fold each
        # system message into the next user message (what Gemma's own later templates
        # do), so a system role anywhere in the chat is removed before rendering, not
        # just a leading one. Empty system prompts are dropped; a system message with
        # no following user message becomes a user turn.
        if self.supports_system_role:
            return messages
        result: list[dict[str, str]] = []
        pending: list[str] = []
        for message in messages:
            if message["role"] == "system":
                if message["content"]:
                    pending.append(message["content"])
                continue
            if pending and message["role"] == "user":
                message = dict(message)
                message["content"] = "\n\n".join([*pending, message["content"]])
                pending = []
            result.append(message)
        if pending:
            result.append({"role": "user", "content": "\n\n".join(pending)})
        return result

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        chats = [
            self._apply_system_role(
                [
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ]
            )
            for prompt in prompts
        ]

        # This cast is valid because list[str] is the return type
        # for batched operation with tokenize=False.
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
                # Overrides the function that templates call to get the current date.
                strftime_now=strftime_fixed_date,
            ),
        )

        if self.settings.response_prefix:
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [
                prompt + self.settings.response_prefix for prompt in chat_prompts
            ]

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            responses.extend(
                self.get_responses(
                    batch,
                    skip_special_tokens=skip_special_tokens,
                )
            )

        return responses

    def get_residuals(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # KV cache is unnecessary here because we only need the hidden states
            # for the first generated token.
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Hidden states for the first (only) generated token.
        # This cast is valid because we passed output_hidden_states=True above.
        hidden_states = cast(tuple[tuple[FloatTensor]], outputs.hidden_states)[0]

        # Models with Hyper-Connections (e.g. DeepSeek V4 and GLM 5.3) have several
        # residual streams, so their hidden states have shape
        # (prompt, position, stream, component). We use the mean of the streams.
        # The last entry has no stream dimension, because Transformers replaces it
        # with the model output, where the streams have already been merged.
        hidden_states = tuple(
            (
                layer_hidden_states.mean(dim=2)
                if layer_hidden_states.dim() == 4
                else layer_hidden_states
            )
            for layer_hidden_states in hidden_states
        )

        hidden_states = _collapse_flattened_residual_streams(
            cast(tuple[FloatTensor, ...], hidden_states),
            self.model.config,
        )

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        residuals = residuals.to(torch.float32)

        if 0 <= winsorization_quantile < 1:
            # Apply symmetric winsorization to each layer of the per-prompt residuals.
            abs_residuals = torch.abs(residuals)
            # Get the (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
            thresholds = torch.quantile(
                abs_residuals,
                winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)

        if self.settings.offload_outputs_to_cpu:
            residuals = residuals.cpu()
            empty_cache()

        return residuals

    def get_residuals_batched(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(
                self.get_residuals(
                    batch,
                    winsorization_quantile=winsorization_quantile,
                )
            )

        return torch.cat(residuals, dim=0)

    def get_residuals_mean(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty")

        running_sum = None
        total_count = 0

        for batch in batchify(prompts, self.settings.batch_size):
            batch_residuals = self.get_residuals(
                batch,
                winsorization_quantile=winsorization_quantile,
            )

            # Accumulate in high precision on CPU to reduce peak VRAM usage.
            batch_sum = batch_residuals.sum(dim=0, dtype=torch.float64).cpu()

            if running_sum is None:
                running_sum = batch_sum
            else:
                running_sum += batch_sum

            total_count += batch_residuals.shape[0]

        assert running_sum is not None

        return (running_sum / total_count).to(torch.float32)

    def get_module_io(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        # The list contains one element per layer.
        # Each element maps from the component name to a (possibly sparse) mapping
        # from the module index to an (input, output) tuple containing the I/O
        # tensors of shape (prompt, component).
        module_io: ModuleIO = []

        def get_hook(
            layer_index: int,
            component: str,
            module_index: int,
        ) -> Callable[[Module, tuple[Tensor, ...], Tensor], None]:
            def hook(
                module: Module,
                inputs: tuple[Tensor, ...],
                outputs: Tensor,
            ) -> None:
                if len(module_io) == layer_index:
                    # First invocation of the hook for this layer.
                    module_io.append({})

                # Layers are invoked in order during inference,
                # so this should always hold.
                assert len(module_io) == layer_index + 1

                if component not in module_io[layer_index]:
                    module_io[layer_index][component] = {}

                # Each module should be invoked at most once per inference step.
                assert module_index not in module_io[layer_index][component]

                # inputs[0] and outputs have shape (prompt, position, component),
                # so this extracts the input/output at the end of each prompt.
                # Move to CPU to decouple from device assignments, which can
                # change between model reloads in multi-GPU configurations.
                input = inputs[0][:, -1, :].detach().clone().cpu()
                output = outputs[:, -1, :].detach().clone().cpu()

                # The modules associated with a component (e.g. expert MLPs)
                # are not necessarily invoked in order, nor are all of them
                # necessarily invoked in each inference step, so we cannot
                # use a list here.
                module_io[layer_index][component][module_index] = (input, output)

            return hook

        hook_handles: list[RemovableHandle] = []

        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    hook_handles.append(
                        module.register_forward_hook(
                            get_hook(layer_index, component, module_index)
                        )
                    )

        self.generate(
            prompts,
            max_new_tokens=1,
            use_cache=False,
        )

        for hook_handle in hook_handles:
            hook_handle.remove()

        return module_io

    def get_module_io_batched(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        # Aggregating batch results is more complicated for module I/O
        # than for other get_*_batched methods, because the structure of the results
        # might differ between batches, as whether individual modules activate
        # can depend on the prompt (in particular for MoE models).
        # In practice, inhomogeneous results should be very rare, but to be fully
        # generic, this logic is required.
        module_io_batches: list[ModuleIO] = [
            self.get_module_io(batch)
            for batch in batchify(prompts, self.settings.batch_size)
        ]

        module_io: ModuleIO = []

        for layer_index in range(len(self.get_layers())):
            module_io.append({})

            for module_io_batch in module_io_batches:
                for component, io_map in module_io_batch[layer_index].items():
                    if component not in module_io[layer_index]:
                        module_io[layer_index][component] = {}

                    for module_index in io_map:
                        if module_index not in module_io[layer_index][component]:
                            # This is a placeholder; the actual aggregation happens below.
                            # We need to iterate over the batches twice because we don't
                            # know in advance which components and module indices are present.
                            module_io[layer_index][component][module_index] = (
                                torch.empty(0),
                                torch.empty(0),
                            )

            for component, io_map in module_io[layer_index].items():
                for module_index in io_map:
                    inputs_outputs = [
                        module_io_batch[layer_index][component][module_index]
                        for module_io_batch in module_io_batches
                        if component in module_io_batch[layer_index]
                        and module_index in module_io_batch[layer_index][component]
                    ]
                    input = torch.cat(
                        [input_output[0] for input_output in inputs_outputs],
                        dim=0,
                    )
                    output = torch.cat(
                        [input_output[1] for input_output in inputs_outputs],
                        dim=0,
                    )

                    # The key already exists, and replacing existing values
                    # in a dictionary while iterating over the same dictionary
                    # is safe in Python.
                    module_io[layer_index][component][module_index] = (input, output)

        return module_io

    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the raw logits over the vocabulary
        # at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # Use raw logits, not processed generation scores; processors can insert
        # -inf for suppressed tokens, which can make KL divergence evaluate to NaN.
        # This cast is valid because we passed output_logits=True above.
        logits = cast(tuple[FloatTensor], outputs.logits)[0]

        # The returned tensor has shape (prompt, token).
        if self.settings.offload_outputs_to_cpu:
            del outputs
            logits = logits.cpu()
            empty_cache()

        return logits

    def get_logits_batched(self, prompts: list[Prompt]) -> Tensor:
        logits = []

        for batch in batchify(prompts, self.settings.batch_size):
            logits.append(self.get_logits(batch))

        return torch.cat(logits, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        chat = self._apply_system_role(chat)
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        # This cast is valid because str is the return type
        # when passing a sequence of token IDs.
        return cast(
            str,
            self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        )
