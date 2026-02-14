# FP8/NVFP4 Quantization Support for Heretic

## Problem

Heretic previously rejected FP8/NVFP4 pre-quantized models (such as `cybermotaz/nemotron3-nano-nvfp4-w4a16`). When a user passed `--dtypes fp8`, the value `"fp8"` was forwarded directly to HuggingFace's `from_pretrained()` as the `dtype=` parameter. This failed because `fp8` is not a valid PyTorch dtype.

Pre-quantized NVFP4 models already carry a `quantization_config` in their `config.json` that HF Transformers (>= 4.57.3) can auto-detect. Heretic just needed to avoid passing an invalid dtype and instead let the auto-detection do its job.

Additionally, the NemotronH architecture (`NemotronHForCausalLM`) used by these models has a fundamentally different internal structure from standard transformers, requiring changes to layer discovery, module extraction, and component scanning.

## Solution

Two categories of changes were needed:

1. **FP8 dtype handling**: Treat `"fp8"` as a special token in the dtypes list. When detected, skip the `dtype=` keyword argument and use `torch_dtype=torch.bfloat16` instead, allowing HuggingFace to auto-detect the model's built-in quantization config.

2. **NemotronH architecture support**: Add fallback paths for the hybrid Mamba/MoE/Attention layer structure that NemotronH uses.

## Usage

```bash
heretic --dtypes fp8 --trust-remote-code true --model cybermotaz/nemotron3-nano-nvfp4-w4a16
```

## Files Changed

Five source files and one build file were modified.

---

### 1. `src/heretic/config.py`

**Added `FP8` variant to `QuantizationMethod` enum**

```python
class QuantizationMethod(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"
    FP8 = "fp8"            # <-- new
```

This enum value exists for future `--quantization fp8` on-the-fly quantization support. It is not required for loading pre-quantized models via `--dtypes fp8`, but keeps the enum complete.

**Updated `quantization` field description** to mention fp8 as an option.

---

### 2. `src/heretic/model.py` (core changes)

This file received the most changes, covering both FP8 dtype handling and NemotronH architecture support.

#### a. Module-level constant

```python
_FP8_DTYPE_TOKEN = "fp8"
```

A named constant used in all FP8 detection checks, avoiding scattered string literals.

#### b. `__init__()` -- dtype loading loop

Added a `self._loaded_dtype` field that records which dtype string successfully loaded the model. This is checked later in `reset_model()` and `get_merged_model()`.

The dtype loop now branches on `dtype == _FP8_DTYPE_TOKEN`:

| Path | `from_pretrained()` call |
|---|---|
| **FP8** | `torch_dtype=torch.bfloat16` (no `dtype=` arg) |
| **Normal** | `dtype=dtype` (original behavior) |

On successful FP8 load, a distinct status message is printed:

```
* Trying dtype fp8... Ok (FP8/NVFP4 pre-quantized)
```

#### c. `get_merged_model()`

The condition that triggers the CPU-reload merge path (required for quantized models where LoRA adapters can't be merged in-place) was expanded:

```python
# Before
if self.settings.quantization == QuantizationMethod.BNB_4BIT:

# After
if (
    self.settings.quantization == QuantizationMethod.BNB_4BIT
    or self._loaded_dtype == _FP8_DTYPE_TOKEN
):
```

FP8/NVFP4 models have quantized weights that can't be merged in-place, so they follow the same CPU-reload path as BNB_4BIT models: save adapter state, reload base model on CPU in full precision, apply adapters, merge.

#### d. `reset_model()`

Added an FP8 branch that uses `torch_dtype=torch.bfloat16` instead of `dtype=dtype` when reloading the model between trials or after `merge_and_unload()`:

```python
if self._loaded_dtype == _FP8_DTYPE_TOKEN:
    self.model = get_model_class(...).from_pretrained(
        ..., torch_dtype=torch.bfloat16, ...
    )
else:
    self.model = get_model_class(...).from_pretrained(
        ..., dtype=dtype, ...
    )
```

#### e. `_get_quantization_config()` -- edge case fix

If a user somehow combines `--quantization bnb_4bit` with `--dtypes fp8`, the original code would call `getattr(torch, "fp8")` and crash. Fixed by treating `"fp8"` the same as `"auto"`:

```python
# Before
if dtype == "auto":
    compute_dtype = torch.bfloat16

# After
if dtype == "auto" or dtype == _FP8_DTYPE_TOKEN:
    compute_dtype = torch.bfloat16
```

#### f. `get_layers()` -- backbone fallback

NemotronH uses `.backbone.layers` instead of `.model.layers`. A new fallback was added between the multimodal and text-only paths:

```python
# NemotronH and other backbone-based models.
with suppress(Exception):
    return model.backbone.layers
```

#### g. `get_layer_modules()` -- NemotronH hybrid layer patterns

NemotronH has 52 layers with 3 different types, all accessed through a unified `mixer` attribute. The actual model (`nemotron3-nano-nvfp4-w4a16`) uses this breakdown:

| Layer type | Count | Description |
|---|---|---|
| Mamba2 SSM | 23 | State-space model layers with `mixer.out_proj` |
| MoE (Mixture of Experts) | 23 | 128 experts + shared expert, each with `down_proj` |
| Attention | 6 | Standard attention with `mixer.o_proj` |

The existing `self_attn.o_proj` hard assertion was changed to `with suppress(Exception)` since NemotronH layers don't have `self_attn`.

Five new `with suppress(Exception)` blocks were added for NemotronH:

```python
# NemotronH attention layers (mixer.o_proj)
with suppress(Exception):
    try_add("attn.o_proj", layer.mixer.o_proj)

# NemotronH simple MLP layers (mixer.down_proj)
with suppress(Exception):
    try_add("mlp.down_proj", layer.mixer.down_proj)

# NemotronH MoE per-expert down_proj (follows heretic's MoE pattern)
with suppress(Exception):
    for expert in layer.mixer.experts:
        try_add("mlp.down_proj", expert.down_proj)

# NemotronH MoE shared expert down_proj
with suppress(Exception):
    try_add("mlp.down_proj", layer.mixer.shared_experts.down_proj)

# NemotronH Mamba2 SSM layers (mixer.out_proj)
with suppress(Exception):
    try_add("mamba.out_proj", layer.mixer.out_proj)
```

The MoE expert handling follows heretic's existing pattern for other MoE models (Qwen3, Phi-3.5-MoE, Granite MoE): all per-expert `down_proj` modules are included under the `"mlp.down_proj"` component name. Optuna's optimizer determines the appropriate abliteration weight.

The `"mamba.out_proj"` component name maps to LoRA target `"out_proj"` via the existing `comp.split(".")[-1]` logic in `_apply_lora()`.

#### h. `get_layer_modules()` -- removed hard assertion

The previous hard assertion `assert total_modules > 0` was removed. Layers with no recognized abliterable modules now return an empty dict instead of crashing. This is necessary because hybrid architectures may have layer types we don't yet support.

#### i. `get_abliterable_components()` -- scan all layers

Previously this only inspected layer 0:

```python
# Before
def get_abliterable_components(self) -> list[str]:
    return list(self.get_layer_modules(0).keys())
```

For NemotronH, layer 0 is a Mamba layer -- it would only return `{"mamba.out_proj"}`, missing the attention and MLP components entirely. This would break the Optuna parameter loop and the LoRA target list.

Now scans all layers to collect the union of component types, with a diagnostic warning for any unrecognized layers:

```python
# After
def get_abliterable_components(self) -> list[str]:
    all_components: dict[str, list[Module]] = {}
    n_layers = len(self.get_layers())
    skipped_layers: list[int] = []
    for layer_index in range(n_layers):
        layer_modules = self.get_layer_modules(layer_index)
        if not layer_modules:
            skipped_layers.append(layer_index)
            continue
        for component, mods in layer_modules.items():
            if component not in all_components:
                all_components[component] = mods

    if skipped_layers:
        # Diagnostic: show the layer type and children for debugging
        sample_idx = skipped_layers[0]
        sample_layer = self.get_layers()[sample_idx]
        child_names = [name for name, _ in sample_layer.named_children()]
        print(f"  Warning: {len(skipped_layers)}/{n_layers} layers ...")

    assert len(all_components) > 0, "No abliterable modules found in any layer."
    return list(all_components.keys())
```

#### j. `_get_hidden_states_via_hooks()` -- new method for hybrid architectures

NemotronH does not return `hidden_states` through `generate()` or `forward()` -- `outputs.hidden_states` is a tuple of `None` values. A new fallback method uses PyTorch forward hooks to capture per-layer outputs directly:

```python
def _get_hidden_states_via_hooks(self, inputs: BatchEncoding) -> list[Tensor]:
    captured: list[Tensor] = []

    def make_hook(idx: int):
        def hook(module: Module, args: Any, output: Any) -> None:
            if isinstance(output, tuple):
                captured.append(output[0].detach())
            else:
                captured.append(output.detach())
        return hook

    embedding_output: list[Tensor] = []

    def embedding_hook(module: Module, args: Any) -> None:
        if isinstance(args, tuple) and len(args) > 0 and isinstance(args[0], Tensor):
            embedding_output.append(args[0].detach())

    layers = self.get_layers()
    handles = []
    handles.append(layers[0].register_forward_pre_hook(embedding_hook))
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        self.model(**inputs)
    finally:
        for h in handles:
            h.remove()

    if embedding_output:
        return [embedding_output[0]] + captured
    return captured
```

Key design decisions:
- Uses `register_forward_hook` on each layer to capture post-layer hidden states
- Uses `register_forward_pre_hook` on the first layer to capture the embedding output (input to layer 0), matching the standard `output_hidden_states` format of `n_layers + 1` entries
- Pre-hooks receive `(module, args)` (NOT `(module, args, output)`) -- this was a bug that was caught and fixed during testing
- All hooks are cleaned up in a `finally` block to prevent leaks

#### k. `get_residuals()` -- hidden states fallback and multi-GPU device handling

The method now has a robust check for usable hidden states before falling back to hooks:

```python
has_hidden_states = (
    outputs.hidden_states is not None
    and len(outputs.hidden_states) > 0
    and outputs.hidden_states[0] is not None  # NemotronH returns tuple of Nones
)

if has_hidden_states:
    hidden_states = outputs.hidden_states[0]
else:
    hidden_states = self._get_hidden_states_via_hooks(inputs)
```

Additionally, multi-GPU setups place different layers on different CUDA devices. The hook-captured tensors retain their original device placement, so `torch.stack()` would fail. Fixed by moving all tensors to the first layer's device before stacking:

```python
target_device = hidden_states[0].device
residuals = torch.stack(
    [layer_hidden_states[:, -1, :].to(target_device) for layer_hidden_states in hidden_states],
    dim=1,
)
```

A diagnostic warning was also added for NaN detection in residuals:

```python
nan_layers = torch.isnan(residuals).any(dim=(0, 2)).nonzero().squeeze(-1).tolist()
if nan_layers:
    print(f"  Warning: NaN residuals in layers: {nan_layers}")
```

#### l. `abliterate()` -- meta-device and NaN guard

With multi-GPU + CPU offloading via accelerate's `device_map="auto"`, modules that don't fit in GPU VRAM are placed on the `meta` device (no actual tensor data). In testing, **1301 out of ~3000 modules** (primarily MoE experts on layers 31+) were on meta device. Writing LoRA weights to these modules would produce garbage/NaN output.

Added a guard that skips meta-device and NaN-weight modules:

```python
# Skip modules whose base weight is on meta device (no actual data)
# or contains NaN values (e.g. improperly dequantized FP8 weights).
if base_weight.device.type == "meta" or torch.isnan(W).any():
    continue
```

These modules keep their zero-initialized LoRA weights (the PEFT default), meaning they pass through the base layer output unmodified. Abliteration only applies to modules with actual GPU-resident weights.

#### m. `get_logprobs()` -- NaN diagnostic

Added a warning when logits contain NaN values after abliteration, to aid debugging:

```python
if torch.isnan(logits).any():
    print("  Warning: NaN values in logits (post-abliteration model corruption)")
```

#### n. `__init__` printout -- per-component layer counts

The startup printout previously assumed all layers have the same components (`"N modules per layer"`). Updated to show how many layers contain each component:

```
* Abliterable components:
  * mamba.out_proj: present in 23 layers
  * mlp.down_proj: present in 23 layers
  * attn.o_proj: present in 6 layers
```

---

### 3. `src/heretic/evaluator.py`

**Fixed division-by-zero in `get_score()`**

When the model has zero initial refusals (`base_refusals == 0`), the refusal score calculation would divide by zero. Added a guard:

```python
# Before
refusals_score = refusals / self.base_refusals

# After
refusals_score = refusals / self.base_refusals if self.base_refusals > 0 else 0.0
```

This can occur with small evaluation prompt sets or models that are already fully uncensored.

---

### 4. `src/heretic/main.py`

**`obtain_merge_strategy()` signature updated** to accept a `Model` instance:

```python
# Before
def obtain_merge_strategy(settings: Settings) -> str | None:

# After
def obtain_merge_strategy(settings: Settings, model: Model) -> str | None:
```

The function now checks for FP8 quantized models alongside BNB_4BIT using a unified `is_quantized` flag:

```python
is_quantized = (
    settings.quantization == QuantizationMethod.BNB_4BIT
    or model._loaded_dtype == _FP8_DTYPE_TOKEN
)
```

This flag drives:
- Whether the RAM warning and memory estimate are shown before merging
- Whether the merge option label includes "(requires sufficient RAM)"

Both call sites (local save and HuggingFace upload) were updated to pass `model`.

---

### 5. `pyproject.toml`

Added an optional dependency group for future on-the-fly FP8 quantization:

```toml
[project.optional-dependencies]
fp8 = [
    "fp-quant>=0.1.0",
]
```

This is **not required** for loading pre-quantized NVFP4 models. It is provided for users who want to install the `fp-quant` package for future on-the-fly FP8 quantization support. Install with:

```bash
pip install heretic-llm[fp8]
```

No changes to core dependencies were needed -- `transformers >= 4.57.3` (already a dependency) handles NVFP4 auto-detection natively.

---

## How It Works End-to-End

1. User runs `heretic --dtypes fp8 --model <nvfp4-model>`
2. The dtype loop hits `"fp8"` and calls `from_pretrained(torch_dtype=torch.bfloat16)` without passing `dtype=`
3. HF Transformers reads `quantization_config` from the model's `config.json` and loads the NVFP4-quantized weights automatically
4. A test generation (`"What is 1+1?"`) verifies the model works
5. `get_layers()` finds layers via `model.backbone.layers` (NemotronH path)
6. `get_abliterable_components()` scans all 52 layers and discovers 3 component types:
   - `mamba.out_proj` (23 Mamba2 SSM layers)
   - `mlp.down_proj` (23 MoE layers, 128 experts + 1 shared expert each)
   - `attn.o_proj` (6 attention layers)
7. LoRA adapters are attached to `out_proj`, `down_proj`, and `o_proj` targets
8. `get_residuals()` captures hidden states via forward hooks (since NemotronH doesn't return `hidden_states` through `generate()`), moving all tensors to the same device for multi-GPU compatibility
9. Abliteration proceeds per-layer -- each layer only abliterates the components it has; modules on meta device (CPU-offloaded) are skipped to prevent NaN corruption
10. Optuna optimizes weights for all 3 component types independently
11. Reset/merge paths use `torch_dtype=torch.bfloat16` consistently

## Verified Output

```
Loading model ./models/nemotron-fp8...
Ok (FP8/NVFP4 pre-quantized)
* LoRA adapters initialized (targets: out_proj, down_proj, o_proj)
* Transformer model with 52 layers
* Abliterable components:
  * mamba.out_proj: present in 23 layers
  * mlp.down_proj: present in 23 layers
  * attn.o_proj: present in 6 layers
```

## NemotronH Architecture Reference

```
NemotronHForCausalLM
  .backbone                          (NemotronHModel)
    .embeddings                      (nn.Embedding)
    .layers                          (nn.ModuleList, 52 layers)
      [Mamba layers]                 NemotronHBlock
        .norm                        NemotronHRMSNorm
        .mixer                       Mamba2Mixer
          .in_proj                   nn.Linear
          .out_proj                  nn.Linear       <-- abliterated
          .conv1d                    nn.Conv1d
      [MoE layers]                   NemotronHBlock
        .norm                        NemotronHRMSNorm
        .mixer                       NemotronHMOE
          .gate                      NemotronHTopkRouter
          .experts                   ModuleList (128 experts)
            [each].up_proj           nn.Linear
            [each].down_proj         nn.Linear       <-- abliterated
          .shared_experts            NemotronHMLP
            .up_proj                 nn.Linear
            .down_proj               nn.Linear       <-- abliterated
      [Attention layers]             NemotronHBlock
        .norm                        NemotronHRMSNorm
        .mixer                       NemotronHAttention
          .q_proj                    nn.Linear
          .k_proj                    nn.Linear
          .v_proj                    nn.Linear
          .o_proj                    nn.Linear       <-- abliterated
    .norm_f                          NemotronHRMSNorm
  .lm_head                           nn.Linear
```

## Known Limitations

- **GPU memory / CPU offloading**: The full model (~34GB weights) exceeds single-GPU VRAM. With dual GPUs (RTX 4080 16GB + RTX A5000 24GB = 40GB), accelerate's `device_map="auto"` still offloads ~1301 MoE expert modules to CPU (meta device). This causes one CPU core to max out at 100% as weights are shuffled between CPU and GPU during inference, resulting in ~11 min/trial. Modules on meta device are skipped during abliteration (their LoRA weights stay at zero). A third GPU or GPUs with more VRAM would eliminate this bottleneck.
- **`torch_dtype` deprecation warning**: The model's `from_pretrained()` prints `` `torch_dtype` is deprecated! Use `dtype` instead! ``. This is cosmetic -- `torch_dtype=torch.bfloat16` is used deliberately to avoid passing `dtype="fp8"` (which is not a valid PyTorch dtype). The warning comes from the model's custom loading code.
- **UNEXPECTED weight_scale/input_scale keys**: The LOAD REPORT lists FP8 quantization scale tensors as "UNEXPECTED". These are metadata from the NVFP4 quantization format that HF Transformers handles internally. They can be safely ignored.
- **On-the-fly FP8 quantization** (`--quantization fp8`) is declared in the enum but not yet implemented. The current support is limited to loading pre-quantized models via `--dtypes fp8`.
- **Mamba layer abliteration**: Abliterating `out_proj` in Mamba2 SSM layers is analogous to abliterating `o_proj` in attention layers, but the effectiveness of directional abliteration on SSM layers is uncharted territory.
- **MoE expert abliteration**: All 128 per-expert `down_proj` modules receive the same abliteration treatment per Heretic's standard MoE pattern. Optuna's dual-objective optimizer (refusals + KL divergence) will find the appropriate weight to avoid damaging expert specialization.
- **Partial abliteration with CPU offloading**: When modules are on meta device, they are skipped during abliteration. This means the abliteration is incomplete -- only GPU-resident modules (~1700 out of ~3000) are modified. This may reduce abliteration effectiveness but preserves model stability.

## Bugs Found and Fixed During Testing

| Bug | Symptom | Fix |
|---|---|---|
| `dtype="fp8"` passed to `from_pretrained()` | `fp8` is not a valid PyTorch dtype | Branch on `_FP8_DTYPE_TOKEN`, use `torch_dtype=torch.bfloat16` |
| `getattr(torch, "fp8")` in `_get_quantization_config()` | Crash when combining `--quantization bnb_4bit` with `--dtypes fp8` | Treat `"fp8"` same as `"auto"` |
| `model.model.layers` on NemotronH | `'NemotronHForCausalLM' has no attribute 'model'` | Add `model.backbone.layers` fallback |
| `assert total_modules > 0` on hybrid layers | `AssertionError` on layers without `self_attn`/`mlp` | Remove assertion, return empty dict |
| Layer 0 only in `get_abliterable_components()` | Only discovers `mamba.out_proj`, misses MoE and attention | Scan all layers for union of components |
| `outputs.hidden_states` is tuple of Nones | `TypeError: NoneType is not iterable` | Add hooks-based fallback `_get_hidden_states_via_hooks()` |
| `register_forward_pre_hook` callback signature | `embedding_hook() missing 1 required positional argument` | Pre-hooks take `(module, args)` not `(module, args, output)` |
| Multi-GPU device mismatch in `torch.stack()` | `tensors on cuda:1, different from cuda:0` | `.to(target_device)` before stacking |
| Meta-device weights in `abliterate()` | NaN KL divergence (NaN logits after abliteration) | Skip modules with `base_weight.device.type == "meta"` |
| `base_refusals == 0` division | `ZeroDivisionError` in evaluator | Guard with `if self.base_refusals > 0 else 0.0` |
