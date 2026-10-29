# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from .model import Model
from .utils import Prompt, batchify


def _map_tensors(value: Any, transform: Callable[[Tensor], Tensor]) -> Any:
    if isinstance(value, Tensor):
        return transform(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, transform) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, transform) for key, item in value.items()}
    if value is None or isinstance(
        value, (bool, int, float, str, torch.dtype, torch.device)
    ):
        return value
    raise ValueError(
        f"Streaming cannot replay layer argument of type {type(value).__name__}. "
        "Layers must support stateless execution with use_cache=False."
    )


@dataclass
class LayerCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    positional_hidden: bool


@dataclass
class BoundaryBatch:
    hidden: Tensor
    calls: list[LayerCall]


class LayerStream:
    """Replay native decoder calls while retaining one full boundary on CPU.

    Capture masks, rotary embeddings and other layer arguments from a native
    forward pass. This preserves architecture-specific calling conventions,
    including hybrid models whose layers receive different attention masks.
    No recurrent or attention cache is reused between replays.

    Collect a projection's I/O, edit it, then collect the next projection.
    Call advance() only after all edits to the current layer are complete.
    """

    def __init__(self, model: Model, prompts: list[Prompt]):
        if not prompts:
            raise ValueError("Streaming requires at least one prompt")
        self.layers = model.get_layers()
        if any(
            parameter.device.type == "meta"
            for layer in self.layers
            for parameter in layer.parameters()
        ):
            raise ValueError("Streaming does not support disk-offloaded decoder layers")
        self.layer_index = 0
        self.batches: list[BoundaryBatch] = []
        model.model.eval()

        for batch in batchify(prompts, model.settings.batch_size):
            calls: list[LayerCall] = []
            first_hidden: list[Tensor] = []
            # Masks and position embeddings are often shared between layers.
            # Keep one CPU copy per tensor and retain its identity during capture.
            copies: dict[int, tuple[Tensor, Tensor]] = {}

            def offload(tensor: Tensor) -> Tensor:
                if id(tensor) not in copies:
                    copies[id(tensor)] = (tensor, tensor.detach().to("cpu", copy=True))
                return copies[id(tensor)][1]

            def capture(index: int):
                def hook(module: Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
                    if index != len(calls):
                        raise ValueError(
                            "Streaming requires decoder layers called once in order"
                        )
                    positional_hidden = bool(args)
                    hidden = (
                        args[0] if positional_hidden else kwargs.get("hidden_states")
                    )
                    if not isinstance(hidden, Tensor) or hidden.ndim != 3:
                        raise ValueError("Streaming requires batch-first hidden states")
                    if index == 0:
                        first_hidden.append(hidden.detach().to("cpu", copy=True))
                    kwargs = dict(kwargs)
                    if not positional_hidden:
                        kwargs.pop("hidden_states")
                    calls.append(
                        LayerCall(
                            args=_map_tensors(
                                args[1:] if positional_hidden else args, offload
                            ),
                            kwargs=_map_tensors(kwargs, offload),
                            positional_hidden=positional_hidden,
                        )
                    )

                return hook

            handles = [
                layer.register_forward_pre_hook(capture(index), with_kwargs=True)
                for index, layer in enumerate(self.layers)
            ]
            try:
                with torch.no_grad():
                    model.generate(batch, max_new_tokens=1, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
                copies.clear()

            if len(calls) != len(self.layers) or not first_hidden:
                raise ValueError("Streaming did not capture every decoder layer")
            self.batches.append(BoundaryBatch(first_hidden[0], calls))

    @torch.no_grad()
    def _run(self, batch: BoundaryBatch) -> Tensor:
        layer = self.layers[self.layer_index]
        # Accelerate hooks handle further dispatch within the layer, if needed.
        device = next(layer.parameters()).device
        if device.type == "meta":
            raise ValueError("Streaming does not support disk-offloaded decoder layers")
        call = batch.calls[self.layer_index]
        args = _map_tensors(call.args, lambda tensor: tensor.to(device))
        kwargs = _map_tensors(call.kwargs, lambda tensor: tensor.to(device))
        hidden = batch.hidden.to(device)
        if call.positional_hidden:
            args = (hidden, *args)
        else:
            kwargs["hidden_states"] = hidden
        output = layer(*args, **kwargs)
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, Tensor) or output.shape != hidden.shape:
            raise ValueError(
                "Streaming requires layers that return batch-first hidden states"
            )
        return output

    def collect(self, module: Module) -> tuple[Tensor, Tensor]:
        """Collect prompt-final I/O using all edits committed so far."""
        inputs_outputs: list[tuple[Tensor, Tensor]] = []

        def hook(module: Module, args: tuple[Tensor, ...], output: Tensor):
            if not args or args[0].ndim != 3 or output.ndim != 3:
                raise ValueError(
                    "Streaming requires projections with one vector per prompt. "
                    "Token-routed expert projections are not supported."
                )
            inputs_outputs.append(
                (
                    args[0][:, -1, :].detach().to("cpu", copy=True),
                    output[:, -1, :].detach().to("cpu", copy=True),
                )
            )

        handle = module.register_forward_hook(hook)
        try:
            for batch in self.batches:
                count = len(inputs_outputs)
                self._run(batch)
                if (
                    len(inputs_outputs) != count + 1
                    or inputs_outputs[-1][0].shape[0] != batch.hidden.shape[0]
                ):
                    raise ValueError(
                        "Streaming requires each projection to run once per prompt"
                    )
        finally:
            handle.remove()

        return (
            torch.cat([item[0] for item in inputs_outputs]),
            torch.cat([item[1] for item in inputs_outputs]),
        )

    def advance(self) -> None:
        """Replay the edited block over every token and release its old boundary."""
        for batch in self.batches:
            batch.hidden = self._run(batch).detach().to("cpu", copy=True)
        self.layer_index += 1
