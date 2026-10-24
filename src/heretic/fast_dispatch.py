# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Speed up Accelerate's model dispatch.

Accelerate clears the CUDA allocator cache for every tensor it moves. For
models with hundreds of thousands of tensors this dominates load time.
Skipping those cache clears is safe when enough VRAM is available; garbage
collection is still performed when requested.
"""

import gc
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def fast_dispatch() -> Iterator[None]:
    try:
        from accelerate import hooks
        from accelerate.utils import memory, modeling
    except ImportError:
        yield
        return

    def clear_device_cache(garbage_collection: bool = False) -> None:
        if garbage_collection:
            gc.collect()

    modules = [hooks, memory, modeling]
    originals = [
        (module, module.clear_device_cache)
        for module in modules
        if hasattr(module, "clear_device_cache")
    ]
    for module, _ in originals:
        module.clear_device_cache = clear_device_cache
    try:
        yield
    finally:
        for module, original in originals:
            module.clear_device_cache = original
