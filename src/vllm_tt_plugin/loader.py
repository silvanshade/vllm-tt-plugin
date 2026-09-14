# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import atexit
import functools
import math

from torch import nn
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture

from vllm_tt_plugin.config import (
    get_tt_config,
    get_tt_data_parallel_size,
    get_tt_max_batch_size,
)
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

_FILL_CAST_INSTALLED = False


def install_paged_fill_cache_cast() -> None:
    """Cast prefill K/V to the paged cache dtype before filling it.

    ``paged_fill_cache`` copies tiles rather than converting, and asserts
    ``input.dtype() == cache.dtype()``. Its sibling ``paged_update_cache``
    refuses low-precision input and owns the repack itself, so the decode leg
    must keep handing it bfloat16 while the fill leg needs the cache dtype:
    the two write paths want different input dtypes against the same cache.

    Models therefore cannot satisfy both by choosing one activation dtype, and
    a low-precision KV cache is startup-fatal at the first prefill without this
    cast -- which is what a bfloat8_b or bfloat4_b cache is for, on a card where
    a bfloat16 cache costs 65,536 B per token.

    The cast lands in a separate short-lived tensor, never in place: the source
    K/V also feed the downstream attention ops, whose programs are compiled for
    their dtype. Deallocating only the materialized copy keeps every live view
    of the source valid.

    Identity when the dtypes already agree, so the default bfloat16 cache pays
    nothing, and installed once per process.
    """
    global _FILL_CAST_INSTALLED
    if _FILL_CAST_INSTALLED:
        return

    import ttnn

    original = ttnn.experimental.paged_fill_cache

    def paged_fill_cache(cache, fill, page_table, *args, **kwargs):
        if fill.dtype == cache.dtype:
            return original(cache, fill, page_table, *args, **kwargs)
        cast = ttnn.clone(
            fill, dtype=cache.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        try:
            return original(cache, cast, page_table, *args, **kwargs)
        finally:
            ttnn.deallocate(cast)

    ttnn.experimental.paged_fill_cache = paged_fill_cache
    _FILL_CAST_INSTALLED = True


_DTYPES = ("bfloat4_b", "bfloat8_b", "bfloat16")


def _resolve_dtype(name):
    import ttnn

    if name not in _DTYPES:
        raise ValueError(f"Unsupported tt dtype {name!r}, expected one of {_DTYPES}")
    return getattr(ttnn, name)


_WEIGHT_DOWNCAST_INSTALLED = False


def install_weight_downcast(dtype_name: str, min_elements: int = 1 << 20) -> None:
    """Store large 2-D weights at ``dtype_name`` instead of bfloat8_b.

    A model's weight dtype is chosen inside its own config, so the only seam a
    serving layer has is the conversion itself: every device weight arrives
    through ``ttnn.as_tensor`` or ``ttnn.from_torch``, and rewriting the dtype
    there reaches all of them without a model edit.

    Only bfloat8_b requests are rewritten, and only for 2-D tensors of at least
    ``min_elements``. That is the matmul weight population -- projections and
    MLP -- while norms, biases, rotary tables and anything the model already
    chose to keep at bfloat16 pass through untouched, which is the split the
    4-bit bring-up validated.

    The cache is keyed on the *requested* dtype, so a downcast run MUST have its
    own ``TT_CACHE_PATH``: a cached tensor is reloaded verbatim and this hook
    never sees it, so pointing a downcast run at a bfloat8_b cache silently
    serves 8-bit weights while reporting a downcast.
    """
    global _WEIGHT_DOWNCAST_INSTALLED
    if _WEIGHT_DOWNCAST_INSTALLED:
        return

    import ttnn

    target = _resolve_dtype(dtype_name)
    counts = {"downcast": 0, "kept": 0}

    def rewrite(tensor, kwargs):
        if kwargs.get("dtype") is not ttnn.bfloat8_b:
            return kwargs
        shape = tuple(getattr(tensor, "shape", ()))
        dims = [d for d in shape if d > 1]
        if len(dims) < 2 or math.prod(shape or (0,)) < min_elements:
            counts["kept"] += 1
            return kwargs
        counts["downcast"] += 1
        return {**kwargs, "dtype": target}

    def wrap(fn):
        @functools.wraps(fn)
        def wrapped(tensor, *args, **kwargs):
            return fn(tensor, *args, **rewrite(tensor, kwargs))

        return wrapped

    ttnn.as_tensor = wrap(ttnn.as_tensor)
    ttnn.from_torch = wrap(ttnn.from_torch)
    _WEIGHT_DOWNCAST_INSTALLED = True
    logger.info(
        "TT weight downcast installed: bfloat8_b -> %s for 2-D weights >= %d elements",
        dtype_name,
        min_elements,
    )
    atexit.register(
        lambda: logger.info(
            "TT weight downcast: %d downcast, %d kept",
            counts["downcast"],
            counts["kept"],
        )
    )


def override_kv_cache_dtype(model, dtype_name: str):
    """Allocate the model's paged KV cache at ``dtype_name``.

    The vLLM entry point on a TT model names its own cache dtype and takes no
    argument for it, so the serving layer substitutes one by calling the
    underlying allocator directly with the same arguments the entry point
    passes. Paired with the fill cast above, which is what makes a cache
    narrower than the prefill activations writable at all.

    Raises rather than degrading if the model is not shaped the way the
    substitution assumes: a silent fallback here would serve a 16-bit cache
    while the run reports 4-bit, and every memory number taken from it would
    be wrong.
    """
    dtype = _resolve_dtype(dtype_name)
    inner = getattr(model, "model", None)
    if not inner or not hasattr(inner[0], "allocate_kv_caches"):
        raise TypeError(
            f"Cannot set KV cache dtype on {type(model).__name__}: "
            "expected a .model list whose entries expose allocate_kv_caches"
        )

    def allocate_kv_cache(kv_cache_shape, _dtype, num_layers):
        target = inner[0]
        return target.allocate_kv_caches(
            kv_cache_shape, dtype, batch_size=target.args.max_batch_size
        )

    model.allocate_kv_cache = allocate_kv_cache
    logger.info("TT paged KV cache dtype overridden to %s", dtype_name)
    return model


class TTModelLoader(BaseModelLoader):
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load a model with the given configurations."""

        device_config = vllm_config.device_config
        model_class, _ = get_model_architecture(model_config)
        tt_config = get_tt_config(vllm_config)
        optimizations = tt_config.get("optimizations", None)
        if optimizations is not None:
            assert optimizations in [
                "performance",
                "accuracy",
            ], f"""Invalid optimizations configuration `{optimizations}`,
            allowed values are 'performance' or 'accuracy'"""

        tt_data_parallel = get_tt_data_parallel_size(vllm_config)
        max_batch_size = get_tt_max_batch_size(vllm_config)

        # Both before the model is built: the downcast has to be in place for
        # the weight conversions the constructor performs, and a model that
        # allocates its cache at load time needs a fill leg that can write
        # into a narrower cache.
        install_paged_fill_cache_cast()
        weight_dtype = tt_config.get("weight_dtype", None)
        if weight_dtype is not None:
            install_weight_downcast(weight_dtype)

        model = model_class.initialize_vllm_model(
            model_config.hf_config,
            device_config.device,
            max_batch_size,
            max_seq_len=model_config.max_model_len,
            tt_data_parallel=tt_data_parallel,
            optimizations=optimizations,
        )

        kv_cache_dtype = tt_config.get("kv_cache_dtype", None)
        if kv_cache_dtype is not None:
            override_kv_cache_dtype(model, kv_cache_dtype)
        return model

    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError
