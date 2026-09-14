# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import contextlib
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

    **Warmup.** A model warms its fill-width-keyed programs before parking a
    prefill trace, because a program compiled after a trace is parked clobbers
    it and the symptom is a hang, not an error. Warmup builds its fill tensor
    at the *cache* dtype, so against a narrowed cache it takes the identity
    branch and the cast never compiles -- then the first real request, whose
    K/V are bfloat16, compiles it at exactly the wrong moment.

    The identity branch therefore compiles the cast itself, once per fill
    shape, whenever the cache is narrower than the bfloat16 the model fills
    with at request time. The probe is cloned from the warmup fill, at that
    fill's own memory config, so it keys a program on the same shape, layout
    and placement the request will present -- which holds exactly as far as
    the warmup fill matches the request K/V, the premise the model's own fill
    warmup already stands on. Callers that match at bfloat16 never enter this
    path.
    """
    global _FILL_CAST_INSTALLED
    if _FILL_CAST_INSTALLED:
        return

    import ttnn

    original = ttnn.experimental.paged_fill_cache
    warmed: set[tuple[int, ...]] = set()

    def paged_fill_cache(cache, fill, page_table, *args, **kwargs):
        if fill.dtype != cache.dtype:
            cast = ttnn.clone(
                fill, dtype=cache.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            try:
                return original(cache, cast, page_table, *args, **kwargs)
            finally:
                ttnn.deallocate(cast)

        shape = tuple(fill.shape)
        if cache.dtype is not ttnn.bfloat16 and shape not in warmed:
            warmed.add(shape)
            probe = ttnn.clone(
                fill, dtype=ttnn.bfloat16, memory_config=fill.memory_config()
            )
            try:
                cast = ttnn.clone(
                    probe, dtype=cache.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.deallocate(cast)
            finally:
                ttnn.deallocate(probe)
        return original(cache, fill, page_table, *args, **kwargs)

    ttnn.experimental.paged_fill_cache = paged_fill_cache
    _FILL_CAST_INSTALLED = True


_DTYPES = ("bfloat4_b", "bfloat8_b", "bfloat16")


def _resolve_dtype(name):
    import ttnn

    if name not in _DTYPES:
        raise ValueError(f"Unsupported tt dtype {name!r}, expected one of {_DTYPES}")
    return getattr(ttnn, name)


_WEIGHT_DOWNCAST_DTYPE: str | None = None


@contextlib.contextmanager
def weight_downcast(dtype_name: str, min_elements: int = 1 << 20):
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

    A tensor cache keyed on the dtype that reaches ``as_tensor`` needs no
    special handling: the rewrite happens before the key is built, so a
    downcast run reads and writes its own entries and cannot pick up an
    8-bit one.

    Scoped to the conversion it exists for, and restored after. Weight loading
    is not the only caller of these two functions -- a model builds device
    tensors at run time too, and some of those are large enough to match the
    predicate -- so leaving the rewrite installed would silently narrow tensors
    that are not weights.
    """
    global _WEIGHT_DOWNCAST_DTYPE
    if _WEIGHT_DOWNCAST_DTYPE is not None:
        # Re-entry can only mean two models in one process wanting different
        # weight dtypes, and the nesting would give the inner one both.
        raise RuntimeError(
            f"Weight downcast to {_WEIGHT_DOWNCAST_DTYPE} is already active; "
            f"cannot nest a downcast to {dtype_name}"
        )

    import ttnn

    target = _resolve_dtype(dtype_name)
    counts = {"downcast": 0, "kept": 0}
    as_tensor, from_torch = ttnn.as_tensor, ttnn.from_torch

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

    # as_tensor resolves from_torch through the module attribute at call time,
    # so the inner call lands on the wrapper too. Both entry points need
    # wrapping -- a model calls from_torch directly as well -- so the outer
    # call claims the conversion and the inner one passes the already-decided
    # dtype through, leaving one decision and one audit entry per tensor.
    depth = 0

    def wrap(fn):
        @functools.wraps(fn)
        def wrapped(tensor, *args, **kwargs):
            nonlocal depth
            if depth:
                return fn(tensor, *args, **kwargs)
            depth += 1
            try:
                return fn(tensor, *args, **rewrite(tensor, kwargs))
            finally:
                depth -= 1

        return wrapped

    ttnn.as_tensor = wrap(as_tensor)
    ttnn.from_torch = wrap(from_torch)
    _WEIGHT_DOWNCAST_DTYPE = dtype_name
    logger.info(
        "TT weight downcast active: bfloat8_b -> %s for 2-D weights >= %d elements",
        dtype_name,
        min_elements,
    )
    try:
        yield
    finally:
        ttnn.as_tensor, ttnn.from_torch = as_tensor, from_torch
        _WEIGHT_DOWNCAST_DTYPE = None
        logger.info(
            "TT weight downcast complete: %d downcast, %d kept",
            counts["downcast"],
            counts["kept"],
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
    if hasattr(model, "allocate_kv_cache_per_layer"):
        # The runner prefers the per-layer entry point where a model has one,
        # and would then never reach the substitution below -- reporting the
        # override while serving the model's own dtype.
        raise TypeError(
            f"{type(model).__name__} exposes allocate_kv_cache_per_layer, "
            "which this substitution does not cover"
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

        # The fill cast stays for the process: it is on the request path, not
        # just the load path. The downcast is scoped to the conversions the
        # constructor performs.
        install_paged_fill_cache_cast()
        weight_dtype = tt_config.get("weight_dtype", None)
        with (
            weight_downcast(weight_dtype)
            if weight_dtype is not None
            else contextlib.nullcontext()
        ):
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
