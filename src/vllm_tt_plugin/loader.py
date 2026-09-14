# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import contextlib
import functools
import math

import torch
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


def compile_prefill_before_traces(model):
    """Run the model's prefill compile in the runner's untraced warmup phase.

    ``TTModelRunner.warmup_model`` is two-phase: Phase 1 calls both warmups with
    ``enable_trace=False`` so every program lands in the cache while no trace is
    parked, then Phase 2 captures traces against an already-compiled cache. A
    program compiled after a trace is parked owns device buffers the replay
    overwrites, and its next launch wedges the device with no error
    (``tt-metal#48536``; the model's ``warmup_prefill_masked_buckets`` docstring
    states the same "before any trace is parked" rule).

    ``Qwen36ForCausalLM.warmup_model_prefill`` returns at once when
    ``enable_trace`` is false, so Phase 1 compiles nothing for prefill and the
    whole prefill program set compiles in Phase 2, after the decode trace is
    parked. ``TT_METAL_TRACE_ALLOC_TRACKING=1`` shows it: 874 buffers alive at
    the first decode replay, 871 allocated from ``warmup_model_prefill``. The
    first prompt runs on intact programs; its decode replays corrupt them; the
    second prompt relaunches them and hangs.

    This makes the untraced call do the model's own warmup with
    ``capture_chunk_trace=False`` (the knob ``capture_prefill_trace_chunked``
    offers for exactly this), and leaves the traced call as it was. The model
    is then in traced-prefill serving state, which only ``trace_mode: all``
    serves; ``load_model`` refuses ``decode_only`` for this model.
    """
    inner = getattr(model, "model", None)
    target = inner[0] if inner else None
    if target is None or not hasattr(target, "capture_prefill_trace_chunked"):
        raise TypeError(
            f"Cannot compile prefill before traces on {type(model).__name__}: "
            "expected .model[0] to expose capture_prefill_trace_chunked"
        )
    traced_warmup = model.warmup_model_prefill

    def warmup_model_prefill(kv_cache, enable_trace, *args, **kwargs):
        if enable_trace:
            return traced_warmup(kv_cache, enable_trace, *args, **kwargs)
        if getattr(model, "already_warmed_up_prefill", False):
            return None
        model.already_warmed_up_prefill = True
        if not kv_cache:
            raise RuntimeError("prefill compile needs the allocated KV cache")
        # Same page-table sizing as the model's traced warmup, so the programs
        # compiled here are the ones the capture and the requests replay.
        num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        logger.info(
            "TT prefill compile before traces: page_table_blocks=%d", num_blocks
        )
        target.capture_prefill_trace_chunked(
            model.mesh_device, page_table, capture_chunk_trace=False
        )
        return None

    model.warmup_model_prefill = warmup_model_prefill
    return model


def override_prefill_chunk_width(model, width: int):
    """Split long prompts into ``width``-token GDN chunks.

    A Qwen3.5 model hands a prompt of at most 1024 tokens to the gated
    delta-net in one piece, and anything longer in chunks of a width the model
    fixes itself -- wider than the longest prompt the short path ever passes.
    The relayout that feeds the delta-rule kernel untilizes into L1 at a size
    linear in that width, next to circular buffers the tree's own comment
    sizes at ~1.36 MB per core, so the wide chunk is the first shape that can
    exhaust a core's L1. Starving a circular buffer stalls the reservation
    rather than failing it, which is why the symptom is a hang with no error.

    Narrowing the chunk keeps every delta-net call at a shape the short path
    already runs, and costs only more chunks per prompt: the window, the
    weights and the cache dtype are untouched.

    Raises rather than degrading if the entry point is missing, and reports
    the width each entry point actually receives rather than the width it was
    asked for: the model logs its own warmup chunk from a module constant, so
    a run whose pin silently failed to reach the call still prints the pinned
    figure and reads as configured.
    """
    if width < 1:
        raise ValueError(f"Prefill chunk width must be positive, got {width}")
    if width % 128:
        # The single-device capture asserts this, and the paged page-table
        # slice floor-divides by a 64-token block, so an unaligned width
        # either trips the assert or silently mismaps blocks.
        raise ValueError(f"Prefill chunk width must be a multiple of 128, got {width}")
    inner = getattr(model, "model", None)
    target = inner[0] if inner else None
    chunked = getattr(target, "prefill_layer_chunked", None)
    capture = getattr(target, "capture_prefill_trace_chunked", None)
    if chunked is None or capture is None:
        raise TypeError(
            f"Cannot set prefill chunk width on {type(model).__name__}: "
            "expected a .model list whose entries expose "
            "prefill_layer_chunked and capture_prefill_trace_chunked"
        )

    def pin(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # Positional chunk_size differs between the two entry points, and
            # both are called by keyword in the tree, so pin the keyword and
            # let a positional caller surface as the TypeError it is.
            logger.info(
                "TT prefill chunk width: %s called at %d (model asked for %s)",
                fn.__name__,
                width,
                kwargs.get("chunk_size", "default"),
            )
            return fn(*args, **{**kwargs, "chunk_size": width})

        return wrapped

    target.prefill_layer_chunked = pin(chunked)
    target.capture_prefill_trace_chunked = pin(capture)
    logger.info("TT prefill chunk width pinned to %d tokens", width)
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

        if hasattr(getattr(model, "model", [None])[0], "capture_prefill_trace_chunked"):
            if tt_config.get("trace_mode", "all") == "decode_only":
                raise ValueError(
                    "trace_mode 'decode_only' is unsupported for this model: its "
                    "eager multi-chunk prefill compiles per prompt shape with no "
                    "warmup, so the first prompt above 1024 tokens compiles under "
                    "the parked decode trace and wedges the device. Use 'all' "
                    "(warmed, traced prefill) or 'none' (no traces)."
                )
            compile_prefill_before_traces(model)

        chunk_width = tt_config.get("prefill_chunk_width", None)
        if chunk_width is not None:
            override_prefill_chunk_width(model, int(chunk_width))
        return model

    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError
