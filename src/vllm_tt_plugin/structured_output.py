# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.v1.structured_output.backend_types import StructuredOutputGrammar
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


logger = init_tt_logger(__name__)

# Track missing request IDs we've already warned about, so each ID logs once.
_warned_missing_request_ids: set[str] = set()


def install_tt_compact_json_patch() -> None:
    """Apply compact JSON to resolved backends without invalidating config.

    # Specification
    - provides: xgrammar and guidance disallow arbitrary JSON whitespace.
    - preserves: configured backend selection, fallback and config revalidation.
    - intension: installs once; defers policy until grammar compilation.
    """
    import vllm.v1.structured_output as structured_output

    if hasattr(structured_output, "_tt_original_create_grammar"):
        return
    original = structured_output.StructuredOutputManager._create_grammar
    structured_output._tt_original_create_grammar = original

    def create_compact_grammar(
        manager: StructuredOutputManager, request: Request
    ) -> StructuredOutputGrammar:
        backend = manager.backend
        if isinstance(
            backend,
            (structured_output.XgrammarBackend, structured_output.GuidanceBackend),
        ):
            backend.disable_any_whitespace = True
        return original(manager, request)

    structured_output.StructuredOutputManager._create_grammar = create_compact_grammar


def has_structured_outputs(
    requests: Mapping[str, CachedRequestState],
    scheduler_output: SchedulerOutput,
    bitmask: torch.Tensor | None,
) -> bool:
    """True if any request scheduled this step constrains its tokens via
    structured outputs: a grammar bitmask, pending structured tokens, or a
    scheduled request carrying ``structured_outputs`` sampling params."""
    if bitmask is not None or scheduler_output.pending_structured_output_tokens:
        return True
    return any(
        (req := requests.get(req_id)) is not None
        and req.sampling_params is not None
        and req.sampling_params.structured_outputs is not None
        for req_id in scheduler_output.num_scheduled_tokens
    )


def reorder_grammar_bitmask_for_tt_batch(
    *,
    bitmask: torch.Tensor,
    structured_output_request_ids: Sequence[str],
    row_req_ids: Sequence[str | None],
    batch_length: int,
) -> torch.Tensor:
    """Reorder scheduler bitmask rows into the TT batch layout.

    Warn once per process for each request ID that is missing a remapped row.
    """
    # region Reorder rows
    grammar_bitmask_length = bitmask.shape[1]
    reordered_bitmask = torch.full(
        (batch_length, grammar_bitmask_length),
        -1,
        dtype=bitmask.dtype,
        device=bitmask.device,
    )

    req_id_to_bitmask_row: dict[str, int] = {
        req_id: i for i, req_id in enumerate(structured_output_request_ids)
    }

    # Collect placeable and placed IDs to log missing IDs if left.
    placeable_ids = {
        req_id
        for req_id in row_req_ids[:batch_length]
        if req_id is not None and req_id in req_id_to_bitmask_row
    }
    placed_ids: set[str] = set()

    for local_row, req_id in enumerate(row_req_ids[:batch_length]):
        scheduler_bitmask_row = req_id_to_bitmask_row.get(req_id)
        if scheduler_bitmask_row is not None:
            reordered_bitmask[local_row, :] = bitmask[scheduler_bitmask_row, :]
            placed_ids.add(req_id)
    # endregion

    # region Log missing request IDs
    missing_ids = placeable_ids - placed_ids
    new_missing_ids = sorted(
        req_id for req_id in missing_ids if req_id not in _warned_missing_request_ids
    )
    if new_missing_ids:
        _warned_missing_request_ids.update(new_missing_ids)
        msg = (
            "Structured-output bitmask remap shortfall: %d new missing "
            "request ID%s did not receive a bitmask row:\n%s"
        )
        args = [
            len(new_missing_ids),
            "" if len(new_missing_ids) == 1 else "s",
            new_missing_ids,
        ]
        logger.warning(msg, *args)
    # endregion

    return reordered_bitmask
