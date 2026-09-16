# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import json
from types import SimpleNamespace

import torch
import xgrammar as xgr
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

from vllm_tt_plugin.structured_output import (
    install_tt_compact_json_patch,
    reorder_grammar_bitmask_for_tt_batch,
)


def test_reorder_grammar_bitmask_uses_forward_row_order():
    bitmask = torch.tensor(
        [
            [10, 11],
            [20, 21],
            [30, 31],
            [40, 41],
        ],
        dtype=torch.int32,
    )

    reordered = reorder_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["req-0", "req-1", "req-2", "req-3"],
        row_req_ids=["req-1", "req-3"],
        batch_length=2,
    )

    assert torch.equal(
        reordered,
        torch.tensor(
            [
                [20, 21],
                [40, 41],
            ],
            dtype=torch.int32,
        ),
    )


def test_reorder_grammar_bitmask_ignores_requests_absent_from_the_forward():
    """A structured request the forward did not run must not claim a row."""
    bitmask = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ],
        dtype=torch.int32,
    )

    reordered = reorder_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["req-0", "req-1"],
        row_req_ids=["req-1", "req-9"],
        batch_length=2,
    )

    assert torch.equal(
        reordered,
        torch.tensor(
            [
                [20, 21],
                [-1, -1],
            ],
            dtype=torch.int32,
        ),
    )


def test_reorder_grammar_bitmask_leaves_uncovered_rows_all_allowed():
    """Decode pads to the wire batch size, so rows can outnumber requests."""
    bitmask = torch.tensor([[10, 11]], dtype=torch.int32)

    reordered = reorder_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["req-0"],
        row_req_ids=["req-0"],
        batch_length=3,
    )

    assert torch.equal(
        reordered,
        torch.tensor([[10, 11], [-1, -1], [-1, -1]], dtype=torch.int32),
    )


def test_reorder_grammar_bitmask_handles_forward_narrower_than_batch():
    """A prefill build can drop rows, so a request's persistent batch index can
    exceed the forward's row count."""
    bitmask = torch.tensor([[10, 11]], dtype=torch.int32)

    # Persistent batch req-0..req-3; the forward kept only req-0 and req-2.
    reordered = reorder_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["req-2"],
        row_req_ids=["req-0", "req-2"],
        batch_length=2,
    )

    assert torch.equal(
        reordered,
        torch.tensor([[-1, -1], [10, 11]], dtype=torch.int32),
    )


def test_compact_json_rejects_whitespace_after_backend_selection() -> None:
    """Automatic backend selection must retain the anti-whitespace-loop policy."""
    vocab = ["[", " ", "1", "]", "<eos>"]
    backend = object.__new__(XgrammarBackend)
    backend.compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, stop_token_ids=[4]),
        max_threads=1,
    )
    backend.disable_any_whitespace = False
    backend.num_speculative_tokens = 0
    backend.vocab_size = len(vocab)
    manager = object.__new__(StructuredOutputManager)
    manager.backend = backend
    request = SimpleNamespace(
        request_id="compact-json",
        sampling_params=None,
        structured_output_request=SimpleNamespace(
            structured_output_key=(
                StructuredOutputOptions.JSON,
                json.dumps(
                    {
                        "type": "array",
                        "items": {"const": 1},
                        "minItems": 1,
                        "maxItems": 1,
                    }
                ),
            )
        ),
    )
    install_tt_compact_json_patch()
    grammar = manager._create_grammar(request)
    assert grammar.validate_tokens([0, 1]) == [0]
    assert grammar.validate_tokens([0, 2, 3]) == [0, 2, 3]
