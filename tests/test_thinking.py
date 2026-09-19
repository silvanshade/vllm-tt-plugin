# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality

from vllm_tt_plugin.thinking import (
    ThinkingBudgetLogitsProcessor,
    install_thinking_budget,
)

THINK_OPEN = 10
THINK_CLOSE = 11
VOCAB = 16


def _vllm_config(*, reasoning: bool = True, tt_cfg: dict | None = None):
    reasoning_config = None
    if reasoning:
        reasoning_config = SimpleNamespace(
            enabled=True,
            reasoning_start_token_ids=[THINK_OPEN],
            reasoning_end_token_ids=[THINK_CLOSE],
            natural_reasoning_end_token_ids=[THINK_CLOSE],
        )
    additional_config: dict = {}
    if tt_cfg is not None:
        additional_config["tt"] = dict(tt_cfg)
    return SimpleNamespace(
        reasoning_config=reasoning_config,
        additional_config=additional_config,
    )


def _params(**kwargs):
    fields = {"thinking_token_budget": None, "extra_args": None}
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _added(index, params, prompt, output):
    return BatchUpdate(
        batch_size=index + 1,
        removed=[],
        added=[(index, params, prompt, output)],
        moved=[],
    )


def _budget_processor(budget, prompt, output, *, index=0):
    processor = ThinkingBudgetLogitsProcessor(
        _vllm_config(), torch.device("cpu"), is_pin_memory=False
    )
    processor.update_state(
        _added(index, _params(thinking_token_budget=budget), prompt, output)
    )
    return processor


def _forced_token(processor, rows=1):
    """The token the processor forces on row 0, or None when it forces none."""
    logits = torch.zeros((rows, VOCAB))
    processor.apply(logits)
    row = logits[0]
    if torch.isneginf(row).all():
        pytest.fail("every candidate was masked")
    if not torch.isneginf(row).any():
        return None
    return int(torch.argmax(row).item())


def test_budget_forces_the_close_once_the_section_spends_it():
    output: list[int] = []
    processor = _budget_processor(3, [THINK_OPEN], output)

    for _ in range(3):
        assert _forced_token(processor) is None
        output.append(7)

    assert _forced_token(processor) == THINK_CLOSE


def test_budget_counts_reasoning_the_prompt_already_opened_from_the_output():
    # The chat template opens ``<think>`` in the generation prompt, so the
    # count starts at the first generated token, not at a generated opener.
    output: list[int] = [7]
    processor = _budget_processor(1, [5, THINK_OPEN], output)

    assert _forced_token(processor) == THINK_CLOSE


def test_budget_stops_forcing_once_the_section_is_closed():
    output: list[int] = [7, 7]
    processor = _budget_processor(2, [THINK_OPEN], output)
    assert _forced_token(processor) == THINK_CLOSE

    output.append(THINK_CLOSE)

    assert _forced_token(processor) is None


def test_budget_is_inert_for_a_request_that_sends_none():
    output: list[int] = [7, 7, 7]
    processor = ThinkingBudgetLogitsProcessor(
        _vllm_config(), torch.device("cpu"), is_pin_memory=False
    )
    processor.update_state(_added(0, _params(), [THINK_OPEN], output))

    assert not processor.has_tracked_rows()
    assert _forced_token(processor) is None


def test_budget_survives_a_speculative_rollback():
    # The native MTP round appends accepted tokens to the request's live output
    # list and truncates them before the commit re-extends it. A rollback must
    # not leave the section counted twice.
    output: list[int] = []
    processor = _budget_processor(3, [THINK_OPEN], output)
    output.extend([7, 7])
    assert _forced_token(processor) is None

    del output[0:]
    assert _forced_token(processor) is None
    output.extend([7, 7])

    assert _forced_token(processor) is None
    output.append(7)
    assert _forced_token(processor) == THINK_CLOSE


def test_budget_follows_its_row_when_the_batch_moves():
    first: list[int] = [7]
    processor = _budget_processor(1, [THINK_OPEN], first, index=1)

    processor.update_state(
        BatchUpdate(
            batch_size=2,
            removed=[],
            added=[],
            moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)],
        )
    )
    logits = torch.zeros((2, VOCAB))
    processor.apply(logits)

    assert int(torch.argmax(logits[0]).item()) == THINK_CLOSE
    assert not torch.isneginf(logits[1]).any()


def test_budget_row_is_dropped_when_its_request_leaves():
    output: list[int] = [7]
    processor = _budget_processor(1, [THINK_OPEN], output)

    processor.update_state(BatchUpdate(batch_size=0, removed=[0], added=[], moved=[]))

    assert not processor.has_tracked_rows()
    assert _forced_token(processor) is None


def test_install_thinking_budget_is_skipped_without_a_reasoning_config():
    logitsprocs = LogitsProcessors()

    assert install_thinking_budget(logitsprocs, _vllm_config(reasoning=False)) is None
    assert list(logitsprocs.all) == []
