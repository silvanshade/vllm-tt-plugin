# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Reasoning-phase sampling controls for the TT sampling path.

vLLM enforces a per-request thinking budget at sample time through
``SamplingMetadata.thinking_budget_state_holder``, which the GPU input batch
constructs. The TT plugin owns its own persistent batch and its own
``SamplingMetadata`` build, so that holder is never constructed and the wire
field ``thinking_token_budget`` is accepted and then ignored. The engine has no
post-thinking sampler switch at all.

Both controls live here, over one incremental scan of a request's output
tokens (:class:`ReasoningPhase`):

* :class:`ThinkingBudgetLogitsProcessor` forces the reasoning-end tokens once a
  request has spent its budget. It is an ordinary logits processor, so it
  reaches both host-sampling paths the plugin has: the persistent batch's
  merged metadata, and the per-request policy the native MTP round builds.
* :class:`PostThinkingSwitch` rewrites a row's sampling parameters to the
  post-thinking preset once the model closes its reasoning block, the way the
  ninfer engine applies its post-reasoning preset mid-generation.

Both are keyed on the reasoning token ids the configured reasoning parser
derives (``--reasoning-parser qwen3`` gives ``<think>`` / ``</think>``); with no
reasoning config neither control is installed.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.sample.logits_processor import LogitsProcessor, LogitsProcessors
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality

from vllm_tt_plugin.config import get_tt_config
from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_tt_logger(__name__)

# How far a phase scan can be rewound without rescanning from the start. The
# native MTP round appends its accepted tokens to the request's live output
# list and truncates them again before the commit re-extends it
# (``mtp.TTNativeMTPController._decode``), so a rewind is never deeper than one
# round's token count; this bound covers that with room to spare.
_UNDO_DEPTH = 64


def _ends_with(tokens: Sequence[int], end: int, needle: Sequence[int]) -> bool:
    """Whether ``tokens[:end]`` ends with ``needle``, without slicing."""
    size = len(needle)
    if size == 0 or end < size:
        return False
    base = end - size
    return all(tokens[base + i] == needle[i] for i in range(size))


def _rfind(tokens: Sequence[int], needle: Sequence[int]) -> int:
    """Index of the last occurrence of ``needle`` in ``tokens``, or -1."""
    size = len(needle)
    if size == 0 or len(tokens) < size:
        return -1
    for start in range(len(tokens) - size, -1, -1):
        if all(tokens[start + i] == needle[i] for i in range(size)):
            return start
    return -1


@dataclass(frozen=True)
class ReasoningTokens:
    """Token sequences that open and close a reasoning section.

    ``end`` is what a budget forces; ``natural_end`` is what the model emits
    when it closes reasoning by itself. They differ when the deployment
    configures a transition phrase before the parser's close marker.
    """

    start: tuple[int, ...]
    end: tuple[int, ...]
    natural_end: tuple[int, ...]

    @classmethod
    def from_config(cls, vllm_config: "VllmConfig") -> "ReasoningTokens | None":
        config = getattr(vllm_config, "reasoning_config", None)
        if config is None or not config.enabled:
            return None
        start = config.reasoning_start_token_ids
        end = config.reasoning_end_token_ids
        natural_end = config.natural_reasoning_end_token_ids or end
        if not start or not end:
            return None
        return cls(tuple(start), tuple(end), tuple(natural_end))

    def starts_in_reasoning(self, prompt_token_ids: Sequence[int] | None) -> bool:
        """Whether the prompt leaves the model inside a reasoning section.

        The Qwen chat template opens ``<think>`` in the generation prompt, so a
        request can be mid-reasoning before it emits a token.
        """
        if not prompt_token_ids:
            return False
        opened = _rfind(prompt_token_ids, self.start)
        if opened < 0:
            return False
        return _rfind(prompt_token_ids, self.natural_end) < opened


class ReasoningPhase:
    """One request's reasoning section, tracked over its live output list.

    The output list is the request's own running list, so each step only has to
    consume the tokens appended since the last one. A shorter list means the
    MTP round rolled its speculative tail back: the scan rewinds to the
    recorded state for that length instead of rescanning.
    """

    __slots__ = (
        "_tokens",
        "_output",
        "_opened_by_prompt",
        "_pos",
        "_undo",
        "in_reasoning",
        "reasoning_tokens",
        "closed",
    )

    def __init__(
        self,
        tokens: ReasoningTokens,
        prompt_token_ids: Sequence[int] | None,
        output_token_ids: Sequence[int],
    ) -> None:
        self._tokens = tokens
        self._output = output_token_ids
        self._opened_by_prompt = tokens.starts_in_reasoning(prompt_token_ids)
        self._undo: deque[tuple[int, bool, int, bool]] = deque(maxlen=_UNDO_DEPTH)
        self._reset()

    def _reset(self) -> None:
        self._pos = 0
        self._undo.clear()
        self.in_reasoning = self._opened_by_prompt
        self.reasoning_tokens = 0
        self.closed = False

    def advance(self) -> None:
        """Consume output tokens appended since the last call."""
        output = self._output
        length = len(output)
        if length < self._pos:
            self._rewind(length)
        while self._pos < length:
            self._undo.append(
                (self._pos, self.in_reasoning, self.reasoning_tokens, self.closed)
            )
            self._pos += 1
            if self.in_reasoning:
                self.reasoning_tokens += 1
                tokens = self._tokens
                if _ends_with(output, self._pos, tokens.natural_end) or _ends_with(
                    output, self._pos, tokens.end
                ):
                    self.in_reasoning = False
                    self.closed = True
            elif _ends_with(output, self._pos, self._tokens.start):
                self.in_reasoning = True
                self.reasoning_tokens = 0

    def _rewind(self, length: int) -> None:
        restored: tuple[int, bool, int, bool] | None = None
        while self._undo and self._undo[-1][0] >= length:
            restored = self._undo.pop()
        if restored is not None and restored[0] == length:
            self._pos, self.in_reasoning, self.reasoning_tokens, self.closed = restored
            return
        # Deeper than the recorded history: rebuild from the start.
        self._reset()
        self.advance()

    def forced_end_token(self, budget: int) -> int | None:
        """The reasoning-end token to force now, or ``None`` when free.

        Once the section has spent its budget the close sequence is emitted one
        token per step; the tail of the output says how much of it already
        landed.
        """
        if not self.in_reasoning or self.reasoning_tokens < budget:
            return None
        end = self._tokens.end
        emitted = 0
        for size in range(len(end) - 1, 0, -1):
            if _ends_with(self._output, self._pos, end[:size]):
                emitted = size
                break
        return end[emitted]


class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    """Force a request's reasoning closed once it spends its budget.

    Only requests that carry ``thinking_token_budget`` are tracked, so a batch
    without one costs a dict lookup per step and leaves device sampling alone
    (``SamplingInputBatch.has_active_logitsprocs``).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        del device, is_pin_memory  # host-side state only
        self._tokens = ReasoningTokens.from_config(vllm_config)
        self._rows: dict[int, tuple[ReasoningPhase, int]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def has_tracked_rows(self) -> bool:
        """Whether any row in this batch carries a budget."""
        return bool(self._rows)

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for index in batch_update.removed:
            self._rows.pop(index, None)
        for index, params, prompt_token_ids, output_token_ids in batch_update.added:
            budget = getattr(params, "thinking_token_budget", None)
            if self._tokens is None or budget is None or budget < 0:
                self._rows.pop(index, None)
                continue
            self._rows[index] = (
                ReasoningPhase(self._tokens, prompt_token_ids, output_token_ids),
                int(budget),
            )
        for first, second, direction in batch_update.moved:
            moved = self._rows.pop(first, None)
            displaced = self._rows.pop(second, None)
            if moved is not None:
                self._rows[second] = moved
            if direction is MoveDirectionality.SWAP and displaced is not None:
                self._rows[first] = displaced

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._rows:
            return logits
        rows = logits.shape[0]
        for index, (phase, budget) in self._rows.items():
            if index >= rows:
                continue
            phase.advance()
            token = phase.forced_end_token(budget)
            if token is None:
                continue
            logits[index].fill_(float("-inf"))
            logits[index, token] = 0.0
        return logits


@dataclass(frozen=True)
class PostThinkingPreset:
    """Sampling applied once a request closes its reasoning block.

    ``min_p`` is absent: the reference preset sets it to zero, which is the
    default, and the TT device sampler carries no ``min_p`` field at all.
    """

    temperature: float
    top_k: int
    top_p: float
    presence_penalty: float

    _FIELDS = ("temperature", "top_k", "top_p", "presence_penalty")

    def merged(self, override: Mapping[str, Any]) -> "PostThinkingPreset":
        """This preset with the recognized keys of ``override`` replaced."""
        changes = {
            field: type(getattr(self, field))(override[field])
            for field in self._FIELDS
            if field in override
        }
        return replace(self, **changes)


# The ninfer engine's post-reasoning preset for this architecture: the answer
# phase is sampled far tighter than the reasoning phase.
NINFER_POST_THINKING = PostThinkingPreset(
    temperature=0.2, top_k=20, top_p=0.95, presence_penalty=0.0
)


def _request_preset(
    params: Any, default: PostThinkingPreset
) -> PostThinkingPreset | None:
    """The preset for one request: its own ``post_thinking`` keys win.

    ``vllm_xargs`` carries scalars only (``str | int | float``; an object is
    rejected at the API boundary and a JSON ``false`` arrives as ``0``), so
    the request-side controls are flat:

    * ``post_thinking: false`` / ``0`` / ``"off"`` opts the request out;
    * ``post_thinking_temperature``, ``post_thinking_top_k``,
      ``post_thinking_top_p``, ``post_thinking_presence_penalty`` override
      single fields of the served preset.
    """
    extra_args = getattr(params, "extra_args", None) or {}
    if not extra_args:
        return default
    enabled = extra_args.get("post_thinking", True)
    if enabled is None or enabled is False or enabled in (0, "0", "false", "off"):
        return None
    override = {
        field: extra_args[f"post_thinking_{field}"]
        for field in PostThinkingPreset._FIELDS
        if f"post_thinking_{field}" in extra_args
    }
    if not override:
        return default
    try:
        return default.merged(override)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring unusable post_thinking override %s; serving the "
            "deployment preset instead",
            override,
        )
        return default


class PostThinkingSwitch:
    """Re-sample the answer phase at the post-thinking preset.

    The switch writes the preset into the persistent batch's per-row sampling
    tensors, so every sampling path reads it with no further plumbing: the
    merged host metadata, the per-DP host metadata, the device sampler's
    per-row parameters, and the row views the MTP round slices out of them.

    A row switches once, at the first step after its reasoning block closes.
    Under speculation the close can land mid-round, so the answer phase starts
    at the preset at most one round late.
    """

    def __init__(self, tokens: ReasoningTokens, preset: PostThinkingPreset) -> None:
        self._tokens = tokens
        self._preset = preset
        self._rows: dict[int, tuple[ReasoningPhase, PostThinkingPreset | None]] = {}
        self._switched: set[int] = set()

    @classmethod
    def from_config(cls, vllm_config: "VllmConfig") -> "PostThinkingSwitch | None":
        """Build the switch this deployment asks for, or ``None``.

        ``--additional-config '{"tt": {"post_thinking": false}}'`` disables it;
        an object overrides fields of the reference preset.
        """
        tokens = ReasoningTokens.from_config(vllm_config)
        if tokens is None:
            return None
        setting = get_tt_config(vllm_config).get("post_thinking", True)
        if setting is False or setting is None:
            return None
        preset = NINFER_POST_THINKING
        if isinstance(setting, Mapping):
            preset = preset.merged(setting)
        elif setting is not True:
            raise ValueError(
                "additional_config['tt']['post_thinking'] must be an object or "
                f"a boolean, got {type(setting).__name__}"
            )
        return cls(tokens, preset)

    @property
    def preset(self) -> PostThinkingPreset:
        return self._preset

    def sync(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for index in batch_update.removed:
            self._drop(index)
        for index, params, prompt_token_ids, output_token_ids in batch_update.added:
            self._drop(index)
            self._rows[index] = (
                ReasoningPhase(self._tokens, prompt_token_ids, output_token_ids),
                _request_preset(params, self._preset),
            )
        for first, second, direction in batch_update.moved:
            moved = self._rows.pop(first, None)
            displaced = self._rows.pop(second, None)
            was_switched = first in self._switched
            displaced_switched = second in self._switched
            self._switched.discard(first)
            self._switched.discard(second)
            if moved is not None:
                self._rows[second] = moved
                if was_switched:
                    self._switched.add(second)
            if direction is MoveDirectionality.SWAP and displaced is not None:
                self._rows[first] = displaced
                if displaced_switched:
                    self._switched.add(first)

    def _drop(self, index: int) -> None:
        self._rows.pop(index, None)
        self._switched.discard(index)

    def apply(self, sampling: Any) -> list[int]:
        """Switch every row whose reasoning closed; return those rows."""
        switched: list[int] = []
        for index, (phase, preset) in self._rows.items():
            if preset is None or index in self._switched:
                continue
            phase.advance()
            if not phase.closed:
                continue
            self._switched.add(index)
            sampling.temperature[index] = preset.temperature
            sampling.top_p[index] = preset.top_p
            sampling.top_k[index] = preset.top_k
            sampling.presence_penalty[index] = preset.presence_penalty
            switched.append(index)
        return switched

    def has_switched(self, index: int) -> bool:
        """Whether this row is already sampling the answer phase."""
        return index in self._switched


def install_thinking_budget(
    logitsprocs: LogitsProcessors, vllm_config: "VllmConfig"
) -> ThinkingBudgetLogitsProcessor | None:
    """Add budget enforcement to a freshly built processor set.

    Returns the installed processor, or ``None`` when the deployment has no
    reasoning configuration and the wire field cannot be honoured.
    """
    if ReasoningTokens.from_config(vllm_config) is None:
        return None
    processor = ThinkingBudgetLogitsProcessor(
        vllm_config, torch.device("cpu"), is_pin_memory=False
    )
    logitsprocs.non_argmax_invariant.append(processor)
    return processor
