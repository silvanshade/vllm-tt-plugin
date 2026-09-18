# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Native captured rounds with vLLM-owned scheduling and target sampling policy."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
import ttnn
from vllm.v1.outputs import (
    DraftTokenIds,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessors,
    build_logitsprocs,
)
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)

from vllm_tt_plugin.model_input import TTModelInput, slice_tt_sampling_params

if TYPE_CHECKING:
    from models.demos.qwen38.tt.mtp_round import Qwen38MTPRound
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

    from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward


class TTNativeMTPController:
    """Serialize shared native frames; keep policy and drafts owned by request ID."""

    def __init__(self, runner: TTModelRunner) -> None:
        self.runner = runner
        # Policy is applied to one target decision at a time, not to vLLM's
        # flattened GPU rejection-sampler batch. Keep all serial processors.
        self.policy_config = copy(runner.vllm_config)
        self.policy_config.speculative_config = None
        self.processors: dict[str, LogitsProcessors] = {}
        self.rope_deltas: dict[str, int] = {}
        self.drafts: dict[str, list[int]] = {}

    @property
    def round(self) -> Qwen38MTPRound:
        return self.runner.model.mtp_round

    def release_request(self, req_id: str) -> None:
        self.processors.pop(req_id, None)
        self.rope_deltas.pop(req_id, None)
        self.drafts.pop(req_id, None)

    def take_draft_token_ids(self) -> DraftTokenIds:
        drafts, self.drafts = self.drafts, {}
        return DraftTokenIds(list(drafts), list(drafts.values()))

    def submit(
        self, model_input: TTModelInput, scheduler_output: SchedulerOutput
    ) -> Callable[[GrammarOutput | None], ModelRunnerOutput]:
        self.drafts.clear()
        if model_input.prompt_lens is None:
            # Native rounds address the authoritative slot directly: no gather ran.
            self.runner._pending_state_slot_settle = None
            return partial(
                self._decode,
                model_input=model_input,
                scheduled_drafts=scheduler_output.scheduled_spec_decode_tokens,
            )
        fwd = self.runner._forward_with_model_input(model_input)
        return partial(self._prefill, model_input=model_input, fwd=fwd)

    def _policy(self, req_id: str) -> LogitsProcessors:
        processors = self.processors.get(req_id)
        if processors is None:
            runner = self.runner
            request = runner.requests[req_id]
            processors = build_logitsprocs(
                vllm_config=self.policy_config,
                device=torch.device("cpu"),
                is_pin_memory=False,
                is_pooling_model=False,
                custom_logitsprocs=runner.model_config.logits_processors or (),
            )
            update = BatchUpdate(
                batch_size=1,
                removed=[],
                moved=[],
                added=[
                    (
                        0,
                        request.sampling_params,
                        request.prompt_token_ids,
                        request.output_token_ids,
                    )
                ],
            )
            for processor in processors.all:
                processor.update_state(update)
            self.processors[req_id] = processors
        return processors

    def _row_input(
        self,
        model_input: TTModelInput,
        row: int,
        req_id: str,
        mask: torch.Tensor | None,
    ) -> TTModelInput:
        request = self.runner.requests[req_id]
        processors = self._policy(req_id)
        for processor in processors.all:
            processor.update_state(None)
        allowed = model_input.allowed_token_ids_mask_list[0]
        bad_words = model_input.bad_words_token_ids_list[0]
        generators = model_input.generators_list[0]
        return replace(
            model_input,
            input_tokens=model_input.input_tokens[row : row + 1],
            input_positions=model_input.input_positions[row : row + 1],
            row_req_ids=[req_id],
            unpadded_batch_size=1,
            tt_sampling_params=slice_tt_sampling_params(
                model_input.tt_sampling_params, [row]
            ),
            perform_device_sampling=False,
            grammar_bitmask=[mask],
            logitsprocs_list=[processors],
            allowed_token_ids_mask_list=[
                None if allowed is None else allowed[row : row + 1]
            ],
            bad_words_token_ids_list=[{0: bad_words[row]} if row in bad_words else {}],
            generators_list=[{0: generators[row]} if row in generators else {}],
            prompt_tokens=torch.tensor([request.prompt_token_ids], dtype=torch.int64),
            output_tokens=torch.tensor([request.output_token_ids], dtype=torch.int64),
        )

    def _sample(
        self,
        logits: torch.Tensor,
        model_input: TTModelInput,
        row: int,
        req_id: str,
        mask: torch.Tensor | None,
        *,
        advance_rng: bool = False,
    ) -> tuple[int, LogprobsTensors | None]:
        row_input = self._row_input(model_input, row, req_id, mask)
        if advance_rng:
            # Match the ordinary runner's per-token generator advancement. The
            # input builder already advanced once for the first target decision.
            for generator in row_input.generators_list[0].values():
                torch.rand(1, generator=generator)
        tokens, logprobs = self.runner._get_output_tokens(
            tt_out=logits.reshape(1, 1, -1),
            tt_log_probs=None,
            sampling_params=row_input.tt_sampling_params,
            model_input=row_input,
            batch_size_per_dp=[1],
            perform_device_sampling=False,
            is_decode=True,
        )
        return int(tokens[0].item()), logprobs[0]

    @staticmethod
    def _grammar_rows(
        grammar_output: GrammarOutput | None, scheduled_drafts: dict[str, list[int]]
    ) -> dict[str, torch.Tensor]:
        masks = {}
        if grammar_output is not None and grammar_output.grammar_bitmask is not None:
            offset = 0
            bitmask = torch.from_numpy(grammar_output.grammar_bitmask)
            for req_id in grammar_output.structured_output_request_ids:
                count = 1 + len(scheduled_drafts.get(req_id, ()))
                masks[req_id] = bitmask[offset : offset + count]
                offset += count
            assert offset == len(bitmask)
        return masks

    def _pure_greedy(
        self, req_id: str, model_input: TTModelInput, masks: dict[str, torch.Tensor]
    ) -> bool:
        params = self.runner.requests[req_id].sampling_params
        known_processors = (
            MinTokensLogitsProcessor,
            LogitBiasLogitsProcessor,
            MinPLogitsProcessor,
        )
        return (
            params.temperature == 0
            and params.presence_penalty == 0
            and params.frequency_penalty == 0
            and params.repetition_penalty == 1
            and not params.min_tokens
            and not params.logit_bias
            and not params.allowed_token_ids
            and not params.bad_words
            and not params.extra_args
            and req_id not in masks
            and model_input.max_num_logprobs[0] is None
            and all(type(p) in known_processors for p in self._policy(req_id).all)
        )

    def _budget(self, req_id: str) -> int:
        request = self.runner.requests[req_id]
        row = self.runner.input_batch.req_id_to_index[req_id]
        return min(
            request.sampling_params.max_tokens - len(request.output_token_ids),
            self.runner.model_config.max_model_len
            - int(self.runner.input_batch.num_tokens[row]),
        )

    def _stops(self, req_id: str, token: int, generated: int = 1) -> bool:
        request = self.runner.requests[req_id]
        params = request.sampling_params
        stops = (
            params.stop_token_ids if params.ignore_eos else params.all_stop_token_ids
        )
        return (
            len(request.output_token_ids) + generated >= params.min_tokens
            and token in stops
        )

    def _read_egress(self, frame: dict[str, ttnn.Tensor]) -> list[int]:
        result = ttnn.to_torch(
            frame["egress"],
            mesh_composer=ttnn.ConcatMeshToTensor(self.round.mesh, dim=0),
        )
        return result.flatten().tolist()

    def _prefill(
        self,
        grammar_output: GrammarOutput | None,
        *,
        model_input: TTModelInput,
        fwd: _SyncForward,
    ) -> ModelRunnerOutput:
        masks = self._grammar_rows(grammar_output, {})
        outputs, logprobs = [], []
        for row, req_id in enumerate(model_input.row_req_ids):
            if bool(model_input.intermediate_prefill_mask[row]):
                outputs.append([])
                logprobs.append([])
                continue
            token, lp = self._sample(
                fwd.tt_out[row, -1], model_input, row, req_id, masks.get(req_id)
            )
            outputs.append([token])
            logprobs.append([lp])
        result = self._commit(model_input.row_req_ids, outputs, logprobs)
        for row, (req_id, tokens) in enumerate(
            zip(model_input.row_req_ids, outputs, strict=True)
        ):
            budget = self._budget(req_id)
            if not tokens or budget < 1 or self._stops(req_id, tokens[-1], generated=0):
                self.drafts[req_id] = []
                continue
            slot = self.runner._req_state_slot[req_id]
            delta = self.round.target.rope.rope_delta
            self.rope_deltas[req_id] = delta
            batch_row = self.runner.input_batch.req_id_to_index[req_id]
            position = int(self.runner.input_batch.num_tokens[batch_row]) - 1
            frame = self.round.bridge(
                slot, tokens[-1], position, model_input.block_tables[row], budget, delta
            )
            message = self._read_egress(frame)
            self.drafts[req_id] = message[3 : 3 + message[2]]
        return result

    def _decode(
        self,
        grammar_output: GrammarOutput | None,
        *,
        model_input: TTModelInput,
        scheduled_drafts: dict[str, list[int]],
    ) -> ModelRunnerOutput:
        masks = self._grammar_rows(grammar_output, scheduled_drafts)
        outputs, logprobs = [], []
        for row, req_id in enumerate(model_input.row_req_ids):
            request = self.runner.requests[req_id]
            budget = self._budget(req_id)
            assert budget > 0
            drafts = scheduled_drafts.get(req_id, [])
            tokens = [int(model_input.input_tokens[row, 0]), *drafts]
            assert len(tokens) <= budget
            self.round.stage(
                self.runner._req_state_slot[req_id],
                tokens,
                int(model_input.input_positions[row]),
                model_input.block_tables[row],
                budget,
                self.rope_deltas.get(req_id, 0),
            )
            greedy = self._pure_greedy(req_id, model_input, masks)
            frame = self.round.execute(greedy=greedy)
            row_logprobs = []
            if greedy:
                message = self._read_egress(frame)
                accepted, correction, extent = message[:3]
                selected = drafts[: accepted - 1] + [correction]
            else:
                logits = ttnn.to_torch(
                    frame["logits"],
                    mesh_composer=ttnn.ConcatMeshToTensor(self.round.mesh, dim=-1),
                )
                logits = logits.reshape(len(tokens), -1)[
                    :, : self.runner.vocab_size
                ].float()
                history = request.output_token_ids
                initial_length = len(history)
                selected = []
                try:
                    for index in range(len(tokens)):
                        mask = masks.get(req_id)
                        if mask is not None:
                            mask = mask[index : index + 1]
                        token, lp = self._sample(
                            logits[index],
                            model_input,
                            row,
                            req_id,
                            mask,
                            advance_rng=index > 0,
                        )
                        selected.append(token)
                        row_logprobs.append(lp)
                        if (
                            self._stops(req_id, token)
                            or index == len(drafts)
                            or token != drafts[index]
                        ):
                            break
                        history.append(token)
                finally:
                    del history[initial_length:]
                frame = self.round.finish(len(selected), selected[-1])
                message = self._read_egress(frame)
                extent = message[2]
            # The engine also clips stops. Do it here to keep the runner's
            # histories exact, and never publish a continuation after a stop.
            for index, token in enumerate(selected):
                if self._stops(req_id, token, generated=index + 1):
                    selected = selected[: index + 1]
                    row_logprobs = row_logprobs[: index + 1]
                    extent = 0
                    break
            outputs.append(selected)
            logprobs.append(row_logprobs)
            self.drafts[req_id] = message[3 : 3 + extent]
        self.runner.note_decode_layout_consumed()
        return self._commit(model_input.row_req_ids, outputs, logprobs)

    def _commit(
        self,
        req_ids: list[str],
        outputs: list[list[int]],
        logprobs: list[list[LogprobsTensors | None]],
    ) -> ModelRunnerOutput:
        runner = self.runner
        writes = []
        for req_id, tokens in zip(req_ids, outputs, strict=True):
            row = runner.input_batch.req_id_to_index[req_id]
            start = int(runner.input_batch.num_tokens[row])
            end = start + len(tokens)
            if end > runner.model_config.max_model_len:
                raise ValueError(
                    "Native MTP output exceeds the request's context budget"
                )
            writes.append((row, start, end, req_id, tokens))
        for row, start, end, req_id, tokens in writes:
            runner.input_batch.token_ids_cpu[row, start:end] = tokens
            runner.input_batch.num_tokens[row] = end
            runner.requests[req_id].output_token_ids.extend(tokens)
        flat = [lp for row in logprobs for lp in row if lp is not None]
        packed = None
        if flat:
            counts = [0]
            for tokens in outputs:
                counts.append(counts[-1] + len(tokens))
            packed = LogprobsLists(
                np.concatenate([lp.logprob_token_ids.numpy() for lp in flat]),
                np.concatenate([lp.logprobs.numpy() for lp in flat]),
                np.concatenate([lp.selected_token_ranks.numpy() for lp in flat]),
                counts,
            )
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=outputs,
            logprobs=packed,
            prompt_logprobs_dict=dict.fromkeys(req_ids, None),
            pooler_output=[],
        )
