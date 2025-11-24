# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Background worker that consumes scheduler snapshots and runs simulation."""

from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from vllm.logger import init_logger
from vllm.v1.core.sched.state_snapshot import (
    RequestStateSnapshot,
    SchedulerConfigSnapshot,
    SchedulerStateSnapshot,
)

logger = init_logger(__name__)

"""
Problems:
1. KV cache management (different waiting states)
2. Speculative decoding
3. Doesn't account for preemptions
"""


@dataclass
class SimulationResult:
    snapshot_version: int
    snapshot_timestamp: float
    simulation_timestamp: float
    num_requests: int
    metadata: dict
    snapshot_build_latency_ms: float = 0.0
    simulation_latency_ms: float = 0.0


@dataclass
class SimRequestState:
    """Mutable view of an individual request within the simulator."""

    snapshot: RequestStateSnapshot
    virtual_tokens_with_spec: int = field(init=False)
    num_computed_tokens: int = field(init=False)
    prefill_processed: int = field(init=False)
    decode_processed: int = field(init=False)
    spec_token_count: int = field(init=False)
    num_output_placeholders: int = field(init=False)
    decode_tokens_staged: int = field(default=0)
    first_scheduled_time_ms: Optional[float] = None
    prefill_done_time_ms: Optional[float] = None
    finished_time_ms: Optional[float] = None

    def __post_init__(self) -> None:
        self.virtual_tokens_with_spec = self.snapshot.num_tokens_with_spec
        self.num_computed_tokens = self.snapshot.num_computed_tokens
        self.prefill_processed = self.snapshot.num_prompt_processed_tokens
        self.decode_processed = self.snapshot.num_output_processed_tokens
        self.spec_token_count = self.snapshot.spec_token_count
        self.num_output_placeholders = self.snapshot.num_output_placeholders

    @property
    def request_id(self) -> str:
        return self.snapshot.request_id

    @property
    def status(self) -> str:
        return self.snapshot.status

    @status.setter
    def status(self, value: str) -> None:
        self.snapshot.status = value

    @property
    def priority(self) -> int:
        return self.snapshot.priority

    @property
    def arrival_time(self) -> float:
        return self.snapshot.arrival_time

    @property
    def prefill_remaining(self) -> int:
        return max(0, self.snapshot.num_prompt_tokens - self.prefill_processed)

    @property
    def decode_remaining(self) -> int:
        return max(0, self.snapshot.num_output_tokens - self.decode_processed)

    def ensure_decode_backlog(self) -> None:
        """Stage one decode token if a request still needs decoding."""
        if self.prefill_remaining > 0:
            return
        outstanding = (
            self.virtual_tokens_with_spec
            + self.num_output_placeholders
            - self.num_computed_tokens
        )
        available_decode = self.decode_remaining - self.decode_tokens_staged
        if outstanding <= 0 and available_decode > 0:
            self.decode_tokens_staged += 1
            self.virtual_tokens_with_spec += 1

    def pending_tokens(self) -> int:
        return max(
            0,
            self.virtual_tokens_with_spec
            + self.num_output_placeholders
            - self.num_computed_tokens,
        )

    def consume(self, num_tokens: int) -> Tuple[int, int]:
        """Advance per-request progress by consuming scheduled tokens."""
        prefill_consumed = min(num_tokens, self.prefill_remaining)
        if prefill_consumed > 0:
            self.prefill_processed += prefill_consumed
            num_tokens -= prefill_consumed

        decode_consumed = 0
        if num_tokens > 0:
            decode_consumed = min(num_tokens, self.decode_remaining)
            if decode_consumed > 0:
                self.decode_processed += decode_consumed
                num_tokens -= decode_consumed
                self.decode_tokens_staged = max(
                    0, self.decode_tokens_staged - decode_consumed
                )

        if num_tokens > 0 and self.spec_token_count > 0:
            spec_consumed = min(num_tokens, self.spec_token_count)
            self.spec_token_count -= spec_consumed
            decode_consumed += spec_consumed
            num_tokens -= spec_consumed

        consumed = prefill_consumed + decode_consumed
        if consumed > 0:
            self.num_computed_tokens += consumed
            if self.num_output_placeholders > 0:
                placeholder_used = min(self.num_output_placeholders, consumed)
                self.num_output_placeholders -= placeholder_used

        return prefill_consumed, decode_consumed

    def mark_scheduled(self, current_time_ms: float) -> None:
        if self.first_scheduled_time_ms is None:
            self.first_scheduled_time_ms = current_time_ms

    def mark_prefill_done(self, completed_time_ms: float) -> None:
        if self.prefill_done_time_ms is None:
            self.prefill_done_time_ms = completed_time_ms

    def mark_finished(self, completed_time_ms: float) -> None:
        if self.finished_time_ms is None:
            self.finished_time_ms = completed_time_ms
            self.status = "FINISHED"

    def is_finished(self) -> bool:
        return (
            self.prefill_remaining == 0
            and self.decode_remaining == 0
            and self.pending_tokens() == 0
            and self.spec_token_count == 0
        )

    def is_active(self) -> bool:
        if self.is_finished():
            return False
        return "RUNNING" in self.status or "WAITING" in self.status


@dataclass
class SimulationContext:
    snapshot: SchedulerStateSnapshot
    requests: Dict[str, SimRequestState]
    running_order: List[SimRequestState]
    waiting_queue: Deque[SimRequestState]
    current_time_ms: float = 0.0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    num_batches: int = 0

    def __init__(self, snapshot: SchedulerStateSnapshot) -> None:
        self.snapshot = snapshot
        self.requests = {
            req.request_id: SimRequestState(req) for req in snapshot.requests
        }
        self.running_order = self._build_running_order()
        self.waiting_queue = self._build_waiting_queue()
        self.current_time_ms = 0.0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.num_batches = 0

    def _build_running_order(self) -> List[SimRequestState]:
        running: List[SimRequestState] = []
        for req_id in self.snapshot.running_request_ids:
            sim_req = self.requests.get(req_id)
            if sim_req and sim_req.is_active():
                running.append(sim_req)
        return running

    def _build_waiting_queue(self) -> Deque[SimRequestState]:
        policy = self.snapshot.config.policy
        queue: Deque[SimRequestState] = deque()
        ordered_ids = self.snapshot.waiting_request_ids
        if not ordered_ids:
            candidates = [
                req
                for req in self.requests.values()
                if "WAITING" in req.status
                and not req.status.upper().startswith("FINISHED")
            ]
            if policy == "priority":
                candidates.sort(key=lambda r: (r.priority, r.arrival_time))
            else:
                candidates.sort(key=lambda r: r.arrival_time)
            ordered_ids = [req.request_id for req in candidates]
        for req_id in ordered_ids:
            sim_req = self.requests.get(req_id)
            if sim_req and "WAITING" in sim_req.status:
                queue.append(sim_req)
        return queue

    def has_pending_requests(self) -> bool:
        if any(req.is_active() for req in self.running_order):
            return True
        return any(req.is_active() for req in self.waiting_queue)

    def remove_finished(self, finished: List[SimRequestState]) -> None:
        finished_ids = {req.request_id for req in finished}
        if not finished_ids:
            return
        self.running_order = [
            req for req in self.running_order if req.request_id not in finished_ids
        ]

    def summary_metadata(self) -> dict:
        per_request: Dict[str, dict] = {}
        for req in self.requests.values():
            per_request[req.request_id] = {
                "status": req.status,
                "first_token_ms": req.first_scheduled_time_ms,
                "prefill_done_ms": req.prefill_done_time_ms,
                "finished_ms": req.finished_time_ms,
                "remaining_prefill_tokens": req.prefill_remaining,
                "remaining_decode_tokens": req.decode_remaining,
            }
        return {
            "num_batches": self.num_batches,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "per_request": per_request,
            # "kv_cache": {
            #     "usage": self.snapshot.kv_cache_usage,
            #     "total_blocks": self.snapshot.kv_cache_total_blocks,
            #     "free_blocks": self.snapshot.kv_cache_free_blocks,
            #     "block_size": self.snapshot.kv_cache_block_size,
            # },
        }

    def estimate_wait_time_ms(self, request_id: Optional[str] = None) -> float:
        if request_id:
            req = self.requests.get(request_id)
            if req and req.first_scheduled_time_ms is not None:
                return req.first_scheduled_time_ms
        for req_id in self.snapshot.waiting_request_ids:
            req = self.requests.get(req_id)
            if req and req.first_scheduled_time_ms is not None:
                return req.first_scheduled_time_ms
        return 0.0


class SchedulerSimulationWorker:
    """Runs deterministic simulations based on scheduler snapshots."""

    def __init__(self, interval_s: float, 
                #  average_prompt_length: float,
                #  average_output_length: float, 
                #  average_max_tokens: float,
                 intercept: float, 
                 prefill_coeff: float,
                 decode_coeff: float) -> None:
        self._interval_s = max(interval_s, 0.001)
        # self._average_prompt_length = average_prompt_length
        # self._average_output_length = average_output_length
        # self._average_max_tokens = average_max_tokens
        self._intercept = intercept
        self._prefill_coeff = prefill_coeff
        self._decode_coeff = decode_coeff
        self._latest_snapshot: Optional[SchedulerStateSnapshot] = None
        self._latest_result: Optional[SimulationResult] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="scheduler-simulator", daemon=True
        )

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def update_snapshot(self, snapshot: SchedulerStateSnapshot) -> None:
        with self._lock:
            self._latest_snapshot = snapshot

    def latest_result(self) -> Optional[SimulationResult]:
        with self._lock:
            return self._latest_result

    def _pop_latest_snapshot(self) -> Optional[SchedulerStateSnapshot]:
        with self._lock:
            snapshot = self._latest_snapshot
            self._latest_snapshot = None
            return snapshot

    def _run(self) -> None:
        logger.info(
            "Scheduler simulation worker started with %.3f s cadence",
            self._interval_s,
        )
        while not self._stop_event.is_set():
            sim_elapsed_ms = 0.0
            snapshot = self._pop_latest_snapshot()
            if snapshot is not None:
                sim_start = time.perf_counter()
                result = self._run_simulation(snapshot)
                sim_elapsed_ms = (time.perf_counter() - sim_start) * 1000.0
                result.simulation_latency_ms = sim_elapsed_ms
                result.snapshot_build_latency_ms = snapshot.build_latency_ms
                logger.debug(
                    "Simulation from snapshot v%04d finished in %.3f ms",
                    snapshot.version,
                    sim_elapsed_ms,
                )
                with self._lock:
                    self._latest_result = result
            
            if sim_elapsed_ms < self._interval_s * 1000.0:
                self._stop_event.wait(self._interval_s - sim_elapsed_ms / 1000.0)
        logger.info("Scheduler simulation worker stopped")

    def _build_next_batch(
        self, state: SimulationContext
    ) -> Tuple[List[SimRequestState], List[int]]:
        config = state.snapshot.config
        token_budget = config.max_num_batched_tokens
        max_model_len = config.max_model_len
        batch_requests: List[SimRequestState] = []
        batch_tokens: List[int] = []
        partial_prefills = sum(1 for req in state.running_order if req.prefill_remaining)
        long_partial_prefills = sum(
            1
            for req in state.running_order
            if req.prefill_remaining and req.snapshot.is_long_prompt
        )

        # Schedule currently running requests first.
        for req in list(state.running_order):
            if token_budget <= 0:
                break
            if req.is_finished():
                continue
            req.ensure_decode_backlog()
            pending = req.pending_tokens()
            if pending <= 0:
                continue
            tokens = min(pending, token_budget, max_model_len - req.num_computed_tokens)
            if tokens <= 0:
                continue
            prefill_before = req.prefill_remaining
            if prefill_before > 0 and tokens >= prefill_before:
                partial_prefills = max(0, partial_prefills - 1)
                if req.snapshot.is_long_prompt:
                    long_partial_prefills = max(0, long_partial_prefills - 1)
            batch_requests.append(req)
            batch_tokens.append(tokens)
            token_budget -= tokens

        # schedule waiting requests while respecting capacity limits
        skipped_requests: List[SimRequestState] = []
        while (
            token_budget > 0
            and len(batch_requests) < config.max_num_seqs
            and state.waiting_queue
        ):
            if len(state.running_order) >= config.max_num_seqs:
                break
            req = state.waiting_queue.popleft()
            if req.is_finished():
                continue
            req.ensure_decode_backlog()
            tokens = self._tokens_for_waiting_request(
                req,
                token_budget,
                config,
                partial_prefills,
                long_partial_prefills,
            )
            if tokens <= 0:
                skipped_requests.append(req)
                continue
            batch_requests.append(req)
            batch_tokens.append(tokens)
            token_budget -= tokens
            req.status = "RUNNING"
            if req not in state.running_order:
                state.running_order.append(req)
            prefill_remaining = req.prefill_remaining
            if prefill_remaining > 0 and tokens < prefill_remaining:
                partial_prefills += 1
                if req.snapshot.is_long_prompt:
                    long_partial_prefills += 1

        if skipped_requests:
            # Maintain FCFS semantics by putting skipped items back at the front.
            state.waiting_queue.extendleft(reversed(skipped_requests))

        return batch_requests, batch_tokens

    def _tokens_for_waiting_request(
        self,
        req: SimRequestState,
        token_budget: int,
        config: SchedulerConfigSnapshot,
        partial_prefills: int,
        long_partial_prefills: int,
    ) -> int:
        prefill_remaining = req.prefill_remaining
        # (BODEN): also needs to use req.max_tokens in computation
        max_tokens = config.max_model_len - req.num_computed_tokens
        if max_tokens <= 0:
            return 0

        if prefill_remaining > 0:
            tokens = prefill_remaining
            if (
                config.long_prefill_token_threshold > 0
                and tokens > config.long_prefill_token_threshold
            ):
                tokens = config.long_prefill_token_threshold
            if not config.chunked_prefill_enabled and tokens > token_budget:
                return 0
            if tokens > token_budget:
                tokens = token_budget
            if tokens <= 0:
                return 0
            will_complete = tokens >= prefill_remaining
            if not will_complete:
                if (
                    config.max_num_partial_prefills > 0
                    and partial_prefills >= config.max_num_partial_prefills
                ):
                    return 0
                if (
                    req.snapshot.is_long_prompt
                    and config.max_long_partial_prefills > 0
                    and long_partial_prefills >= config.max_long_partial_prefills
                ):
                    return 0
        else:
            tokens = min(req.pending_tokens(), token_budget)

        return min(tokens, max_tokens)

    def _apply_batch_result(
        self,
        state: SimulationContext,
        batch_requests: List[SimRequestState],
        batch_tokens: List[int],
    ) -> Tuple[int, int, List[SimRequestState], List[SimRequestState]]:
        total_prefill = 0
        total_decode = 0
        prefill_done: List[SimRequestState] = []
        finished: List[SimRequestState] = []
        for req, tokens in zip(batch_requests, batch_tokens):
            prefill_before = req.prefill_remaining
            decode_before = req.decode_remaining
            consumed_prefill, consumed_decode = req.consume(tokens)
            total_prefill += consumed_prefill
            total_decode += consumed_decode
            if prefill_before > 0 and req.prefill_remaining == 0:
                prefill_done.append(req)
            if (
                decode_before > 0
                and req.decode_remaining == 0
                and req.spec_token_count == 0
            ):
                finished.append(req)
        state.remove_finished(finished)
        return total_prefill, total_decode, prefill_done, finished

    def _estimate_batch_time(self, num_prefill_tokens: int, num_decode_tokens: int, intercept: float, prefill_coeff: float, decode_coeff: float) -> float:
        """Simple deterministic estimate for per-batch latency in milliseconds."""
        assert num_prefill_tokens > 0 or num_decode_tokens > 0
        return intercept + (prefill_coeff * num_prefill_tokens) + (decode_coeff * num_decode_tokens)

    def _run_simulation(self, snapshot: SchedulerStateSnapshot) -> SimulationResult:
        simulation_timestamp = time.monotonic()
        state = SimulationContext(snapshot)

        idle_ticks = 0
        max_idle_ticks = 4
        while state.has_pending_requests():
            batch_requests, batch_tokens = self._build_next_batch(state)
            
            # Dummy request first scheduled
            if any(req.request_id == "__DUMMY__" for req in batch_requests):
                break
            
            if not batch_requests:
                idle_ticks += 1
                if idle_ticks >= max_idle_ticks:
                    logger.debug("Simulation stalled after %d idle ticks", idle_ticks)
                    break
                continue
            idle_ticks = 0

            start_time = state.current_time_ms
            for req in batch_requests:
                req.mark_scheduled(start_time)
            (
                batch_prefill,
                batch_decode,
                prefill_done_reqs,
                finished_reqs,
            ) = self._apply_batch_result(state, batch_requests, batch_tokens)
            batch_time = self._estimate_batch_time(
                batch_prefill,
                batch_decode,
                self._intercept,
                self._prefill_coeff,
                self._decode_coeff,
            )
            end_time = start_time + batch_time
            for req in prefill_done_reqs:
                req.mark_prefill_done(end_time)
            for req in finished_reqs:
                req.mark_finished(end_time)
            state.total_prefill_tokens += batch_prefill
            state.total_decode_tokens += batch_decode
            state.current_time_ms = end_time
            state.num_batches += 1

        metadata = state.summary_metadata()
        metadata["estimated_wait_ms"] = state.estimate_wait_time_ms()
        return SimulationResult(
            snapshot_version=snapshot.version,
            snapshot_timestamp=snapshot.created_at,
            simulation_timestamp=simulation_timestamp,
            num_requests=len(snapshot.requests),
            metadata=metadata,
        )
