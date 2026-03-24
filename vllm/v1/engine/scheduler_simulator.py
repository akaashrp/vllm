# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Background worker that consumes scheduler snapshots and runs simulation."""

from __future__ import annotations
import cProfile
import io
import os
import pstats
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple, Type

from vllm.logger import init_logger
from vllm.v1.core.sched.state_snapshot import (
    SchedulerKVCacheSnapshot,
    RequestStateSnapshot,
    SchedulerConfigSnapshot,
    SchedulerStateSnapshot,
)

from vllm.v1.core.sched.request_queue import SchedulingPolicy, FCFSRequestQueue, PriorityRequestQueue, create_request_queue
from vllm.v1.request import Request
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.core.sched.snapshot_serialization import (
    encode_scheduler_state_snapshot,
    decode_scheduler_state_snapshot,
)

"""
Config files we don't care about:
1. compilation
2. device
3. kv_events
4. kv_transfer
5. load
6. lora
7. model
8. multimodal
9. observability
10. parallel (only care about dcp_world_size)
11. pooler
12. speculative
13. speech_to_text
14. structured_outputs
15. utils

KV cache remote transfers would require estimating remote transfer time
Speculative decoding would require estimating draft model time per speculative decoding method
"""

logger = init_logger(__name__)

try:
    from vllm.v1.engine import _scheduler_sim as _scheduler_sim_native
except ImportError:
    logger.info("Native scheduler simulator module not found")
    _scheduler_sim_native = None

"""
Problems:
1. KV cache management (different waiting states)
2. Speculative decoding

Note: including LoRAs seems relatively trivial
"""


class SimulationProfiler:
    """Optional cProfile hook controlled by env vars.

    Enable by setting VLLM_SIM_PROFILE=1. Results are written to
    VLLM_SIM_PROFILE_OUT (default: scheduler_sim.prof). Set
    VLLM_SIM_PROFILE_PRINT=N to also log the top-N cumulative entries.
    """

    def __init__(self) -> None:
        self.enabled = bool(os.getenv("VLLM_SIM_PROFILE"))
        self.output_path = os.getenv("VLLM_SIM_PROFILE_OUT", "scheduler_sim.prof")
        self.print_limit = int(os.getenv("VLLM_SIM_PROFILE_PRINT", "0"))
        self.profiler: Optional[cProfile.Profile] = None

    def __enter__(self) -> "SimulationProfiler":
        if self.enabled:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self.profiler:
            return False

        self.profiler.disable()
        stats = pstats.Stats(self.profiler)
        if os.path.exists(self.output_path):
            try:
                stats.add(pstats.Stats(self.output_path))
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to merge existing simulator profile %s: %s",
                    self.output_path,
                    exc,
                )
        stats.dump_stats(self.output_path)

        if self.print_limit > 0:
            buffer = io.StringIO()
            stats.stream = buffer
            stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(self.print_limit)
            logger.info(
                "Scheduler simulator profile (top %d, saved to %s):\n%s",
                self.print_limit,
                self.output_path,
                buffer.getvalue(),
            )
        else:
            logger.info("Scheduler simulator profile saved to %s", self.output_path)
        return False


@dataclass
class SimulationResult:
    snapshot_version: int
    snapshot_timestamp: float
    simulation_timestamp: float
    num_requests: int
    metadata: dict
    snapshot_build_latency_ms: float = 0.0
    simulation_latency_ms: float = 0.0


def _snapshot_backlog_summary(snapshot: SchedulerStateSnapshot) -> dict[str, int]:
    return {
        "resident_set_size": int(snapshot.resident_set_size),
        "waiting_set_size": int(snapshot.waiting_set_size),
        "prefill_backlog_running_tokens": int(snapshot.prefill_backlog_running_tokens),
        "prefill_backlog_running_sq_sum_tokens": int(
            snapshot.prefill_backlog_running_sq_sum_tokens
        ),
        "prefill_backlog_waiting_tokens": int(snapshot.prefill_backlog_waiting_tokens),
        "prefill_backlog_waiting_sq_sum_tokens": int(
            snapshot.prefill_backlog_waiting_sq_sum_tokens
        ),
        "prefill_backlog_total_tokens": int(snapshot.prefill_backlog_total_tokens),
        "prefill_backlog_total_sq_sum_tokens": int(
            snapshot.prefill_backlog_total_sq_sum_tokens
        ),
        "decode_backlog_total_tokens": int(snapshot.decode_backlog_total_tokens),
        "running_context_length_sum_snapshot": int(
            snapshot.running_context_length_sum_snapshot
        ),
        "running_context_length_sq_sum_snapshot": int(
            snapshot.running_context_length_sq_sum_snapshot
        ),
    }


@dataclass
class SimRequestState:
    """Mutable view of an individual request within the simulator."""

    snapshot: RequestStateSnapshot
    num_computed_tokens: int = field(init=False)
    prefill_processed: int = field(init=False)
    decode_processed: int = field(init=False)
    num_preemptions: int = field(init=False)
    num_cached_tokens: int = field(init=False, default=-1)
    first_scheduled_time_ms: Optional[float] = None
    prefill_done_time_ms: Optional[float] = None
    finished_time_ms: Optional[float] = None

    def __post_init__(self) -> None:
        self.status = self._normalize_status(self.snapshot.status)
        self.num_computed_tokens = self.snapshot.num_computed_tokens
        self.prefill_processed = self.snapshot.num_prompt_processed_tokens
        self.decode_processed = self.snapshot.num_output_processed_tokens
        self.num_preemptions = self.snapshot.num_preemptions
        self.num_cached_tokens = self.snapshot.num_cached_tokens
        
        # logger.info(
        #     "Initialized SimRequestState for request %s: status=%s, num_computed_tokens=%d, prefill_processed=%d, decode_processed=%d",
        #     self.request_id,
        #     self.status,
        #     self.num_computed_tokens,
        #     self.prefill_processed,
        #     self.decode_processed,
        # )

    @property
    def request_id(self) -> str:
        return self.snapshot.request_id

    def __hash__(self) -> int:
        # Requests are uniquely identified by request_id.
        return hash(self.request_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimRequestState):
            return False
        return self.request_id == other.request_id

    @property
    def priority(self) -> int:
        return self.snapshot.priority

    @property
    def arrival_time(self) -> float:
        return self.snapshot.arrival_time

    @property
    def num_tokens(self) -> int:
        return self.snapshot.num_prompt_tokens + self.decode_processed

    @property
    def prefill_remaining(self) -> int:
        return max(0, self.snapshot.num_prompt_tokens - self.prefill_processed)

    @property
    def decode_remaining(self) -> int:
        return max(
            0, self.snapshot.num_output_target_tokens - self.decode_processed
        )

    def ensure_decode_backlog(self) -> None:
        # Async scheduling/spec decoding are ignored; nothing to stage.
        return

    def num_new_tokens(self) -> int:
        if self.prefill_remaining > 0:
            return self.prefill_remaining
        # Decode progresses autoregressively; without spec/async we schedule
        # at most one decode token per step.
        return min(1, self.decode_remaining)

    def ensure_decode_backlog(self) -> None:
        """
        The live scheduler always maintains a 1-token gap during decoding
        (num_tokens_with_spec == num_computed_tokens + 1) because the next
        token is emitted after compute. Snapshots may collapse this gap
        (num_tokens == num_computed_tokens), especially if captured right
        after prefill. Reintroduce the gap logically so the simulator will
        schedule the next decode token even when the snapshot shows no gap.
        """
        if self.prefill_remaining == 0 and self.decode_remaining > 0:
            target_prefix = self.snapshot.num_prompt_tokens + self.decode_processed
            if self.num_computed_tokens >= target_prefix:
                # Force a logical 1-token backlog for scheduling.
                self.num_computed_tokens = target_prefix - 1

    def consume(self, num_tokens: int) -> Tuple[int, int]:
        """
        Advance per-request progress by consuming scheduled tokens.

        Mirrors the live scheduler's `_update_after_schedule`: advance
        `num_computed_tokens` immediately when work is scheduled, but do NOT
        advance `decode_processed` here. The emitted decode tokens are added
        later (after the batch "completes") so that `num_tokens` stays ahead
        of `num_computed_tokens` during decoding, matching the live scheduler's
        gap of 1 token.
        """
        prefill_consumed = min(num_tokens, self.prefill_remaining)
        if prefill_consumed:
            self.prefill_processed += prefill_consumed
            num_tokens -= prefill_consumed

        decode_consumed = min(num_tokens, self.decode_remaining)
        if decode_consumed:
            num_tokens -= decode_consumed

        consumed = prefill_consumed + decode_consumed
        if consumed:
            self.num_computed_tokens += consumed

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
        # return self.status == "FINISHED"
        return self.prefill_remaining == 0 and self.decode_remaining == 0

    @staticmethod
    def _normalize_status(status: str) -> str:
        if "WAITING" in status:
            return "WAITING"
        if "FINISHED" in status:
            return "FINISHED"
        return status

@dataclass
class SimulationContext:
    snapshot: SchedulerStateSnapshot
    requests: Dict[str, SimRequestState]
    running: List[SimRequestState]
    waiting: Deque[SimRequestState]
    kv_allocations: Dict[str, List[int]]
    kv_free_blocks: int
    current_time_ms: float = 0.0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    num_batches: int = 0
    queued_at_snapshot: int = 0
    running_at_snapshot: int = 0

    def __init__(self, snapshot: SchedulerStateSnapshot) -> None:
        self.snapshot = snapshot
        self.requests = {
            request_id: SimRequestState(req) for request_id, req in snapshot.requests.items()
        }
        self.running = self._build_running()
        self.waiting = self._build_waiting()
        self.running_at_snapshot = len(snapshot.running_request_ids)
        # exclude dummy request for snapshot count
        self.queued_at_snapshot = max(len(snapshot.waiting_request_ids) - 1, 0)
        self.kv_allocations = self._init_kv_allocations()
        self.kv_free_blocks = snapshot.kv_cache_config.kv_cache_free_blocks
        self.current_time_ms = 0.0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.num_batches = 0

    def summary_metadata(self) -> dict:
        # Aggregate backlog/state fields only; keep payload compact by default.
        non_dummy_running = [req for req in self.running if req.request_id != "__DUMMY__"]
        non_dummy_waiting = [req for req in self.waiting if req.request_id != "__DUMMY__"]
        prefill_backlog_running = sum(req.prefill_remaining for req in non_dummy_running)
        prefill_backlog_running_sq_sum = sum(
            req.prefill_remaining * req.prefill_remaining for req in non_dummy_running
        )
        prefill_backlog_waiting = sum(req.prefill_remaining for req in non_dummy_waiting)
        prefill_backlog_waiting_sq_sum = sum(
            req.prefill_remaining * req.prefill_remaining for req in non_dummy_waiting
        )
        decode_backlog_total = sum(req.decode_remaining for req in non_dummy_running) + sum(
            req.decode_remaining for req in non_dummy_waiting
        )
        return {
            "num_batches": self.num_batches,
            "num_running": len(self.running),
            "num_waiting": len(self.waiting),
            "running_at_snapshot": self.running_at_snapshot,
            "queued_at_snapshot": self.queued_at_snapshot,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "resident_set_size": len(non_dummy_running),
            "waiting_set_size": len(non_dummy_waiting),
            "prefill_backlog_running_tokens": int(prefill_backlog_running),
            "prefill_backlog_running_sq_sum_tokens": int(prefill_backlog_running_sq_sum),
            "prefill_backlog_waiting_tokens": int(prefill_backlog_waiting),
            "prefill_backlog_waiting_sq_sum_tokens": int(prefill_backlog_waiting_sq_sum),
            "prefill_backlog_total_tokens": int(
                prefill_backlog_running + prefill_backlog_waiting
            ),
            "prefill_backlog_total_sq_sum_tokens": int(
                prefill_backlog_running_sq_sum + prefill_backlog_waiting_sq_sum
            ),
            "decode_backlog_total_tokens": int(decode_backlog_total),
        }

    def estimate_wait_time_ms(self, request_id: Optional[str] = None) -> float:
        target_id = request_id or "__DUMMY__"
        req = self.requests.get(target_id)
        if req and req.first_scheduled_time_ms is not None:
            return req.first_scheduled_time_ms
        return 0.0

    def _build_running(self) -> List[SimRequestState]:
        # running: List[SimRequestState] = []
        # for req_id in self.snapshot.running_request_ids:
        #     sim_req = self.requests.get(req_id)
        #     if not sim_req or sim_req.status == "FINISHED":
        #         continue
        #     if sim_req.status == "RUNNING":
        #         running.append(sim_req)
        # return running
        return [self.requests[req_id] for req_id in self.snapshot.running_request_ids]

    def _build_waiting(self) -> Deque[SimRequestState]:
        # queue: Deque[SimRequestState] = deque()
        # for req_id in self.snapshot.waiting_request_ids:
        #     sim_req = self.requests.get(req_id)
        #     if not sim_req or sim_req.status == "FINISHED":
        #         continue
        #     if sim_req.status in ("WAITING", "PREEMPTED"):
        #         queue.append(sim_req)
        # return queue
        return deque([self.requests[req_id] for req_id in self.snapshot.waiting_request_ids])

    def has_pending_requests(self) -> bool:
        if any(request.status != "FINISHED" for request in self.running):
            return True
        return any(request.status != "FINISHED" for request in self.waiting)

    def _init_kv_allocations(self) -> Dict[str, List[int]]:
        allocs: Dict[str, List[int]] = {}
        for req in self.requests.values():
            allocs[req.request_id] = list(req.snapshot.kv_block_counts)
        return allocs

    @property
    def num_kv_groups(self) -> int:
        return len(self.snapshot.kv_cache_config.kv_cache_groups)

    @property
    def block_size(self) -> int:
        return self.snapshot.kv_cache_config.block_size

    def allocated_blocks(self, req: SimRequestState) -> List[int]:
        return self.kv_allocations.get(req.request_id, [0] * self.num_kv_groups)

    def free_request_blocks(self, req: SimRequestState) -> None:
        current = self.allocated_blocks(req)
        reclaimed = sum(current)
        self.kv_free_blocks += reclaimed
        self.kv_allocations[req.request_id] = [0] * self.num_kv_groups

    def create_empty_block_list(self) -> list[list[int]]:
        return [[] for _ in range(self.num_kv_groups)]

    def get_computed_blocks(self, req: SimRequestState) -> Tuple[list[list[int]], int]:
        # Prefix caching/local computed blocks are ignored in the simulator.
        return self.create_empty_block_list(), 0

    def get_blocks(self, request_id: str) -> List[int]:
        return self.kv_allocations.get(
            request_id, [0 for _ in range(self.num_kv_groups)]
        )

    """
    Might need to support this call:
    new_computed_blocks, num_new_local_computed_tokens = (
        self.kv_cache_manager.get_computed_blocks(request)
    )
    """
    def allocate_slots(
        self,
        req: SimRequestState,
        num_new_tokens: int, # always equal to num_new_tokens for us
        num_new_computed_tokens: int = 0, # look at condition in scheduler.py
        new_computed_blocks = None, # look at condition in scheduler.py
        num_lookahead_tokens = 0, # always 0 for us
        delay_cache_blocks = False, # always False for us
        num_encoder_tokens = 0, # always 0 for us
    ) -> Optional[List[int]]:
        block_size = self.block_size
        if block_size <= 0:
            return None

        current_blocks = self.allocated_blocks(req)
        total_tokens_after = req.num_computed_tokens + num_new_tokens
        required_blocks = [
            (total_tokens_after + block_size - 1) // block_size
            for _ in range(self.num_kv_groups)
        ]
        additional = [max(0, reqd - curr) for reqd, curr in zip(required_blocks, current_blocks)]
        additional_total = sum(additional)

        if additional_total > self.kv_free_blocks:
            return None

        # Commit allocation.
        self.kv_free_blocks -= additional_total
        new_blocks = [curr + add for curr, add in zip(current_blocks, additional)]
        self.kv_allocations[req.request_id] = new_blocks
        return new_blocks
    
    def can_allocate_slots(
        self,
        req: SimRequestState,
        num_new_tokens: int,
    ) -> bool:
        block_size = self.block_size
        if block_size <= 0:
            return False

        current_blocks = self.allocated_blocks(req)
        total_tokens_after = req.num_computed_tokens + num_new_tokens
        required_blocks = [
            (total_tokens_after + block_size - 1) // block_size
            for _ in range(self.num_kv_groups)
        ]
        additional = [max(0, reqd - curr) for reqd, curr in zip(required_blocks, current_blocks)]
        additional_total = sum(additional)

        if additional_total > self.kv_free_blocks:
            return False

        return True

class PythonSchedulerSimulationWorker:
    """Runs deterministic simulations based on scheduler snapshots."""

    def __init__(
        self,
        interval_s: float,
        intercept: float,
        prefill_coeff: float,
        decode_coeff: float,
        sum_coeff: float,
        prefill_sq_coeff: float = 0.0,
        sum_sq_coeff: float = 0.0,
    ) -> None:
        self._interval_s = max(interval_s, 0.001)
        self._intercept = intercept
        self._prefill_coeff = prefill_coeff
        self._prefill_sq_coeff = prefill_sq_coeff
        self._decode_coeff = decode_coeff
        self._sum_coeff = sum_coeff
        self._sum_sq_coeff = sum_sq_coeff
        self._latest_snapshot: Optional[SchedulerStateSnapshot] = None
        self._latest_snapshot_summary: Optional[dict[str, int]] = None
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
        snapshot_summary = _snapshot_backlog_summary(snapshot)
        with self._lock:
            self._latest_snapshot = snapshot
            self._latest_snapshot_summary = snapshot_summary

    def latest_result(self) -> Optional[SimulationResult]:
        with self._lock:
            return self._latest_result

    def latest_snapshot_summary(self) -> Optional[dict[str, int]]:
        with self._lock:
            if self._latest_snapshot_summary is None:
                return None
            return dict(self._latest_snapshot_summary)


    def clear_latest_snapshot(self) -> None:
        with self._lock:
            self._latest_snapshot = None
            self._latest_snapshot_summary = None

    def clear_latest_result(self) -> None:
        with self._lock:
            self._latest_result = None

    def _pop_latest_snapshot(self) -> Optional[SchedulerStateSnapshot]:
        with self._lock:
            snapshot = self._latest_snapshot
            self._latest_snapshot = None
            return snapshot

    def _run(self) -> None:
        # logger.info(
        #     "Scheduler simulation worker started with %.3f s cadence",
        #     self._interval_s,
        # )
        while not self._stop_event.is_set():
            sim_elapsed_ms = 0.0
            snapshot = self._pop_latest_snapshot()
            
            # logger.info(
            #     "Num running: %d, Num waiting: %d",
            #     snapshot.num_running if snapshot else -1,
            #     snapshot.num_waiting if snapshot else -1,
            # )
            
            if snapshot is not None:
                sim_start = time.perf_counter()
                with SimulationProfiler():
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
        # logger.info("Scheduler simulation worker stopped")

    def _build_next_batch(
        self, 
        state: SimulationContext
    ) -> Tuple[List[SimRequestState], List[int], int, int]:
        batch_requests: List[SimRequestState] = []
        batch_tokens: List[int] = []
        sum_context_length: int = 0
        sum_sq_tokens: int = 0
        config = state.snapshot.config
        
        max_model_len = config.max_model_len
        
        scheduled_new_reqs = []
        scheduled_resumed_reqs = []
        scheduled_running_reqs = []
        preempted_reqs = []
        
        req_to_new_blocks = {}
        num_scheduled_tokens = {}
        token_budget = config.max_num_batched_tokens
        
        scheduled_timestamp = time.monotonic()
        
        req_index = 0
        # Schedule RUNNING requests first.
        while req_index < len(state.running):
            request = state.running[req_index]
            request.ensure_decode_backlog()
            
            if token_budget <= 0:
                break
            if request.is_finished():
                continue
            
            num_new_tokens = request.num_tokens - request.num_computed_tokens
            if num_new_tokens <= 0:
                raise RuntimeError("Request has no new tokens to schedule")
                # logger.info("Request %s has no new tokens to schedule: num_computed_tokens=%d, num_tokens=%d", request.request_id, request.num_computed_tokens, request.num_tokens)
            
            if 0 < config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget, max_model_len - request.num_computed_tokens)

            if num_new_tokens <= 0:
                req_index += 1
                continue
            
            while True:
                new_blocks = state.allocate_slots(request, num_new_tokens)
                if new_blocks is not None:
                    # The request can be scheduled
                    break
                
                # The request cannot be scheduled.
                # Preempt the lowest-priority request.
                preempted_req: Optional[SimRequestState] = None
                if config.policy == "priority":
                    # preempted_req = max(
                    #     state.running,
                    #     key=lambda r: (r.priority, r.arrival_time),
                    # )
                    # state.running.remove(preempted_req)
                    # if preempted_req in scheduled_running_reqs:
                    #     scheduled_running_reqs.remove(preempted_req)
                    raise NotImplementedError("Priority scheduling not implemented yet")
                elif config.policy == "fcfs":
                    preempted_req = state.running.pop()
                
                state.free_request_blocks(preempted_req)
                preempted_req.status = "PREEMPTED"
                preempted_req.num_computed_tokens = 0
                preempted_req.prefill_processed = 0
                preempted_req.num_preemptions += 1
                
                if config.policy == "priority":
                    # state.waiting.append(preempted_req)
                    raise NotImplementedError("Priority scheduling not implemented yet")
                elif config.policy == "fcfs":
                    state.waiting.appendleft(preempted_req)
                    
                preempted_reqs.append(preempted_req)
                if preempted_req == request:
                    # No more request to preempt. Cannot schedule this request.
                    break
                
            if new_blocks is None:
                # Could not schedule this request.
                break
            
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1
            
            batch_requests.append(request)
            batch_tokens.append(num_new_tokens)
            request_tokens = request.num_tokens
            sum_context_length += request_tokens
            sum_sq_tokens += request_tokens * request_tokens
            
        # BODEN: handle differently based on FCFS vs priority
        if config.policy == "priority":
            raise NotImplementedError("Priority scheduling not implemented yet")
        elif config.policy == "fcfs":
            skipped_waiting_requests = deque() 
        
        if not preempted_reqs:
            while state.waiting:
                request = None # to prevent unbound variable error
                if token_budget <= 0:
                    break
                if len(state.running) == config.max_num_seqs:
                    break
                
                # request = state.waiting.peek_request()
                if config.policy == "priority":
                    raise NotImplementedError("Priority scheduling not implemented yet")
                elif config.policy == "fcfs":
                    request = state.waiting[0]
                request.ensure_decode_backlog()
                
                if request.num_computed_tokens == 0:
                    # BODEN
                    new_computed_blocks, num_new_local_computed_tokens = (
                        state.get_computed_blocks(request)
                    )
                    num_computed_tokens = (
                        num_new_local_computed_tokens
                    )
                else:
                    # BODEN
                    new_computed_blocks = (
                        state.create_empty_block_list()
                    )
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                num_new_tokens = request.num_tokens - num_computed_tokens
                if 0 < config.long_prefill_token_threshold < num_new_tokens:
                    num_new_tokens = (
                        config.long_prefill_token_threshold
                    )

                # no chunked prefill
                if not config.chunked_prefill_enabled and num_new_tokens > token_budget:
                    # state.waiting.pop_request()
                    # skipped_waiting_requests.prepend_request(request)
                    if config.policy == "priority":
                        raise NotImplementedError("Priority scheduling not implemented yet")
                    elif config.policy == "fcfs":
                        request = state.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                    
                    continue
                
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0
                
                new_blocks = state.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                )
                
                if new_blocks is None:
                    break
                
                # request = state.waiting.pop_request()
                if config.policy == "priority":
                    raise NotImplementedError("Priority scheduling not implemented yet")
                elif config.policy == "fcfs":
                    request = state.waiting.popleft()
                
                state.running.append(request)
                
                if request.status == "WAITING":
                    scheduled_new_reqs.append(request)
                elif request.status == "PREEMPTED":
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")
                
                req_to_new_blocks[request.request_id] = (
                    state.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = "RUNNING"
                request.num_computed_tokens = num_computed_tokens    
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                    
                batch_requests.append(request)
                batch_tokens.append(num_new_tokens)
                request_tokens = request.num_tokens
                sum_context_length += request_tokens
                sum_sq_tokens += request_tokens * request_tokens
                
        if skipped_waiting_requests:
            if config.policy == "priority":
                raise NotImplementedError("Priority scheduling not implemented yet")
            elif config.policy == "fcfs":
                state.waiting.extendleft(skipped_waiting_requests)
            
        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= config.max_num_batched_tokens
        assert token_budget >= 0
        assert len(state.running) <= config.max_num_seqs
        
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of schedul`ed requests can be smaller than
        # len(state.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(state.running)

        # # Get the longest common prefix among all requests in the running queue.
        # # This can be potentially used for cascade attention.
        # num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        # if self.running:
        #     any_request = self.running[0]
        #     num_common_prefix_blocks = (
        #         self.kv_cache_manager.get_num_common_prefix_blocks(
        #             any_request, len(self.running)
        #         )
        #     )
        
        # BODEN: summary metadata is still empty
            
        return batch_requests, batch_tokens, sum_context_length, sum_sq_tokens
        
    def _apply_batch_result(
        self,
        state: SimulationContext,
        batch_requests: List[SimRequestState],
        batch_tokens: List[int],
    ) -> Tuple[int, int, int, List[SimRequestState], List[SimRequestState]]:
        total_prefill = 0
        total_prefill_sq_sum = 0
        total_decode = 0
        prefill_done: List[SimRequestState] = []
        finished: List[SimRequestState] = []
        
        """
        Need to update: 
        self.status = self._normalize_status(self.snapshot.status) -> DONE
        self.num_computed_tokens = self.snapshot.num_computed_tokens -> DONE
        self.prefill_processed = self.snapshot.num_prompt_processed_tokens -> DONE
        self.decode_processed = self.snapshot.num_output_processed_tokens -> DONE
        self.num_preemptions = self.snapshot.num_preemptions -> nothing to do
        self.num_cached_tokens = self.snapshot.num_cached_tokens -> nothing to do
        
        running: List[SimRequestState] -> DONE
        waiting: Deque[SimRequestState] -> DONE
        kv_allocations: Dict[str, List[int]] -> DONE
        kv_free_blocks: int -> DONE
        current_time_ms: float = 0.0 -> DONE in _run_simulation
        total_prefill_tokens: int = 0 -> DONE in _run_simulation
        total_decode_tokens: int = 0 -> DONE in _run_simulation
        num_batches: int = 0 -> DONE in _run_simulation
        """
        
        config = state.snapshot.config
        
        stopped_running_reqs: set[SimRequestState] = set()
        stopped_preempted_reqs: set[SimRequestState] = set()
        for request, num_tokens_scheduled in zip(batch_requests, batch_tokens):
            assert num_tokens_scheduled > 0
            if request is None:
                continue
                    
            prefill_before = request.prefill_remaining
            decode_before = request.decode_remaining
            consumed_prefill, consumed_decode = request.consume(num_tokens_scheduled)
            total_prefill += consumed_prefill
            total_prefill_sq_sum += consumed_prefill * consumed_prefill
            total_decode += consumed_decode

            # Simulate emission of decode tokens after the batch finishes:
            # - The first decode token arrives when prefill completes.
            # - Each scheduled decode token produces one emitted token.
            emitted_decode = consumed_decode
            if (
                prefill_before > 0
                and request.prefill_remaining == 0
                and decode_before > 0
            ):
                # logger.info("Request %s prefill completed; emitting first decode token", request.request_id)
                emitted_decode += 1

            if emitted_decode:
                emitted_decode = min(emitted_decode, request.decode_remaining)
                request.decode_processed += emitted_decode

            if prefill_before > 0 and request.prefill_remaining == 0:
                prefill_done.append(request)
            if decode_before > 0 and request.is_finished():
                status_before_stop = request.status
                
                state.free_request_blocks(request)

                request.status = "FINISHED"
                
                if status_before_stop == "RUNNING":
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)
        
        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            state.running = remove_all(state.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            if config.policy == "priority":
                raise NotImplementedError("Priority scheduling not implemented yet")
            elif config.policy == "fcfs":
                state.waiting = remove_all(state.waiting, stopped_preempted_reqs)

        return (
            total_prefill,
            total_prefill_sq_sum,
            total_decode,
            prefill_done,
            stopped_running_reqs.union(stopped_preempted_reqs),
        )

    def _estimate_batch_time(
        self,
        num_prefill_tokens: int,
        num_prefill_sq_sum: int,
        num_decode_tokens: int,
        sum_context_length: int,
        sum_sq_tokens: int,
        intercept: float,
        prefill_coeff: float,
        prefill_sq_coeff: float,
        decode_coeff: float,
        sum_coeff: float,
        sum_sq_coeff: float,
    ) -> float:
        """Simple deterministic estimate for per-batch latency in milliseconds."""
        assert num_prefill_tokens > 0 or num_decode_tokens > 0
        return (
            intercept
            + (prefill_coeff * num_prefill_tokens)
            + (prefill_sq_coeff * num_prefill_sq_sum)
            + (decode_coeff * num_decode_tokens)
            + (sum_coeff * sum_context_length)
            + (sum_sq_coeff * sum_sq_tokens)
        ) * 1000.0

    def _run_simulation(self, snapshot: SchedulerStateSnapshot) -> SimulationResult:
        simulation_timestamp = time.monotonic()
        state = SimulationContext(snapshot)

        idle_ticks = 0
        max_idle_ticks = 4
        while True:
            (
                batch_requests,
                batch_tokens,
                sum_context_length,
                sum_sq_tokens,
            ) = self._build_next_batch(state)
            if any(req.request_id == "__DUMMY__" for req in batch_requests):
                break
            
            if not batch_requests:
                idle_ticks += 1
                if idle_ticks >= max_idle_ticks:
                    raise RuntimeError("Simulation stalled: no requests can be scheduled")
                continue
            idle_ticks = 0

            start_time = state.current_time_ms
            (
                batch_prefill,
                batch_prefill_sq_sum,
                batch_decode,
                _,
                finished_reqs,
            ) = self._apply_batch_result(state, batch_requests, batch_tokens)
            batch_time = self._estimate_batch_time(
                batch_prefill,
                batch_prefill_sq_sum,
                batch_decode,
                sum_context_length,
                sum_sq_tokens,
                self._intercept,
                self._prefill_coeff,
                self._prefill_sq_coeff,
                self._decode_coeff,
                self._sum_coeff,
                self._sum_sq_coeff,
            )
            end_time = start_time + batch_time
            
            for req in finished_reqs:
                req.mark_finished(end_time)
            
            state.total_prefill_tokens += batch_prefill
            state.total_decode_tokens += batch_decode
            state.current_time_ms = end_time
            state.num_batches += 1

        metadata = state.summary_metadata()
        metadata["estimated_wait_ms"] = state.current_time_ms
        return SimulationResult(
            snapshot_version=snapshot.version,
            snapshot_timestamp=snapshot.created_at,
            simulation_timestamp=simulation_timestamp,
            num_requests=len(snapshot.requests),
            metadata=metadata,
        )


class NativeSchedulerSimulationWorker:
    """Wrapper around the C++ scheduler simulation worker."""

    def __init__(
        self,
        interval_s: float,
        intercept: float,
        prefill_coeff: float,
        decode_coeff: float,
        sum_coeff: float,
        prefill_sq_coeff: float = 0.0,
        sum_sq_coeff: float = 0.0,
    ) -> None:
        if _scheduler_sim_native is None:
            raise RuntimeError(
                "Native scheduler simulator extension is not available."
            )
        try:
            self._worker = _scheduler_sim_native.SchedulerSimulationWorker(
                interval_s=interval_s,
                intercept=intercept,
                prefill_coeff=prefill_coeff,
                prefill_sq_coeff=prefill_sq_coeff,
                decode_coeff=decode_coeff,
                sum_coeff=sum_coeff,
                sum_sq_coeff=sum_sq_coeff,
            )
        except TypeError:
            if abs(prefill_sq_coeff) > 0.0 or abs(sum_sq_coeff) > 0.0:
                logger.warning(
                    "Native scheduler simulator extension does not yet support "
                    "quadratic coefficients; prefill_sq_coeff and "
                    "sum_sq_coeff will be ignored."
                )
            self._worker = _scheduler_sim_native.SchedulerSimulationWorker(
                interval_s=interval_s,
                intercept=intercept,
                prefill_coeff=prefill_coeff,
                decode_coeff=decode_coeff,
                sum_coeff=sum_coeff,
            )
        self._latest_snapshot_summary: Optional[dict[str, int]] = None
        logger.info("Scheduler simulator using native backend")

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self._worker.stop()

    def update_snapshot(self, snapshot: SchedulerStateSnapshot) -> None:
        self._latest_snapshot_summary = _snapshot_backlog_summary(snapshot)
        serialized = encode_scheduler_state_snapshot(snapshot)
        self._worker.update_snapshot(serialized)

    def latest_result(self) -> Optional[SimulationResult]:
        summary = self._worker.latest_result_summary()
        return self._result_from_summary(summary)

    @staticmethod
    def _result_from_summary(summary: Any) -> Optional[SimulationResult]:
        if summary is None:
            return None
        (
            snapshot_version,
            snapshot_timestamp,
            simulation_timestamp,
            num_requests,
            snapshot_build_latency_ms,
            simulation_latency_ms,
            metadata,
        ) = summary
        result = SimulationResult(
            snapshot_version=snapshot_version,
            snapshot_timestamp=snapshot_timestamp,
            simulation_timestamp=simulation_timestamp,
            num_requests=num_requests,
            metadata=metadata,
        )
        result.snapshot_build_latency_ms = snapshot_build_latency_ms
        result.simulation_latency_ms = simulation_latency_ms
        return result

    def run_simulation_on_latest_snapshot(
        self,
        prompt_tokens: int,
    ) -> Optional[SimulationResult]:
        run_critical_path = getattr(
            self._worker, "run_simulation_on_latest_snapshot", None
        )
        if run_critical_path is None:
            return None
        summary = run_critical_path(int(prompt_tokens))
        return self._result_from_summary(summary)

    def latest_snapshot_summary(self) -> Optional[dict[str, int]]:
        if self._latest_snapshot_summary is None:
            return None
        return dict(self._latest_snapshot_summary)

    def latest_snapshot(self) -> Optional[dict]:
        snapshot_bytes = self._worker.latest_snapshot()
        if snapshot_bytes is None:
            return None
        return decode_scheduler_state_snapshot(snapshot_bytes)

    def clear_latest_snapshot(self) -> None:
        self._latest_snapshot_summary = None
        self._worker.clear_latest_snapshot()

    def clear_latest_result(self) -> None:
        self._worker.clear_latest_result()

    def run_simulation(self, snapshot: SchedulerStateSnapshot) -> dict:
        serialized = encode_scheduler_state_snapshot(snapshot)
        return self._worker.run_simulation_for_test(serialized)


SchedulerSimulationWorker: Type[Any]
if _scheduler_sim_native is not None:
    SchedulerSimulationWorker = NativeSchedulerSimulationWorker
else:
    SchedulerSimulationWorker = PythonSchedulerSimulationWorker
