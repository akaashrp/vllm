# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import statistics
import time

import pytest

from vllm.v1.core.sched.snapshot_serialization import (
    decode_scheduler_state_snapshot,
    encode_scheduler_state_snapshot,
    serialize_scheduler_state_snapshot,
)
from vllm.v1.core.sched.state_snapshot import (
    RequestStateSnapshot,
    SchedulerConfigSnapshot,
    SchedulerKVCacheSnapshot,
    SchedulerParallelSnapshot,
    SchedulerStateSnapshot,
)
from vllm.v1.engine.scheduler_simulator import (
    NativeSchedulerSimulationWorker,
    PythonSchedulerSimulationWorker,
    SchedulerSimulationWorker,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec

_scheduler_sim_native = pytest.importorskip(
    "vllm.v1.engine._scheduler_sim",
    reason="scheduler simulator native extension is not built",
)


def _build_snapshot() -> SchedulerStateSnapshot:
    req_snapshot = RequestStateSnapshot(
        request_id="req-1",
        status="WAITING",
        priority=1,
        arrival_time=1.5,
        num_prompt_tokens=4,
        num_computed_tokens=2,
        num_output_target_tokens=8,
        num_prompt_processed_tokens=2,
        num_output_processed_tokens=1,
        max_tokens=16,
        num_preemptions=0,
        num_cached_tokens=0,
        is_long_prompt=False,
        kv_block_counts=(1,),
    )
    config_snapshot = SchedulerConfigSnapshot(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        max_model_len=2048,
        long_prefill_token_threshold=256,
        chunked_prefill_enabled=True,
        policy="fcfs",
    )
    kv_cache_snapshot = SchedulerKVCacheSnapshot(
        num_gpu_blocks=1024,
        block_size=16,
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer0"],
                kv_cache_spec=KVCacheSpec(block_size=16),
            )
        ],
        kv_cache_usage=0.5,
        kv_cache_total_blocks=2048,
        kv_cache_free_blocks=512,
    )
    parallel_snapshot = SchedulerParallelSnapshot(
        decode_context_parallel_size=1
    )
    dummy_snapshot = RequestStateSnapshot(
        request_id="__DUMMY__",
        status="WAITING",
        priority=0,
        arrival_time=10.0,
        num_prompt_tokens=32,
        num_computed_tokens=0,
        num_output_target_tokens=32,
        num_prompt_processed_tokens=0,
        num_output_processed_tokens=0,
        max_tokens=64,
        num_preemptions=0,
        num_cached_tokens=0,
        is_long_prompt=False,
        kv_block_counts=(0,),
    )
    return SchedulerStateSnapshot(
        version=1,
        created_at=0.0,
        num_running=0,
        num_waiting=2,
        running_request_ids=[],
        waiting_request_ids=["req-1", "__DUMMY__"],
        requests={"req-1": req_snapshot, "__DUMMY__": dummy_snapshot},
        config=config_snapshot,
        kv_cache_config=kv_cache_snapshot,
        parallel_config=parallel_snapshot,
    )


def _build_multi_request_snapshot() -> SchedulerStateSnapshot:
    running_req = RequestStateSnapshot(
        request_id="run-req",
        status="RUNNING",
        priority=0,
        arrival_time=1.0,
        num_prompt_tokens=6,
        num_computed_tokens=3,
        num_output_target_tokens=12,
        num_prompt_processed_tokens=3,
        num_output_processed_tokens=1,
        max_tokens=32,
        num_preemptions=0,
        num_cached_tokens=3,
        is_long_prompt=False,
        kv_block_counts=(1,),
    )
    waiting_req = RequestStateSnapshot(
        request_id="wait-req",
        status="WAITING",
        priority=0,
        arrival_time=2.0,
        num_prompt_tokens=5,
        num_computed_tokens=0,
        num_output_target_tokens=10,
        num_prompt_processed_tokens=0,
        num_output_processed_tokens=0,
        max_tokens=32,
        num_preemptions=0,
        num_cached_tokens=0,
        is_long_prompt=False,
        kv_block_counts=(0,),
    )
    dummy_req = RequestStateSnapshot(
        request_id="__DUMMY__",
        status="WAITING",
        priority=0,
        arrival_time=10.0,
        num_prompt_tokens=32,
        num_computed_tokens=0,
        num_output_target_tokens=32,
        num_prompt_processed_tokens=0,
        num_output_processed_tokens=0,
        max_tokens=64,
        num_preemptions=0,
        num_cached_tokens=0,
        is_long_prompt=False,
        kv_block_counts=(0,),
    )
    config_snapshot = SchedulerConfigSnapshot(
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=2048,
        long_prefill_token_threshold=256,
        chunked_prefill_enabled=True,
        policy="fcfs",
    )
    kv_cache_snapshot = SchedulerKVCacheSnapshot(
        num_gpu_blocks=1024,
        block_size=16,
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer0"],
                kv_cache_spec=KVCacheSpec(block_size=16),
            )
        ],
        kv_cache_usage=0.25,
        kv_cache_total_blocks=2048,
        kv_cache_free_blocks=800,
    )
    parallel_snapshot = SchedulerParallelSnapshot(
        decode_context_parallel_size=1
    )
    return SchedulerStateSnapshot(
        version=2,
        created_at=5.0,
        num_running=1,
        num_waiting=2,
        running_request_ids=["run-req"],
        waiting_request_ids=["wait-req", "__DUMMY__"],
        requests={
            "run-req": running_req,
            "wait-req": waiting_req,
            "__DUMMY__": dummy_req,
        },
        config=config_snapshot,
        kv_cache_config=kv_cache_snapshot,
        parallel_config=parallel_snapshot,
    )


def _build_heavy_snapshot(num_running: int = 16,
                          num_waiting: int = 512) -> SchedulerStateSnapshot:
    """Creates a deterministic workload large enough for timing."""
    running_requests = {}
    waiting_requests = {}
    running_ids = []
    waiting_ids = []

    for idx in range(num_running):
        request_id = f"run-{idx}"
        running_ids.append(request_id)
        running_requests[request_id] = RequestStateSnapshot(
            request_id=request_id,
            status="RUNNING",
            priority=idx % 4,
            arrival_time=idx * 0.25,
            num_prompt_tokens=128,
            num_computed_tokens=64,
            num_output_target_tokens=512,
            num_prompt_processed_tokens=64,
            num_output_processed_tokens=32,
            max_tokens=1024,
            num_preemptions=0,
            num_cached_tokens=64,
            is_long_prompt=False,
            kv_block_counts=(4,),
        )

    for idx in range(num_waiting):
        request_id = f"wait-{idx}"
        waiting_ids.append(request_id)
        waiting_requests[request_id] = RequestStateSnapshot(
            request_id=request_id,
            status="WAITING",
            priority=idx % 4,
            arrival_time=5.0 + idx * 0.1,
            num_prompt_tokens=256,
            num_computed_tokens=0,
            num_output_target_tokens=512,
            num_prompt_processed_tokens=0,
            num_output_processed_tokens=0,
            max_tokens=1024,
            num_preemptions=0,
            num_cached_tokens=0,
            is_long_prompt=False,
            kv_block_counts=(0,),
        )

    # Ensure the simulator eventually terminates by appending a dummy request.
    dummy_request_id = "__DUMMY__"
    waiting_ids.append(dummy_request_id)
    waiting_requests[dummy_request_id] = RequestStateSnapshot(
        request_id=dummy_request_id,
        status="WAITING",
        priority=0,
        arrival_time=10.0,
        num_prompt_tokens=32,
        num_computed_tokens=0,
        num_output_target_tokens=32,
        num_prompt_processed_tokens=0,
        num_output_processed_tokens=0,
        max_tokens=64,
        num_preemptions=0,
        num_cached_tokens=0,
        is_long_prompt=False,
        kv_block_counts=(0,),
    )

    config_snapshot = SchedulerConfigSnapshot(
        max_num_batched_tokens=4096,
        max_num_seqs=128,
        max_model_len=8192,
        long_prefill_token_threshold=512,
        chunked_prefill_enabled=True,
        policy="fcfs",
    )
    kv_cache_snapshot = SchedulerKVCacheSnapshot(
        num_gpu_blocks=8192,
        block_size=16,
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer0"],
                kv_cache_spec=KVCacheSpec(block_size=16),
            )
        ],
        kv_cache_usage=0.6,
        kv_cache_total_blocks=16384,
        kv_cache_free_blocks=4000,
    )
    parallel_snapshot = SchedulerParallelSnapshot(
        decode_context_parallel_size=1
    )
    requests = {}
    requests.update(running_requests)
    requests.update(waiting_requests)
    return SchedulerStateSnapshot(
        version=3,
        created_at=5.0,
        num_running=num_running,
        num_waiting=num_waiting + 1,
        running_request_ids=running_ids,
        waiting_request_ids=waiting_ids,
        requests=requests,
        config=config_snapshot,
        kv_cache_config=kv_cache_snapshot,
        parallel_config=parallel_snapshot,
    )


def test_native_worker_receives_python_snapshot():
    snapshot = _build_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)
    expected = serialize_scheduler_state_snapshot(snapshot)

    worker = _scheduler_sim_native.SchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        decode_coeff=0.2,
    )
    worker.update_snapshot(encoded)

    latest_bytes = worker.latest_snapshot()
    assert isinstance(latest_bytes, (bytes, bytearray))
    decoded = decode_scheduler_state_snapshot(latest_bytes)
    assert decoded == expected

    parsed = worker.parsed_snapshot()
    assert parsed == expected


def test_native_wrapper_round_trip_snapshot():
    snapshot = _build_snapshot()
    expected = serialize_scheduler_state_snapshot(snapshot)

    worker = NativeSchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        decode_coeff=0.2,
    )
    worker.update_snapshot(snapshot)

    latest = worker.latest_snapshot()
    assert latest == expected


def test_native_run_simulation_matches_python():
    snapshot = _build_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)

    py_worker = PythonSchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        decode_coeff=0.2,
    )
    py_result = py_worker._run_simulation(snapshot)

    native_worker = _scheduler_sim_native.SchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        decode_coeff=0.2,
    )
    native_metadata = native_worker.run_simulation_for_test(encoded)
    
    assert native_metadata == py_result.metadata


def test_native_simulation_multiple_real_batches():
    snapshot = _build_multi_request_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)

    py_worker = PythonSchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        decode_coeff=0.2,
    )
    py_result = py_worker._run_simulation(snapshot)

    native_worker = _scheduler_sim_native.SchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        decode_coeff=0.2,
    )
    native_metadata = native_worker.run_simulation_for_test(encoded)
        
    assert native_metadata == py_result.metadata
    # Ensure at least one real batch happened before dummy termination.
    assert native_metadata["num_batches"] >= 2
    # max_num_seqs=1 should restrict concurrency.
    assert native_metadata["num_running"] <= 1


def test_python_worker_invokes_native_simulator():
    snapshot = _build_snapshot()
    py_worker = PythonSchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        decode_coeff=0.2,
    )
    if SchedulerSimulationWorker is not NativeSchedulerSimulationWorker:
        pytest.skip("Native scheduler simulator unavailable")
    native_worker = SchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        decode_coeff=0.2,
    )
    metadata = native_worker.run_simulation(snapshot)
    encoded = encode_scheduler_state_snapshot(snapshot)
    backend_worker = _scheduler_sim_native.SchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        decode_coeff=0.2,
    )
    native_metadata = backend_worker.run_simulation_for_test(encoded)
    assert metadata == native_metadata
    native_worker.stop()


_TIMING_ENV = "VLLM_RUN_SCHEDULER_SIM_TIMING"
_TIMING_WARMUP = 3
_TIMING_RUNS = 10

def test_native_scheduler_simulator_timing():
    """Collects repeatable timing numbers for the native simulator.

    The test is opt-in to avoid slowing down CI runs; set
    VLLM_RUN_SCHEDULER_SIM_TIMING=1 when invoking pytest to record results.
    """
    if not os.environ.get(_TIMING_ENV):
        pytest.skip(f"Set {_TIMING_ENV}=1 to run timing measurement")

    snapshot = _build_heavy_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)
    worker = _scheduler_sim_native.SchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        decode_coeff=0.2,
    )

    for _ in range(_TIMING_WARMUP):
        worker.run_simulation_for_test(encoded)

    durations = []
    for _ in range(_TIMING_RUNS):
        start = time.perf_counter()
        worker.run_simulation_for_test(encoded)
        durations.append(time.perf_counter() - start)

    mean_ms = statistics.fmean(durations) * 1000.0
    median_ms = statistics.median(durations) * 1000.0

    print(  # noqa: T201 - intentional diagnostic output.
        f"[scheduler_sim_native] runs={_TIMING_RUNS} "
        f"mean={mean_ms:.6f}ms median={median_ms:.6f}ms"
    )

    assert len(durations) == _TIMING_RUNS
    assert mean_ms > 0.0
