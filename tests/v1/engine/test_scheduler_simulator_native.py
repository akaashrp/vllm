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
from vllm.v1.engine.snapshot_shm import SnapshotShmPublisher
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec

_scheduler_sim_native = pytest.importorskip(
    "vllm.v1.engine._scheduler_sim",
    reason="scheduler simulator native extension is not built",
)


def _make_request(
    request_id: str,
    *,
    status: str,
    arrival_time: float,
    num_prompt_tokens: int,
    num_computed_tokens: int,
    num_output_target_tokens: int,
    num_prompt_processed_tokens: int,
    num_output_processed_tokens: int,
    max_tokens: int,
    priority: int = 0,
    num_cached_tokens: int = 0,
    kv_block_count: int = 0,
) -> RequestStateSnapshot:
    return RequestStateSnapshot(
        request_id=request_id,
        status=status,
        priority=priority,
        arrival_time=arrival_time,
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=num_computed_tokens,
        num_output_target_tokens=num_output_target_tokens,
        num_prompt_processed_tokens=num_prompt_processed_tokens,
        num_output_processed_tokens=num_output_processed_tokens,
        max_tokens=max_tokens,
        num_preemptions=0,
        num_cached_tokens=num_cached_tokens,
        is_long_prompt=False,
        kv_block_counts=(kv_block_count,),
    )


def _make_snapshot(
    *,
    version: int,
    created_at: float,
    running_requests: list[RequestStateSnapshot],
    waiting_requests: list[RequestStateSnapshot],
    max_num_batched_tokens: int = 128,
    max_num_seqs: int = 4,
    num_gpu_blocks: int = 1024,
    kv_cache_usage: float = 0.5,
    kv_cache_total_blocks: int = 2048,
    kv_cache_free_blocks: int = 512,
) -> SchedulerStateSnapshot:
    running_request_ids = [request.request_id for request in running_requests]
    waiting_request_ids = [request.request_id for request in waiting_requests]
    requests = {
        request.request_id: request
        for request in [*running_requests, *waiting_requests]
    }

    prefill_backlog_running_tokens = sum(
        max(
            request.num_prompt_tokens - request.num_prompt_processed_tokens,
            0,
        )
        for request in running_requests
    )
    prefill_backlog_waiting_tokens = sum(
        max(request.num_prompt_tokens, 0) for request in waiting_requests
    )
    running_context_length_sum_snapshot = sum(
        max(request.num_computed_tokens, 0) for request in running_requests
    )

    return SchedulerStateSnapshot(
        version=version,
        created_at=created_at,
        num_running=len(running_requests),
        num_waiting=len(waiting_requests),
        running_request_ids=running_request_ids,
        waiting_request_ids=waiting_request_ids,
        requests=requests,
        config=SchedulerConfigSnapshot(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=2048,
            long_prefill_token_threshold=256,
            chunked_prefill_enabled=True,
            policy="fcfs",
        ),
        kv_cache_config=SchedulerKVCacheSnapshot(
            num_gpu_blocks=num_gpu_blocks,
            block_size=16,
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["layer0"],
                    kv_cache_spec=KVCacheSpec(block_size=16),
                )
            ],
            kv_cache_usage=kv_cache_usage,
            kv_cache_total_blocks=kv_cache_total_blocks,
            kv_cache_free_blocks=kv_cache_free_blocks,
        ),
        parallel_config=SchedulerParallelSnapshot(
            decode_context_parallel_size=1,
        ),
        resident_set_size=len(running_requests),
        waiting_set_size=len(waiting_requests),
        prefill_backlog_running_tokens=prefill_backlog_running_tokens,
        prefill_backlog_waiting_tokens=prefill_backlog_waiting_tokens,
        prefill_backlog_total_tokens=(
            prefill_backlog_running_tokens + prefill_backlog_waiting_tokens
        ),
        running_context_length_sum_snapshot=running_context_length_sum_snapshot,
    )


def _build_snapshot() -> SchedulerStateSnapshot:
    return _make_snapshot(
        version=1,
        created_at=0.0,
        running_requests=[],
        waiting_requests=[
            _make_request(
                "req-1",
                status="WAITING",
                arrival_time=1.5,
                num_prompt_tokens=4,
                num_computed_tokens=2,
                num_output_target_tokens=8,
                num_prompt_processed_tokens=2,
                num_output_processed_tokens=1,
                max_tokens=16,
                priority=1,
                kv_block_count=1,
            )
        ],
    )


def _build_multi_request_snapshot() -> SchedulerStateSnapshot:
    return _make_snapshot(
        version=2,
        created_at=5.0,
        max_num_seqs=1,
        kv_cache_usage=0.25,
        kv_cache_free_blocks=800,
        running_requests=[
            _make_request(
                "run-req",
                status="RUNNING",
                arrival_time=1.0,
                num_prompt_tokens=6,
                num_computed_tokens=3,
                num_output_target_tokens=12,
                num_prompt_processed_tokens=3,
                num_output_processed_tokens=1,
                max_tokens=32,
                num_cached_tokens=3,
                kv_block_count=1,
            )
        ],
        waiting_requests=[
            _make_request(
                "wait-req",
                status="WAITING",
                arrival_time=2.0,
                num_prompt_tokens=5,
                num_computed_tokens=0,
                num_output_target_tokens=10,
                num_prompt_processed_tokens=0,
                num_output_processed_tokens=0,
                max_tokens=32,
            )
        ],
    )


def _build_prompt_override_snapshot() -> SchedulerStateSnapshot:
    return _make_snapshot(
        version=4,
        created_at=9.0,
        max_num_seqs=2,
        num_gpu_blocks=8,
        kv_cache_usage=0.75,
        kv_cache_total_blocks=8,
        kv_cache_free_blocks=2,
        running_requests=[
            _make_request(
                "run-req",
                status="RUNNING",
                arrival_time=1.0,
                num_prompt_tokens=64,
                num_computed_tokens=64,
                num_output_target_tokens=68,
                num_prompt_processed_tokens=64,
                num_output_processed_tokens=0,
                max_tokens=68,
                num_cached_tokens=64,
                kv_block_count=6,
            )
        ],
        waiting_requests=[],
    )


def _build_heavy_snapshot(
    num_running: int = 16,
    num_waiting: int = 512,
) -> SchedulerStateSnapshot:
    running_requests = [
        _make_request(
            f"run-{idx}",
            status="RUNNING",
            arrival_time=idx * 0.25,
            num_prompt_tokens=128,
            num_computed_tokens=64,
            num_output_target_tokens=512,
            num_prompt_processed_tokens=64,
            num_output_processed_tokens=32,
            max_tokens=1024,
            priority=idx % 4,
            num_cached_tokens=64,
            kv_block_count=4,
        )
        for idx in range(num_running)
    ]
    waiting_requests = [
        _make_request(
            f"wait-{idx}",
            status="WAITING",
            arrival_time=5.0 + idx * 0.1,
            num_prompt_tokens=256,
            num_computed_tokens=0,
            num_output_target_tokens=512,
            num_prompt_processed_tokens=0,
            num_output_processed_tokens=0,
            max_tokens=1024,
            priority=idx % 4,
        )
        for idx in range(num_waiting)
    ]
    return _make_snapshot(
        version=3,
        created_at=5.0,
        running_requests=running_requests,
        waiting_requests=waiting_requests,
        max_num_batched_tokens=4096,
        max_num_seqs=128,
        num_gpu_blocks=8192,
        kv_cache_usage=0.6,
        kv_cache_total_blocks=16384,
        kv_cache_free_blocks=4000,
    )


def _make_worker():
    return _scheduler_sim_native.SchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        prefill_sq_coeff=0.0,
        decode_coeff=0.2,
        sum_coeff=0.0,
        sum_sq_coeff=0.0,
    )


def _make_python_worker() -> PythonSchedulerSimulationWorker:
    return PythonSchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        prefill_sq_coeff=0.0,
        decode_coeff=0.2,
        sum_coeff=0.0,
        sum_sq_coeff=0.0,
    )


def _publisher_name() -> str:
    return f"vllm_native_snapshot_test_{os.getpid()}_{time.time_ns()}"


def _build_watcher_snapshot(version: int, created_at: float) -> SchedulerStateSnapshot:
    return _make_snapshot(
        version=version,
        created_at=created_at,
        running_requests=[],
        waiting_requests=[
            _make_request(
                f"watch-{version}",
                status="WAITING",
                arrival_time=created_at,
                num_prompt_tokens=16 + version,
                num_computed_tokens=0,
                num_output_target_tokens=32,
                num_prompt_processed_tokens=0,
                num_output_processed_tokens=0,
                max_tokens=32,
            )
        ],
    )


def _wait_for_parsed_summary(worker, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        summary = worker.latest_parsed_snapshot_summary()
        if summary is not None:
            return summary
        time.sleep(0.01)
    return None


def _wait_for_parsed_version(worker, version: int, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        summary = worker.latest_parsed_snapshot_summary()
        if summary is not None and int(summary[0]) == int(version):
            return summary
        time.sleep(0.01)
    return None


def test_native_worker_receives_python_snapshot():
    snapshot = _build_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)
    expected = serialize_scheduler_state_snapshot(snapshot)

    worker = _make_worker()
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
        prefill_sq_coeff=0.0,
        decode_coeff=0.2,
        sum_coeff=0.0,
        sum_sq_coeff=0.0,
    )
    worker.update_snapshot(snapshot)

    latest = worker.latest_snapshot()
    assert latest == expected


def test_native_run_simulation_matches_python_drain():
    snapshot = _build_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)

    py_metadata = _make_python_worker().run_simulation(snapshot)
    native_metadata = _make_worker().run_simulation_for_test(encoded)

    assert native_metadata == py_metadata


def test_native_simulation_multiple_real_batches_without_dummy():
    snapshot = _build_multi_request_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)

    py_metadata = _make_python_worker().run_simulation(snapshot)
    native_metadata = _make_worker().run_simulation_for_test(encoded)

    assert native_metadata == py_metadata
    assert native_metadata["num_batches"] >= 2
    assert native_metadata["num_running"] <= 1
    assert native_metadata["num_waiting"] == 0


def test_native_prompt_override_first_schedule_vs_prefill_done():
    snapshot = _build_prompt_override_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)
    worker = _make_worker()
    worker.update_snapshot(encoded)

    first_schedule = worker.run_simulation_on_latest_snapshot(16, "first_schedule")
    prefill_done = worker.run_simulation_on_latest_snapshot(16, "prefill_done")

    assert first_schedule is not None
    assert prefill_done is not None
    first_schedule_metadata = first_schedule[6]
    prefill_done_metadata = prefill_done[6]

    assert (
        float(prefill_done_metadata["estimated_wait_ms"])
        > float(first_schedule_metadata["estimated_wait_ms"])
    )
    assert int(first_schedule_metadata["queued_at_snapshot"]) == 0
    assert int(prefill_done_metadata["queued_at_snapshot"]) == 0


def test_native_prompt_override_wait_is_monotonic_in_prompt_size():
    snapshot = _build_prompt_override_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)
    worker = _make_worker()
    worker.update_snapshot(encoded)

    small = worker.run_simulation_on_latest_snapshot(16, "prefill_done")
    large = worker.run_simulation_on_latest_snapshot(128, "prefill_done")

    assert small is not None
    assert large is not None
    assert float(large[6]["estimated_wait_ms"]) > float(small[6]["estimated_wait_ms"])


def test_native_prompt_override_does_not_mutate_cached_snapshot_state():
    snapshot = _build_prompt_override_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)
    worker = _make_worker()
    worker.update_snapshot(encoded)

    baseline = worker.run_simulation_for_test(encoded)
    worker.run_simulation_on_latest_snapshot(128, "prefill_done")
    after_override = worker.run_simulation_for_test(encoded)

    assert after_override == baseline


def test_native_snapshot_watcher_parses_initial_snapshot():
    snapshot = _build_watcher_snapshot(version=21, created_at=12.5)
    payload = encode_scheduler_state_snapshot(snapshot)
    publisher = SnapshotShmPublisher(
        name=_publisher_name(),
        size_bytes=1024 * 1024,
        simulation_intercept=1.0,
        simulation_prefill_coeff=0.1,
        simulation_prefill_sq_coeff=0.0,
        simulation_decode_coeff=0.2,
        simulation_sum_coeff=0.0,
        simulation_sum_sq_coeff=0.0,
    )
    worker = _make_worker()
    try:
        assert publisher.publish(snapshot, payload) is True
        worker.start_snapshot_shm_watcher(
            publisher.name,
            publisher.size_bytes,
            1,
        )

        summary = _wait_for_parsed_summary(worker)
        assert summary is not None
        assert int(summary[0]) == snapshot.version
        assert float(summary[1]) == snapshot.created_at
        assert int(summary[2]) == len(snapshot.requests)
        assert float(summary[3]) == snapshot.build_latency_ms
    finally:
        worker.stop()
        publisher.close()


def test_native_snapshot_watcher_converges_to_latest_version():
    publisher = SnapshotShmPublisher(
        name=_publisher_name(),
        size_bytes=1024 * 1024,
        simulation_intercept=1.0,
        simulation_prefill_coeff=0.1,
        simulation_prefill_sq_coeff=0.0,
        simulation_decode_coeff=0.2,
        simulation_sum_coeff=0.0,
        simulation_sum_sq_coeff=0.0,
    )
    worker = _make_worker()
    try:
        initial = _build_watcher_snapshot(version=30, created_at=30.0)
        assert publisher.publish(initial, encode_scheduler_state_snapshot(initial))
        worker.start_snapshot_shm_watcher(
            publisher.name,
            publisher.size_bytes,
            1,
        )
        assert _wait_for_parsed_version(worker, 30) is not None

        for version in range(31, 36):
            snapshot = _build_watcher_snapshot(
                version=version,
                created_at=float(version),
            )
            assert publisher.publish(snapshot, encode_scheduler_state_snapshot(snapshot))

        summary = _wait_for_parsed_version(worker, 35)
        assert summary is not None
        assert int(summary[0]) == 35
    finally:
        worker.stop()
        publisher.close()


def test_native_snapshot_watcher_critical_path_uses_latest_parsed_snapshot():
    publisher = SnapshotShmPublisher(
        name=_publisher_name(),
        size_bytes=1024 * 1024,
        simulation_intercept=1.0,
        simulation_prefill_coeff=0.1,
        simulation_prefill_sq_coeff=0.0,
        simulation_decode_coeff=0.2,
        simulation_sum_coeff=0.0,
        simulation_sum_sq_coeff=0.0,
    )
    worker = _make_worker()
    try:
        first = _build_watcher_snapshot(version=40, created_at=40.0)
        assert publisher.publish(first, encode_scheduler_state_snapshot(first))
        worker.start_snapshot_shm_watcher(
            publisher.name,
            publisher.size_bytes,
            1,
        )
        assert _wait_for_parsed_version(worker, 40) is not None

        summary = worker.run_simulation_on_latest_snapshot(32, "prefill_done")
        assert summary is not None
        assert int(summary[0]) == 40

        second = _build_watcher_snapshot(version=41, created_at=41.0)
        assert publisher.publish(second, encode_scheduler_state_snapshot(second))
        assert _wait_for_parsed_version(worker, 41) is not None

        summary = worker.run_simulation_on_latest_snapshot(32, "prefill_done")
        assert summary is not None
        assert int(summary[0]) == 41
    finally:
        worker.stop()
        publisher.close()


def test_python_worker_invokes_native_simulator():
    snapshot = _build_snapshot()
    if SchedulerSimulationWorker is not NativeSchedulerSimulationWorker:
        pytest.skip("Native scheduler simulator unavailable")

    native_worker = SchedulerSimulationWorker(
        interval_s=0.01,
        intercept=1.0,
        prefill_coeff=0.1,
        prefill_sq_coeff=0.0,
        decode_coeff=0.2,
        sum_coeff=0.0,
        sum_sq_coeff=0.0,
    )
    metadata = native_worker.run_simulation(
        snapshot,
        prompt_tokens=32,
        stop_mode="prefill_done",
    )
    encoded = encode_scheduler_state_snapshot(snapshot)
    backend_worker = _make_worker()
    native_metadata = backend_worker.run_simulation_for_test(
        encoded,
        32,
        "prefill_done",
    )
    assert metadata == native_metadata
    native_worker.stop()


_TIMING_ENV = "VLLM_RUN_SCHEDULER_SIM_TIMING"
_TIMING_WARMUP = 3
_TIMING_RUNS = 10


def test_native_scheduler_simulator_timing_light():
    """Collect repeatable timing numbers for the native simulator."""
    if not os.environ.get(_TIMING_ENV):
        pytest.skip(f"Set {_TIMING_ENV}=1 to run timing measurement")

    snapshot = _build_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)
    worker = _make_worker()

    for _ in range(_TIMING_WARMUP):
        worker.run_simulation_for_test(encoded)

    durations = []
    for _ in range(_TIMING_RUNS):
        start = time.perf_counter()
        worker.run_simulation_for_test(encoded)
        durations.append(time.perf_counter() - start)

    mean_ms = statistics.fmean(durations) * 1000.0
    median_ms = statistics.median(durations) * 1000.0

    print(  # noqa: T201
        f"[scheduler_sim_native] runs={_TIMING_RUNS} "
        f"mean={mean_ms:.6f}ms median={median_ms:.6f}ms"
    )

    assert len(durations) == _TIMING_RUNS
    assert mean_ms > 0.0


def test_native_scheduler_simulator_timing():
    """Collect repeatable timing numbers for the native simulator."""
    if not os.environ.get(_TIMING_ENV):
        pytest.skip(f"Set {_TIMING_ENV}=1 to run timing measurement")

    snapshot = _build_heavy_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)
    worker = _make_worker()

    for _ in range(_TIMING_WARMUP):
        worker.run_simulation_for_test(encoded)

    durations = []
    for _ in range(_TIMING_RUNS):
        start = time.perf_counter()
        worker.run_simulation_for_test(encoded)
        durations.append(time.perf_counter() - start)

    mean_ms = statistics.fmean(durations) * 1000.0
    median_ms = statistics.median(durations) * 1000.0

    print(  # noqa: T201
        f"[scheduler_sim_native] runs={_TIMING_RUNS} "
        f"mean={mean_ms:.6f}ms median={median_ms:.6f}ms"
    )

    assert len(durations) == _TIMING_RUNS
    assert mean_ms > 0.0


def test_native_scheduler_simulator_critical_path_timing():
    """Measure the cached-snapshot critical-path timing used by the router."""
    if not os.environ.get(_TIMING_ENV):
        pytest.skip(f"Set {_TIMING_ENV}=1 to run timing measurement")

    snapshot = _build_heavy_snapshot()
    encoded = encode_scheduler_state_snapshot(snapshot)
    worker = _make_worker()
    worker.update_snapshot(encoded)
    prompt_tokens = 128

    for _ in range(_TIMING_WARMUP):
        summary = worker.run_simulation_on_latest_snapshot(
            prompt_tokens,
            "prefill_done",
        )
        assert summary is not None

    outer_durations = []
    native_latencies_ms = []
    for _ in range(_TIMING_RUNS):
        start = time.perf_counter()
        summary = worker.run_simulation_on_latest_snapshot(
            prompt_tokens,
            "prefill_done",
        )
        outer_durations.append(time.perf_counter() - start)
        assert summary is not None
        native_latencies_ms.append(float(summary[5]))

    outer_mean_ms = statistics.fmean(outer_durations) * 1000.0
    outer_median_ms = statistics.median(outer_durations) * 1000.0
    native_mean_ms = statistics.fmean(native_latencies_ms)
    native_median_ms = statistics.median(native_latencies_ms)

    print(  # noqa: T201
        f"[scheduler_sim_native_critical_path] runs={_TIMING_RUNS} "
        f"outer_mean={outer_mean_ms:.6f}ms outer_median={outer_median_ms:.6f}ms "
        f"native_mean={native_mean_ms:.6f}ms native_median={native_median_ms:.6f}ms"
    )

    assert len(outer_durations) == _TIMING_RUNS
    assert len(native_latencies_ms) == _TIMING_RUNS
    assert outer_mean_ms > 0.0
    assert native_mean_ms > 0.0
