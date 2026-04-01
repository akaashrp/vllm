# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
import time
from multiprocessing import shared_memory

from vllm.v1.core.sched.snapshot_serialization import (
    decode_scheduler_state_snapshot,
    encode_scheduler_state_snapshot,
)
from vllm.v1.core.sched.state_snapshot import (
    RequestStateSnapshot,
    SchedulerConfigSnapshot,
    SchedulerKVCacheSnapshot,
    SchedulerParallelSnapshot,
    SchedulerStateSnapshot,
)
from vllm.v1.engine.snapshot_shm import (
    SNAPSHOT_SHM_HEADER_SIZE,
    SnapshotShmPublisher,
    read_snapshot_shm_header_once,
    read_snapshot_shm_once,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec


def _make_request(
    request_id: str,
    *,
    status: str,
    arrival_time: float,
    num_prompt_tokens: int,
) -> RequestStateSnapshot:
    return RequestStateSnapshot(
        request_id=request_id,
        status=status,
        priority=0,
        arrival_time=arrival_time,
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=0,
        num_output_target_tokens=max(num_prompt_tokens, 1),
        num_prompt_processed_tokens=0,
        num_output_processed_tokens=0,
        max_tokens=max(num_prompt_tokens, 1),
        num_preemptions=0,
        num_cached_tokens=0,
        is_long_prompt=False,
        kv_block_counts=(0,),
    )


def _build_snapshot(
    *,
    version: int,
    created_at: float,
    prefill_backlog_total_tokens: int,
    build_latency_ms: float = 0.0,
) -> SchedulerStateSnapshot:
    waiting_request = _make_request(
        f"wait-{version}",
        status="WAITING",
        arrival_time=created_at,
        num_prompt_tokens=max(prefill_backlog_total_tokens, 1),
    )
    return SchedulerStateSnapshot(
        version=version,
        created_at=created_at,
        num_running=0,
        num_waiting=1,
        running_request_ids=[],
        waiting_request_ids=[waiting_request.request_id],
        requests={waiting_request.request_id: waiting_request},
        config=SchedulerConfigSnapshot(
            max_num_batched_tokens=128,
            max_num_seqs=4,
            max_model_len=2048,
            long_prefill_token_threshold=256,
            chunked_prefill_enabled=True,
            policy="fcfs",
        ),
        kv_cache_config=SchedulerKVCacheSnapshot(
            num_gpu_blocks=1024,
            block_size=16,
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["layer0"],
                    kv_cache_spec=KVCacheSpec(block_size=16),
                )
            ],
            kv_cache_usage=0.1,
            kv_cache_total_blocks=2048,
            kv_cache_free_blocks=1800,
        ),
        parallel_config=SchedulerParallelSnapshot(
            decode_context_parallel_size=1,
        ),
        waiting_set_size=1,
        prefill_backlog_waiting_tokens=prefill_backlog_total_tokens,
        prefill_backlog_total_tokens=prefill_backlog_total_tokens,
        build_latency_ms=build_latency_ms,
    )


def _publisher_name() -> str:
    return f"vllm_snapshot_test_{os.getpid()}_{time.time_ns()}"


def test_snapshot_shm_publish_and_read_round_trip():
    publisher = SnapshotShmPublisher(
        name=_publisher_name(),
        size_bytes=1024 * 1024,
        simulation_intercept=1.0,
        simulation_prefill_coeff=0.1,
        simulation_prefill_sq_coeff=0.01,
        simulation_decode_coeff=0.2,
        simulation_sum_coeff=0.3,
        simulation_sum_sq_coeff=0.02,
    )
    reader = None
    try:
        snapshot = _build_snapshot(
            version=7,
            created_at=3.5,
            prefill_backlog_total_tokens=123,
            build_latency_ms=4.25,
        )
        payload = encode_scheduler_state_snapshot(snapshot)
        assert publisher.publish(snapshot, payload) is True

        reader = shared_memory.SharedMemory(name=publisher.name)
        read_result = read_snapshot_shm_once(reader)
        assert read_result is not None
        header, read_payload = read_result
        decoded = decode_scheduler_state_snapshot(read_payload)

        assert header.snapshot_version == snapshot.version
        assert header.created_at == snapshot.created_at
        assert header.prefill_backlog_total_tokens == 123
        assert header.build_latency_ms == 4.25
        assert decoded["version"] == snapshot.version
        assert decoded["prefill_backlog_total_tokens"] == 123

        header_only = read_snapshot_shm_header_once(reader)
        assert header_only is not None
        assert header_only.snapshot_version == snapshot.version
        assert header_only.build_latency_ms == 4.25
    finally:
        if reader is not None:
            reader.close()
        publisher.close()


def test_snapshot_shm_version_visibility_is_monotonic():
    publisher = SnapshotShmPublisher(
        name=_publisher_name(),
        size_bytes=1024 * 1024,
        simulation_intercept=1.0,
        simulation_prefill_coeff=0.1,
        simulation_prefill_sq_coeff=0.0,
        simulation_decode_coeff=0.2,
        simulation_sum_coeff=0.3,
        simulation_sum_sq_coeff=0.0,
    )
    reader = None
    try:
        reader = shared_memory.SharedMemory(name=publisher.name)
        last_version = 0
        for version in range(1, 6):
            snapshot = _build_snapshot(
                version=version,
                created_at=float(version),
                prefill_backlog_total_tokens=version * 10,
            )
            payload = encode_scheduler_state_snapshot(snapshot)
            assert publisher.publish(snapshot, payload) is True

            read_result = read_snapshot_shm_once(reader)
            assert read_result is not None
            header, read_payload = read_result
            decoded = decode_scheduler_state_snapshot(read_payload)

            assert header.snapshot_version >= last_version
            assert decoded["version"] == header.snapshot_version
            last_version = header.snapshot_version
        assert last_version == 5
    finally:
        if reader is not None:
            reader.close()
        publisher.close()


def test_snapshot_shm_oversize_publish_retains_last_valid_snapshot():
    snapshot = _build_snapshot(
        version=11,
        created_at=11.0,
        prefill_backlog_total_tokens=64,
    )
    payload = encode_scheduler_state_snapshot(snapshot)
    publisher = SnapshotShmPublisher(
        name=_publisher_name(),
        size_bytes=SNAPSHOT_SHM_HEADER_SIZE + len(payload) + 16,
        simulation_intercept=1.0,
        simulation_prefill_coeff=0.1,
        simulation_prefill_sq_coeff=0.0,
        simulation_decode_coeff=0.2,
        simulation_sum_coeff=0.3,
        simulation_sum_sq_coeff=0.0,
    )
    reader = None
    try:
        assert publisher.publish(snapshot, payload) is True

        oversize_payload = b"x" * (len(payload) + 32)
        oversize_snapshot = _build_snapshot(
            version=12,
            created_at=12.0,
            prefill_backlog_total_tokens=128,
        )
        assert publisher.publish(oversize_snapshot, oversize_payload) is False

        reader = shared_memory.SharedMemory(name=publisher.name)
        read_result = read_snapshot_shm_once(reader)
        assert read_result is not None
        header, read_payload = read_result
        decoded = decode_scheduler_state_snapshot(read_payload)

        assert header.snapshot_version == 11
        assert header.prefill_backlog_total_tokens == 64
        assert decoded["version"] == 11
    finally:
        if reader is not None:
            reader.close()
        publisher.close()


def test_snapshot_shm_reads_consistent_payloads_during_rapid_updates():
    publisher = SnapshotShmPublisher(
        name=_publisher_name(),
        size_bytes=1024 * 1024,
        simulation_intercept=1.0,
        simulation_prefill_coeff=0.1,
        simulation_prefill_sq_coeff=0.0,
        simulation_decode_coeff=0.2,
        simulation_sum_coeff=0.3,
        simulation_sum_sq_coeff=0.0,
    )
    reader = None
    try:
        reader = shared_memory.SharedMemory(name=publisher.name)
        final_version = 25
        writer_done = threading.Event()

        def _writer() -> None:
            for version in range(1, final_version + 1):
                snapshot = _build_snapshot(
                    version=version,
                    created_at=float(version),
                    prefill_backlog_total_tokens=version * 7,
                )
                payload = encode_scheduler_state_snapshot(snapshot)
                publisher.publish(snapshot, payload)
                time.sleep(0.001)
            writer_done.set()

        thread = threading.Thread(target=_writer, daemon=True)
        thread.start()

        last_seen_version = 0
        deadline = time.time() + 5.0
        while time.time() < deadline and (
            not writer_done.is_set() or last_seen_version < final_version
        ):
            read_result = read_snapshot_shm_once(reader)
            if read_result is None:
                continue
            header, read_payload = read_result
            decoded = decode_scheduler_state_snapshot(read_payload)
            assert decoded["version"] == header.snapshot_version
            assert (
                decoded["prefill_backlog_total_tokens"]
                == header.prefill_backlog_total_tokens
            )
            last_seen_version = max(last_seen_version, header.snapshot_version)

        thread.join(timeout=1.0)
        assert last_seen_version == final_version
    finally:
        if reader is not None:
            reader.close()
        publisher.close()
