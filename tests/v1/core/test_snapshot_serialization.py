# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.sched.snapshot_serialization import (
    serialize_scheduler_state_snapshot,
)
from vllm.v1.core.sched.state_snapshot import (
    RequestStateSnapshot,
    SchedulerConfigSnapshot,
    SchedulerKVCacheSnapshot,
    SchedulerParallelSnapshot,
    SchedulerStateSnapshot,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec


def _build_snapshot() -> SchedulerStateSnapshot:
    req_snapshot = RequestStateSnapshot(
        request_id="req-1",
        status="RUNNING",
        priority=3,
        arrival_time=2.5,
        num_prompt_tokens=8,
        num_computed_tokens=4,
        num_output_target_tokens=16,
        num_prompt_processed_tokens=4,
        num_output_processed_tokens=2,
        max_tokens=64,
        num_preemptions=1,
        num_cached_tokens=4,
        is_long_prompt=True,
        kv_block_counts=(2, 1),
    )
    config_snapshot = SchedulerConfigSnapshot(
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=4096,
        long_prefill_token_threshold=512,
        chunked_prefill_enabled=False,
        policy="fcfs",
    )
    kv_cache_snapshot = SchedulerKVCacheSnapshot(
        num_gpu_blocks=4096,
        block_size=32,
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer_a", "layer_b"],
                kv_cache_spec=KVCacheSpec(block_size=32),
            )
        ],
        kv_cache_usage=0.75,
        kv_cache_total_blocks=8192,
        kv_cache_free_blocks=1024,
    )
    parallel_snapshot = SchedulerParallelSnapshot(
        decode_context_parallel_size=2
    )
    return SchedulerStateSnapshot(
        version=7,
        created_at=123.45,
        num_running=1,
        num_waiting=0,
        running_request_ids=["req-1"],
        waiting_request_ids=[],
        requests={"req-1": req_snapshot},
        config=config_snapshot,
        kv_cache_config=kv_cache_snapshot,
        parallel_config=parallel_snapshot,
    )


def test_serialize_scheduler_state_snapshot_handles_nested_dataclasses():
    snapshot = _build_snapshot()
    serialized = serialize_scheduler_state_snapshot(snapshot)
    
    expected = {
        "version": 7,
        "created_at": 123.45,
        "num_running": 1,
        "num_waiting": 0,
        "running_request_ids": ["req-1"],
        "waiting_request_ids": [],
        "requests": {
            "req-1": {
                "request_id": "req-1",
                "status": "RUNNING",
                "priority": 3,
                "arrival_time": 2.5,
                "num_prompt_tokens": 8,
                "num_computed_tokens": 4,
                "num_output_target_tokens": 16,
                "num_prompt_processed_tokens": 4,
                "num_output_processed_tokens": 2,
                "max_tokens": 64,
                "num_preemptions": 1,
                "num_cached_tokens": 4,
                "is_long_prompt": True,
                "kv_block_counts": [2, 1],
            }
        },
        "config": {
            "max_num_batched_tokens": 256,
            "max_num_seqs": 8,
            "max_model_len": 4096,
            "long_prefill_token_threshold": 512,
            "chunked_prefill_enabled": False,
            "policy": "fcfs",
        },
        "kv_cache_config": {
            "num_gpu_blocks": 4096,
            "block_size": 32,
            "kv_cache_groups": [
                {
                    "layer_names": ["layer_a", "layer_b"],
                    "kv_cache_spec": {"block_size": 32},
                }
            ],
            "kv_cache_usage": 0.75,
            "kv_cache_total_blocks": 8192,
            "kv_cache_free_blocks": 1024,
        },
        "parallel_config": {"decode_context_parallel_size": 2},
        "build_latency_ms": 0.0,
    }
    assert serialized == expected
