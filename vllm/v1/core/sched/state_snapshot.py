# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight scheduler snapshot structures for background simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
# from kv_cache_manager import KVCacheBlocks

@dataclass(slots=True)
class RequestStateSnapshot:
    request_id: str
    status: str
    priority: int
    arrival_time: float
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_target_tokens: int
    num_prompt_processed_tokens: int
    num_output_processed_tokens: int
    # num_tokens_with_spec: int # speculative decoding
    # num_output_placeholders: int # async scheduling
    # spec_token_count: int # speculative decoding
    max_tokens: int
    num_preemptions: int
    num_cached_tokens: int # kv
    # num_pending_tokens: int
    is_long_prompt: bool
    # has_encoder_inputs: bool
    kv_block_counts: Tuple[int, ...] # kv block counts per kv cache group for a given request


@dataclass(slots=True)
class SchedulerConfigSnapshot:
    max_num_batched_tokens: int # equivalent to max_num_scheduled_tokens
    max_num_seqs: int # equivalent to max_num_running_reqs
    max_model_len: int
    # max_num_partial_prefills: int # not used
    # max_long_partial_prefills: int # not used
    long_prefill_token_threshold: int
    chunked_prefill_enabled: bool
    # num_lookahead_slots: int # speculative decoding
    # num_lookahead_tokens: int # speculative decoding
    policy: str

@dataclass(slots=True)
class SchedulerKVCacheSnapshot:
    num_gpu_blocks: int # total pool size
    block_size: int # size of each block
    kv_cache_groups: list[KVCacheGroupSpec]
    kv_cache_usage: float
    kv_cache_total_blocks: int
    kv_cache_free_blocks: int

@dataclass(slots=True)
class SchedulerParallelSnapshot:
    decode_context_parallel_size: int

@dataclass(slots=True)
class SchedulerStateSnapshot:
    version: int
    created_at: float
    num_running: int
    num_waiting: int
    running_request_ids: List[str]
    waiting_request_ids: List[str]
    requests: Dict[str, RequestStateSnapshot]
    config: SchedulerConfigSnapshot
    kv_cache_config: SchedulerKVCacheSnapshot
    parallel_config: SchedulerParallelSnapshot
    resident_set_size: int = 0
    waiting_set_size: int = 0
    prefill_backlog_running_tokens: int = 0
    prefill_backlog_running_sq_sum_tokens: int = 0
    prefill_backlog_waiting_tokens: int = 0
    prefill_backlog_waiting_sq_sum_tokens: int = 0
    prefill_backlog_total_tokens: int = 0
    prefill_backlog_total_sq_sum_tokens: int = 0
    decode_backlog_total_tokens: int = 0
    running_context_length_sum_snapshot: int = 0
    running_context_length_sq_sum_snapshot: int = 0
    build_latency_ms: float = 0.0
