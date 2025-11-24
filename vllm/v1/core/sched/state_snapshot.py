# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight scheduler snapshot structures for background simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(slots=True)
class RequestStateSnapshot:
    request_id: str
    status: str
    priority: int
    arrival_time: float
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    num_prompt_processed_tokens: int
    num_output_processed_tokens: int
    num_tokens_with_spec: int
    num_output_placeholders: int
    spec_token_count: int
    max_tokens: int
    # num_preemptions: int
    num_cached_tokens: int
    # num_pending_tokens: int
    is_long_prompt: bool
    # has_encoder_inputs: bool
    kv_block_counts: Tuple[int, ...]


@dataclass(slots=True)
class SchedulerConfigSnapshot:
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int
    max_num_partial_prefills: int
    max_long_partial_prefills: int
    long_prefill_token_threshold: int
    chunked_prefill_enabled: bool
    num_lookahead_slots: int
    num_lookahead_tokens: int
    policy: str
    

@dataclass(slots=True)
class SchedulerStateSnapshot:
    version: int
    created_at: float
    num_running: int
    num_waiting: int
    kv_cache_usage: float
    kv_cache_total_blocks: int
    kv_cache_free_blocks: int
    kv_cache_block_size: Optional[int]
    running_request_ids: List[str]
    waiting_request_ids: List[str]
    requests: List[RequestStateSnapshot]
    config: SchedulerConfigSnapshot
    build_latency_ms: float = 0.0
