# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import math
import time
from bisect import bisect_left, insort
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Callable, Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    compute_encoder_budget,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.state_snapshot import (
    RequestStateSnapshot,
    SchedulerKVCacheSnapshot,
    SchedulerParallelSnapshot,
    SchedulerConfigSnapshot,
    SchedulerStateSnapshot,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
        self._snapshot_consumer: Optional[
            Callable[[SchedulerStateSnapshot], None]
        ] = None
        self._snapshot_consumer_interval_s: Optional[float] = None
        self._next_snapshot_emit_time: float = 0.0
        self._snapshot_version = 0
        # Online positive residuals of (actual_decode_tokens - predicted_decode_tokens),
        # used to inflate decode targets under uncertain long-tail workloads.
        self._decode_tail_residual_samples: deque[float] = deque()
        self._decode_tail_residual_sorted: list[float] = []
        self._decode_tail_residual_max_samples = 4096
        self._decode_tail_min_samples = 200
        self._decode_tail_alpha = 1.25

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors"
            )
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER
            )

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): The scheduler’s block_size must be multiplied
        # by dcp_world_size, since block hashes are computed on the
        # original full token sequence at a granularity of
        # original_block_size × dcp_world_size.
        if self.dcp_world_size > 1:
            self.block_size *= self.dcp_world_size

        self._config_snapshot_template = SchedulerConfigSnapshot(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.scheduler_config.max_model_len,
            long_prefill_token_threshold=self.scheduler_config.long_prefill_token_threshold,
            chunked_prefill_enabled=self.scheduler_config.chunked_prefill_enabled,
            policy=self.scheduler_config.policy,
        )
        self._parallel_snapshot_template = SchedulerParallelSnapshot(
            decode_context_parallel_size=self.dcp_world_size
        )

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            )
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

    @staticmethod
    def _split_prefill_decode_tokens(
        prompt_len: int, start_pos: int, num_tokens: int
    ) -> tuple[int, int]:
        """Return (prefill_tokens, decode_tokens) for scheduled range."""
        if num_tokens <= 0:
            return 0, 0
        prefill_end = min(prompt_len, start_pos + num_tokens)
        prefill_start = min(prompt_len, start_pos)
        prefill = max(0, prefill_end - prefill_start)
        decode = num_tokens - prefill
        return prefill, decode

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        batch_prefill_tokens = 0
        batch_prefill_tokens_sq = 0
        batch_decode_tokens = 0
        batch_decode_tokens_sq = 0
        batch_num_active_seqs = 0
        batch_total_context_len = 0
        batch_sq_sum_context_len = 0
        batch_max_context_len = 0
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )

                if new_blocks is not None:
                    # The request can be scheduled.
                    break

                # The request cannot be scheduled.
                # Preempt the lowest-priority request.
                if self.policy == SchedulingPolicy.PRIORITY:
                    preempted_req = max(
                        self.running,
                        key=lambda r: (r.priority, r.arrival_time),
                    )
                    self.running.remove(preempted_req)
                    if preempted_req in scheduled_running_reqs:
                        scheduled_running_reqs.remove(preempted_req)
                else:
                    preempted_req = self.running.pop()

                self.kv_cache_manager.free(preempted_req)
                self.encoder_cache_manager.free(preempted_req)
                preempted_req.status = RequestStatus.PREEMPTED
                preempted_req.num_computed_tokens = 0
                preempted_req.num_preemptions += 1
                if self.log_stats:
                    preempted_req.record_event(
                        EngineCoreEventType.PREEMPTED, scheduled_timestamp
                    )

                self.waiting.prepend_request(preempted_req)
                preempted_reqs.append(preempted_req)
                if preempted_req == request:
                    # No more request to preempt. Cannot schedule this request.
                    break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            prefill, decode = self._split_prefill_decode_tokens(
                request.num_prompt_tokens, request.num_computed_tokens, num_new_tokens
            )
            batch_prefill_tokens += prefill
            batch_prefill_tokens_sq += prefill * prefill
            batch_decode_tokens += decode
            batch_decode_tokens_sq += decode * decode
            ctx_len = len(request.all_token_ids)
            batch_num_active_seqs += 1
            batch_total_context_len += ctx_len
            batch_sq_sum_context_len += ctx_len * ctx_len
            batch_max_context_len = max(batch_max_context_len, ctx_len)
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens + request.num_computed_tokens - request.num_tokens
                )
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests (only if no preemptions occurred).
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # (BODEN): I don't think there's a good way to include this in our simulator
                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # (BODEN): can we ignore structured outputs / grammar constraining?
                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if num_external_computed_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list()
                    )
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (
                        0
                        < self.scheduler_config.long_prefill_token_threshold
                        < num_new_tokens
                    ):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold
                        )

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.chunked_prefill_enabled
                        and num_new_tokens > token_budget
                    ):
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    # basically chunking where chunk size = remaining token_budget
                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    num_encoder_tokens = (
                        self.scheduler_config.max_num_encoder_input_tokens
                    )
                else:
                    num_encoder_tokens = 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                prefill, decode = self._split_prefill_decode_tokens(
                    request.num_prompt_tokens, num_computed_tokens, num_new_tokens
                )
                batch_prefill_tokens += prefill
                batch_prefill_tokens_sq += prefill * prefill
                batch_decode_tokens += decode
                batch_decode_tokens_sq += decode * decode
                ctx_len = len(request.all_token_ids)
                batch_num_active_seqs += 1
                batch_total_context_len += ctx_len
                batch_sq_sum_context_len += ctx_len * ctx_len
                batch_max_context_len = max(batch_max_context_len, ctx_len)
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)
                )
            )

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids()
            )
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        scheduled_requests = (
            scheduled_new_reqs + scheduled_running_reqs + scheduled_resumed_reqs
        )
        structured_output_request_ids, grammar_bitmask = self.get_grammar_bitmask(
            scheduled_requests, scheduled_spec_decode_tokens
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            batch_prefill_tokens=batch_prefill_tokens,
            batch_prefill_tokens_sq=batch_prefill_tokens_sq,
            batch_decode_tokens=batch_decode_tokens,
            batch_decode_tokens_sq=batch_decode_tokens_sq,
            batch_num_active_seqs=batch_num_active_seqs,
            batch_total_context_len=batch_total_context_len,
            batch_sq_sum_context_len=batch_sq_sum_context_len,
            batch_avg_context_len=(
                batch_total_context_len / batch_num_active_seqs
                if batch_num_active_seqs
                else 0.0
            ),
            batch_max_context_len=batch_max_context_len,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        self._emit_snapshot()
        return scheduler_output

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = num_scheduled_tokens[req_id] - len(
                spec_decode_tokens.get(req_id, ())
            )
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(len(req.output_token_ids))
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_tokens_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if request.mm_features[i].identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_tokens_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            num_tokens_to_schedule += num_encoder_tokens
            encoder_compute_budget -= num_encoder_tokens
            mm_hashes_to_schedule.add(request.mm_features[i].identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
        )

    def get_grammar_bitmask(
        self,
        requests: list[Request],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ):
        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to its index in the batch.
        # This will help us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}
        for i, req in enumerate(requests):
            if req.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[req.request_id] = i

        if not structured_output_request_ids:
            bitmask = None
        else:
            bitmask = self.structured_output_manager.grammar_bitmask(
                self.requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )
        return structured_output_request_ids, bitmask

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
        batch_schedule_time_s: float | None = None,
        batch_execute_time_s: float | None = None,
        batch_interval_s: float | None = None,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )

        batch_prefill_tokens = scheduler_output.batch_prefill_tokens
        batch_prefill_tokens_sq = scheduler_output.batch_prefill_tokens_sq
        batch_decode_tokens = scheduler_output.batch_decode_tokens
        batch_decode_tokens_sq = scheduler_output.batch_decode_tokens_sq
        batch_total_tokens = scheduler_output.total_num_scheduled_tokens
        batch_num_active_seqs = scheduler_output.batch_num_active_seqs
        batch_total_context_len = scheduler_output.batch_total_context_len
        batch_sq_sum_context_len = scheduler_output.batch_sq_sum_context_len
        batch_avg_context_len = scheduler_output.batch_avg_context_len
        batch_max_context_len = scheduler_output.batch_max_context_len

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                request.num_computed_tokens -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len, pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids
                )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats,
                kv_connector_stats,
                batch_prefill_tokens=batch_prefill_tokens,
                batch_prefill_tokens_sq=batch_prefill_tokens_sq,
                batch_decode_tokens=batch_decode_tokens,
                batch_decode_tokens_sq=batch_decode_tokens_sq,
                batch_total_tokens=batch_total_tokens,
                batch_schedule_time_s=batch_schedule_time_s,
                batch_execute_time_s=batch_execute_time_s,
                batch_interval_s=batch_interval_s,
                batch_num_active_seqs=batch_num_active_seqs,
                batch_total_context_len=batch_total_context_len,
                batch_sq_sum_context_len=batch_sq_sum_context_len,
                batch_avg_context_len=batch_avg_context_len,
                batch_max_context_len=batch_max_context_len,
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if not spec_token_ids:
                # NOTE(woosuk): request.spec_token_ids should be updated.
                request.spec_token_ids.clear()
            elif self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids
                )
            else:
                request.spec_token_ids = spec_token_ids

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def set_snapshot_consumer(
        self,
        consumer: Optional[Callable[[SchedulerStateSnapshot], None]],
        interval_s: Optional[float] = None,
    ) -> None:
        """Registers a consumer that receives scheduler state snapshots."""
        self._snapshot_consumer = consumer
        if consumer is None:
            self._snapshot_consumer_interval_s = None
            self._next_snapshot_emit_time = 0.0
        else:
            self._snapshot_consumer_interval_s = interval_s
            self._next_snapshot_emit_time = time.monotonic()

    def _emit_snapshot(self) -> None:
        consumer = self._snapshot_consumer
        if consumer is None:
            return
        interval = self._snapshot_consumer_interval_s
        current_time = time.monotonic()
        if interval is not None and current_time < self._next_snapshot_emit_time:
            return
        snapshot_start = time.perf_counter()
        snapshot = self._build_state_snapshot()
        snapshot_elapsed_ms = (time.perf_counter() - snapshot_start) * 1000.0
        snapshot.build_latency_ms = snapshot_elapsed_ms
        logger.debug(
            "Built scheduler snapshot v%04d in %.3f ms",
            snapshot.version,
            snapshot_elapsed_ms,
        )
        consumer(snapshot)
        if interval is not None:
            self._next_snapshot_emit_time = time.monotonic() + interval

    def _append_decode_tail_residual(self, residual_tokens: float) -> None:
        if not math.isfinite(residual_tokens) or residual_tokens < 0:
            return

        max_samples = self._decode_tail_residual_max_samples
        if len(self._decode_tail_residual_samples) >= max_samples:
            oldest = self._decode_tail_residual_samples.popleft()
            idx = bisect_left(self._decode_tail_residual_sorted, oldest)
            if (
                idx < len(self._decode_tail_residual_sorted)
                and self._decode_tail_residual_sorted[idx] == oldest
            ):
                self._decode_tail_residual_sorted.pop(idx)

        self._decode_tail_residual_samples.append(residual_tokens)
        insort(self._decode_tail_residual_sorted, residual_tokens)

    def _record_decode_tail_residual(self, request: Request) -> None:
        if request.status not in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ):
            return

        prediction = request.output_length_prediction
        if prediction is None:
            return

        predicted_total = int(math.ceil(prediction.mean_tokens))
        if predicted_total <= 0:
            return

        residual = float(max(0, request.num_output_tokens - predicted_total))
        self._append_decode_tail_residual(residual)

    def _decode_residual_quantile(self, q: float) -> float:
        n = len(self._decode_tail_residual_sorted)
        if n < self._decode_tail_min_samples:
            return 0.0
        q = min(max(float(q), 0.0), 1.0)
        idx = int(math.ceil(q * n)) - 1
        idx = min(max(idx, 0), n - 1)
        return float(self._decode_tail_residual_sorted[idx])

    @staticmethod
    def _decode_floor_tokens(
        kv_usage: float,
        num_waiting: int,
        num_residuals: int,
        min_samples: int,
    ) -> int:
        warm = num_residuals >= min_samples

        if kv_usage > 0.95 and num_waiting > 0:
            return 256 if warm else 512
        if kv_usage > 0.85:
            return 128 if warm else 256
        if kv_usage > 0.70:
            return 64 if warm else 128
        return 16 if warm else 32

    @staticmethod
    def _decode_tail_quantile_for_congestion(
        kv_usage: float,
        num_waiting: int,
        num_residuals: int,
        min_samples: int,
    ) -> float:
        warm = num_residuals >= min_samples

        if kv_usage > 0.95 and num_waiting > 0:
            return 0.97 if warm else 0.95
        if kv_usage > 0.85:
            return 0.94 if warm else 0.92
        return 0.90 if warm else 0.88

    def _compute_adaptive_decode_reserve(
        self,
        *,
        predicted_target: int,
        num_output_processed_tokens: int,
        snapshot_base_reserve_tokens: float,
    ) -> int:
        base_reserve = max(0.0, float(snapshot_base_reserve_tokens))
        overshoot = max(
            0.0, float(num_output_processed_tokens) - float(predicted_target)
        )
        reserve = max(
            base_reserve,
            float(self._decode_tail_alpha) * overshoot,
        )
        return max(1, int(math.ceil(reserve)))

    def _snapshot_decode_reserve_base_tokens(
        self,
        *,
        kv_usage: float,
        num_waiting: int,
    ) -> float:
        num_residuals = len(self._decode_tail_residual_samples)
        min_samples = self._decode_tail_min_samples
        floor_tokens = float(
            self._decode_floor_tokens(
                kv_usage=kv_usage,
                num_waiting=num_waiting,
                num_residuals=num_residuals,
                min_samples=min_samples,
            )
        )
        tail_quantile = self._decode_tail_quantile_for_congestion(
            kv_usage=kv_usage,
            num_waiting=num_waiting,
            num_residuals=num_residuals,
            min_samples=min_samples,
        )
        tail_tokens = self._decode_residual_quantile(tail_quantile)
        return max(0.0, floor_tokens, tail_tokens)

    def _build_state_snapshot(self) -> SchedulerStateSnapshot:
        self._snapshot_version += 1
        created_at = time.monotonic()
        requests: dict[str, RequestStateSnapshot] = {}
        long_threshold = self.scheduler_config.long_prefill_token_threshold

        if self.scheduler_config.policy == "priority":
            # Will need to manage a heapq for waiting queue
            raise NotImplementedError("Priority scheduling not implemented yet")
        if self.scheduler_config.policy != "fcfs":
            raise NotImplementedError(
                f"Unsupported scheduling policy: {self.scheduler_config.policy}"
            )

        running_request_ids = [req.request_id for req in self.running]
        waiting_request_ids = [req.request_id for req in self.waiting]
        running_id_set = set(running_request_ids)
        waiting_id_set = set(waiting_request_ids)
        kv_usage = float(self.kv_cache_manager.usage)
        num_waiting = len(self.waiting)
        snapshot_base_reserve_tokens = self._snapshot_decode_reserve_base_tokens(
            kv_usage=kv_usage,
            num_waiting=num_waiting,
        )

        prefill_backlog_running = 0
        prefill_backlog_waiting = 0
        decode_backlog_total = 0
        running_context_length_sum_snapshot = 0

        max_arrival_time = -1
        for request_id, request in self.requests.items():
            kv_blocks = self.kv_cache_manager.get_blocks(request_id)
            kv_block_counts = tuple(len(block_group) for block_group in kv_blocks.blocks)

            num_prompt_tokens = request.num_prompt_tokens
            num_computed_tokens = request.num_computed_tokens
            num_prompt_processed_tokens = min(num_computed_tokens, num_prompt_tokens)
            num_output_processed_tokens = request.num_output_tokens

            # might need to artificially ensure that num_computed_tokens and num_tokens
            # match expectations (we calculate num_tokens as num_prompt_tokens + decode_processed)

            predicted_target = None
            if request.output_length_prediction is not None:
                predicted_target = int(
                    math.ceil(request.output_length_prediction.mean_tokens)
                )
            reserve_anchor = (
                predicted_target
                if predicted_target is not None and predicted_target > 0
                else num_output_processed_tokens
            )
            adaptive_reserve = self._compute_adaptive_decode_reserve(
                predicted_target=int(reserve_anchor),
                num_output_processed_tokens=int(num_output_processed_tokens),
                snapshot_base_reserve_tokens=float(snapshot_base_reserve_tokens),
            )
            if predicted_target is not None and predicted_target > 0:
                num_output_target_tokens = predicted_target
            else:
                num_output_target_tokens = (
                    max(num_output_processed_tokens, request.max_tokens // 2) + 1
                )
            num_output_target_tokens = max(
                num_output_target_tokens,
                num_output_processed_tokens + adaptive_reserve,
            )
            decode_budget = max(self.max_model_len - num_prompt_tokens, 1)
            num_output_target_tokens = min(
                num_output_target_tokens,
                max(request.max_tokens, num_output_processed_tokens + 1),
                decode_budget,
            )

            assert num_output_target_tokens > num_output_processed_tokens, "target number of tokens must be greater than processed number of tokens"
            assert num_output_target_tokens > 0, "target number of tokens must be positive"
            assert request.num_tokens == num_prompt_tokens + num_output_processed_tokens, "total number of tokens so far mismatch"

            prefill_remaining = max(num_prompt_tokens - num_prompt_processed_tokens, 0)
            decode_remaining = max(num_output_target_tokens - num_output_processed_tokens, 0)
            if request_id in running_id_set:
                prefill_backlog_running += prefill_remaining
                decode_backlog_total += decode_remaining
                running_context_length_sum_snapshot += (
                    num_prompt_tokens + num_output_processed_tokens
                )
            elif request_id in waiting_id_set:
                prefill_backlog_waiting += prefill_remaining
                decode_backlog_total += decode_remaining

            requests[request_id] = RequestStateSnapshot(
                request_id=request_id,
                status=request.status.name,
                priority=request.priority,
                arrival_time=request.arrival_time,
                num_prompt_tokens=num_prompt_tokens,
                num_computed_tokens=num_computed_tokens,
                num_output_target_tokens=num_output_target_tokens,
                num_prompt_processed_tokens=num_prompt_processed_tokens,
                num_output_processed_tokens=num_output_processed_tokens,
                max_tokens=request.max_tokens,
                num_preemptions=request.num_preemptions,
                num_cached_tokens=request.num_cached_tokens,
                is_long_prompt=long_threshold > 0 and num_prompt_tokens >= long_threshold,
                kv_block_counts=kv_block_counts,
            )
            max_arrival_time = max(max_arrival_time, request.arrival_time)

        num_prompt_tokens = 8192
        dummy_request_id = "__DUMMY__"
        requests[dummy_request_id] = RequestStateSnapshot(
            request_id=dummy_request_id,
            status="WAITING",
            priority=0,
            arrival_time=max_arrival_time + 1,
            num_prompt_tokens=num_prompt_tokens,
            num_computed_tokens=0,
            num_output_target_tokens=1024,
            num_prompt_processed_tokens=0,
            num_output_processed_tokens=0,
            max_tokens=2048,
            num_preemptions=0,
            num_cached_tokens=-1,
            is_long_prompt=long_threshold > 0 and num_prompt_tokens >= long_threshold,
            kv_block_counts=tuple(0 for _ in self.kv_cache_config.kv_cache_groups)
        )

        waiting_request_ids = waiting_request_ids + [dummy_request_id]
        prefill_backlog_total = prefill_backlog_running + prefill_backlog_waiting

        block_pool = self.kv_cache_manager.block_pool
        kv_cache_config_snapshot = SchedulerKVCacheSnapshot(
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            # Prefix caching can be disabled, in which case KVCacheManager keeps
            # block_size unset. Snapshots must still provide an integer.
            block_size=self.block_size,
            kv_cache_groups=self.kv_cache_config.kv_cache_groups,
            kv_cache_usage=self.kv_cache_manager.usage,
            kv_cache_total_blocks=block_pool.num_gpu_blocks,
            kv_cache_free_blocks=block_pool.get_num_free_blocks(),
        )
        parallel_config_snapshot = replace(self._parallel_snapshot_template)
        scheduler_config_snapshot = replace(self._config_snapshot_template)

        return SchedulerStateSnapshot(
            version=self._snapshot_version,
            created_at=created_at,
            num_running=len(self.running),
            num_waiting=len(self.waiting) + 1,
            running_request_ids=running_request_ids,
            waiting_request_ids=waiting_request_ids,
            requests=requests,
            config=scheduler_config_snapshot,
            kv_cache_config=kv_cache_config_snapshot,
            parallel_config=parallel_config_snapshot,
            resident_set_size=len(running_id_set),
            waiting_set_size=len(waiting_id_set),
            prefill_backlog_running_tokens=int(prefill_backlog_running),
            prefill_backlog_waiting_tokens=int(prefill_backlog_waiting),
            prefill_backlog_total_tokens=int(prefill_backlog_total),
            decode_backlog_total_tokens=int(decode_backlog_total),
            running_context_length_sum_snapshot=int(running_context_length_sum_snapshot),
        )

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        self._record_decode_tail_residual(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        batch_prefill_tokens: int = 0,
        batch_prefill_tokens_sq: int = 0,
        batch_decode_tokens: int = 0,
        batch_decode_tokens_sq: int = 0,
        batch_total_tokens: int = 0,
        batch_schedule_time_s: float | None = None,
        batch_execute_time_s: float | None = None,
        batch_interval_s: float | None = None,
        batch_num_active_seqs: int = 0,
        batch_total_context_len: int = 0,
        batch_sq_sum_context_len: int = 0,
        batch_avg_context_len: float = 0.0,
        batch_max_context_len: int = 0,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        total_queries = (
            prefix_cache_stats.queries + prefix_cache_stats.preempted_queries
        )
        total_hits = prefix_cache_stats.hits + prefix_cache_stats.preempted_hits
        hit_rate = (total_hits / total_queries) if total_queries else 0.0
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            batch_total_tokens=batch_total_tokens,
            batch_prefill_tokens=batch_prefill_tokens,
            batch_prefill_tokens_sq=batch_prefill_tokens_sq,
            batch_decode_tokens=batch_decode_tokens,
            batch_decode_tokens_sq=batch_decode_tokens_sq,
            batch_schedule_time_s=batch_schedule_time_s or 0.0,
            batch_execute_time_s=batch_execute_time_s or 0.0,
            batch_interval_s=batch_interval_s or 0.0,
            batch_num_active_seqs=batch_num_active_seqs,
            batch_total_context_len=batch_total_context_len,
            batch_sq_sum_context_len=batch_sq_sum_context_len,
            batch_avg_context_len=batch_avg_context_len,
            batch_max_context_len=batch_max_context_len,
            kv_cache_usage=self.kv_cache_manager.usage,
            kv_cache_hit_rate=hit_rate,
            kv_cache_queries=total_queries,
            kv_cache_hits=total_hits,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted for req in self.running),
            kv_connector_stats=kv_connector_stats.data if kv_connector_stats else None,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> SpecDecodingStats | None:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
            num_computed_tokens = len(block_ids) * self.block_size
            # Handle the case where num request tokens less than one block.
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # Update the request state for scheduling.
            request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            if req_id not in self.requests:
                logger.warning(
                    "Got finished sending KV transfer for request %s,"
                    "but the request is already freed.",
                    req_id,
                )
            else:
                self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self, requests: Iterable[Request], invalid_block_ids: set[int]
    ) -> tuple[set[str], int]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests (for observability).
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # Async loading. If num_computed_tokens is set it implies we
                # already processed some block failures for it in a prior step
                req_num_computed_tokens = (
                    request.num_computed_tokens
                    if req_id in self.failed_recving_kv_req_ids
                    else len(req_block_ids) * self.block_size
                )
            else:
                # Sync loading. num_computed_tokens includes new tokens
                req_num_computed_tokens = request.num_cached_tokens

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                total_affected_tokens += (
                    req_num_computed_tokens - request.num_computed_tokens
                )

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens

                affected_req_ids.add(request.request_id)

        return (affected_req_ids, total_affected_tokens)

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        total_requests_to_reschedule = 0
        total_tokens_to_reschedule = 0

        # --- Handle async KV loads (WAITING_FOR_REMOTE_KVS) ---
        async_load_reqs = (
            req
            for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_affected_req_ids, num_tokens_to_reschedule = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids
            )
        )

        total_requests_to_reschedule += len(async_affected_req_ids)
        total_tokens_to_reschedule += num_tokens_to_reschedule

        # Mark requests with async KV load failures; they will be rescheduled
        # once loading completes
        self.failed_recving_kv_req_ids |= async_affected_req_ids

        # --- Handle sync KV loads (running requests) ---
        sync_affected_req_ids, num_tokens_to_reschedule = (
            self._update_requests_with_invalid_blocks(self.running, invalid_block_ids)
        )

        total_requests_to_reschedule += len(sync_affected_req_ids)
        total_tokens_to_reschedule += num_tokens_to_reschedule

        if total_requests_to_reschedule:
            logger.warning(
                "Recovered from KV load failure: "
                "%d request(s) rescheduled (%d tokens affected).",
                total_requests_to_reschedule,
                total_tokens_to_reschedule,
            )

        # Return the IDs of affected running requests to skip in
        # update_from_output.
        return sync_affected_req_ids
