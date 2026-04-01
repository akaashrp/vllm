# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""POSIX shared-memory transport for scheduler snapshots."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from struct import Struct
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.sched.state_snapshot import SchedulerStateSnapshot

logger = init_logger(__name__)

SNAPSHOT_SHM_MAGIC = b"VLLMSHM1"
SNAPSHOT_SHM_VERSION = 1
SNAPSHOT_SHM_HEADER_STRUCT = Struct("<8sIIQQdQdddddddd")
SNAPSHOT_SHM_HEADER_SIZE = SNAPSHOT_SHM_HEADER_STRUCT.size
DEFAULT_SNAPSHOT_SHM_SIZE_BYTES = 8 * 1024 * 1024


@dataclass(slots=True)
class SnapshotShmHeader:
    sequence: int
    snapshot_version: int
    created_at: float
    payload_size: int
    prefill_backlog_total_tokens: float
    build_latency_ms: float
    simulation_intercept: float
    simulation_prefill_coeff: float
    simulation_prefill_sq_coeff: float
    simulation_decode_coeff: float
    simulation_sum_coeff: float
    simulation_sum_sq_coeff: float


def encode_snapshot_shm_header(header: SnapshotShmHeader) -> bytes:
    return SNAPSHOT_SHM_HEADER_STRUCT.pack(
        SNAPSHOT_SHM_MAGIC,
        SNAPSHOT_SHM_VERSION,
        SNAPSHOT_SHM_HEADER_SIZE,
        int(header.sequence),
        int(header.snapshot_version),
        float(header.created_at),
        int(header.payload_size),
        float(header.prefill_backlog_total_tokens),
        float(header.build_latency_ms),
        float(header.simulation_intercept),
        float(header.simulation_prefill_coeff),
        float(header.simulation_prefill_sq_coeff),
        float(header.simulation_decode_coeff),
        float(header.simulation_sum_coeff),
        float(header.simulation_sum_sq_coeff),
    )


def decode_snapshot_shm_header(data: bytes) -> SnapshotShmHeader:
    (
        magic,
        version,
        header_size,
        sequence,
        snapshot_version,
        created_at,
        payload_size,
        prefill_backlog_total_tokens,
        build_latency_ms,
        simulation_intercept,
        simulation_prefill_coeff,
        simulation_prefill_sq_coeff,
        simulation_decode_coeff,
        simulation_sum_coeff,
        simulation_sum_sq_coeff,
    ) = SNAPSHOT_SHM_HEADER_STRUCT.unpack(data)
    if magic != SNAPSHOT_SHM_MAGIC:
        raise ValueError("Snapshot SHM magic mismatch.")
    if version != SNAPSHOT_SHM_VERSION:
        raise ValueError(
            f"Unsupported snapshot SHM version {version}; "
            f"expected {SNAPSHOT_SHM_VERSION}."
        )
    if header_size != SNAPSHOT_SHM_HEADER_SIZE:
        raise ValueError(
            f"Snapshot SHM header_size mismatch {header_size}; "
            f"expected {SNAPSHOT_SHM_HEADER_SIZE}."
        )
    return SnapshotShmHeader(
        sequence=int(sequence),
        snapshot_version=int(snapshot_version),
        created_at=float(created_at),
        payload_size=int(payload_size),
        prefill_backlog_total_tokens=float(prefill_backlog_total_tokens),
        build_latency_ms=float(build_latency_ms),
        simulation_intercept=float(simulation_intercept),
        simulation_prefill_coeff=float(simulation_prefill_coeff),
        simulation_prefill_sq_coeff=float(simulation_prefill_sq_coeff),
        simulation_decode_coeff=float(simulation_decode_coeff),
        simulation_sum_coeff=float(simulation_sum_coeff),
        simulation_sum_sq_coeff=float(simulation_sum_sq_coeff),
    )


class SnapshotShmPublisher:
    """Publishes scheduler snapshots to a named POSIX shared-memory region."""

    def __init__(
        self,
        *,
        name: str,
        size_bytes: int,
        simulation_intercept: float,
        simulation_prefill_coeff: float,
        simulation_prefill_sq_coeff: float,
        simulation_decode_coeff: float,
        simulation_sum_coeff: float,
        simulation_sum_sq_coeff: float,
    ) -> None:
        if not name:
            raise ValueError("snapshot SHM name must be non-empty")
        if size_bytes <= SNAPSHOT_SHM_HEADER_SIZE:
            raise ValueError(
                "snapshot SHM size must be larger than the fixed header size"
            )

        self._name = str(name)
        self._size_bytes = int(size_bytes)
        self._coefficients = {
            "simulation_intercept": float(simulation_intercept),
            "simulation_prefill_coeff": float(simulation_prefill_coeff),
            "simulation_prefill_sq_coeff": float(simulation_prefill_sq_coeff),
            "simulation_decode_coeff": float(simulation_decode_coeff),
            "simulation_sum_coeff": float(simulation_sum_coeff),
            "simulation_sum_sq_coeff": float(simulation_sum_sq_coeff),
        }
        self._lock = threading.Lock()
        self._sequence = 0
        self._shm = self._create_shared_memory()
        self._write_header(
            sequence=self._sequence,
            snapshot_version=0,
            created_at=0.0,
            payload_size=0,
            prefill_backlog_total_tokens=0.0,
            build_latency_ms=0.0,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    def close(self) -> None:
        try:
            self._shm.close()
        finally:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def publish(self, snapshot: SchedulerStateSnapshot, payload: bytes) -> bool:
        if len(payload) > self._payload_capacity:
            logger.warning(
                "Skipping snapshot SHM publish for %s: payload %d exceeds "
                "capacity %d",
                self._name,
                len(payload),
                self._payload_capacity,
            )
            return False

        with self._lock:
            self._sequence += 1
            odd_sequence = self._sequence
            self._write_header(
                sequence=odd_sequence,
                snapshot_version=snapshot.version,
                created_at=snapshot.created_at,
                payload_size=len(payload),
                prefill_backlog_total_tokens=snapshot.prefill_backlog_total_tokens,
                build_latency_ms=snapshot.build_latency_ms,
            )
            self._shm.buf[
                SNAPSHOT_SHM_HEADER_SIZE : SNAPSHOT_SHM_HEADER_SIZE + len(payload)
            ] = payload
            self._sequence += 1
            self._write_header(
                sequence=self._sequence,
                snapshot_version=snapshot.version,
                created_at=snapshot.created_at,
                payload_size=len(payload),
                prefill_backlog_total_tokens=snapshot.prefill_backlog_total_tokens,
                build_latency_ms=snapshot.build_latency_ms,
            )
        return True

    @property
    def _payload_capacity(self) -> int:
        return self._size_bytes - SNAPSHOT_SHM_HEADER_SIZE

    def _create_shared_memory(self) -> shared_memory.SharedMemory:
        try:
            return shared_memory.SharedMemory(
                name=self._name,
                create=True,
                size=self._size_bytes,
            )
        except FileExistsError:
            stale = shared_memory.SharedMemory(name=self._name)
            try:
                stale.close()
                stale.unlink()
            finally:
                return shared_memory.SharedMemory(
                    name=self._name,
                    create=True,
                    size=self._size_bytes,
                )

    def _write_header(
        self,
        *,
        sequence: int,
        snapshot_version: int,
        created_at: float,
        payload_size: int,
        prefill_backlog_total_tokens: float,
        build_latency_ms: float,
    ) -> None:
        header = SnapshotShmHeader(
            sequence=int(sequence),
            snapshot_version=int(snapshot_version),
            created_at=float(created_at),
            payload_size=int(payload_size),
            prefill_backlog_total_tokens=float(prefill_backlog_total_tokens),
            build_latency_ms=float(build_latency_ms),
            simulation_intercept=self._coefficients["simulation_intercept"],
            simulation_prefill_coeff=self._coefficients["simulation_prefill_coeff"],
            simulation_prefill_sq_coeff=self._coefficients[
                "simulation_prefill_sq_coeff"
            ],
            simulation_decode_coeff=self._coefficients[
                "simulation_decode_coeff"
            ],
            simulation_sum_coeff=self._coefficients["simulation_sum_coeff"],
            simulation_sum_sq_coeff=self._coefficients["simulation_sum_sq_coeff"],
        )
        self._shm.buf[:SNAPSHOT_SHM_HEADER_SIZE] = encode_snapshot_shm_header(header)


def read_snapshot_shm_once(
    shm: shared_memory.SharedMemory,
    *,
    max_retries: int = 32,
) -> Optional[tuple[SnapshotShmHeader, bytes]]:
    """Read a consistent snapshot payload using a single-writer seqlock."""

    for _ in range(max_retries):
        header_before = bytes(shm.buf[:SNAPSHOT_SHM_HEADER_SIZE])
        parsed_before = decode_snapshot_shm_header(header_before)
        if parsed_before.sequence % 2 == 1:
            continue
        payload_size = parsed_before.payload_size
        if payload_size < 0 or payload_size > shm.size - SNAPSHOT_SHM_HEADER_SIZE:
            raise ValueError(
                f"Snapshot SHM payload_size {payload_size} is out of bounds"
            )
        payload = bytes(
            shm.buf[
                SNAPSHOT_SHM_HEADER_SIZE : SNAPSHOT_SHM_HEADER_SIZE + payload_size
            ]
        )
        header_after = bytes(shm.buf[:SNAPSHOT_SHM_HEADER_SIZE])
        if header_before != header_after:
            continue
        parsed_after = decode_snapshot_shm_header(header_after)
        if parsed_after.sequence % 2 == 1:
            continue
        return parsed_after, payload
    return None


def read_snapshot_shm_header_once(
    shm: shared_memory.SharedMemory,
    *,
    max_retries: int = 32,
) -> Optional[SnapshotShmHeader]:
    """Read only the fixed SHM header using the same seqlock protocol."""

    for _ in range(max_retries):
        header_before = bytes(shm.buf[:SNAPSHOT_SHM_HEADER_SIZE])
        parsed_before = decode_snapshot_shm_header(header_before)
        if parsed_before.sequence % 2 == 1:
            continue
        header_after = bytes(shm.buf[:SNAPSHOT_SHM_HEADER_SIZE])
        if header_before != header_after:
            continue
        parsed_after = decode_snapshot_shm_header(header_after)
        if parsed_after.sequence % 2 == 1:
            continue
        return parsed_after
    return None
