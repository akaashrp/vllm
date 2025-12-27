# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for serializing scheduler snapshots for native consumers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict

import msgspec
import torch

from vllm.v1.core.sched.state_snapshot import SchedulerStateSnapshot

__all__ = [
    "serialize_scheduler_state_snapshot",
    "encode_scheduler_state_snapshot",
    "decode_scheduler_state_snapshot",
]


def _serialize_value(value: Any) -> Any:
    """Recursively convert dataclasses and torch types into primitives."""
    if is_dataclass(value):
        return {
            field.name: _serialize_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, torch.dtype):
        # Some torch builds don't expose ``dtype.name``. Fall back to the
        # string representation so serialization still succeeds.
        name = getattr(value, "name", None)
        if name is not None:
            return name
        dtype_str = str(value)
        if dtype_str.startswith("torch."):
            return dtype_str.split(".", 1)[1]
        return dtype_str
    return value


def serialize_scheduler_state_snapshot(
    snapshot: SchedulerStateSnapshot,
) -> Dict[str, Any]:
    """Convert a scheduler snapshot into primitive Python structures."""
    return _serialize_value(snapshot)


def encode_scheduler_state_snapshot(snapshot: SchedulerStateSnapshot) -> bytes:
    """Serialize a scheduler snapshot to msgpack bytes."""
    payload = serialize_scheduler_state_snapshot(snapshot)
    return msgspec.msgpack.encode(payload)


def decode_scheduler_state_snapshot(data: bytes) -> Dict[str, Any]:
    """Decode a msgpack-encoded scheduler snapshot payload."""
    if not data:
        return {}
    return msgspec.msgpack.decode(data)
