# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime validation for externally generated Franka Pour reset datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import torch

FRANKA_POUR_RESET_DATASET_FORMAT = "franka_pour_reset_dataset"
FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION = 12

_STATE_TENSOR_SPECS = {
    "arm_joint_position": (torch.float32, (7,)),
    "arm_joint_velocity": (torch.float32, (7,)),
    "finger_joint_position": (torch.float32, (2,)),
    "finger_joint_velocity": (torch.float32, (2,)),
    "finger_joint_target": (torch.float32, (2,)),
    "source_root_pose": (torch.float32, (7,)),
    "source_root_velocity": (torch.float32, (6,)),
    "target_root_pose": (torch.float32, (7,)),
    "target_root_velocity": (torch.float32, (6,)),
    "category": (torch.int8, ()),
    "objective": (torch.float32, ()),
    "reset_region": (torch.int8, ()),
    "difficulty": (torch.float32, ()),
    "particle_layout_id": (torch.int32, ()),
}
RESET_DATASET_STATE_NAMES = tuple(_STATE_TENSOR_SPECS)


class _HashWriter:
    """Adapt a hashlib digest to the byte-writer interface used by the encoder."""

    def __init__(self):
        self.digest = hashlib.sha256()

    def write(self, value: bytes):
        """Append bytes to the digest."""
        self.digest.update(value)


def reset_dataset_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for nested primitive and tensor data.

    Mapping insertion order and tensor device do not affect the result. Concrete sequence types,
    tensor dtypes, and tensor shapes remain part of the digest.

    Args:
        value: Nested mappings, lists, tuples, primitive values, or dense tensors.

    Returns:
        The lowercase hexadecimal SHA-256 digest.

    Raises:
        TypeError: If the value contains an unsupported type or tensor layout.
    """
    writer = _HashWriter()
    _write_value(writer, value)
    return writer.digest.hexdigest()


def reset_dataset_content_digest(
    payload: Mapping[str, Any],
    *,
    digest_key: str = "content_sha256",
) -> str:
    """Return a dataset payload digest while excluding its own digest field.

    Args:
        payload: Dataset payload to hash.
        digest_key: Top-level field that stores the resulting content digest.

    Returns:
        The content SHA-256 digest.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("Reset dataset payload must be a mapping.")
    if not isinstance(digest_key, str) or not digest_key:
        raise ValueError("digest_key must be a non-empty string.")
    return reset_dataset_digest({key: value for key, value in payload.items() if key != digest_key})


def reset_dataset_validate_runtime(
    payload: Mapping[str, Any],
    *,
    expected_content_sha256: str | None = None,
    expected_task_contract: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """Validate a production Franka Pour reset dataset for runtime replay.

    Dataset generation is an offline workflow. This validator keeps the runtime boundary limited to
    the safety-critical artifact digest, task contract, tensor schema, shapes, and finite values.
    When supplied, :paramref:`expected_task_contract` compares only its keys so callers can bind the
    artifact to the current task contract.

    Args:
        payload: Safely loaded reset-dataset payload.
        expected_content_sha256: Optional configured artifact content digest.
        expected_task_contract: Optional subset of task-contract fields to compare.

    Returns:
        The validated metadata, state tensors, and particle-layout tensors.

    Raises:
        TypeError: If a required container or tensor has the wrong type.
        ValueError: If the artifact does not match the task or is malformed.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("Reset dataset payload must be a mapping.")
    if expected_content_sha256 is not None:
        _validate_sha256(expected_content_sha256, "expected_content_sha256")
    if expected_task_contract is not None and not isinstance(expected_task_contract, Mapping):
        raise TypeError("expected_task_contract must be a mapping or None.")

    if payload.get("format") != FRANKA_POUR_RESET_DATASET_FORMAT:
        raise ValueError(f"Expected reset dataset format {FRANKA_POUR_RESET_DATASET_FORMAT!r}.")
    if payload.get("schema_version") != FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION:
        raise ValueError(f"Expected reset dataset schema version {FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION}.")

    metadata = _require_mapping(payload, "metadata")
    states = _require_mapping(payload, "states")
    particle_layouts = _require_mapping(payload, "particle_layouts")
    task_contract = _require_mapping(metadata, "task_contract", path="metadata")

    content_sha256 = payload.get("content_sha256")
    _validate_sha256(content_sha256, "content_sha256")
    if content_sha256 != reset_dataset_content_digest(payload):
        raise ValueError("Reset dataset content digest does not match its payload.")
    if expected_content_sha256 is not None and content_sha256 != expected_content_sha256:
        raise ValueError("Reset dataset content digest does not match the configured digest.")

    if expected_task_contract is not None:
        _validate_contract_subset(task_contract, expected_task_contract, path="metadata.task_contract")

    _validate_state_tensors(states)
    layout_count = _validate_particle_layouts(particle_layouts)
    layout_ids = states["particle_layout_id"]
    if bool(torch.any((layout_ids < 0) | (layout_ids >= layout_count))):
        raise ValueError("Reset dataset particle-layout identifiers are outside the layout table.")

    return metadata, states, particle_layouts


def _write_bytes(sink: Any, value: bytes):
    """Write one length-delimited byte string to a hash or byte buffer."""
    sink.write(len(value).to_bytes(8, byteorder="big", signed=False))
    sink.write(value)


def _write_value(sink: Any, value: Any):
    """Write the canonical representation of one supported value."""
    if value is None:
        sink.write(b"none")
    elif isinstance(value, bool):
        sink.write(b"bool")
        sink.write(b"1" if value else b"0")
    elif isinstance(value, int):
        sink.write(b"int")
        _write_bytes(sink, str(value).encode("ascii"))
    elif isinstance(value, float):
        sink.write(b"float")
        _write_bytes(sink, value.hex().encode("ascii"))
    elif isinstance(value, str):
        sink.write(b"str")
        _write_bytes(sink, value.encode("utf-8"))
    elif isinstance(value, bytes):
        sink.write(b"bytes")
        _write_bytes(sink, value)
    elif isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError("Reset-dataset hashes support only dense, strided tensors.")
        tensor = value.detach().cpu().contiguous()
        sink.write(b"tensor")
        _write_bytes(sink, str(tensor.dtype).encode("ascii"))
        _write_value(sink, tuple(tensor.shape))
        raw_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        _write_bytes(sink, raw_bytes)
    elif isinstance(value, Mapping):
        sink.write(b"mapping")
        encoded_items: list[tuple[bytes, Any]] = []
        for key, item in value.items():
            key_buffer = BytesIO()
            _write_value(key_buffer, key)
            encoded_items.append((key_buffer.getvalue(), item))
        encoded_items.sort(key=lambda pair: pair[0])
        _write_value(sink, len(encoded_items))
        for encoded_key, item in encoded_items:
            _write_bytes(sink, encoded_key)
            _write_value(sink, item)
    elif isinstance(value, tuple):
        sink.write(b"tuple")
        _write_value(sink, len(value))
        for item in value:
            _write_value(sink, item)
    elif isinstance(value, list):
        sink.write(b"list")
        _write_value(sink, len(value))
        for item in value:
            _write_value(sink, item)
    else:
        raise TypeError(f"Unsupported reset-dataset hash value type: {type(value).__name__}.")


def _require_mapping(container: Mapping[str, Any], key: str, *, path: str = "payload") -> Mapping[str, Any]:
    """Return one required mapping field."""
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}.{key} must be a mapping.")
    return value


def _validate_sha256(value: Any, name: str):
    """Validate one lowercase hexadecimal SHA-256 string."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def _validate_contract_subset(actual: Mapping[str, Any], expected: Mapping[str, Any], *, path: str):
    """Recursively compare only expected task-contract fields."""
    for key, expected_value in expected.items():
        field_path = f"{path}.{key}"
        if key not in actual:
            raise ValueError(f"Reset dataset task contract is missing {field_path!r}.")
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                raise ValueError(f"Reset dataset task contract field {field_path!r} is not a mapping.")
            _validate_contract_subset(actual_value, expected_value, path=field_path)
        elif reset_dataset_digest(actual_value) != reset_dataset_digest(expected_value):
            raise ValueError(f"Reset dataset task contract field {field_path!r} does not match the runtime.")


def _validate_state_tensors(states: Mapping[str, Any]):
    """Validate tensors consumed by reset replay and adaptive sampling."""
    state_count: int | None = None
    for name, (dtype, trailing_shape) in _STATE_TENSOR_SPECS.items():
        value = states.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"states.{name} must be a torch.Tensor.")
        if state_count is None:
            state_count = int(value.shape[0]) if value.ndim > 0 else 0
            if state_count <= 0:
                raise ValueError("Reset dataset must contain at least one state.")
        expected_shape = (state_count, *trailing_shape)
        if value.dtype != dtype or tuple(value.shape) != expected_shape:
            raise ValueError(
                f"states.{name} must have dtype {dtype} and shape {expected_shape}, "
                f"got {value.dtype} and {tuple(value.shape)}."
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"states.{name} must contain only finite values.")
    category = states["category"]
    if not bool(torch.all((category == 0) | (category == 1))):
        raise ValueError("states.category must contain only non-grasping and grasping identifiers.")
    reset_region = states["reset_region"]
    if not bool(torch.all((reset_region >= 0) & (reset_region < 4))):
        raise ValueError("states.reset_region must contain only reaching through near-goal identifiers.")


def _validate_particle_layouts(particle_layouts: Mapping[str, Any]) -> int:
    """Validate source-local particle position and velocity layouts."""
    position = particle_layouts.get("local_position")
    velocity = particle_layouts.get("local_velocity")
    for name, value in (("local_position", position), ("local_velocity", velocity)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"particle_layouts.{name} must be a torch.Tensor.")
        if value.dtype != torch.float32 or value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 0:
            raise ValueError(f"particle_layouts.{name} must be a non-empty float32 tensor with shape (L, P, 3).")
        if value.shape[2] != 3 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"particle_layouts.{name} must contain finite three-dimensional vectors.")
    if position.shape != velocity.shape:
        raise ValueError("Particle position and velocity layouts must have identical shapes.")
    return int(position.shape[0])
