# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime validation for externally generated Franka Pour reset datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any

import torch

FRANKA_POUR_RESET_DATASET_FORMAT = "franka_pour_reset_dataset"
FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION = 12

_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
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
    "difficulty": (torch.float32, ()),
    "particle_layout_id": (torch.int32, ()),
}


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
    expected_sampling_profile: str,
    expected_grasp_side_ids: Sequence[int],
    expected_content_sha256: str | None = None,
    expected_task_contract: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """Validate a production Franka Pour reset dataset for runtime replay.

    Dataset generation and calibration are external workflows. This validator checks the immutable
    runtime boundary: envelope integrity, production provenance, metadata consumed by replay, and
    the tensors restored by the environment. When supplied, :paramref:`expected_task_contract`
    compares only its keys so callers can bind the artifact to a compact compatibility contract.

    Args:
        payload: Safely loaded reset-dataset payload.
        expected_sampling_profile: Required external generation profile.
        expected_grasp_side_ids: Required grasp-side identifiers.
        expected_content_sha256: Optional configured artifact content digest.
        expected_task_contract: Optional subset of task-contract fields to compare.

    Returns:
        The validated metadata, state tensors, and particle-layout tensors.

    Raises:
        TypeError: If a required container or tensor has the wrong type.
        ValueError: If the artifact is incompatible, malformed, or not production-calibrated.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("Reset dataset payload must be a mapping.")
    if not isinstance(expected_sampling_profile, str) or not expected_sampling_profile:
        raise ValueError("expected_sampling_profile must be a non-empty string.")
    grasp_side_ids = _validate_expected_grasp_side_ids(expected_grasp_side_ids)
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
    sampler_cfg = _require_mapping(metadata, "sampler_cfg", path="metadata")
    task_contract = _require_mapping(metadata, "task_contract", path="metadata")

    content_sha256 = payload.get("content_sha256")
    _validate_sha256(content_sha256, "content_sha256")
    if content_sha256 != reset_dataset_content_digest(payload):
        raise ValueError("Reset dataset content digest does not match its payload.")
    if expected_content_sha256 is not None and content_sha256 != expected_content_sha256:
        raise ValueError("Reset dataset content digest does not match the configured digest.")

    if expected_task_contract is not None:
        _validate_contract_subset(task_contract, expected_task_contract, path="metadata.task_contract")

    if sampler_cfg.get("sampling_profile") != expected_sampling_profile:
        raise ValueError("Reset dataset sampling profile does not match the configured profile.")
    try:
        actual_grasp_side_ids = _validate_expected_grasp_side_ids(sampler_cfg.get("grasp_side_ids"))
    except (TypeError, ValueError) as error:
        raise ValueError("Reset dataset grasp-side identifiers are invalid.") from error
    if actual_grasp_side_ids != grasp_side_ids:
        raise ValueError("Reset dataset grasp-side identifiers do not match the configured identifiers.")

    _validate_runtime_metadata(metadata)
    _validate_production_marker(metadata)
    state_count = metadata["state_count"]
    _validate_state_tensors(states, state_count)
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


def _validate_expected_grasp_side_ids(values: Sequence[int]) -> tuple[int, ...]:
    """Validate and canonicalize configured grasp-side identifiers."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("expected_grasp_side_ids must be a sequence of integers.")
    result = tuple(values)
    if (
        not result
        or any(not isinstance(value, int) or isinstance(value, bool) for value in result)
        or len(set(result)) != len(result)
        or any(value < 0 or value > 3 for value in result)
    ):
        raise ValueError("expected_grasp_side_ids must contain unique integers.")
    return result


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


def _validate_runtime_metadata(metadata: Mapping[str, Any]):
    """Validate metadata consumed directly by runtime replay."""
    state_count = metadata.get("state_count")
    if not isinstance(state_count, int) or isinstance(state_count, bool) or state_count <= 0:
        raise ValueError("metadata.state_count must be a positive integer.")
    joint_names = metadata.get("joint_names")
    if not isinstance(joint_names, (tuple, list)) or tuple(joint_names) != _ARM_JOINT_NAMES + _FINGER_JOINT_NAMES:
        raise ValueError("Reset dataset joint order does not match the Franka runtime joint order.")
    if metadata.get("frame") != "environment" or metadata.get("quaternion_order") != "xyzw":
        raise ValueError("Reset dataset poses must use environment-frame XYZW representation.")
    if metadata.get("particle_solver_state") != "fresh_zero":
        raise ValueError("Reset dataset particles must use fresh zero solver state.")


def _validate_production_marker(metadata: Mapping[str, Any]):
    """Require an externally calibrated production marker."""
    marker = _require_mapping(metadata, "static_validation", path="metadata")
    if (
        marker.get("policy") != "analytic_static_v1"
        or marker.get("all_rows_statically_validated") is not True
        or marker.get("per_row_mpm_rollout") is not False
    ):
        raise ValueError("Reset dataset does not contain supported all-row static validation.")
    manifold = _require_mapping(marker, "terminal_pour_manifold", path="metadata.static_validation")
    if manifold.get("policy") != "relative_source_receiver_v1":
        raise ValueError("Reset dataset terminal-pour manifold policy is unsupported.")
    calibration = _require_mapping(
        manifold,
        "calibration",
        path="metadata.static_validation.terminal_pour_manifold",
    )
    if calibration.get("policy") != "bounded_terminal_gpu_sweep_v2" or calibration.get("status") != "passed":
        raise ValueError("Reset dataset terminal-pour calibration has not passed.")
    measured_result = _require_mapping(
        calibration,
        "measured_result",
        path="metadata.static_validation.terminal_pour_manifold.calibration",
    )
    if measured_result.get("passed") is not True:
        raise ValueError("Reset dataset terminal-pour calibration result has not passed.")
    source_sha256 = calibration.get("source_content_sha256")
    _validate_sha256(source_sha256, "terminal calibration source_content_sha256")
    physics_sha256 = manifold.get("physics_contract_sha256")
    _validate_sha256(physics_sha256, "terminal manifold physics_contract_sha256")
    if calibration.get("physics_contract_sha256") != physics_sha256:
        raise ValueError("Reset dataset terminal-pour calibration physics contract is inconsistent.")
    if measured_result.get("physics_contract_sha256") != physics_sha256:
        raise ValueError("Reset dataset measured terminal calibration physics contract is inconsistent.")
    if measured_result.get("source_content_sha256") != source_sha256:
        raise ValueError("Reset dataset measured terminal calibration source digest is inconsistent.")


def _validate_state_tensors(states: Mapping[str, Any], state_count: int):
    """Validate tensors consumed by reset replay and adaptive sampling."""
    for name, (dtype, trailing_shape) in _STATE_TENSOR_SPECS.items():
        value = states.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"states.{name} must be a torch.Tensor.")
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
