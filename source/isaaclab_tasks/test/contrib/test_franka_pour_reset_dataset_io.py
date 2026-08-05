# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Franka Pour runtime reset-dataset boundary."""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.contrib.franka_pour.reset_dataset_io import (
    FRANKA_POUR_RESET_DATASET_FORMAT,
    FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION,
    reset_dataset_content_digest,
    reset_dataset_digest,
    reset_dataset_validate_runtime,
)

_SAMPLING_PROFILE = "full"
_GRASP_SIDE_IDS = (0, 1, 2, 3)
_PHYSICS_SHA256 = "1" * 64
_SOURCE_SHA256 = "2" * 64


def _payload() -> dict:
    """Build one compact production payload spanning every runtime field."""
    state_count = 3
    identity_pose = torch.tensor((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=torch.float32)
    measured_result = {
        "passed": True,
        "source_content_sha256": _SOURCE_SHA256,
        "physics_contract_sha256": _PHYSICS_SHA256,
    }
    sampler_cfg = {
        "sampling_profile": _SAMPLING_PROFILE,
        "grasp_side_ids": _GRASP_SIDE_IDS,
    }
    task_contract = {
        "robot_asset": "franka.usda",
        "simulation_dt": 1.0 / 120.0,
        "media_material": {"density": 1000.0, "friction": 0.45},
    }
    metadata = {
        "state_count": state_count,
        "joint_names": tuple(f"panda_joint{index}" for index in range(1, 8))
        + ("panda_finger_joint1", "panda_finger_joint2"),
        "frame": "environment",
        "quaternion_order": "xyzw",
        "particle_solver_state": "fresh_zero",
        "sampler_cfg": sampler_cfg,
        "task_contract": task_contract,
        "static_validation": {
            "policy": "analytic_static_v1",
            "all_rows_statically_validated": True,
            "per_row_mpm_rollout": False,
            "terminal_pour_manifold": {
                "policy": "relative_source_receiver_v1",
                "physics_contract_sha256": _PHYSICS_SHA256,
                "calibration": {
                    "policy": "bounded_terminal_gpu_sweep_v2",
                    "status": "passed",
                    "source_content_sha256": _SOURCE_SHA256,
                    "physics_contract_sha256": _PHYSICS_SHA256,
                    "result_sha256": reset_dataset_digest(measured_result),
                    "measured_result": measured_result,
                },
            },
        },
    }
    states = {
        "arm_joint_position": torch.zeros((state_count, 7), dtype=torch.float32),
        "arm_joint_velocity": torch.zeros((state_count, 7), dtype=torch.float32),
        "finger_joint_position": torch.zeros((state_count, 2), dtype=torch.float32),
        "finger_joint_velocity": torch.zeros((state_count, 2), dtype=torch.float32),
        "finger_joint_target": torch.zeros((state_count, 2), dtype=torch.float32),
        "source_root_pose": identity_pose.repeat(state_count, 1),
        "source_root_velocity": torch.zeros((state_count, 6), dtype=torch.float32),
        "target_root_pose": identity_pose.repeat(state_count, 1),
        "target_root_velocity": torch.zeros((state_count, 6), dtype=torch.float32),
        "category": torch.tensor((0, 1, 1), dtype=torch.int8),
        "objective": torch.tensor((-1.0, 0.2, 0.8), dtype=torch.float32),
        "difficulty": torch.tensor((0.0, 0.5, 1.0), dtype=torch.float32),
        "particle_layout_id": torch.tensor((0, 0, 1), dtype=torch.int32),
    }
    payload = {
        "format": FRANKA_POUR_RESET_DATASET_FORMAT,
        "schema_version": FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION,
        "metadata": metadata,
        "states": states,
        "particle_layouts": {
            "local_position": torch.zeros((2, 4, 3), dtype=torch.float32),
            "local_velocity": torch.zeros((2, 4, 3), dtype=torch.float32),
        },
        "contract_sha256": reset_dataset_digest({"sampler_cfg": sampler_cfg, "task_contract": task_contract}),
    }
    payload["content_sha256"] = reset_dataset_content_digest(payload)
    return payload


def _seal(payload: dict):
    """Recompute the content hash after one intentional semantic mutation."""
    payload["content_sha256"] = reset_dataset_content_digest(payload)


def _validate(payload: dict, **kwargs):
    """Validate with the canonical task expectations."""
    return reset_dataset_validate_runtime(
        payload,
        expected_sampling_profile=_SAMPLING_PROFILE,
        expected_grasp_side_ids=_GRASP_SIDE_IDS,
        **kwargs,
    )


def test_runtime_validator_accepts_valid_production_payload():
    payload = _payload()

    metadata, states, particle_layouts = _validate(
        payload,
        expected_content_sha256=payload["content_sha256"],
        expected_task_contract={
            "robot_asset": "franka.usda",
            "media_material": {"density": 1000.0},
        },
    )

    assert metadata is payload["metadata"]
    assert states is payload["states"]
    assert particle_layouts is payload["particle_layouts"]


def test_runtime_validator_rejects_bad_schema_and_content_hash():
    payload = _payload()
    payload["schema_version"] += 1
    with pytest.raises(ValueError, match="schema version"):
        _validate(payload)

    payload = _payload()
    payload["states"]["difficulty"][0] = 0.25
    with pytest.raises(ValueError, match="content digest"):
        _validate(payload)

    payload = _payload()
    with pytest.raises(ValueError, match="configured digest"):
        _validate(payload, expected_content_sha256="f" * 64)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("arm_joint_position", torch.zeros((3, 6), dtype=torch.float32), "arm_joint_position"),
        ("category", torch.tensor((0, 1, 2), dtype=torch.int8), "category"),
        ("particle_layout_id", torch.tensor((0, 1, 2), dtype=torch.int32), "layout identifiers"),
    ],
)
def test_runtime_validator_rejects_bad_state_shape_or_values(field, replacement, message):
    payload = _payload()
    payload["states"][field] = replacement
    _seal(payload)

    with pytest.raises(ValueError, match=message):
        _validate(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["metadata"].update(frame="world"), "environment-frame"),
        (
            lambda payload: payload["metadata"]["sampler_cfg"].update(sampling_profile="diagnostic"),
            "sampling profile",
        ),
        (
            lambda payload: payload["metadata"]["sampler_cfg"].update(grasp_side_ids=(0, 1)),
            "grasp-side identifiers",
        ),
        (
            lambda payload: payload["metadata"]["static_validation"]["terminal_pour_manifold"]["calibration"].update(
                status="pending"
            ),
            "has not passed",
        ),
    ],
)
def test_runtime_validator_rejects_bad_metadata_and_production_marker(mutation, message):
    payload = _payload()
    mutation(payload)
    _seal(payload)

    with pytest.raises(ValueError, match=message):
        _validate(payload)


def test_runtime_validator_rejects_task_contract_subset_mismatch():
    payload = _payload()
    with pytest.raises(ValueError, match="media_material.density"):
        _validate(payload, expected_task_contract={"media_material": {"density": 900.0}})
