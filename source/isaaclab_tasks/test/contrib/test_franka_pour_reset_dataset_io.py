# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the safety-critical Franka Pour reset-artifact boundary."""

import pytest
import torch

from isaaclab_tasks.contrib.franka_pour.reset_dataset_io import (
    _STATE_TENSOR_SPECS,
    FRANKA_POUR_RESET_DATASET_FORMAT,
    FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION,
    reset_dataset_content_digest,
    reset_dataset_validate_runtime,
)


def _payload() -> dict:
    state_count = 2
    states = {
        name: torch.zeros((state_count, *shape), dtype=dtype) for name, (dtype, shape) in _STATE_TENSOR_SPECS.items()
    }
    states["source_root_pose"][:, 6] = 1.0
    states["target_root_pose"][:, 6] = 1.0
    payload = {
        "format": FRANKA_POUR_RESET_DATASET_FORMAT,
        "schema_version": FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION,
        "metadata": {"task_contract": {"particle_count": 3}},
        "states": states,
        "particle_layouts": {
            "local_position": torch.zeros((1, 3, 3), dtype=torch.float32),
            "local_velocity": torch.zeros((1, 3, 3), dtype=torch.float32),
        },
    }
    payload["content_sha256"] = reset_dataset_content_digest(payload)
    return payload


def test_minimal_runtime_artifact_is_accepted():
    """Runtime replay does not require redundant generator-provenance metadata."""
    payload = _payload()

    metadata, states, layouts = reset_dataset_validate_runtime(
        payload,
        expected_task_contract={"particle_count": 3},
    )

    assert metadata is payload["metadata"]
    assert states is payload["states"]
    assert layouts is payload["particle_layouts"]


def test_nonfinite_state_is_rejected_after_digest_verification():
    payload = _payload()
    payload["states"]["arm_joint_position"][0, 0] = torch.nan
    payload["content_sha256"] = reset_dataset_content_digest(payload)

    with pytest.raises(ValueError, match="states.arm_joint_position must contain only finite values"):
        reset_dataset_validate_runtime(payload)


def test_content_mutation_is_rejected():
    payload = _payload()
    payload["states"]["difficulty"][0] = 0.5

    with pytest.raises(ValueError, match="content digest does not match"):
        reset_dataset_validate_runtime(payload)
