# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less unit tests for Newton actuator kernels and defaults."""

import types

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.actuators.kernels import sync_torque_telemetry


def test_sync_torque_telemetry_reads_backend_effort_buffers_in_user_order() -> None:
    """Report torque telemetry in public joint order from backend-order effort buffers."""
    joint_pos = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
    zeros = wp.zeros_like(joint_pos)
    effort_limit = wp.full((1, 3), 1000.0, dtype=wp.float32, device="cpu")
    joint_modes = wp.array(np.asarray([0, 1, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([2, 0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
    backend_effort = wp.array(np.asarray([[100.0, 200.0, 300.0]], dtype=np.float32), device="cpu")
    actuator_effort = wp.array(np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32), device="cpu")
    computed = wp.zeros_like(joint_pos)
    applied = wp.zeros_like(joint_pos)

    wp.launch(
        sync_torque_telemetry,
        dim=joint_pos.shape,
        inputs=[
            joint_pos,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            effort_limit,
            joint_modes,
            backend_effort,
            actuator_effort,
            user_to_backend,
            True,
        ],
        outputs=[computed, applied],
        device="cpu",
    )

    np.testing.assert_allclose(computed.numpy(), np.asarray([[30.0, 100.0, 20.0]], dtype=np.float32))
    np.testing.assert_allclose(applied.numpy(), np.asarray([[300.0, 100.0, 200.0]], dtype=np.float32))


def test_sync_torque_telemetry_keeps_user_order_effort_buffers_unmapped() -> None:
    """Report torque telemetry directly from user-order actuator buffers."""
    joint_pos = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
    zeros = wp.zeros_like(joint_pos)
    joint_modes = wp.array(np.asarray([0, 1, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([2, 0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_effort = wp.array(np.asarray([[100.0, 200.0, 300.0]], dtype=np.float32), device="cpu")
    user_computed_effort = wp.array(np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32), device="cpu")
    computed = wp.zeros_like(joint_pos)
    applied = wp.zeros_like(joint_pos)

    wp.launch(
        sync_torque_telemetry,
        dim=joint_pos.shape,
        inputs=[
            joint_pos,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            wp.full((1, 3), 1000.0, dtype=wp.float32, device="cpu"),
            joint_modes,
            user_effort,
            user_computed_effort,
            user_to_backend,
            False,
        ],
        outputs=[computed, applied],
        device="cpu",
    )

    np.testing.assert_allclose(computed.numpy(), np.asarray([[10.0, 200.0, 30.0]], dtype=np.float32))
    np.testing.assert_allclose(applied.numpy(), np.asarray([[100.0, 200.0, 300.0]], dtype=np.float32))


def test_newton_actuator_defaults_follow_requested_public_joint_order() -> None:
    """Convert Newton actuator gain snapshots and managed IDs into public joint order."""
    from isaaclab_newton.actuators.adapter import build_newton_actuator_defaults

    controller = types.SimpleNamespace(
        kp=wp.array((10.0, 30.0, 11.0, 31.0), dtype=wp.float32, device="cpu"),
        kd=wp.array((1.0, 3.0, 1.1, 3.1), dtype=wp.float32, device="cpu"),
    )
    actuator = types.SimpleNamespace(
        controller=controller,
        indices=wp.array((0, 2, 3, 5), dtype=wp.uint32, device="cpu"),
    )

    stiffness, damping, managed = build_newton_actuator_defaults(
        actuators=[actuator],
        num_envs=2,
        num_joints=3,
        dof_offset=0,
        env_stride=3,
        device="cpu",
        joint_user_to_backend_indices=(2, 0, 1),
    )

    torch.testing.assert_close(stiffness, torch.tensor([[30.0, 10.0, 0.0], [31.0, 11.0, 0.0]]))
    torch.testing.assert_close(damping, torch.tensor([[3.0, 1.0, 0.0], [3.1, 1.1, 0.0]]))
    torch.testing.assert_close(managed, torch.tensor([0, 1], dtype=torch.int32))


def test_newton_actuator_defaults_reject_incomplete_joint_permutation() -> None:
    """Reject malformed actuator-default ordering maps with an actionable error."""
    from isaaclab_newton.actuators.adapter import build_newton_actuator_defaults

    with pytest.raises(
        ValueError,
        match=(
            r"joint_user_to_backend_indices must contain each backend joint index exactly once; "
            r"expected a permutation of 0\.\.2, got \(0, 0, 2\)\."
        ),
    ):
        build_newton_actuator_defaults(
            actuators=[],
            num_envs=1,
            num_joints=3,
            dof_offset=0,
            env_stride=3,
            device="cpu",
            joint_user_to_backend_indices=(0, 0, 2),
        )
