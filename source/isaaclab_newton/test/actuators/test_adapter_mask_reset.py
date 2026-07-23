# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for mask-based Newton actuator adapter resets."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import warp as wp
from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter


def test_adapter_expands_environment_mask_without_allocating() -> None:
    """Adapter state masks should select every actuator DOF in masked environments."""
    adapter = NewtonActuatorAdapter.__new__(NewtonActuatorAdapter)
    actuator = Mock()
    actuator.indices = wp.array([0, 1, 2, 3], dtype=wp.uint32, device="cpu")
    state_a = Mock()
    state_b = Mock()
    adapter.actuators = [actuator]
    adapter._states_a = [state_a]
    adapter._states_b = [state_b]
    adapter._num_envs = 2
    adapter._dof_offset = 0
    adapter.num_joints = 2
    adapter._device = "cpu"
    adapter._reset_env_mask = wp.zeros(2, dtype=wp.bool, device="cpu")
    adapter._reset_dof_masks = [wp.zeros(4, dtype=wp.bool, device="cpu")]
    env_mask = wp.array([False, True], dtype=wp.bool, device="cpu")

    adapter.reset(env_mask=env_mask)

    state_a.reset.assert_called_once()
    state_b.reset.assert_called_once()
    np.testing.assert_array_equal(state_a.reset.call_args.args[0].numpy(), [False, False, True, True])
    assert state_a.reset.call_args.args[0] is adapter._reset_dof_masks[0]


def test_adapter_reuses_id_mask_without_stale_selection() -> None:
    """Sequential ID resets should clear the reusable environment mask."""
    adapter = NewtonActuatorAdapter.__new__(NewtonActuatorAdapter)
    actuator = Mock()
    actuator.indices = wp.array([0, 1, 2, 3], dtype=wp.uint32, device="cpu")
    state_a = Mock()
    adapter.actuators = [actuator]
    adapter._states_a = [state_a]
    adapter._states_b = [None]
    adapter._num_envs = 2
    adapter._dof_offset = 0
    adapter.num_joints = 2
    adapter._device = "cpu"
    adapter._reset_env_mask = wp.zeros(2, dtype=wp.bool, device="cpu")
    adapter._reset_dof_masks = [wp.zeros(4, dtype=wp.bool, device="cpu")]

    adapter.reset(env_ids=[0])
    first_mask = state_a.reset.call_args.args[0].numpy().copy()
    adapter.reset(env_ids=[1])
    second_mask = state_a.reset.call_args.args[0].numpy().copy()

    np.testing.assert_array_equal(first_mask, [True, True, False, False])
    np.testing.assert_array_equal(second_mask, [False, False, True, True])
