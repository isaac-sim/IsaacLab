# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the PhysX surface-gripper command and property payloads."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import warp as wp
from isaaclab_physx.assets.surface_gripper.surface_gripper import SurfaceGripper


def test_command_filter_and_partial_property_update_use_literal_view_payloads() -> None:
    """Submit only open/close commands and preserve property selector ordering."""
    # Skip USD initialization: a three-environment CPU gripper around a recording view.
    gripper = object.__new__(SurfaceGripper)
    gripper._device = "cpu"
    gripper._num_envs = 3
    gripper._ALL_INDICES = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
    gripper._gripper_command = wp.zeros(3, dtype=wp.float32, device="cpu")
    gripper._max_grip_distance = wp.zeros(3, dtype=wp.float32, device="cpu")
    gripper._coaxial_force_limit = wp.zeros(3, dtype=wp.float32, device="cpu")
    gripper._shear_force_limit = wp.zeros(3, dtype=wp.float32, device="cpu")
    gripper._retry_interval = wp.zeros(3, dtype=wp.float32, device="cpu")
    gripper._gripper_view = SimpleNamespace(
        apply_gripper_action=Mock(),
        set_surface_gripper_properties=Mock(),
    )

    gripper.set_grippers_command_index(wp.array([0.5, 0.0, -0.5], dtype=wp.float32, device="cpu"))

    gripper.write_data_to_sim()

    gripper.gripper_view.apply_gripper_action.assert_called_once_with([0.5, 0.0, -0.5], [[0], [2]])

    env_ids = wp.array([2, 0], dtype=wp.int32, device="cpu")
    gripper.update_gripper_properties_index(
        max_grip_distance=wp.array([0.2, 0.4], dtype=wp.float32, device="cpu"),
        env_ids=env_ids,
    )

    np.testing.assert_array_equal(gripper._max_grip_distance.numpy(), np.asarray([0.4, 0.0, 0.2], dtype=np.float32))
    properties = gripper.gripper_view.set_surface_gripper_properties.call_args.kwargs
    np.testing.assert_array_equal(properties.pop("max_grip_distance"), np.asarray([0.4, 0.0, 0.2], dtype=np.float32))
    assert properties == {
        "coaxial_force_limit": [0.0, 0.0, 0.0],
        "shear_force_limit": [0.0, 0.0, 0.0],
        "retry_interval": [0.0, 0.0, 0.0],
        "indices": [2, 0],
    }
