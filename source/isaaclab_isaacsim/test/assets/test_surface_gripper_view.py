# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the PhysX surface-gripper view boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_isaacsim.assets import SurfaceGripper

pytestmark = pytest.mark.kitless


def _make_gripper() -> SurfaceGripper:
    """Initialize the real command and state buffers around a recording CPU view."""
    gripper = object.__new__(SurfaceGripper)
    gripper._device = "cpu"
    gripper._num_envs = 3
    gripper._create_buffers()
    gripper._gripper_view = SimpleNamespace(
        apply_gripper_action=Mock(),
        set_surface_gripper_properties=Mock(),
    )
    return gripper


def test_command_filter_and_partial_property_update_use_literal_view_payloads() -> None:
    """Submit only open/close commands and preserve property selector ordering."""
    gripper = _make_gripper()

    gripper.set_grippers_command_index(torch.tensor([-0.5, 0.5]), env_ids=[2, 0])

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


@pytest.mark.parametrize("as_warp", [False, True], ids=["torch", "warp"])
def test_selected_reset_opens_only_selected_grippers(as_warp: bool) -> None:
    """A selected reset sends only open actions and leaves other commands staged."""
    gripper = _make_gripper()
    gripper.state.fill_(1.0)
    gripper.set_grippers_command_index(torch.tensor([0.5, 0.5, -0.5]))
    mask = torch.tensor([True, False, True])
    gripper.reset(env_ids=[1], env_mask=wp.from_torch(mask) if as_warp else mask)

    gripper.gripper_view.apply_gripper_action.assert_called_once_with([-1.0, 0.5, -1.0], [[0], [2]])
    np.testing.assert_array_equal(gripper.command.numpy(), [0.0, 0.5, 0.0])
    np.testing.assert_array_equal(gripper.state.numpy(), [-1.0, 1.0, -1.0])

    gripper.reset(env_ids=[])
    assert gripper.gripper_view.apply_gripper_action.call_count == 1
