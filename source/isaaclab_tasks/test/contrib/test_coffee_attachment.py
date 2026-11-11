# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mounted spoon pose and link twist published through the asset boundary."""

from math import sqrt
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import warp as wp
from newton import JointType

from isaaclab_tasks.contrib.rizon_sharpa_coffee.attachment import MountedSpoonAttachment

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


def test_spoon_asset_sync_rotates_the_mount_and_selects_worlds() -> None:
    """A rotated, moving palm transfers link velocity without mutating robot state."""
    pose = torch.tensor(
        [
            [[0.0] * 7, [1.0, 2.0, 3.0, 0.0, 0.0, sqrt(0.5), sqrt(0.5)]],
            [[0.0] * 7, [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]],
        ]
    )
    velocity = torch.tensor([[[0.0] * 6, [1.0, 2.0, 3.0, 0.0, 0.0, 2.0]], [[0.0] * 6, [7.0, 8.0, 9.0, 0.0, 1.0, 0.0]]])
    robot = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Robot"),
        device="cpu",
        num_instances=2,
        find_bodies=Mock(return_value=([1], ["r_palm_ctrl"])),
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=pose), body_link_vel_w=SimpleNamespace(torch=velocity)
        ),
    )
    spoon = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Spoon"),
        write_root_link_pose_to_sim_index=Mock(),
        write_root_link_velocity_to_sim_index=Mock(),
    )
    model = SimpleNamespace(
        device="cpu",
        body_world=wp.array([0, 0, 1, 1], dtype=int, device="cpu"),
        body_label=[
            "/World/envs/env_0/Robot/r_palm_ctrl",
            "/World/envs/env_0/Spoon",
            "/World/envs/env_1/Robot/r_palm_ctrl",
            "/World/envs/env_1/Spoon",
        ],
        joint_child=wp.array([1, 3], dtype=int, device="cpu"),
        joint_type=wp.array([int(JointType.FREE)] * 2, dtype=int, device="cpu"),
        joint_q_start=wp.array([0, 7], dtype=int, device="cpu"),
        joint_qd_start=wp.array([0, 6], dtype=int, device="cpu"),
    )
    manager = SimpleNamespace(backend=SimpleNamespace(model=model), register_post_actuator_callback=Mock())
    attachment = MountedSpoonAttachment(robot, spoon, (0.5, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), backend_manager=manager)
    original_velocity = velocity.clone()
    attachment.sync_to_sim()
    all_pose = spoon.write_root_link_pose_to_sim_index.call_args.kwargs["root_pose"]
    all_velocity = spoon.write_root_link_velocity_to_sim_index.call_args.kwargs["root_velocity"]
    torch.testing.assert_close(all_pose[:, :3], torch.tensor([[1.0, 2.5, 3.0], [4.5, 5.0, 6.0]]))
    torch.testing.assert_close(all_pose[:, 3:], pose[:, 1, 3:])
    torch.testing.assert_close(
        all_velocity, torch.tensor([[0.0, 2.0, 3.0, 0.0, 0.0, 2.0], [7.0, 8.0, 8.5, 0.0, 1.0, 0.0]])
    )
    torch.testing.assert_close(velocity, original_velocity)

    selected = torch.tensor([1])
    attachment.sync_to_sim(selected)
    pose_write = spoon.write_root_link_pose_to_sim_index.call_args.kwargs
    velocity_write = spoon.write_root_link_velocity_to_sim_index.call_args.kwargs
    assert pose_write["env_ids"] is selected
    assert velocity_write["env_ids"] is selected
    torch.testing.assert_close(pose_write["root_pose"], all_pose[1:])
    torch.testing.assert_close(velocity_write["root_velocity"], all_velocity[1:])
    attachment.close()
