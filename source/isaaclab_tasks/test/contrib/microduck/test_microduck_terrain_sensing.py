# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source terrain-clearance behavior at the retained sole sites, independent of body height."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.microduck.terrain import foot_clearance, foot_height_safe

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def view(value):
    return SimpleNamespace(torch=torch.as_tensor(value, dtype=torch.float32))


def test_clearance_reduces_three_rays_and_preserves_source_miss_backface_rules():
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(max_distance=1.0),
        data=SimpleNamespace(
            pos_w=view([[0, 0, 1.0], [0, 0, 1.4], [0, 0, 0.2]]),
            ray_hits_w=view(
                [[[0, 0, 0.8], [0, 0, 0.9], [0, 0, float("inf")]], [[0, 0, float("inf")]] * 3, [[0, 0, 0.1]] * 3]
            ),
            ray_normals_w=view([[[0, 0, 1]] * 3, [[0, 0, 1]] * 3, [[0, 0, 1], [0, 0, -1], [0, 0, 1]]]),
        ),
    )
    env = SimpleNamespace(scene={"foot": sensor})
    # World-frame site z fallback is the source rule even on elevated terrain.
    torch.testing.assert_close(foot_height_safe(env, ("foot",)), torch.tensor([[0.1], [1.0], [0.0]]))


def test_clearance_cost_uses_site_velocity_including_rotation_about_ankle():
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(max_distance=1.0),
        data=SimpleNamespace(
            pos_w=view([[0.1, 0.0, 0.04]]),
            ray_hits_w=view([[[0.1, 0.0, 0.0]] * 3]),
            ray_normals_w=view([[[0.0, 0.0, 1.0]] * 3]),
        ),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=view([[[0.0, 0.0, 0.04]]]),
            body_link_lin_vel_w=view([[[0.0, 0.0, 0.0]]]),
            body_link_ang_vel_w=view([[[0.0, 0.0, 2.0]]]),
        )
    )
    env = SimpleNamespace(
        scene={"foot": sensor, "robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[1.0, 0.0, 0.0]])),
    )
    # A stationary ankle rotating at2rad/s moves its10cm-offset sole site at.2m/s.
    ac = SceneEntityCfg("robot", body_ids=[0])
    value = foot_clearance(env, 0.02, "base_velocity", ac, 0.01, ("foot",))
    torch.testing.assert_close(value, torch.tensor([0.004]))
