# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.manipulation.pick_place.mdp import observations


class _Scene(dict):
    pass


class _Proxy:
    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


def _make_env(body_link_pose_w: torch.Tensor, body_link_vel_w: torch.Tensor):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=_Proxy(body_link_pose_w),
            body_link_vel_w=_Proxy(body_link_vel_w),
        )
    )
    return SimpleNamespace(scene=_Scene(robot=robot))


def test_robot_link_pose_and_velocity_terms_return_explicit_components():
    pose = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7)
    velocity = torch.arange(36, dtype=torch.float32).reshape(2, 3, 6)
    env = _make_env(pose, velocity)

    assert observations.get_all_robot_link_pose(env) is pose
    assert observations.get_all_robot_link_velocity(env) is velocity


def test_deprecated_robot_link_state_composes_pose_and_velocity():
    pose = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7)
    velocity = torch.arange(36, dtype=torch.float32).reshape(2, 3, 6)
    env = _make_env(pose, velocity)

    with pytest.warns(DeprecationWarning, match="get_all_robot_link_state"):
        state = observations.get_all_robot_link_state(env)

    assert state.shape == (2, 3, 13)
    assert torch.equal(state, torch.cat((pose, velocity), dim=-1))
