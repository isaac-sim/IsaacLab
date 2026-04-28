# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp import observations
from isaaclab.managers import SceneEntityCfg


class _Scene(dict):
    def __init__(self, env_origins: torch.Tensor, **assets):
        super().__init__(assets)
        self.env_origins = env_origins


class _Proxy:
    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


def _make_env(body_link_pose_w: torch.Tensor, body_link_vel_w: torch.Tensor):
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=_Proxy(body_link_pose_w),
            body_link_vel_w=_Proxy(body_link_vel_w),
        )
    )
    env_origins = torch.tensor([[10.0, 0.0, 1.0], [-3.0, 2.0, 0.5]])
    return SimpleNamespace(
        num_envs=2,
        scene=_Scene(env_origins, robot=asset),
    )


def test_body_link_pose_w_returns_selected_env_origin_relative_poses():
    """Test that body link poses are selected, origin-relative, and flattened."""
    body_link_pose_w = torch.tensor(
        [
            [
                [10.5, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                [11.5, 2.0, 4.0, 0.0, 1.0, 0.0, 0.0],
                [12.5, 3.0, 5.0, 0.0, 0.0, 1.0, 0.0],
            ],
            [
                [-2.5, 4.0, 1.5, 0.0, 0.0, 0.0, 1.0],
                [-1.5, 5.0, 2.5, 1.0, 0.0, 0.0, 0.0],
                [-0.5, 6.0, 3.5, 0.0, 1.0, 0.0, 0.0],
            ],
        ]
    )
    body_link_vel_w = torch.zeros(2, 3, 6)
    env = _make_env(body_link_pose_w, body_link_vel_w)
    original_body_link_pose_w = body_link_pose_w.clone()

    pose = observations.body_link_pose_w(env, SceneEntityCfg("robot", body_ids=[0, 2]))

    expected = torch.tensor(
        [
            [0.5, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 2.5, 3.0, 4.0, 0.0, 0.0, 1.0, 0.0],
            [0.5, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.5, 4.0, 3.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    assert pose.shape == (2, 14)
    assert torch.allclose(pose, expected)
    assert torch.allclose(body_link_pose_w, original_body_link_pose_w)


def test_body_link_vel_w_returns_selected_velocities():
    """Test that body link velocities are selected and flattened."""
    body_link_pose_w = torch.zeros(2, 3, 7)
    body_link_vel_w = torch.tensor(
        [
            [
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            ],
            [
                [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
                [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
                [5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
            ],
        ]
    )
    env = _make_env(body_link_pose_w, body_link_vel_w)

    velocity = observations.body_link_vel_w(env, SceneEntityCfg("robot", body_ids=[1, 2]))

    expected = torch.tensor(
        [
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
        ]
    )
    assert velocity.shape == (2, 12)
    assert torch.allclose(velocity, expected)


def test_body_link_pose_w_returns_single_body_pose():
    """Test that a single selected body link pose keeps one pose payload per environment."""
    body_link_pose_w = torch.tensor(
        [
            [
                [10.5, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                [11.5, 2.0, 4.0, 0.0, 1.0, 0.0, 0.0],
                [12.5, 3.0, 5.0, 0.0, 0.0, 1.0, 0.0],
            ],
            [
                [-2.5, 4.0, 1.5, 0.0, 0.0, 0.0, 1.0],
                [-1.5, 5.0, 2.5, 1.0, 0.0, 0.0, 0.0],
                [-0.5, 6.0, 3.5, 0.0, 1.0, 0.0, 0.0],
            ],
        ]
    )
    body_link_vel_w = torch.zeros(2, 3, 6)
    env = _make_env(body_link_pose_w, body_link_vel_w)

    pose = observations.body_link_pose_w(env, SceneEntityCfg("robot", body_ids=1))

    expected = torch.tensor(
        [
            [1.5, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0],
            [1.5, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    assert pose.shape == (2, 7)
    assert torch.allclose(pose, expected)


def test_body_link_vel_w_returns_single_body_velocity():
    """Test that a single selected body link velocity keeps one velocity payload per environment."""
    body_link_pose_w = torch.zeros(2, 3, 7)
    body_link_vel_w = torch.tensor(
        [
            [
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            ],
            [
                [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
                [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
                [5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
            ],
        ]
    )
    env = _make_env(body_link_pose_w, body_link_vel_w)

    velocity = observations.body_link_vel_w(env, SceneEntityCfg("robot", body_ids=1))

    expected = torch.tensor(
        [
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
        ]
    )
    assert velocity.shape == (2, 6)
    assert torch.allclose(velocity, expected)
