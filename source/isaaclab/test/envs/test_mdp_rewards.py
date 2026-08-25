# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.rewards import base_height_l2
from isaaclab.managers import SceneEntityCfg


def _make_env(root_heights: list[float], ray_hit_heights: list[list[float]]) -> SimpleNamespace:
    root_pos_w = torch.zeros((len(root_heights), 3))
    root_pos_w[:, 2] = torch.tensor(root_heights)

    ray_hits_w = torch.zeros((len(ray_hit_heights), len(ray_hit_heights[0]), 3))
    ray_hits_w[..., 2] = torch.tensor(ray_hit_heights)

    scene = {
        "robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=root_pos_w))),
        "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=ray_hits_w))),
    }
    return SimpleNamespace(scene=scene)


def test_base_height_l2_ignores_non_finite_ray_hits():
    """The terrain height excludes missed and otherwise invalid ray hits."""
    env = _make_env(
        root_heights=[1.0, 1.0],
        ray_hit_heights=[[0.0, 0.2, float("inf")], [0.4, float("nan"), 0.6]],
    )

    penalty = base_height_l2(
        env,
        target_height=0.5,
        sensor_cfg=SceneEntityCfg("height_scanner"),
    )

    torch.testing.assert_close(penalty, torch.tensor([0.16, 0.0]))


def test_base_height_l2_returns_zero_when_all_rays_miss():
    """The penalty is neutral when the terrain height cannot be observed."""
    env = _make_env(
        root_heights=[1.0, 2.0],
        ray_hit_heights=[[float("inf"), float("inf")], [float("nan"), float("inf")]],
    )

    penalty = base_height_l2(
        env,
        target_height=0.5,
        sensor_cfg=SceneEntityCfg("height_scanner"),
    )

    torch.testing.assert_close(penalty, torch.zeros(2))


def test_base_height_l2_does_not_modify_ray_hits():
    """Filtering invalid ray hits does not modify the shared sensor buffer."""
    env = _make_env(
        root_heights=[1.0],
        ray_hit_heights=[[0.0, float("nan"), float("inf")]],
    )
    ray_hits_w = env.scene["height_scanner"].data.ray_hits_w.torch
    original_ray_hits_w = ray_hits_w.clone()

    base_height_l2(
        env,
        target_height=0.5,
        sensor_cfg=SceneEntityCfg("height_scanner"),
    )

    torch.testing.assert_close(ray_hits_w, original_ray_hits_w, equal_nan=True)
