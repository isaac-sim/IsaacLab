# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for root-state event terms that sample from range dictionaries."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp import push_by_setting_velocity, reset_root_state_uniform
from isaaclab.managers import EventTermCfg, SceneEntityCfg

pytestmark = pytest.mark.unit


class _Scene(dict):
    """Scene double exposing assets by name and the environment origins."""

    def __init__(self, asset, env_origins: torch.Tensor):
        super().__init__(robot=asset)
        self.env_origins = env_origins


def _make_env(device: str, num_envs: int = 3) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Create an environment double whose robot records the written root state."""
    default_root_pose = torch.zeros(num_envs, 7, device=device)
    default_root_pose[:, 2] = 0.4
    default_root_pose[:, 6] = 1.0
    asset = SimpleNamespace(
        device=device,
        data=SimpleNamespace(
            default_root_pose=SimpleNamespace(torch=default_root_pose),
            default_root_vel=SimpleNamespace(torch=torch.zeros(num_envs, 6, device=device)),
            root_vel_w=SimpleNamespace(torch=torch.zeros(num_envs, 6, device=device)),
        ),
        written={},
    )
    asset.write_root_pose_to_sim_index = lambda root_pose, env_ids: asset.written.update(pose=root_pose)
    asset.write_root_velocity_to_sim_index = lambda root_velocity, env_ids: asset.written.update(vel=root_velocity)
    env_origins = torch.arange(num_envs * 3, dtype=torch.float, device=device).reshape(num_envs, 3)
    env = SimpleNamespace(num_envs=num_envs, device=device, scene=_Scene(asset, env_origins))
    return env, asset


def test_reset_root_state_uniform_uses_call_ranges():
    """The term samples from the ranges passed at call time, not the ones present at construction."""
    device = "cpu"
    env, asset = _make_env(device)
    asset_cfg = SceneEntityCfg("robot")
    cfg = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": asset_cfg},
    )
    term = reset_root_state_uniform(cfg, env)
    env_ids = torch.arange(env.num_envs, device=device)

    term(env, env_ids, pose_range={"x": (1.0, 1.0), "z": (0.5, 0.5)}, velocity_range={"yaw": (2.0, 2.0)})

    expected_pos = env.scene.env_origins + torch.tensor([1.0, 0.0, 0.9], device=device)
    torch.testing.assert_close(asset.written["pose"][:, :3], expected_pos)
    torch.testing.assert_close(asset.written["pose"][:, 3:], asset.data.default_root_pose.torch[:, 3:])
    expected_vel = torch.zeros(env.num_envs, 6, device=device)
    expected_vel[:, 5] = 2.0
    torch.testing.assert_close(asset.written["vel"], expected_vel)


def test_push_by_setting_velocity_follows_range_edits():
    """Ranges are matched to components by name, and in-place range edits still take effect."""
    device = "cpu"
    env, asset = _make_env(device)
    env_ids = torch.arange(env.num_envs, device=device)
    # Dictionary insertion order differs from the command's component order.
    velocity_range = {"yaw": (-0.2, 0.8), "x": (-2.0, 1.0)}
    push_by_setting_velocity(env, env_ids, velocity_range)
    velocity = asset.written["vel"]
    assert ((velocity[:, 0] >= -2.0) & (velocity[:, 0] <= 1.0)).all()
    assert ((velocity[:, 5] >= -0.2) & (velocity[:, 5] <= 0.8)).all()
    assert (velocity[:, 1:5] == 0.0).all()

    # a curriculum may edit the range dictionary in place
    velocity_range["x"] = (-2.0, -2.0)
    push_by_setting_velocity(env, env_ids, velocity_range)
    torch.testing.assert_close(asset.written["vel"][:, 0], torch.full((env.num_envs,), -2.0, device=device))
