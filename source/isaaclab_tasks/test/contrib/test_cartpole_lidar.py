# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Cartpole lidar tasks return one tiled scan per environment."""

import pytest
import torch

from isaaclab.test.utils import launch_test_simulation, resolve_test_sim_device

_DEVICE = resolve_test_sim_device()
if not str(_DEVICE).startswith("cuda") or not torch.cuda.is_available():
    pytest.skip("tiled lidar rendering requires CUDA", allow_module_level=True)


def _launch_cfg():
    """Launch Newton and OVRTX before the task imports the runtime."""
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg

    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass

    @configclass
    class _Cfg:
        sim: SimulationCfg = SimulationCfg(physics=NewtonCfg())
        renderer: OVRTXRendererCfg = OVRTXRendererCfg(motion_bvh="auto")

    return _Cfg()


launch_test_simulation(_launch_cfg(), device=_DEVICE)

import gymnasium as gym

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.cartpole_lidar.cartpole_lidar_common import (
    CARTPOLE_LIDAR_AZIMUTH_COUNT,
    CARTPOLE_LIDAR_ELEVATION_COUNT,
)
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.ov]


@pytest.mark.parametrize("task", ("IsaacContrib-Cartpole-Lidar", "IsaacContrib-Cartpole-Lidar-Direct"))
def test_tiled_lidar_returns_are_local_to_each_environment(task):
    """Sensor-frame scans agree across environments, and the policy is a stacked range image."""
    env_cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"))
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = _DEVICE
    env_cfg.seed = 0

    sim_utils.create_new_stage()
    env = None
    try:
        env = gym.make(task, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None
        obs, _ = env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        reward = None
        with torch.inference_mode():
            for _ in range(8):
                obs, reward, _, _, _ = env.step(actions)

        scene = env.unwrapped.scene
        points = scene["lidar"].data.points.torch
        valid = scene["lidar"].data.valid.torch
        assert points.shape == (4, valid.shape[1], 3)
        assert valid.shape == (4, points.shape[1])
        assert points.shape[1] > 1
        assert torch.isfinite(reward).all()
        policy = obs["policy"]
        assert torch.isfinite(policy).all()
        assert policy.shape == (4, 2, CARTPOLE_LIDAR_ELEVATION_COUNT, CARTPOLE_LIDAR_AZIMUTH_COUNT)
        assert float(policy.min()) >= 0.0
        assert float(policy.max()) <= 1.0
        assert float(policy.max()) > 0.2
        assert obs["critic"].shape[-1] == 4
        assert env.unwrapped.single_observation_space["policy"].shape == policy.shape[1:]
        assert env.unwrapped.single_observation_space["critic"].shape[-1] == 4

        centers = []
        for index in range(points.shape[0]):
            hits = points[index][valid[index]]
            assert hits.numel() > 0, f"environment {index} returned no lidar hits"
            assert torch.isfinite(hits).all()
            centers.append(hits.mean(dim=0))
        centers = torch.stack(centers)
        # Sensor-frame scans of the same cart agree. World-frame scans sit on origins 20 m apart.
        spread = torch.linalg.norm(centers - centers.mean(dim=0), dim=-1).max()
        assert spread < 5.0, f"sensor-frame hit centers differ by {float(spread):.2f} m"
        assert "tiled_camera" not in scene.sensors

        kept = points[1].clone()
        kept_valid = valid[1].clone()
        scene["lidar"].reset(env_ids=[0])
        reset_points = scene["lidar"].data.points.torch
        reset_valid = scene["lidar"].data.valid.torch
        assert not reset_valid[0].any()
        assert not torch.isfinite(reset_points[0]).any()
        torch.testing.assert_close(reset_points[1], kept, equal_nan=True)
        assert torch.equal(reset_valid[1], kept_valid)
    finally:
        if env is not None:
            env.close()
