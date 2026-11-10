# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The cartpole lidar preset returns one tiled scan per environment."""

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
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.ov]


@pytest.mark.parametrize("task", ("Isaac-Cartpole-Camera", "Isaac-Cartpole-Camera-Direct"))
def test_tiled_lidar_returns_are_local_to_each_environment(task):
    """Each environment's points land on its own cart, and the shared step still returns the camera."""
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, {"lidar"})
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
        origins = scene.env_origins.to(device=points.device, dtype=points.dtype)
        assert points.shape == (4, valid.shape[1], 3)
        assert valid.shape == (4, points.shape[1])
        assert points.shape[1] > 1
        assert torch.isfinite(reward).all()
        assert torch.isfinite(obs["policy"]).all()
        assert obs["policy"].shape[-1] == 4 + points.shape[1] * 3
        assert obs["critic"].shape[-1] == 4
        assert env.unwrapped.single_observation_space["policy"].shape[-1] == obs["policy"].shape[-1]
        assert env.unwrapped.single_observation_space["critic"].shape[-1] == 4

        for index in range(points.shape[0]):
            hits = points[index][valid[index]]
            assert hits.numel() > 0, f"environment {index} returned no lidar hits"
            assert torch.isfinite(hits).all()
            center = hits.mean(dim=0)
            own = torch.linalg.norm(center - origins[index])
            other = torch.linalg.norm(center.unsqueeze(0) - origins, dim=-1)
            other[index] = float("inf")
            assert own < other.min(), f"environment {index} hits are closer to another environment"

        image = scene["tiled_camera"].data.output["rgb"].torch
        assert image.shape[0] == 4 and image.shape[-1] == 3
        assert image.shape[1] > 1 and image.shape[2] > 1
        assert int(image.abs().sum()) > 0
    finally:
        if env is not None:
            env.close()
