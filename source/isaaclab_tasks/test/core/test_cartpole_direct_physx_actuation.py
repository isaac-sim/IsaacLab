# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test PhysX actuation in the direct-workflow Cartpole environment."""

from isaaclab.app import AppLauncher

# launch the simulator
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import gymnasium as gym
import pytest
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


@pytest.mark.isaacsim_ci
def test_cartpole_direct_physx_actuation_is_consistent_across_environments():
    """Verify that cloned PhysX carts respond consistently to a constant effort."""
    cfg = resolve_presets(CartpoleEnvCfg(), ("physx",))
    cfg.scene.num_envs = 4
    cfg.sim.device = "cuda:0"
    cfg.seed = 0
    cfg.initial_cart_position_range = (0.0, 0.0)
    cfg.initial_cart_velocity_range = (0.0, 0.0)
    cfg.initial_pole_angle_range = (0.0, 0.0)
    cfg.initial_pole_velocity_range = (0.0, 0.0)

    env = gym.make("Isaac-Cartpole-Direct", cfg=cfg)
    try:
        env.reset()
        actions = torch.ones(env.action_space.shape, device=env.unwrapped.device)
        with torch.inference_mode():
            for _ in range(20):
                env.step(actions)

        cart_idx = env.unwrapped._cart_dof_idx[0]
        cart_vel = env.unwrapped.cartpole.data.joint_vel.torch[:, cart_idx]
        assert torch.all(cart_vel > 1.0), f"Expected every cart to accelerate, got velocities {cart_vel.tolist()}."
        torch.testing.assert_close(cart_vel, cart_vel[0].expand_as(cart_vel))
    finally:
        env.close()
