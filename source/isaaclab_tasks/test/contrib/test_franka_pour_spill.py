# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spilled media falls to the ground and remains classified as spilled."""

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourResetDatasetEnvCfg

ENV_CFG = FrankaPourResetDatasetEnvCfg()
ENV_CFG.play_mode()
ENV_CFG.scene.num_envs = 4
launch_test_simulation(ENV_CFG)

import newton
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonMPMManager

from isaaclab.utils import clone

from isaaclab_tasks.contrib.franka_pour.mdp.observations import particle_fractions_obs, particle_transfer_obs
from isaaclab_tasks.contrib.franka_pour.mdp.terminations import excessive_spill, particle_out_of_bounds
from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_spilled_particles_land_on_ground_without_escaping_workspace(device):
    """An outside-cup pile settles on the floor, not an invisible tabletop shelf."""
    cfg = clone(ENV_CFG)
    cfg.sim.device = device
    env = FrankaPourEnv(cfg)
    try:
        env.reset()
        points = env._media.data.particle_pos_w.torch.clone() - env.env_origins[:, None]
        points[..., :2] -= points[..., :2].mean(dim=1, keepdim=True)
        points[..., :2] += torch.tensor((-0.4, 0.6), device=device)
        points[..., 2] -= points[..., 2].amin(dim=1, keepdim=True)
        points[..., 2] += 0.035
        env_ids = torch.arange(env.num_envs, device=device)
        env._media.write_particle_pos_to_sim_index(points + env.env_origins[:, None], env_ids=env_ids)
        env._media.write_particle_velocity_to_sim_index(torch.zeros_like(points), env_ids=env_ids)
        world_mask = torch.ones(env.num_envs + 1, dtype=torch.bool, device=device)
        world_mask[-1] = False
        NewtonMPMManager.reset_solver_state(
            world_mask=wp.from_torch(world_mask, dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )
        # Step physics without automatic episode resets so the spill can finish falling.
        for _ in range(120):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(cfg.sim.dt)
            env.common_step_counter += 1

        heights = env.particle_pos_e()[..., 2]
        ground_height = cfg.scene.plane.init_state.pos[2]
        assert torch.all(heights > ground_height - 0.005)
        assert torch.all(heights < ground_height + 0.05)
        assert excessive_spill(env).all()
        assert not particle_out_of_bounds(env).any()
        assert (particle_fractions_obs(env)[:, -1] == 1.0).all()
        assert torch.count_nonzero(particle_transfer_obs(env)[:, -1]) == 0
    finally:
        env.close()
