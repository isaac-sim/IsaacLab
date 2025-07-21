# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test texture randomization in the cartpole scene using pytest."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import numpy as np

import omni.usd
import pytest

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_camera_env_cfg import (
    CartpoleDepthCameraEnvCfg,
    CartpoleRGBCameraEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg


@pytest.mark.parametrize(
    "env_cfg_cls",
    [CartpoleRGBCameraEnvCfg, CartpoleDepthCameraEnvCfg, AnymalCRoughEnvCfg],
    ids=["RGB", "Depth", "RayCaster"],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_obs_space_follows_clip_contraint(env_cfg_cls, device):
    """Ensure curriculum terms apply correctly after the fallback and replacement."""
    # new USD stage
    omni.usd.get_context().new_stage()

    # configure the cartpole env
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = 2  # keep num_envs small for testing
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.sim.device = device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    for group_name, group_space in env.observation_space.spaces.items():
        for term_name, term_space in group_space.spaces.items():
            term_cfg = getattr(getattr(env_cfg.observations, group_name), term_name)
            low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
            high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
            assert isinstance(
                term_space, gym.spaces.Box
            ), f"Expected Box space for {term_name} in {group_name}, got {type(term_space)}"
            assert np.all(term_space.low == low)
            assert np.all(term_space.high == high)

    env.close()
