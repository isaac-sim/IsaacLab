# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preserve task-backed manager-based RL observation-space integration coverage.

This temporary relocation handoff intentionally remains in the core test tree until the
task-backed reset/step, camera, and ray-caster scenarios can move to the task package.
"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.utils import resolve_task_config

pytestmark = pytest.mark.integration


# Gym spaces are built on the host, so one device covers them. Camera image terms have no clip and share the
# unbounded branch that test_manager_based_rl_env_unit.py covers, so the ray-caster task with a clipped height
# scan is the task-backed row.
@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_obs_space_follows_clip_constraint(device):
    """Ensure observation space bounds reflect the clip constraint on each term, and that non-concatenated
    groups contain all their terms (issue #3133)."""
    # new USD stage
    sim_utils.create_new_stage()

    env_cfg, _ = resolve_task_config("IsaacContrib-Velocity-Rough-AnymalC", "", overrides=())
    env_cfg.scene.num_envs = 2  # keep num_envs small for testing
    for group_cfg in vars(env_cfg.observations).values():
        if isinstance(group_cfg, ObservationGroupCfg):
            group_cfg.concatenate_terms = False
    env_cfg.sim.device = device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        for group_name, group_space in env.observation_space.spaces.items():
            assert isinstance(group_space, gym.spaces.Dict)
            for term_name, term_space in group_space.spaces.items():
                term_cfg = getattr(getattr(env_cfg.observations, group_name), term_name)
                low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                assert isinstance(term_space, gym.spaces.Box), (
                    f"Expected Box space for {term_name} in {group_name}, got {type(term_space)}"
                )
                assert np.all(term_space.low == low)
                assert np.all(term_space.high == high)

        # every term of a non-concatenated group is present in its space and in the stepped observations
        expected_policy_terms = env.observation_manager.active_terms["policy"]
        assert len(expected_policy_terms) > 1
        # gymnasium sorts Dict space keys, so compare the term sets
        assert sorted(env.observation_space.spaces["policy"].spaces) == sorted(expected_policy_terms)
        env.reset()
        action = torch.tensor(env.action_space.sample(), device=env.device)
        obs, _, _, _, _ = env.step(action)
        assert list(obs["policy"]) == expected_policy_terms
    finally:
        env.close()
