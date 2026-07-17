# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.test.env_cfgs import make_empty_manager_based_env_cfg
from isaaclab.utils.configclass import configclass

pytestmark = pytest.mark.integration


def dummy_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Return a dummy observation."""
    return torch.randn((env.num_envs, 1), device=env.device)


@configclass
class EmptyObservationWithHistoryCfg:
    """Empty observation with history specifications for the environment."""

    @configclass
    class EmptyObservationGroupWithHistoryCfg(ObsGroup):
        """Empty observation with history specifications for the environment."""

        dummy_term: ObsTerm = ObsTerm(func=dummy_observation)

        def __post_init__(self):
            self.history_length = 5

    empty_observation: EmptyObservationGroupWithHistoryCfg = EmptyObservationGroupWithHistoryCfg()


def make_empty_manager_based_env_with_history_cfg(
    device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0
) -> ManagerBasedEnvCfg:
    """Create an empty environment configuration with observation history."""
    cfg = make_empty_manager_based_env_cfg(device=device, num_envs=num_envs, env_spacing=env_spacing)
    cfg.observations = EmptyObservationWithHistoryCfg()
    return cfg


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_initialization(device):
    """Test initialization of ManagerBasedEnv."""
    # create a new stage
    sim_utils.create_new_stage()
    # create environment
    env = ManagerBasedEnv(cfg=make_empty_manager_based_env_cfg(device=device))
    # check size of action manager terms
    assert env.action_manager.total_action_dim == 0
    assert len(env.action_manager.active_terms) == 0
    assert len(env.action_manager.action_term_dim) == 0
    # check size of observation manager terms
    assert len(env.observation_manager.active_terms) == 0
    assert len(env.observation_manager.group_obs_dim) == 0
    assert len(env.observation_manager.group_obs_term_dim) == 0
    assert len(env.observation_manager.group_obs_concatenate) == 0
    # create actions of correct size (1,0)
    act = torch.randn_like(env.action_manager.action)
    # step environment to verify setup
    for _ in range(2):
        obs, ext = env.step(action=act)
    # close the environment
    env.close()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_step_updates_observation_history(device):
    """Test that a real environment step advances observation history."""
    # create a new stage
    sim_utils.create_new_stage()
    # create environment with history length of 5
    env = ManagerBasedEnv(cfg=make_empty_manager_based_env_with_history_cfg(device=device))
    history = env.observation_manager._group_obs_term_history_buffer["empty_observation"]["dummy_term"]

    torch.testing.assert_close(
        history.current_length,
        torch.zeros((env.num_envs,), device=device, dtype=torch.int64),
    )

    # step the environment and verify that history advances
    env.step(torch.randn_like(env.action_manager.action))
    torch.testing.assert_close(
        history.current_length,
        torch.ones((env.num_envs,), device=device, dtype=torch.int64),
    )

    env.close()
