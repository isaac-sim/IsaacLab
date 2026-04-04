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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class EmptyManagerCfg:
    """Empty manager specifications for the environment."""

    pass


@configclass
class EmptyObservationWithHistoryCfg:
    """Empty observation with history specifications for the environment."""

    @configclass
    class EmptyObservationGroupWithHistoryCfg(ObsGroup):
        """Empty observation with history specifications for the environment."""

        dummy_term: ObsTerm = ObsTerm(func=lambda env: torch.randn(env.num_envs, 1, device=env.device))

        def __post_init__(self):
            self.history_length = 5

    empty_observation: EmptyObservationGroupWithHistoryCfg = EmptyObservationGroupWithHistoryCfg()


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene."""

    pass


def get_empty_base_env_cfg(device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0):
    """Generate base environment config based on device"""

    @configclass
    class EmptyEnvCfg(ManagerBasedEnvCfg):
        """Configuration for the empty test environment."""

        # Scene settings
        scene: EmptySceneCfg = EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        # Basic settings
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()

        def __post_init__(self):
            """Post initialization."""
            # step settings
            self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
            # simulation settings
            self.sim.dt = 0.005  # sim step every 5ms: 200Hz
            self.sim.render_interval = self.decimation  # render every 4 sim steps
            # pass device down from test
            self.sim.device = device

    return EmptyEnvCfg()


def get_empty_base_env_cfg_with_history(device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0):
    """Generate base environment config based on device"""

    @configclass
    class EmptyEnvWithHistoryCfg(ManagerBasedEnvCfg):
        """Configuration for the empty test environment."""

        # Scene settings
        scene: EmptySceneCfg = EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        # Basic settings
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyObservationWithHistoryCfg = EmptyObservationWithHistoryCfg()

        def __post_init__(self):
            """Post initialization."""
            # step settings
            self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
            # simulation settings
            self.sim.dt = 0.005  # sim step every 5ms: 200Hz
            self.sim.render_interval = self.decimation  # render every 4 sim steps
            # pass device down from test
            self.sim.device = device

    return EmptyEnvWithHistoryCfg()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_initialization(device):
    """Test initialization of ManagerBasedEnv."""
    # create a new stage
    sim_utils.create_new_stage()
    # create environment
    env = ManagerBasedEnv(cfg=get_empty_base_env_cfg(device=device))
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
def test_observation_history_changes_only_after_step(device):
    """Test observation history of ManagerBasedEnv.

    The history buffer should only change after a step is taken.
    """
    # create a new stage
    sim_utils.create_new_stage()
    # create environment with history length of 5
    env = ManagerBasedEnv(cfg=get_empty_base_env_cfg_with_history(device=device))

    # check if history buffer is empty
    for group_name in env.observation_manager._group_obs_term_names:
        group_term_names = env.observation_manager._group_obs_term_names[group_name]
        for term_name in group_term_names:
            torch.testing.assert_close(
                env.observation_manager._group_obs_term_history_buffer[group_name][term_name].current_length,
                torch.zeros((env.num_envs,), device=device, dtype=torch.int64),
            )

    # check if history buffer is empty after compute
    env.observation_manager.compute()
    for group_name in env.observation_manager._group_obs_term_names:
        group_term_names = env.observation_manager._group_obs_term_names[group_name]
        for term_name in group_term_names:
            torch.testing.assert_close(
                env.observation_manager._group_obs_term_history_buffer[group_name][term_name].current_length,
                torch.zeros((env.num_envs,), device=device, dtype=torch.int64),
            )

    # check if history buffer is not empty after step
    act = torch.randn_like(env.action_manager.action)
    env.step(act)
    group_obs = dict()
    for group_name in env.observation_manager._group_obs_term_names:
        group_term_names = env.observation_manager._group_obs_term_names[group_name]
        group_obs[group_name] = dict()
        for term_name in group_term_names:
            torch.testing.assert_close(
                env.observation_manager._group_obs_term_history_buffer[group_name][term_name].current_length,
                torch.ones((env.num_envs,), device=device, dtype=torch.int64),
            )
            group_obs[group_name][term_name] = env.observation_manager._group_obs_term_history_buffer[group_name][
                term_name
            ].buffer

    # check if history buffer is not empty after compute and is the same as the buffer after step
    env.observation_manager.compute()
    for group_name in env.observation_manager._group_obs_term_names:
        group_term_names = env.observation_manager._group_obs_term_names[group_name]
        for term_name in group_term_names:
            torch.testing.assert_close(
                env.observation_manager._group_obs_term_history_buffer[group_name][term_name].current_length,
                torch.ones((env.num_envs,), device=device, dtype=torch.int64),
            )
            assert torch.allclose(
                group_obs[group_name][term_name],
                env.observation_manager._group_obs_term_history_buffer[group_name][term_name].buffer,
            )

    # close the environment
    env.close()
