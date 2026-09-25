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
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass

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


# Both devices, so the CPU and GPU simulation pipelines each build and step an environment.
@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_step_updates_observation_history(device):
    """Test that real environment steps advance observation history."""
    # create a new stage
    sim_utils.create_new_stage()
    # create environment with history length of 5
    env = ManagerBasedEnv(cfg=make_empty_manager_based_env_with_history_cfg(device=device, num_envs=3))
    assert env.action_manager.total_action_dim == 0
    history = env.observation_manager._group_obs_term_history_buffer["empty_observation"]["dummy_term"]

    torch.testing.assert_close(
        history.current_length,
        torch.zeros((env.num_envs,), device=device, dtype=torch.int64),
    )

    # step the environment repeatedly and verify that history advances with each step
    for num_steps in (1, 2):
        env.step(torch.randn_like(env.action_manager.action))
        torch.testing.assert_close(
            history.current_length,
            torch.full((env.num_envs,), num_steps, device=device, dtype=torch.int64),
        )

    # A slice must reach every manager as device indices, including observation history reset.
    env.reset(env_ids=slice(1, None, 2))
    torch.testing.assert_close(history.current_length, torch.tensor([3, 1, 3], device=device))
    env.reset_to({}, env_ids=slice(0, None, 2))
    torch.testing.assert_close(history.current_length, torch.tensor([1, 2, 1], device=device))
    for invalid in (None, [0, 2]):
        with pytest.raises(TypeError, match="env_ids"):
            env.reset(invalid)
        with pytest.raises(TypeError, match="env_ids"):
            env.reset(env_ids=invalid)
        with pytest.raises(TypeError, match="env_ids"):
            env.reset_to({}, env_ids=invalid)
    torch.testing.assert_close(history.current_length, torch.tensor([1, 2, 1], device=device))
    env.reset()
    torch.testing.assert_close(history.current_length, torch.ones_like(history.current_length))
    env.reset(slice(None), seed=42)
    env.reset_to({})
    torch.testing.assert_close(history.current_length, torch.ones_like(history.current_length))
    assert not hasattr(env.scene, "resolve_env_ids")
    env.close()
