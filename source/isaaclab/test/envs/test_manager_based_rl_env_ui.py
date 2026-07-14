# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest

from isaacsim.core.experimental.utils.app import enable_extension

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.test.env_cfgs import make_empty_manager_based_rl_env_cfg

pytestmark = pytest.mark.integration

enable_extension("isaacsim.gui.components")


def make_empty_manager_based_rl_env_ui_cfg(
    device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0
) -> ManagerBasedRLEnvCfg:
    """Create an empty reinforcement-learning environment configuration with a UI window."""
    cfg = make_empty_manager_based_rl_env_cfg(device=device, num_envs=num_envs, env_spacing=env_spacing)
    cfg.ui_window_class_type = ManagerBasedRLEnvWindow
    return cfg


def test_ui_window():
    """Test UI window of ManagerBasedRLEnv."""
    device = "cuda:0"
    # override sim setting to enable UI
    from isaaclab.app.settings_manager import get_settings_manager

    get_settings_manager().set_bool("/app/window/enabled", True)
    # create a new stage
    sim_utils.create_new_stage()
    # create environment
    env = ManagerBasedRLEnv(cfg=make_empty_manager_based_rl_env_ui_cfg(device=device))
    # close the environment
    env.close()
