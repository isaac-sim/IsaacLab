# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import carb
import omni.usd
from isaacsim.core.utils.extensions import enable_extension

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

enable_extension("isaacsim.gui.components")


@configclass
class EmptyManagerCfg:
    """Empty manager specifications for the environment."""

    pass


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene."""

    pass


def get_empty_base_env_cfg(device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0):
    """Generate base environment config based on device"""

    @configclass
    class EmptyEnvCfg(ManagerBasedRLEnvCfg):
        """Configuration for the empty test environment."""

        # Scene settings
        scene: EmptySceneCfg = EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        # Basic settings
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()
        rewards: EmptyManagerCfg = EmptyManagerCfg()
        terminations: EmptyManagerCfg = EmptyManagerCfg()
        # Define window
        ui_window_class_type: type[ManagerBasedRLEnvWindow] = ManagerBasedRLEnvWindow

        def __post_init__(self):
            """Post initialization."""
            # step settings
            self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
            # simulation settings
            self.sim.dt = 0.005  # sim step every 5ms: 200Hz
            self.sim.render_interval = self.decimation  # render every 4 sim steps
            # pass device down from test
            self.sim.device = device
            # episode length
            self.episode_length_s = 5.0

    return EmptyEnvCfg()


def test_ui_window():
    """Test UI window of ManagerBasedRLEnv."""
    device = "cuda:0"
    # override sim setting to enable UI
    carb.settings.get_settings().set_bool("/app/window/enabled", True)
    # create a new stage
    omni.usd.get_context().new_stage()
    # create environment
    env = ManagerBasedRLEnv(cfg=get_empty_base_env_cfg(device=device))
    # close the environment
    env.close()
