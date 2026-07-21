# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

from typing import ClassVar

import pytest

import omni.kit.app

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.test.env_cfgs import make_empty_manager_based_rl_env_cfg

pytestmark = pytest.mark.integration

# The minimal app does not include ``isaacsim.core.experimental``, so enable the
# GUI dependency directly through Kit after AppLauncher has started.
extension_manager = omni.kit.app.get_app().get_extension_manager()
if not extension_manager.is_extension_enabled("isaacsim.gui.components"):
    extension_manager.set_extension_enabled_immediate("isaacsim.gui.components", True)

_HAS_GUI_SETTING = "/isaaclab/has_gui"


class TrackingManagerBasedRLEnvWindow(ManagerBasedRLEnvWindow):
    """Manager-based RL window that tracks successful construction."""

    construction_count: ClassVar[int] = 0

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "IsaacLab") -> None:
        super().__init__(env, window_name=window_name)
        type(self).construction_count += 1


def make_empty_manager_based_rl_env_ui_cfg(
    device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0
) -> ManagerBasedRLEnvCfg:
    """Create an empty reinforcement-learning environment configuration with a UI window."""
    cfg = make_empty_manager_based_rl_env_cfg(device=device, num_envs=num_envs, env_spacing=env_spacing)
    cfg.ui_window_class_type = TrackingManagerBasedRLEnvWindow
    return cfg


@pytest.mark.parametrize("has_gui", [False, True])
def test_ui_window_follows_has_gui_gate(has_gui: bool):
    """Create the RL environment window only when the GUI setting is enabled."""
    settings = get_settings_manager()
    previous_has_gui = bool(settings.get(_HAS_GUI_SETTING, False))
    env = None
    TrackingManagerBasedRLEnvWindow.construction_count = 0

    try:
        settings.set_bool(_HAS_GUI_SETTING, has_gui)
        sim_utils.create_new_stage()
        env = ManagerBasedRLEnv(cfg=make_empty_manager_based_rl_env_ui_cfg())

        assert env.sim.has_gui is has_gui
        assert TrackingManagerBasedRLEnvWindow.construction_count == int(has_gui)
    finally:
        if env is not None:
            env.close()
        settings.set_bool(_HAS_GUI_SETTING, previous_has_gui)
