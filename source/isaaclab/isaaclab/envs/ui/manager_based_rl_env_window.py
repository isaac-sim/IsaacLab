# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from .base_env_window import BaseEnvWindow

if TYPE_CHECKING:
    from ..manager_based_rl_env import ManagerBasedRLEnv


class ManagerBasedRLEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment.

    On top of the basic environment window, this class adds controls for the RL environment.
    This includes visualization of the command manager.
    """

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)

        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._visualize_manager(title="Commands", class_name="command_manager")
                    self._visualize_manager(title="Rewards", class_name="reward_manager")
                    self._visualize_manager(title="Curriculum", class_name="curriculum_manager")
                    self._visualize_manager(title="Termination", class_name="termination_manager")
