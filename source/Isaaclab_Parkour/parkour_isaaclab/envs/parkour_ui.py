from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.ui.base_env_window import BaseEnvWindow
import omni.kit.app
if TYPE_CHECKING:
    from parkour_isaaclab.envs.parkour_manager_based_env import ParkourManagerBasedEnv

class ParkourManagerBasedRLEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment.

    On top of the basic environment window, this class adds controls for the RL environment.
    This includes visualization of the command manager.
    """

    def __init__(self, env: ParkourManagerBasedEnv, window_name: str = "IsaacLab"):
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
                    self._visualize_manager(title="Parkour", class_name="parkour_manager")

