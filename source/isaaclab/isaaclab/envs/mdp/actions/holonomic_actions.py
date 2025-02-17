from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class HolonomicBaseAction(ActionTerm):
    """Action term for holonomic base control using velocity commands.

    Takes a 3D action vector representing:
    - x velocity in world frame
    - y velocity in world frame
    - yaw angular velocity
    """

    cfg: actions_cfg.HolonomicBaseActionCfg

    _asset: Articulation

    def __init__(self, cfg: actions_cfg.HolonomicBaseActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Get joint IDs
        x_joint_id, x_joint_name = self._asset.find_joints(self.cfg.x_joint_name)
        y_joint_id, y_joint_name = self._asset.find_joints(self.cfg.y_joint_name)
        yaw_joint_id, yaw_joint_name = self._asset.find_joints(self.cfg.yaw_joint_name)

        # Validate joints
        for joint_id, name, joint_type in [
            (x_joint_id, "x", self.cfg.x_joint_name),
            (y_joint_id, "y", self.cfg.y_joint_name),
            (yaw_joint_id, "yaw", self.cfg.yaw_joint_name),
        ]:
            if len(joint_id) != 1:
                raise ValueError(f"Expected single joint match for {name} joint name: {joint_type}")

        # Store joint IDs and names
        self._joint_ids = [x_joint_id[0], y_joint_id[0], yaw_joint_id[0]]
        self._joint_names = [x_joint_name[0], y_joint_name[0], yaw_joint_name[0]]

        # Debug: Log which joints are used
        print(
            f"[HolonomicBaseAction] Initialized with joints: X -> {self._joint_names[0]}, Y -> {self._joint_names[1]}, Yaw -> {self._joint_names[2]}"
        )

        # Initialize tensors
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Scale and offset as tensors
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions with scaling and offset."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale + self._offset
        # Debug: Log raw and processed actions
        print(f"[HolonomicBaseAction] Raw actions: {self._raw_actions}")
        print(f"[HolonomicBaseAction] Processed actions: {self._processed_actions}")

    def apply_actions(self):
        """Apply velocity commands directly to the joints."""
        # Debug: Log before applying joint commands
        print(f"[HolonomicBaseAction] Applying commands: {self._processed_actions} to joints: {self._joint_ids}")
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset actions to zero."""
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
