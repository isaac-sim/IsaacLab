# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.assets.surface_gripper import SurfaceGripper
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class SurfaceGripperBinaryAction(ActionTerm):
    """Surface gripper binary action.

    This action term maps a binary action to the *open* or *close* surface gripper configurations.
    The surface gripper behavior is as follows:
    - [-1, -0.3] --> Gripper is Opening
    - [-0.3, 0.3] --> Gripper is Idle (do nothing)
    - [0.3, 1] --> Gripper is Closing

    Based on above, we follow the following convention for the binary action:

    1. Open action: 1 (bool) or positive values (float).
    2. Close action: 0 (bool) or negative values (float).

    The action term is specifically designed for surface grippers, which use a different
    interface than joint-based grippers.
    """

    cfg: actions_cfg.SurfaceGripperBinaryActionCfg
    """The configuration of the action term."""
    _asset: SurfaceGripper
    """The surface gripper asset on which the action term is applied."""

    def __init__(self, cfg: actions_cfg.SurfaceGripperBinaryActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # log the resolved asset name for debugging
        omni.log.info(
            f"Resolved surface gripper asset for the action term {self.__class__.__name__}: {self.cfg.asset_name}"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 1, device=self.device)

        # parse open command
        self._open_command = torch.tensor(self.cfg.open_command, device=self.device)
        # parse close command
        self._close_command = torch.tensor(self.cfg.close_command, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions < 0
        # compute the command
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)

    def apply_actions(self):
        """Apply the processed actions to the surface gripper."""
        self._asset.set_grippers_command(self._processed_actions.view(-1))
        self._asset.write_data_to_sim()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
