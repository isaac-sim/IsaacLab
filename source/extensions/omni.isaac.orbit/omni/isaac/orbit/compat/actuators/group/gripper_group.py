# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import torch
from typing import Sequence

from omni.isaac.core.articulations import ArticulationView

from .actuator_group import ActuatorGroup
from .actuator_group_cfg import GripperActuatorGroupCfg


class GripperActuatorGroup(ActuatorGroup):
    """
    A mimicking actuator group to format a binary open/close command into the joint commands.

    The input actions are processed as a scalar which sign determines whether to open or close the gripper.
    We consider the following convention:

    1. Positive value (> 0): open command
    2. Negative value (< 0): close command

    The mimicking actuator group has only two valid command types: absolute positions (``"p_abs"``) or absolute
    velocity (``"v_abs"``). Based on the chosen command type, the joint commands are computed by multiplying the
    reference command with the mimicking multiplier.

    * **position mode:** The reference command is resolved as the joint position target for opening or closing the
      gripper. These targets are read from the :class:`GripperActuatorGroupCfg` class.
    * **velocity mode:** The reference command is resolved as the joint velocity target based on the configured speed
      of the gripper. The reference commands are added to the previous command and clipped to the range of (-1.0, 1.0).

    Tip:
        In general, we recommend using the velocity mode, as it simulates the delay in opening or closing the gripper.

    """

    cfg: GripperActuatorGroupCfg
    """The configuration of the actuator group."""

    def __init__(self, cfg: GripperActuatorGroupCfg, view: ArticulationView):
        """Initialize the actuator group.

        Args:
            cfg (GripperActuatorGroupCfg): The configuration of the actuator group.
            view (ArticulationView): The simulation articulation view.

        Raises:
            ValueError: When command type is not "p_abs" or "v_abs".
            RuntimeError: When the articulation view is not initialized.
            ValueError: When not able to find a match for all DOF names in the configuration.
            ValueError: When the actuator model configuration is invalid, i.e. not "explicit" or "implicit".
        """
        # check valid config
        if cfg.control_cfg.command_types[0] not in ["p_abs", "v_abs"] or len(cfg.control_cfg.command_types) > 1:
            raise ValueError(f"Gripper mimicking does not support command types: '{cfg.control_cfg.command_types}'.")
        # initialize parent
        super().__init__(cfg, view)

        # --from actuator dof names
        self._mimic_multiplier = torch.ones(self.num_articulation, self.num_actuators, device=self.device)
        for index, dof_name in enumerate(self.dof_names):
            # multiplier command
            for re_key, value in self.cfg.mimic_multiplier.items():
                if re.fullmatch(re_key, dof_name):
                    self._mimic_multiplier[:, index] = value
        # -- dof positions
        self._open_dof_pos = self._mimic_multiplier * self.cfg.open_dof_pos
        self._close_dof_pos = self._mimic_multiplier * self.cfg.close_dof_pos

        # create buffers
        self._previous_dof_targets = torch.zeros(self.num_articulation, self.num_actuators, device=self.device)
        # constants
        self._ALL_INDICES = torch.arange(self.view.count, device=self.view._device, dtype=torch.long)

    def __str__(self) -> str:
        """String representation of the actuator group."""
        msg = super().__str__() + "\n"
        msg = msg.replace("ActuatorGroup", "GripperActuatorGroup")
        msg += (
            f"\tNumber of DOFs: {self.num_actuators}\n"
            f"\tMimic multiply: {self._mimic_multiplier}\n"
            f"\tOpen position : {self.cfg.open_dof_pos}\n"
            f"\tClose position: {self.cfg.close_dof_pos}"
        )
        return msg

    """
    Properties
    """

    @property
    def control_dim(self) -> int:
        """Dimension of control actions."""
        return 1

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int]):
        # reset super
        super().reset(env_ids=env_ids)
        # buffers
        self._previous_dof_targets[env_ids] = 0.0

    def _format_command(self, command: torch.Tensor) -> torch.Tensor:
        """Pre-processing of the commands given to actuators.

        We consider the following convention:

        * Non-negative command (includes 0): open grippers
        * Negative command: close grippers

        Returns:
            torch.Tensor: The target joint commands for the gripper.
        """
        # FIXME: mimic joint positions -- Gazebo plugin seems to do this.
        # The following is commented out because Isaac Sim doesn't support setting joint positions
        # of particular dof indices properly. It sets joint positions and joint position targets for
        # the whole robot, i.e. all previous position targets is lost. This affects resetting the robot
        # to a particular joint position.
        # self.view._physics_sim_view.enable_warnings(False)
        # # get current joint positions
        # new_dof_pos = self.view._physics_view.get_dof_positions()
        # # set joint positions of the mimic joints
        # new_dof_pos[:, self.dof_indices] = self._dof_pos[:, 0].unsqueeze(1) * self._mimic_multiplier
        # # set joint positions to the physics view
        # self.view.set_joint_positions(new_dof_pos, self._ALL_INDICES)
        # self.view._physics_sim_view.enable_warnings(True)

        # process actions
        if self.control_mode == "velocity":
            # compute new command
            dof_vel_targets = torch.sign(command) * self.cfg.speed * self._mimic_multiplier
            dof_vel_targets = self._previous_dof_targets[:] + dof_vel_targets
            dof_vel_targets = torch.clamp(dof_vel_targets, -1.0, 1.0)
            # store new command
            self._previous_dof_targets[:] = dof_vel_targets
            # return command
            return dof_vel_targets
        else:
            # compute new command
            dof_pos_targets = torch.where(command >= 0, self._open_dof_pos, self._close_dof_pos)
            # store new command
            self._previous_dof_targets[:] = dof_pos_targets
            # return command
            return dof_pos_targets
