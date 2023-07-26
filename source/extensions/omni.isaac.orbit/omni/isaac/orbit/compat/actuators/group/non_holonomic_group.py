# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from omni.isaac.core.articulations import ArticulationView

from omni.isaac.orbit.utils.math import euler_xyz_from_quat

from .actuator_group import ActuatorGroup
from .actuator_group_cfg import NonHolonomicKinematicsGroupCfg


class NonHolonomicKinematicsGroup(ActuatorGroup):
    r"""
    An actuator group that formulates the 2D-base constraint for vehicle kinematics control.

    In simulation, it is easier to consider the mobile base as a floating link controlled by three dummy joints
    (prismatic joints along x and y, and revolute joint along z), in comparison to simulating wheels which is at times
    is tricky because of friction settings. Thus, this class implements the non-holonomic kinematic constraint for
    translating the input velocity command into joint velocity commands.

    A skid-steering base is under-actuated, i.e. the commands are forward velocity :math:`v_{B,x}` and the turning rate
    :\omega_{B,z}: in the base frame. Using the current base orientation, the commands are transformed into dummy
    joint velocity targets as:

    .. math::

        \dot{q}_{0, des} &= v_{B,x} \cos(\theta) \\
        \dot{q}_{1, des} &= v_{B,x} \sin(\theta) \\
        \dot{q}_{2, des} &= \omega_{B,z}

    where :math:`\theta` is the yaw of the 2-D base. Since the base is simulated as a dummy joint, the yaw is directly
    the value of the revolute joint along z, i.e., :math:`q_2 = \theta`.

    Tip:
        For velocity control of the base with dummy mechanism, we recommend setting high damping gains to the joints.
        This ensures that the base remains unperturbed from external disturbances, such as an arm mounted on the base.

    """

    cfg: NonHolonomicKinematicsGroupCfg
    """The configuration of the actuator group."""

    def __init__(self, cfg: NonHolonomicKinematicsGroupCfg, view: ArticulationView):
        """Initialize the actuator group.

        Args:
            cfg (NonHolonomicKinematicsGroupCfg): The configuration of the actuator group.
            view (ArticulationView): The simulation articulation view.

        Raises:
            ValueError: When command type is not "v_abs".
            RuntimeError: When the articulation view is not initialized.
            ValueError: When not able to find a match for all DOF names in the configuration.
            ValueError: When the actuator model configuration is invalid, i.e. not "explicit" or "implicit".
        """
        # check valid config
        if cfg.control_cfg.command_types[0] not in ["v_abs"] or len(cfg.control_cfg.command_types) > 1:
            raise ValueError(
                f"Non-holonomic kinematics group does not support command types: '{cfg.control_cfg.command_types}'."
            )
        # initialize parent
        super().__init__(cfg, view)
        # check that the encapsulated joints are three
        if self.num_actuators != 3:
            raise ValueError(f"Non-holonomic kinematics group requires three joints, but got {self.num_actuators}.")

    """
    Properties
    """

    @property
    def control_dim(self) -> int:
        """Dimension of control actions."""
        return 2

    """
    Operations.
    """

    def _format_command(self, command: torch.Tensor) -> torch.Tensor:
        """Pre-processing of commands given to actuators.

        The input command is the base velocity and turning rate command.

        Returns:
            torch.Tensor: The target joint commands for the gripper.
        """
        # obtain current heading
        quat_w = self.view.get_world_poses(clone=False)[1]
        yaw_w = euler_xyz_from_quat(quat_w)[2]
        # compute new command
        dof_vel_targets = torch.zeros(self.num_articulation, 3, device=self.device)
        dof_vel_targets[:, 0] = torch.cos(yaw_w) * command[:, 0]
        dof_vel_targets[:, 1] = torch.sin(yaw_w) * command[:, 0]
        dof_vel_targets[:, 2] = command[:, 1]
        # return command
        return dof_vel_targets
