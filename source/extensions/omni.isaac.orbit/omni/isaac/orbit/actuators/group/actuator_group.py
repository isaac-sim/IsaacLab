# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import torch
from typing import List, Optional, Sequence

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationActions

from omni.isaac.orbit.actuators.group.actuator_group_cfg import ActuatorGroupCfg
from omni.isaac.orbit.actuators.model import *  # noqa: F403, F401
from omni.isaac.orbit.actuators.model import DCMotor, IdealActuator, ImplicitActuatorCfg


class ActuatorGroup:
    """A class for applying actuator models over a collection of actuated joints in an articulation.

    The default actuator group for applying the same actuator model over a collection of actuated joints in
    an articulation. It is possible to specify multiple joint-level command types (position, velocity or
    torque control).

    The joint names are specified in the configuration through a list of regular expressions. The regular
    expressions are matched against the joint names in the articulation. The first match is used to determine
    the joint indices in the articulation.

    The command types are applied in the order they are specified in the configuration. For each command, the
    scaling and offset can be configured through the :class:`ActuatorControlCfg` class.

    In the default actuator group, no constraints or formatting is performed over the input actions. Thus, the
    input actions are directly used to compute the joint actions in the :meth:`compute`.
    """

    cfg: ActuatorGroupCfg
    """The configuration of the actuator group."""
    view: ArticulationView
    """The simulation articulation view."""
    num_articulations: int
    """Number of articulations in the view."""
    device: str
    """Device used for processing."""
    dof_names: List[str]
    """Articulation's DOF names that are part of the group."""
    dof_indices: List[int]
    """Articulation's DOF indices that are part of the group."""
    model: Optional[IdealActuator]
    """Actuator model used by the group.

    If model type is "implicit" (i.e., when :obj:`ActuatorGroupCfg.model_cfg` is instance of
    :class:`ImplicitActuatorCfg`), then `model` is set to :obj:`None`.
    """
    dof_pos_offset: torch.Tensor
    """DOF position offsets used for processing commands."""
    dof_pos_scale: torch.Tensor
    """DOF position scale used for processing commands."""
    dof_vel_scale: torch.Tensor
    """DOF velocity scale  used for processing commands."""
    dof_torque_scale: torch.Tensor
    """DOF torque scale used for processing commands."""

    def __init__(self, cfg: ActuatorGroupCfg, view: ArticulationView):
        """Initialize the actuator group.

        Args:
            cfg (ActuatorGroupCfg): The configuration of the actuator group.
            view (ArticulationView): The simulation articulation view.

        Raises:
            ValueError: When no command types are specified in the configuration.
            RuntimeError: When the articulation view is not initialized.
            ValueError: When not able to find a match for all DOF names in the configuration.
            ValueError: When the actuator model configuration is invalid, i.e. not "explicit" or "implicit".
        """
        # check valid inputs
        if len(cfg.control_cfg.command_types) < 1:
            raise ValueError("Each actuator group must have at least one command type. Received: 0.")
        if not view.initialized:
            raise RuntimeError("Actuator group expects an initialized articulation view.")
        # save parameters
        self.cfg = cfg
        self.view = view
        # extract useful quantities
        self.num_articulation = self.view.count
        self.device = self.view._device

        # TODO (@mayank): Make regex resolving safer by throwing errors for keys that didn't find a match.
        # -- from articulation dof names
        self.dof_names = list()
        self.dof_indices = list()
        for dof_name in self.view.dof_names:
            # dof-names and indices
            for re_key in self.cfg.dof_names:
                if re.fullmatch(re_key, dof_name):
                    # add to group details
                    self.dof_names.append(dof_name)
                    self.dof_indices.append(self.view.get_dof_index(dof_name))
        # check that group is valid
        if len(self.dof_names) == 0:
            raise ValueError(f"Unable to find any joints associated with actuator group. Input: {self.cfg.dof_names}.")

        # process configuration
        # -- create actuator model
        if self.model_type == "explicit":
            actuator_model_cls = eval(self.cfg.model_cfg.cls_name)
            self.model = actuator_model_cls(
                cfg=self.cfg.model_cfg,
                num_actuators=self.num_actuators,
                num_envs=self.num_articulation,
                device=self.device,
            )
        elif self.model_type == "implicit":
            self.model = None
        else:
            raise ValueError(
                f"Invalid articulation model type. Received: '{self.model_type}'. Expected: 'explicit' or 'implicit'."
            )
        # -- action transforms
        self._process_action_transforms_cfg()
        # -- gains
        self._process_actuator_gains_cfg()
        # -- torque limits
        self._process_actuator_torque_limit_cfg()
        # -- control mode
        self.view.switch_control_mode(self.control_mode, joint_indices=self.dof_indices)

        # create buffers for allocation
        # -- state
        self._dof_pos = torch.zeros(self.num_articulation, self.num_actuators, device=self.device)
        self._dof_vel = torch.zeros_like(self._dof_pos)
        # -- commands
        self._computed_torques = torch.zeros_like(self._dof_pos)
        self._applied_torques = torch.zeros_like(self._dof_pos)
        self._gear_ratio = torch.ones_like(self._dof_pos)

    def __str__(self) -> str:
        """A string representation of the actuator group."""
        return (
            "<class ActuatorGroup> object:\n"
            f"\tNumber of DOFs: {self.num_actuators}\n"
            f"\tDOF names     : {self.dof_names}\n"
            f"\tDOF indices   : {self.dof_indices}\n"
            f"\tActuator model: {self.model_type}\n"
            f"\tCommand types : {self.command_types}\n"
            f"\tControl mode  : {self.control_mode}\n"
            f"\tControl dim   : {self.control_dim}"
        )

    """
    Properties- Group.
    """

    @property
    def num_actuators(self) -> int:
        """Number of actuators in the group."""
        return len(self.dof_names)

    @property
    def model_type(self) -> str:
        """Type of actuator model: "explicit" or "implicit".

        - **explicit**: Computes joint torques to apply into the simulation.
        - **implicit**: Uses the in-built PhysX joint controller.
        """
        return self.cfg.model_cfg.model_type

    @property
    def command_types(self) -> List[str]:
        """Command type applied on the DOF in the group.

        It contains a list of strings. Each string has two sub-strings joined by underscore:
        - type of command mode: "p" (position), "v" (velocity), "t" (torque)
        - type of command resolving: "abs" (absolute), "rel" (relative)
        """
        return self.cfg.control_cfg.command_types

    @property
    def control_dim(self) -> int:
        """Dimension of control actions.

        The number of control actions is a product of number of actuated joints in the group and the
        number of command types provided in the control confgiruation.
        """
        return self.num_actuators * len(self.command_types)

    @property
    def control_mode(self) -> str:
        """Simulation drive control mode.

        For explicit actuator models, the control mode is always "effort". For implicit actuators, the
        control mode is prioritized by the first command type in the control configuration. It is either:

        - "position" (position-controlled)
        - "velocity" (velocity-controlled)
        - "effort" (torque-controlled).

        """
        if self.model_type == "explicit":
            return "effort"
        elif self.model_type == "implicit":
            # get first command type.
            drive_type = self.command_types[0]
            # check valid drive type
            if "p" in drive_type:
                return "position"
            elif "v" in drive_type:
                return "velocity"
            elif "t" in drive_type:
                return "effort"
            else:
                raise ValueError(f"Invalid primary control mode '{drive_type}'. Expected substring: 'p', 'v', 't'.")
        else:
            raise ValueError(
                f"Invalid actuator model in group '{self.model_type}'. Expected: 'explicit' or 'implicit'."
            )

    """
    Properties- Actuator Model.
    """

    @property
    def gear_ratio(self) -> torch.Tensor:
        """Gear-box conversion factor from motor axis to joint axis."""
        return self._gear_ratio

    @property
    def computed_torques(self) -> torch.Tensor:
        """DOF torques computed from the actuator model (before clipping).

        Note: The torques are zero for implicit actuator models.
        """
        return self._computed_torques

    @property
    def applied_torques(self) -> torch.Tensor:
        """DOF torques applied from the actuator model to simulation (after clipping).

        Note: The torques are zero for implicit actuator models.
        """
        return self._applied_torques

    @property
    def velocity_limit(self) -> Optional[torch.Tensor]:
        """DOF velocity limits from actuator.

        Returns :obj:`None` for implicit and ideal actuator.
        """
        if isinstance(self.model, DCMotor):
            return self.model.cfg.motor_velocity_limit / self.model.gear_ratio
        else:
            return None

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int]):
        """Reset the internals within the group.

        Args:
            env_ids (Sequence[int]): List of environment IDs to reset.
        """
        # actuator
        if self.model is not None:
            self.model.reset(env_ids)

    def compute(self, group_actions: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> ArticulationActions:
        """Process the actuator group actions and compute the articulation actions.

        It performs the following operations:

        1. formats the input actions to apply any extra constraints
        2. splits the formatted actions into individual tensors corresponding to each command type
        3. applies offset and scaling to individual commands based on absolute or relative command
        4. computes the articulation actions based on the actuator model type

        Args:
            group_actions (torch.Tensor): The actuator group actions of shape (num_articulation, control_dim).
            dof_pos (torch.Tensor): The current joint positions of the joints in the group.
            dof_vel (torch.Tensor): The current joint velocities of the joints in the group.

        Raises:
            ValueError: When the group actions has the wrong shape. Expected shape: (num_articulation, control_dim).
            ValueError: Invalid command type. Valid: 'p_abs', 'p_rel', 'v_abs', 'v_rel', 't_abs'.
            ValueError: Invalid actuator model type in group. Expected: 'explicit' or 'implicit'.

        Returns:
            ArticulationActions: An instance of the articulation actions.
              a. for explicit actuator models, it returns the computed joint torques (after clipping)
              b. for implicit actuator models, it returns the actions with the processed desired joint
                 positions, velocities, and efforts (based on the command types)
        """
        # check that group actions has the right dimension
        control_shape = (self.num_articulation, self.control_dim)
        if tuple(group_actions.shape) != control_shape:
            raise ValueError(
                f"Invalid actuator group actions shape '{group_actions.shape}'. Expected: '{control_shape}'."
            )
        # store current dof state
        self._dof_pos[:] = dof_pos
        self._dof_vel[:] = dof_vel
        # buffers for applying actions.
        group_des_dof_pos = None
        group_des_dof_vel = None
        group_des_dof_torque = None
        # pre-processing of commands based on group.
        group_actions = self._format_command(group_actions)
        # reshape actions for sub-groups
        group_actions = group_actions.split([self.num_actuators] * len(self.command_types), dim=-1)
        # pre-process relative commands
        for command_type, command_value in zip(self.command_types, group_actions):
            if command_type == "p_rel":
                group_des_dof_pos = self.dof_pos_scale * command_value + self._dof_pos
            elif command_type == "p_abs":
                group_des_dof_pos = self.dof_pos_scale * command_value + self.dof_pos_offset
            elif command_type == "v_rel":
                group_des_dof_vel = self.dof_vel_scale * command_value + self._dof_vel
            elif command_type == "v_abs":
                group_des_dof_vel = self.dof_vel_scale * command_value  # offset = 0
            elif command_type == "t_abs":
                group_des_dof_torque = self.dof_torque_scale * command_value  # offset = 0
            else:
                raise ValueError(f"Invalid action command type for actuators: '{command_type}'.")

        # process commands based on actuators
        if self.model_type == "explicit":
            # assert that model is created
            assert self.model is not None, "Actuator model is not created."
            # update state and command
            self.model.set_command(dof_pos=group_des_dof_pos, dof_vel=group_des_dof_vel, torque_ff=group_des_dof_torque)
            # compute torques
            self._computed_torques = self.model.compute_torque(dof_pos=self._dof_pos, dof_vel=self._dof_vel)
            self._applied_torques = self.model.clip_torques(
                self._computed_torques, dof_pos=self._dof_pos, dof_vel=self._dof_vel
            )
            # store updated quantities
            self._gear_ratio[:] = self.model.gear_ratio
            # wrap into isaac-sim command
            control_action = ArticulationActions(joint_efforts=self._applied_torques, joint_indices=self.dof_indices)
        elif self.model_type == "implicit":
            # wrap into isaac-sim command
            control_action = ArticulationActions(
                joint_positions=group_des_dof_pos,
                joint_velocities=group_des_dof_vel,
                joint_efforts=group_des_dof_torque,
                joint_indices=self.dof_indices,
            )
        else:
            raise ValueError(
                f"Invalid articulation model type. Received: '{self.model_type}'. Expected: 'explicit' or 'implicit'."
            )
        # return the computed actions
        return control_action

    """
    Implementation specifics.
    """

    def _format_command(self, command: torch.Tensor) -> torch.Tensor:
        """Formatting of commands given to the group.

        In the default actuator group, an identity mapping is performed between the input
        commands and the joint commands.

        Returns:
            torch.Tensor: Desired commands for the DOFs in the group.
                Shape is ``(num_articulations, num_actuators * len(command_types))``.
        """
        return command

    """
    Helper functions.
    """

    def _process_action_transforms_cfg(self):
        """Resolve the action scaling and offsets for different command types."""
        # default values
        # -- scale
        self.dof_pos_scale = torch.ones(self.num_actuators, device=self.device)
        self.dof_vel_scale = torch.ones(self.num_actuators, device=self.device)
        self.dof_torque_scale = torch.ones(self.num_actuators, device=self.device)
        # -- offset
        self.dof_pos_offset = torch.zeros(self.num_actuators, device=self.device)
        # parse configuration
        for index, dof_name in enumerate(self.dof_names):
            # dof pos scale
            if self.cfg.control_cfg.dof_pos_scale is not None:
                for re_key, value in self.cfg.control_cfg.dof_pos_scale.items():
                    if re.fullmatch(re_key, dof_name):
                        self.dof_pos_scale[index] = value
            # dof vel scale
            if self.cfg.control_cfg.dof_vel_scale is not None:
                for re_key, value in self.cfg.control_cfg.dof_vel_scale.items():
                    if re.fullmatch(re_key, dof_name):
                        self.dof_vel_scale[index] = value
            # dof torque scale
            if self.cfg.control_cfg.dof_torque_scale is not None:
                for re_key, value in self.cfg.control_cfg.dof_torque_scale.items():
                    if re.fullmatch(re_key, dof_name):
                        self.dof_torque_scale[index] = value
            # dof pos offset
            if self.cfg.control_cfg.dof_pos_offset is not None:
                for re_key, value in self.cfg.control_cfg.dof_pos_offset.items():
                    if re.fullmatch(re_key, dof_name):
                        self.dof_pos_offset[index] = value

    def _process_actuator_gains_cfg(self):
        """Resolve the PD gains for the actuators and set them into actuator model.

        If the actuator model is implicit, then the gains are applied into the simulator.
        If the actuator model is explicit, then the gains are applied into the actuator model.
        """
        # get the default values from simulator/USD
        stiffness, damping = self.view.get_gains(joint_indices=self.dof_indices, clone=False)
        # parse configuration
        for index, dof_name in enumerate(self.dof_names):
            # stiffness
            if self.cfg.control_cfg.stiffness is not None:
                for re_key, value in self.cfg.control_cfg.stiffness.items():
                    if re.fullmatch(re_key, dof_name):
                        if value is not None:
                            stiffness[:, index] = value
            # damping
            if self.cfg.control_cfg.damping is not None:
                for re_key, value in self.cfg.control_cfg.damping.items():
                    if re.fullmatch(re_key, dof_name):
                        if value is not None:
                            damping[:, index] = value
        # set values into model
        if self.model_type == "explicit":
            # assert that model is created
            assert self.model is not None, "Actuator model is not created."
            # set values into the explicit models
            self.model.set_command(p_gains=stiffness, d_gains=damping)
        elif self.model_type == "implicit":
            # set gains into simulation (implicit)
            self.view.set_gains(kps=stiffness, kds=damping, joint_indices=self.dof_indices)
        else:
            raise ValueError(
                f"Invalid articulation model type. Received: '{self.model_type}'. Expected: 'explicit' or 'implicit'."
            )

    def _process_actuator_torque_limit_cfg(self):
        """Process the torque limit of the actuators and set them into simulation.

        If the actuator model is implicit, then the torque limits are applied into the simulator based on the parsed
        model configuration. If the actuator model is explicit, the torque limits are set to high values since the
        torque range is handled by the saturation model.
        """
        # get the default values from simulator/USD
        torque_limit = self.view.get_max_efforts(joint_indices=self.dof_indices, clone=False)
        # parse configuration
        if self.model_type == "explicit":
            # for explicit actuators, clipping is handled by the forward model.
            # so we set this high for PhysX to not do anything.
            torque_limit[:] = 1.0e9
        elif self.model_type == "implicit":
            # for implicit actuators, we let PhysX handle the actuator limits.
            if isinstance(self.cfg.model_cfg, ImplicitActuatorCfg):
                if self.cfg.model_cfg.torque_limit is not None:
                    torque_limit[:] = self.cfg.model_cfg.torque_limit
        else:
            raise ValueError(
                f"Invalid articulation model type. Received: '{self.model_type}'. Expected: 'explicit' or 'implicit'."
            )
        # set values into simulator
        self.view.set_max_efforts(torque_limit, joint_indices=self.dof_indices)
