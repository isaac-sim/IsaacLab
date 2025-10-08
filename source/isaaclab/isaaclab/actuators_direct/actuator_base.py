# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import isaaclab.utils.string as string_utils
from isaaclab.actuators_direct.kernels import clip_efforts_with_limits
from isaaclab.assets.articulation_direct.kernels import update_joint_array_with_value, update_joint_array_with_value_int, populate_empty_array, update_joint_array_with_value_array

if TYPE_CHECKING:
    from .actuator_cfg import ActuatorBaseDirectCfg
    from isaaclab.assets.articulation_direct.articulation import ArticulationDataDirect


class ActuatorBaseDirect(ABC):
    """Base class for actuator models over a collection of actuated joints in an articulation.

    Actuator models augment the simulated articulation joints with an external drive dynamics model.
    The model is used to convert the user-provided joint commands (positions, velocities and efforts)
    into the desired joint positions, velocities and efforts that are applied to the simulated articulation.

    The base class provides the interface for the actuator models. It is responsible for parsing the
    actuator parameters from the configuration and storing them as buffers. It also provides the
    interface for resetting the actuator state and computing the desired joint commands for the simulation.

    For each actuator model, a corresponding configuration class is provided. The configuration class
    is used to parse the actuator parameters from the configuration. It also specifies the joint names
    for which the actuator model is applied. These names can be specified as regular expressions, which
    are matched against the joint names in the articulation.

    To see how the class is used, check the :class:`isaaclab.assets.Articulation` class.
    """

    is_implicit_model: ClassVar[bool] = False
    """Flag indicating if the actuator is an implicit or explicit actuator model.

    If a class inherits from :class:`ImplicitActuator`, then this flag should be set to :obj:`True`.
    """

    data: ArticulationDataDirect
    """The data of the articulation."""

    _DEFAULT_MAX_EFFORT_SIM: ClassVar[float] = 1.0e9
    """The default maximum effort for the actuator joints in the simulation. Defaults to 1.0e9.

    If the :attr:`ActuatorBaseCfg.effort_limit_sim` is not specified and the actuator is an explicit
    actuator, then this value is used.
    """


    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        joint_names: list[str],
        joint_mask: wp.array,
        env_mask: wp.array,
        articulation_data: ArticulationDataDirect,
        device: str,
    ):
        """Initialize the actuator.

        The actuator parameters are parsed from the configuration and stored as buffers. If the parameters
        are not specified in the configuration, then their values provided in the constructor are used.

        .. note::
            The values in the constructor are typically obtained through the USD schemas corresponding
            to the joints in the actuator model.

        Args:
            cfg: The configuration of the actuator model.
            joint_names: The joint names in the articulation.
            joint_mask: The mask of joints to use.
            env_mask: The mask of environments to use.
            articulation_data: The data for the articulation.
            device: Device used for processing.
        """
        # save parameters
        self.cfg = cfg
        self._device = device
        self._joint_names = joint_names
        self._joint_mask = joint_mask
        self._env_mask = env_mask
        # Get the number of environments and joints from the articulation data
        self._num_envs = env_mask.shape[0]
        self._num_joints = joint_mask.shape[0]
        # Get the data from the articulation
        self.data = articulation_data

        # For explicit models, we do not want to enforce the effort limit through the solver
        # (unless it is explicitly set)
        if not self.is_implicit_model and self.cfg.effort_limit_sim is None:
            self.cfg.effort_limit_sim = self._DEFAULT_MAX_EFFORT_SIM

        # Parse the joint commands:
        if self.cfg.control_mode == "position":
            self.cfg.control_mode = 1
        elif self.cfg.control_mode == "velocity":
            self.cfg.control_mode = 2
        elif self.cfg.control_mode == "none":
            self.cfg.control_mode = 0

        # resolve usd, actuator configuration values
        # case 1: if usd_value == actuator_cfg_value: all good,
        # case 2: if usd_value != actuator_cfg_value: we use actuator_cfg_value
        # case 3: if actuator_cfg_value is None: we use usd_value

        to_check = [
            ("velocity_limit_sim", self.data.sim_bind_joint_vel_limits_sim),
            ("effort_limit_sim", self.data.sim_bind_joint_effort_limits_sim),
            ("stiffness", self.data.joint_stiffness),
            ("damping", self.data.joint_damping),
            ("armature", self.data.sim_bind_joint_armature),
            ("friction", self.data.sim_bind_joint_friction_coeff),
            ("dynamic_friction", self.data.joint_dynamic_friction),
            ("viscous_friction", self.data.joint_viscous_friction),
            ("control_mode", self.data.joint_control_mode),
        ]
        for param_name, newton_val in to_check:
            cfg_val = getattr(self.cfg, param_name)
            self._parse_joint_parameter(cfg_val, newton_val)

    def __str__(self) -> str:
        """Returns: A string representation of the actuator group."""
        # resolve model type (implicit or explicit)
        model_type = "implicit" if self.is_implicit_model else "explicit"

        return (
            f"<class {self.__class__.__name__}> object:\n"
            f"\tModel type            : {model_type}\n"
            f"\tNumber of joints      : {self.num_joints}\n"
            f"\tJoint names expression: {self.cfg.joint_names_expr}\n"
            f"\tJoint names           : {self.joint_names}\n"
            f"\tJoint mask            : {self.joint_mask}\n"
        )

    """
    Properties.
    """

    @property
    def num_joints(self) -> int:
        """Number of actuators in the group."""
        return len(self._joint_names)

    @property
    def joint_names(self) -> list[str]:
        """Articulation's joint names that are part of the group."""
        return self._joint_names

    @property
    def joint_mask(self) -> wp.array:
        """Articulation's masked indices that denote which joints are part of the group."""
        return self._joint_mask

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_mask: wp.array(dtype=wp.int32)):
        """Reset the internals within the group.

        Args:
            env_mask: Mask of environments to reset.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        """Process the actuator group actions and compute the articulation actions.

        It computes the articulation actions based on the actuator model type. To do so, it reads
        the following quantities from the articulation data:
        - sim_bind_joint_pos, the current joint positions
        - sim_bind_joint_vel, the current joint velocities
        - joint_control_mode, the current joint control mode
        - joint_stiffness, the current joint stiffness
        - joint_damping, the current joint damping

        With these, it updates the following quantities:

        """
        raise NotImplementedError

    """
    Helper functions.
    """

    def _parse_joint_parameter(
        self, cfg_value: float | dict[str, float] | None, original_value: wp.array | None
    ):
        """Parse the joint parameter from the configuration.

        Args:
            cfg_value: The parameter value from the configuration. If None, then use the default value.
            default_value: The default value to use if the parameter is None. If it is also None,
                then an error is raised.

        Returns:
            The parsed parameter value.

        Raises:
            TypeError: If the parameter value is not of the expected type.
            TypeError: If the default value is not of the expected type.
            ValueError: If the parameter value is None and no default value is provided.
            ValueError: If the default value tensor is the wrong shape.
        """
        # parse the parameter
        if cfg_value is not None:
            if isinstance(cfg_value, float):
                # if float, then use the same value for all joints
                wp.launch(
                    update_joint_array_with_value,
                    dim=(self._num_envs, self._num_joints),
                    inputs=[
                        float(cfg_value),
                        original_value,
                        self._env_mask,
                        self._joint_mask,
                    ]
                )
            elif isinstance(cfg_value, int):
                # if int, then use the same value for all joints
                wp.launch(
                    update_joint_array_with_value_int,
                    dim=(self._num_envs, self._num_joints),
                    inputs=[
                        cfg_value,
                        original_value,
                        self._env_mask,
                        self._joint_mask,
                    ]
                )
            elif isinstance(cfg_value, dict):
                # if dict, then parse the regular expression
                indices, _, values = string_utils.resolve_matching_names_values(cfg_value, self.joint_names)
                tmp_param =wp.zeros((self._num_joints,), dtype=wp.float32, device=self._device)
                wp.launch(
                    populate_empty_array,
                    dim=(self._num_joints,),
                    inputs=[
                        wp.array(values, dtype=wp.float32, device=self._device),
                        tmp_param,
                        wp.array(indices, dtype=wp.int32, device=self._device),
                    ]
                )
                wp.launch(
                    update_joint_array_with_value_array,
                    dim=(self._num_envs, self._num_joints),
                    inputs=[
                        tmp_param,
                        original_value,
                        self._env_mask,
                        self._joint_mask,
                    ]
                )
            else:
                raise TypeError(
                    f"Invalid type for parameter value: {type(cfg_value)} for "
                    + f"actuator on joints {self.joint_names}. Expected float or dict."
                )
        elif original_value is None:
            raise ValueError("The parameter value is None and no newton value is provided.")

    def _clip_effort(self, effort: wp.array, clipped_effort: wp.array) -> None:
        """Clip the desired torques based on the motor limits.

        .. note:: The array is modified in place.

        Args:
            desired_torques: The desired torques to clip. Expected shape is (num_envs, num_joints).

        Returns:
            The clipped torques.
        """
        wp.launch(
            clip_efforts_with_limits,
            dim=(self._num_envs, self._num_joints),
            inputs=[
                self.data.joint_effort_limits_sim,
                effort,
                clipped_effort,
                self._env_mask,
                self.joint_mask,
            ]
        )
