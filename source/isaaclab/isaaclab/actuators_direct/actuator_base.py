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
from isaaclab.utils.types import ArticulationActions
from .actuator_data import ActuatorData
from isaaclab.assets.articulation_direct.kernels import *

if TYPE_CHECKING:
    from .actuator_cfg import ActuatorBaseCfg


class ActuatorBase(ABC):
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

    data: ActuatorData
    """The data for the actuator group. Shape is (num_envs, num_joints)."""

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
        articulation_data: ActuatorData,
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
            joint_ids: The joint indices in the articulation. If :obj:`slice(None)`, then all
                the joints in the articulation are part of the group.
            num_envs: Number of articulations in the view.
            num_joints: Number of joints in the articulation.
            device: Device used for processing.
            joint_mask: The mask of joints to use.
            articulation_data: The data for the articulation.
        """
        # save parameters
        self.cfg = cfg
        self._device = device
        self._joint_names = joint_names
        self._joint_mask = joint_mask
        # Get the number of environments and joints from the articulation data
        self._num_envs = articulation_data.all_env_mask.shape[0]
        self._num_joints = articulation_data.all_joint_mask.shape[0]

        # For explicit models, we do not want to enforce the effort limit through the solver
        # (unless it is explicitly set)
        if not self.is_implicit_model and self.cfg.effort_limit_sim is None:
            self.cfg.effort_limit_sim = self._DEFAULT_MAX_EFFORT_SIM

        # resolve usd, actuator configuration values
        # case 1: if usd_value == actuator_cfg_value: all good,
        # case 2: if usd_value != actuator_cfg_value: we use actuator_cfg_value
        # case 3: if actuator_cfg_value is None: we use usd_value

        to_check = [
            ("velocity_limit_sim", self.data.velocity_limit_sim),
            ("effort_limit_sim", self.data.effort_limit_sim),
            ("stiffness", self.data.stiffness),
            ("damping", self.data.damping),
            ("armature", self.data.armature),
            ("friction", self.data.friction),
            ("dynamic_friction", self.data.dynamic_friction),
            ("viscous_friction", self.data.viscous_friction),
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
    def compute(
        self, control_action: ArticulationActions, joint_pos: wp.array, joint_vel: wp.array) -> ArticulationActions:
        """Process the actuator group actions and compute the articulation actions.

        It computes the articulation actions based on the actuator model type

        Args:
            control_action: The joint action instance comprising of the desired joint positions, joint velocities
                and (feed-forward) joint efforts.
            joint_pos: The current joint positions of the joints in the group. Shape is (num_envs, num_joints).
            joint_vel: The current joint velocities of the joints in the group. Shape is (num_envs, num_joints).

        Returns:
            The computed desired joint positions, joint velocities and joint efforts.
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
                        self.data.all_env_mask,
                        self.data.all_joint_mask,
                    ]
                )
            elif isinstance(cfg_value, int):
                # if int, then use the same value for all joints
                wp.launch(
                    update_joint_array_with_value_int,
                    dim=(self._num_envs, self._num_joints),
                    inputs=[
                        float(cfg_value),
                        original_value,
                        self.data.all_env_mask,
                        self.data.all_joint_mask,
                    ]
                )
            elif isinstance(cfg_value, dict):
                # if dict, then parse the regular expression
                indices, _, values = string_utils.resolve_matching_names_values(cfg_value, self.joint_names)
                tmp_param =wp.zeros((self._num_joints,), dtype=wp.float32, device=self._device)
                wp.launch(
                    populate_empty_array,
                    dim=(self._num_envs, self._num_joints),
                    inputs=[
                        tmp_param,
                        wp.array(values, dtype=wp.float32, device=self._device),
                        wp.array(indices, dtype=wp.int32, device=self._device),
                    ]
                )
                wp.launch(
                    update_joint_array_with_value_array,
                    dim=(self._num_envs, self._num_joints),
                    inputs=[
                        tmp_param,
                        original_value,
                        self.data.all_env_mask,
                        self.data.all_joint_mask,
                    ]
                )
            else:
                raise TypeError(
                    f"Invalid type for parameter value: {type(cfg_value)} for "
                    + f"actuator on joints {self.joint_names}. Expected float or dict."
                )
        else:
            raise ValueError("The parameter value is None and no newton value is provided.")

    def _clip_effort(self, effort: wp.array) -> None:
        """Clip the desired torques based on the motor limits.

        .. note:: The array is modified in place.

        Args:
            desired_torques: The desired torques to clip. Expected shape is (num_envs, num_joints).

        Returns:
            The clipped torques.
        """
        wp.launch(
            clip_joint_array_with_limits_masked,
            dim=(self._num_envs, self._num_joints),
            inputs=[
                self.data.effort_limit_sim,
                self.data.effort_limit_sim,
                effort,
                self.data.all_env_mask,
                self.joint_mask,
            ]
        )
