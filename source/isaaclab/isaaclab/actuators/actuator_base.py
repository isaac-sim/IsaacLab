# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import isaaclab.utils.string as string_utils
from isaaclab.utils.types import ArticulationActions

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

    computed_effort: torch.Tensor
    """The computed effort for the actuator group. Shape is (num_envs, num_joints)."""

    applied_effort: torch.Tensor
    """The applied effort for the actuator group. Shape is (num_envs, num_joints).

    This is the effort obtained after clipping the :attr:`computed_effort` based on the
    actuator characteristics.
    """

    effort_limit: torch.Tensor
    """The effort limit for the actuator group. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`effort_limit` and :attr:`effort_limit_sim` are the same.
    """

    effort_limit_sim: torch.Tensor
    """The effort limit for the actuator group in the simulation. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`effort_limit` and :attr:`effort_limit_sim` are the same.
    """

    velocity_limit: torch.Tensor
    """The velocity limit for the actuator group. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`velocity_limit` and :attr:`velocity_limit_sim` are the same.
    """

    velocity_limit_sim: torch.Tensor
    """The velocity limit for the actuator group in the simulation. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`velocity_limit` and :attr:`velocity_limit_sim` are the same.
    """

    stiffness: torch.Tensor
    """The stiffness (P gain) of the PD controller. Shape is (num_envs, num_joints)."""

    damping: torch.Tensor
    """The damping (D gain) of the PD controller. Shape is (num_envs, num_joints)."""

    armature: torch.Tensor
    """The armature of the actuator joints. Shape is (num_envs, num_joints)."""

    friction: torch.Tensor
    """The joint static friction of the actuator joints. Shape is (num_envs, num_joints)."""

    dynamic_friction: torch.Tensor
    """The joint dynamic friction of the actuator joints. Shape is (num_envs, num_joints)."""

    viscous_friction: torch.Tensor
    """The joint viscous friction of the actuator joints. Shape is (num_envs, num_joints)."""

    _DEFAULT_MAX_EFFORT_SIM: ClassVar[float] = 1.0e9
    """The default maximum effort for the actuator joints in the simulation. Defaults to 1.0e9.

    If the :attr:`ActuatorBaseCfg.effort_limit_sim` is not specified and the actuator is an explicit
    actuator, then this value is used.
    """

    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        dynamic_friction: torch.Tensor | float = 0.0,
        viscous_friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
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
            device: Device used for processing.
            stiffness: The default joint stiffness (P gain). Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            damping: The default joint damping (D gain). Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            armature: The default joint armature. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            friction: The default joint static friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            dynamic_friction: The default joint dynamic friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            viscous_friction: The default joint viscous friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            effort_limit: The default effort limit. Defaults to infinity.
                If a tensor, then the shape is (num_envs, num_joints).
            velocity_limit: The default velocity limit. Defaults to infinity.
                If a tensor, then the shape is (num_envs, num_joints).
        """
        # save parameters
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._joint_names = joint_names
        self._joint_indices = joint_ids
        self.joint_property_resolution_table: dict[str, list] = {}
        # For explicit models, we do not want to enforce the effort limit through the solver
        # (unless it is explicitly set)
        if not self.is_implicit_model and self.cfg.effort_limit_sim is None:
            self.cfg.effort_limit_sim = self._DEFAULT_MAX_EFFORT_SIM

        # resolve usd, actuator configuration values
        # case 1: if usd_value == actuator_cfg_value: all good,
        # case 2: if usd_value != actuator_cfg_value: we use actuator_cfg_value
        # case 3: if actuator_cfg_value is None: we use usd_value

        to_check = [
            ("velocity_limit_sim", velocity_limit),
            ("effort_limit_sim", effort_limit),
            ("stiffness", stiffness),
            ("damping", damping),
            ("armature", armature),
            ("friction", friction),
            ("dynamic_friction", dynamic_friction),
            ("viscous_friction", viscous_friction),
        ]
        for param_name, usd_val in to_check:
            cfg_val = getattr(self.cfg, param_name)
            setattr(self, param_name, self._parse_joint_parameter(cfg_val, usd_val))
            new_val = getattr(self, param_name)

            allclose = (
                torch.all(new_val == usd_val) if isinstance(usd_val, (float, int)) else torch.allclose(new_val, usd_val)
            )
            if cfg_val is None or not allclose:
                self._record_actuator_resolution(
                    cfg_val=getattr(self.cfg, param_name),
                    new_val=new_val[0],  # new val always has the shape of (num_envs, num_joints)
                    usd_val=usd_val,
                    joint_names=joint_names,
                    joint_ids=joint_ids,
                    actuator_param=param_name,
                )

        self.velocity_limit = self._parse_joint_parameter(self.cfg.velocity_limit, self.velocity_limit_sim)
        self.effort_limit = self._parse_joint_parameter(self.cfg.effort_limit, self.effort_limit_sim)

        # create commands buffers for allocation
        self.computed_effort = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        self.applied_effort = torch.zeros_like(self.computed_effort)

    def __str__(self) -> str:
        """Returns: A string representation of the actuator group."""
        # resolve joint indices for printing
        joint_indices = self.joint_indices
        if joint_indices == slice(None):
            joint_indices = list(range(self.num_joints))
        # resolve model type (implicit or explicit)
        model_type = "implicit" if self.is_implicit_model else "explicit"

        return (
            f"<class {self.__class__.__name__}> object:\n"
            f"\tModel type            : {model_type}\n"
            f"\tNumber of joints      : {self.num_joints}\n"
            f"\tJoint names expression: {self.cfg.joint_names_expr}\n"
            f"\tJoint names           : {self.joint_names}\n"
            f"\tJoint indices         : {joint_indices}\n"
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
    def joint_indices(self) -> slice | torch.Tensor:
        """Articulation's joint indices that are part of the group.

        Note:
            If :obj:`slice(None)` is returned, then the group contains all the joints in the articulation.
            We do this to avoid unnecessary indexing of the joints for performance reasons.
        """
        return self._joint_indices

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int]):
        """Reset the internals within the group.

        Args:
            env_ids: List of environment IDs to reset.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
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

    def _record_actuator_resolution(self, cfg_val, new_val, usd_val, joint_names, joint_ids, actuator_param: str):
        if actuator_param not in self.joint_property_resolution_table:
            self.joint_property_resolution_table[actuator_param] = []
        table = self.joint_property_resolution_table[actuator_param]

        ids = joint_ids if isinstance(joint_ids, torch.Tensor) else list(range(len(joint_names)))
        for idx, name in enumerate(joint_names):
            cfg_val_log = "Not Specified" if cfg_val is None else float(new_val[idx])
            default_usd_val = usd_val if isinstance(usd_val, (float, int)) else float(usd_val[0][idx])
            applied_val_log = default_usd_val if cfg_val is None else float(new_val[idx])
            table.append([name, int(ids[idx]), default_usd_val, cfg_val_log, applied_val_log])

    def _parse_joint_parameter(
        self, cfg_value: float | dict[str, float] | None, default_value: float | torch.Tensor | None
    ) -> torch.Tensor:
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
        # create parameter buffer
        param = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        # parse the parameter
        if cfg_value is not None:
            if isinstance(cfg_value, (float, int)):
                # if float, then use the same value for all joints
                param[:] = float(cfg_value)
            elif isinstance(cfg_value, dict):
                # if dict, then parse the regular expression
                indices, _, values = string_utils.resolve_matching_names_values(cfg_value, self.joint_names)
                # note: need to specify type to be safe (e.g. values are ints, but we want floats)
                param[:, indices] = torch.tensor(values, dtype=torch.float, device=self._device)
            else:
                raise TypeError(
                    f"Invalid type for parameter value: {type(cfg_value)} for "
                    + f"actuator on joints {self.joint_names}. Expected float or dict."
                )
        elif default_value is not None:
            if isinstance(default_value, (float, int)):
                # if float, then use the same value for all joints
                param[:] = float(default_value)
            elif isinstance(default_value, torch.Tensor):
                # if tensor, then use the same tensor for all joints
                if default_value.shape == (self._num_envs, self.num_joints):
                    param = default_value.float()
                else:
                    raise ValueError(
                        "Invalid default value tensor shape.\n"
                        f"Got: {default_value.shape}\n"
                        f"Expected: {(self._num_envs, self.num_joints)}"
                    )
            else:
                raise TypeError(
                    f"Invalid type for default value: {type(default_value)} for "
                    + f"actuator on joints {self.joint_names}. Expected float or Tensor."
                )
        else:
            raise ValueError("The parameter value is None and no default value is provided.")

        return param

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip the desired torques based on the motor limits.

        Args:
            desired_torques: The desired torques to clip.

        Returns:
            The clipped torques.
        """
        return torch.clip(effort, min=-self.effort_limit, max=self.effort_limit)
