# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import torch

import isaaclab.utils.string as string_utils
from isaaclab.utils.types import ArticulationActions

from ._compat import _limits_equal, _resolve_limit_aliases

if TYPE_CHECKING:
    from .actuator_base_cfg import ActuatorBaseCfg


def resolve_joint_parameter(
    cfg_value: float | dict[str, float] | None,
    default_value: float | torch.Tensor | None,
    joint_names: list[str],
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Resolve one group-shaped joint parameter from configuration and defaults.

    The single source of joint-parameter resolution semantics, shared by the actuator
    models and by :class:`~isaaclab.actuators.ActuatorCollection` when it resolves the
    construction-time joint properties.

    Args:
        cfg_value: The parameter value from the configuration, a scalar or a
            joint-name-pattern dictionary. If None, then the default value is used.
        default_value: The default value, a scalar or a ``(num_envs, len(joint_names))``
            tensor. If it is also None, then an error is raised.
        joint_names: The group's joint names, defining the column order.
        num_envs: Number of articulation instances.
        device: Torch device string.

    Returns:
        The resolved parameter value, shape ``(num_envs, len(joint_names))``.

    Raises:
        TypeError: If the parameter or default value is not of the expected type.
        ValueError: If both values are None, or the default tensor has the wrong shape.
    """
    num_joints = len(joint_names)
    param = torch.zeros(num_envs, num_joints, device=device)
    if cfg_value is not None:
        if isinstance(cfg_value, (float, int, dict)):
            dense_values = string_utils._resolve_matching_values_dense(cfg_value, joint_names)
            param[:] = torch.tensor(dense_values, dtype=torch.float, device=device)
        else:
            raise TypeError(
                f"Invalid type for parameter value: {type(cfg_value)} for "
                + f"actuator on joints {joint_names}. Expected float or dict."
            )
    elif default_value is not None:
        if isinstance(default_value, (float, int)):
            # if float, then use the same value for all joints
            param[:] = float(default_value)
        elif isinstance(default_value, torch.Tensor):
            # if tensor, then use the same tensor for all joints
            if default_value.shape == (num_envs, num_joints):
                param = default_value.float()
            else:
                raise ValueError(
                    "Invalid default value tensor shape.\n"
                    f"Got: {default_value.shape}\n"
                    f"Expected: {(num_envs, num_joints)}"
                )
        else:
            raise TypeError(
                f"Invalid type for default value: {type(default_value)} for "
                + f"actuator on joints {joint_names}. Expected float or Tensor."
            )
    else:
        raise ValueError("The parameter value is None and no default value is provided.")

    return param


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
    """The computed effort [N or N·m, depending on joint type] for the actuator group.

    Shape is (num_envs, num_joints).
    """

    applied_effort: torch.Tensor
    """The applied effort [N or N·m, depending on joint type] for the actuator group.

    Shape is (num_envs, num_joints).

    This is the effort obtained after clipping the :attr:`computed_effort` based on the
    actuator characteristics.
    """

    actuator_velocity_limit: torch.Tensor
    """The actuator velocity limit [m/s or rad/s, depending on joint type]. Shape is (num_envs, num_joints).

    The peak velocity of the actuated joint (the actuator's rated speed reflected at the joint,
    after any gearbox). Feeds the articulation data buffers (e.g. soft joint velocity limits) and
    explicit-model effort clipping; it is not pushed to the physics solver. Defaults to
    ``joint_velocity_limit`` when only the solver constraint is configured.
    """

    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        actuator_effort_limit: torch.Tensor | float | None = None,
        actuator_velocity_limit: torch.Tensor | float | None = None,
        effort_limit: torch.Tensor | float | None = None,  # TODO: Deprecated. Remove in 4.0.
        velocity_limit: torch.Tensor | float | None = None,  # TODO: Deprecated. Remove in 4.0.
    ):
        """Initialize the actuator.

        The actuator parameters are parsed from the configuration and stored as buffers. If the parameters
        are not specified in the configuration, then their values provided in the constructor are used.

        .. note::
            The constructor defaults are typically read from the backend's authored joint properties.

        Args:
            cfg: The configuration of the actuator model.
            joint_names: The joint names in the articulation.
            joint_ids: The joint indices in the articulation. If :obj:`slice(None)`, then all
                the joints in the articulation are part of the group.
            num_envs: Number of articulations in the view.
            device: Device used for processing.
            actuator_effort_limit: Default actuator-model effort clipping limit
                [N or N·m, depending on joint type]. Defaults to infinity.
                If a tensor, then the shape is (num_envs, num_joints).
            actuator_velocity_limit: Default actuator velocity limit
                [m/s or rad/s, depending on joint type]. Defaults to infinity.
                If a tensor, then the shape is (num_envs, num_joints).
            effort_limit: Deprecated alias for :paramref:`actuator_effort_limit`.
            velocity_limit: Deprecated alias for :paramref:`actuator_velocity_limit`.
        """
        # save parameters
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._joint_names = joint_names
        self._joint_indices = joint_ids

        # create commands buffers for allocation
        self.computed_effort = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        self.applied_effort = torch.zeros_like(self.computed_effort)

        # normalize deprecated configuration aliases for direct construction
        # TODO: Deprecated. Remove in 4.0.
        if (
            self.cfg.effort_limit is not None
            or self.cfg.effort_limit_sim is not None
            or self.cfg.velocity_limit is not None
            or self.cfg.velocity_limit_sim is not None
        ):
            _resolve_limit_aliases(type(self).__name__, self.cfg, self.joint_names)

        # normalize deprecated constructor aliases
        # TODO: Deprecated. Remove in 4.0.
        if effort_limit is not None:
            warnings.warn(
                "The effort_limit constructor argument is deprecated. Use actuator_effort_limit instead; "
                "effort_limit will be removed in 4.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if actuator_effort_limit is not None and not _limits_equal(actuator_effort_limit, effort_limit):
                raise ValueError(
                    "Received conflicting actuator_effort_limit and deprecated effort_limit constructor arguments."
                )
            actuator_effort_limit = effort_limit
        if velocity_limit is not None:
            warnings.warn(
                "The velocity_limit constructor argument is deprecated. Use actuator_velocity_limit instead; "
                "velocity_limit will be removed in 4.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if actuator_velocity_limit is not None and not _limits_equal(actuator_velocity_limit, velocity_limit):
                raise ValueError(
                    "Received conflicting actuator_velocity_limit and deprecated velocity_limit constructor arguments."
                )
            actuator_velocity_limit = velocity_limit

        # parse the actuator-model limits. Implicit models expose their effort limit as a live
        # projection of the articulation joint effort limit instead of a local buffer.
        if not self.is_implicit_model:
            if actuator_effort_limit is None:
                actuator_effort_limit = torch.inf
            self.actuator_effort_limit = resolve_joint_parameter(
                self.cfg.actuator_effort_limit, actuator_effort_limit, joint_names, num_envs, device
            )
        if actuator_velocity_limit is None:
            actuator_velocity_limit = torch.inf
        self.actuator_velocity_limit = resolve_joint_parameter(
            self.cfg.actuator_velocity_limit, actuator_velocity_limit, joint_names, num_envs, device
        )

    def __str__(self) -> str:
        """Returns: A string representation of the actuator group."""
        # resolve joint indices for printing
        joint_indices = self.joint_indices
        if isinstance(joint_indices, slice):
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

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip the desired torques based on the motor limits.

        Args:
            effort: The effort to clip [N or N·m, depending on joint type].

        Returns:
            The clipped effort [N or N·m, depending on joint type].
        """
        return torch.clip(effort, min=-self.actuator_effort_limit, max=self.actuator_effort_limit)

    @property
    def effort_limit(self) -> torch.Tensor:
        """Deprecated actuator effort limit [N or N·m, depending on joint type].

        .. deprecated:: 3.0
            Use :attr:`actuator_effort_limit` instead. This alias will be removed in 4.0.
        """
        warnings.warn(
            "ActuatorBase.effort_limit is deprecated. Use actuator_effort_limit instead; "
            "effort_limit will be removed in 4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.actuator_effort_limit

    @effort_limit.setter
    def effort_limit(self, value: torch.Tensor) -> None:
        warnings.warn(
            "ActuatorBase.effort_limit is deprecated. Use actuator_effort_limit instead; "
            "effort_limit will be removed in 4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.actuator_effort_limit = value

    @property
    def velocity_limit(self) -> torch.Tensor:
        """Deprecated actuator velocity limit [m/s or rad/s, depending on joint type].

        .. deprecated:: 3.0
            Use :attr:`actuator_velocity_limit` instead. This alias will be removed in 4.0.
        """
        warnings.warn(
            "ActuatorBase.velocity_limit is deprecated. Use actuator_velocity_limit instead; "
            "velocity_limit will be removed in 4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.actuator_velocity_limit

    @velocity_limit.setter
    def velocity_limit(self, value: torch.Tensor) -> None:
        warnings.warn(
            "ActuatorBase.velocity_limit is deprecated. Use actuator_velocity_limit instead; "
            "velocity_limit will be removed in 4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.actuator_velocity_limit = value
