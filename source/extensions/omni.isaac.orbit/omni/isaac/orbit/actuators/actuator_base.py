# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from omni.isaac.core.utils.types import ArticulationActions

import omni.isaac.orbit.utils.string as string_utils

if TYPE_CHECKING:
    from .actuator_cfg import ActuatorBaseCfg


class ActuatorBase(ABC):
    """Base class for applying actuator models over a collection of actuated joints in an articulation.

    The default actuator for applying the same actuator model over a collection of actuated joints in
    an articulation.

    The joint names are specified in the configuration through a list of regular expressions. The regular
    expressions are matched against the joint names in the articulation. The first match is used to determine
    the joint indices in the articulation.

    In the default actuator, no constraints or formatting is performed over the input actions. Thus, the
    input actions are directly used to compute the joint actions in the :meth:`compute`.
    """

    computed_effort: torch.Tensor
    """The computed effort for the actuator group. Shape is ``(num_envs, num_joints)``."""
    applied_effort: torch.Tensor
    """The applied effort for the actuator group. Shape is ``(num_envs, num_joints)``."""
    effort_limit: float = torch.inf
    """The effort limit for the actuator group. Shape is ``(num_envs, num_joints)``."""
    velocity_limit: float = torch.inf
    """The velocity limit for the actuator group. Shape is ``(num_envs, num_joints)``."""
    stiffness: torch.Tensor
    """The stiffness (P gain) of the PD controller. Shape is ``(num_envs, num_joints)``."""
    damping: torch.Tensor
    """The damping (D gain) of the PD controller. Shape is ``(num_envs, num_joints)``."""

    def __init__(
        self, cfg: ActuatorBaseCfg, joint_names: list[str], joint_ids: Sequence[int], num_envs: int, device: str
    ):
        """Initialize the actuator.

        Args:
            cfg: The configuration of the actuator model.
            joint_names: The joint names in the articulation.
            joint_ids: The joint indices in the articulation. If :obj:`slice(None)`, then all
                the joints in the articulation are part of the group.
            num_envs: Number of articulations in the view.
            device: Device used for processing.
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
        self.stiffness = torch.zeros_like(self.computed_effort)
        self.damping = torch.zeros_like(self.computed_effort)

        # parse joint limits
        if self.cfg.effort_limit is not None:
            self.effort_limit = self.cfg.effort_limit
        if self.cfg.velocity_limit is not None:
            self.velocity_limit = self.cfg.velocity_limit
        # parse joint stiffness and damping
        # -- stiffness
        if self.cfg.stiffness is not None:
            if isinstance(self.cfg.stiffness, float):
                # if float, then use the same value for all joints
                self.stiffness[:] = self.cfg.stiffness
            else:
                # if dict, then parse the regular expression
                indices, _, values = string_utils.resolve_matching_names_values(self.cfg.stiffness, self.joint_names)
                self.stiffness[:, indices] = torch.tensor(values, device=self._device)
        # -- damping
        if self.cfg.damping is not None:
            if isinstance(self.cfg.damping, float):
                # if float, then use the same value for all joints
                self.damping[:] = self.cfg.damping
            else:
                # if dict, then parse the regular expression
                indices, _, values = string_utils.resolve_matching_names_values(self.cfg.stiffness, self.joint_names)
                self.damping[:, indices] = torch.tensor(values, device=self._device)

    def __str__(self) -> str:
        """Returns: A string representation of the actuator group."""
        # resolve joint indices for printing
        joint_indices = self.joint_indices
        if joint_indices == slice(None):
            joint_indices = list(range(self.num_joints))
        return (
            f"<class {self.__class__.__name__}> object:\n"
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
    def joint_indices(self) -> Sequence[int]:
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
            joint_pos: The current joint positions of the joints in the group. Shape is ``(num_envs, num_joints)``.
            joint_vel: The current joint velocities of the joints in the group. Shape is ``(num_envs, num_joints)``.

        Returns:
            The computed desired joint positions, joint velocities and joint efforts.
        """
        raise NotImplementedError
