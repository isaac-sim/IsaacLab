# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from omni.isaac.core.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .actuator_cfg import ActuatorBaseCfg, DCMotorCfg, IdealPDActuatorCfg


class ActuatorBase(ABC):
    """A class for applying actuator models over a collection of actuated joints in an articulation.

    The default actuator for applying the same actuator model over a collection of actuated joints in
    an articulation.

    The joint names are specified in the configuration through a list of regular expressions. The regular
    expressions are matched against the joint names in the articulation. The first match is used to determine
    the joint indices in the articulation.

    In the default actuator, no constraints or formatting is performed over the input actions. Thus, the
    input actions are directly used to compute the joint actions in the :meth:`compute`.
    """

    cfg: ActuatorBaseCfg
    """The configuration of the actuator group."""
    _num_envs: int
    """Number of articulations."""
    _device: str
    """Device used for processing."""
    computed_effort: torch.Tensor
    """The computed effort for the actuator group."""
    applied_effort: torch.Tensor
    """The applied effort for the actuator group."""
    effort_limit: float = torch.inf
    """The effort limit for the actuator group."""
    velocity_limit: float = torch.inf
    """The velocity limit for the actuator group."""
    stiffness: torch.Tensor
    """The stiffness (P gain) of the PD controller."""
    damping: torch.Tensor
    """The damping (D gain) of the PD controller."""

    def __init__(self, cfg: ActuatorBaseCfg, dof_names: list[str], dof_ids: list[int], num_envs: int, device: str):
        """Initialize the actuator.

        Args:
            cfg (ActuatorBaseCfg): The configuration of the actuator.
            find_dofs_func (callable): A function that takes in a list of regular expressions and returns
                                       the joint indices and names that match the regular expressions.
            num_envs (int): Number of articulations in the view.
            device (str): Device used for processing.

        Raises:
            ValueError: When not able to find a match for all DOF names in the configuration.
        """
        # save parameters
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        # extract useful quantities
        # find actuator names and indices
        self._dof_names = dof_names
        self._dof_indices = dof_ids
        # check that group is valid
        if len(self._dof_names) == 0:
            raise ValueError(
                f"Unable to find any joints associated with actuator group. Input: {self.cfg.dof_name_expr}."
            )
        # -- create commands buffers for allocation
        self.computed_effort = torch.zeros(self._num_envs, self.num_actuators, device=self._device)
        self.applied_effort = torch.zeros(self._num_envs, self.num_actuators, device=self._device)
        # parse configuration
        if self.cfg.effort_limit is not None:
            self.effort_limit = self.cfg.effort_limit
        if self.cfg.velocity_limit is not None:
            self.velocity_limit = self.cfg.velocity_limit
        # pd gains
        self.stiffness = torch.zeros(self._num_envs, self.num_actuators, device=self._device)
        self.damping = torch.zeros(self._num_envs, self.num_actuators, device=self._device)
        for index, dof_name in enumerate(self.dof_names):
            # stiffness
            if self.cfg.stiffness is not None:
                for re_key, value in self.cfg.stiffness.items():
                    if re.fullmatch(re_key, dof_name):
                        if value is not None:
                            self.stiffness[:, index] = value
            # damping
            if self.cfg.damping is not None:
                for re_key, value in self.cfg.damping.items():
                    if re.fullmatch(re_key, dof_name):
                        if value is not None:
                            self.damping[:, index] = value

    def __str__(self) -> str:
        """A string representation of the actuator group."""
        return (
            f"<class {self.__class__.__name__}> object:\n"
            f"\tNumber of DOFs: {self.num_actuators}\n"
            f"\tDOF names     : {self.dof_names}\n"
            f"\tDOF indices   : {self.dof_indices}\n"
        )

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int]):
        """Reset the internals within the group.

        Args:
            env_ids (Sequence[int]): List of environment IDs to reset.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self, control_action: ArticulationActions, dof_pos: torch.Tensor, dof_vel: torch.Tensor
    ) -> ArticulationActions:
        """Process the actuator group actions and compute the articulation actions.

        It computes the articulation actions based on the actuator model type

        Args:
            control_action (ArticulationActions): desired joint positions, velocities and (feedforward) efforts.
            dof_pos (torch.Tensor): The current joint positions of the joints in the group.
            dof_vel (torch.Tensor): The current joint velocities of the joints in the group.

        Returns:
            ArticulationActions: modified joint positions, velocities and efforts.
        """
        raise NotImplementedError

    @abstractmethod
    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def num_actuators(self) -> int:
        """Number of actuators in the group."""
        return len(self._dof_names)

    @property
    def dof_names(self) -> list[str]:
        """Articulation's DOF names that are part of the group."""
        return self._dof_names

    @property
    def dof_indices(self) -> list[int]:
        """Articulation's DOF indices that are part of the group."""
        return self._dof_indices

    @property
    def num_envs(self) -> int:
        """Number of articulations."""
        return self._num_envs

    @property
    def device(self) -> str:
        """Device used for processing."""
        return self._device


class IdealPDActuator(ActuatorBase):
    """A class for applying ideal PD actuator models over a collection of actuated joints in an articulation."""

    cfg: IdealPDActuatorCfg

    def compute(
        self, control_action: ArticulationActions, dof_pos: torch.Tensor, dof_vel: torch.Tensor
    ) -> ArticulationActions:
        # calculate the desired joint torques
        self.computed_effort = self.stiffness * (control_action.joint_positions - dof_pos) - self.damping * (
            control_action.joint_velocities - dof_vel
        )
        self.applied_effort = self._clip_effort(self.computed_effort.clip(-self.effort_limit, self.effort_limit))

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    """
    Helper functions.
    """

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        return torch.clip(effort, min=-self.effort_limit, max=self.effort_limit)


class ImplicitActuator(ActuatorBase):
    """A class for applying implicit PD actuator models over a collection of actuated joints in an articulation.
    Same as IdealPDActuator, but the PD control is handled implicitly by the simulation.

    The robot class set the stiffness and damping parameters of this actuator into the simulation.
    """

    def compute(
        self, control_action: ArticulationActions, dof_pos: torch.Tensor, dof_vel: torch.Tensor
    ) -> ArticulationActions:
        raise NotImplementedError("ImplicitActuators are handled directly by the simulation.")


class DCMotor(IdealPDActuator):
    """A class for applying a DC motor actuator models over a collection of actuated joints in an articulation.
    This actuator computes torques based on a PD controller and then clips the torques based on a DC motor model:
    max_torque(dof_vel) = saturation_torque * (1 - dof_vel/saturation_dof_vel)
    """

    cfg: DCMotorCfg

    _saturation_effort: float = torch.inf
    """The saturation effort of the DC motor model."""

    def __init__(self, cfg: DCMotorCfg, dof_names: list[str], dof_ids: list[int], num_envs: int, device: str):
        super().__init__(cfg, dof_names, dof_ids, num_envs, device)
        # parse configuration
        if self.cfg.saturation_effort is not None:
            self._saturation_effort = self.cfg.saturation_effort
        # prepare dof vel buffer for max effort computation
        self._dof_vel = torch.zeros(self._num_envs, self.num_actuators, device=self._device)

    def compute(
        self, control_action: ArticulationActions, dof_pos: torch.Tensor, dof_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current dof vel
        self._dof_vel[:] = dof_vel
        # calculate the desired joint torques
        return super().compute(control_action, dof_pos, dof_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        max_effort = self.cfg.saturation_effort * (1.0 - self._dof_vel / self.cfg.velocity_limit)
        max_effort = torch.clip(max_effort, min=0.0, max=self.effort_limit)
        # -- min limit
        min_effort = self.cfg.saturation_effort * (-1.0 - self._dof_vel / self.cfg.velocity_limit)
        min_effort = torch.clip(min_effort, min=-self.effort_limit, max=0.0)

        return torch.clip(effort, min=min_effort, max=max_effort)
