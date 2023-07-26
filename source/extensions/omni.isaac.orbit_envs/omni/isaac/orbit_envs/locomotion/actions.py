from __future__ import annotations

import re
import torch
from dataclasses import MISSING

from omni.isaac.orbit.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.orbit.robots.robot_base import RobotBase
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import euler_xyz_from_quat


# -- Joint Action
class JointAction(ActionTerm):
    """Joint action that maps a vector of actions to a joint configuration."""

    _asset: RobotBase
    _scale: torch.Tensor | float = 1.0
    _offset: torch.Tensor | float = 0.0

    @property
    def action_dim(self) -> int:
        """Dimension of control actions."""
        return self._num_dofs

    def __init__(self, cfg: JointActionCfg, env: object) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the action dimension
        self._dof_ids, dof_names = self._asset.find_dofs(cfg.joint_name_expr)
        self._num_dofs = len(self._dof_ids)
        # Avoid indexing across all dofs for efficiency
        if self._num_dofs == self._asset.num_dof:
            self._dof_ids = ...

        # create tensors for raw and processed actions
        self.raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, float):
            self._scale = cfg.scale
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(1, self.action_dim, device=self.device)
            for index, dof_name in enumerate(dof_names):
                for re_expr, value in cfg.scale.items():
                    if re.fullmatch(re_expr, dof_name):
                        self._scale[:, index] = value
        # parse offset
        if isinstance(cfg.offset, float):
            self._offset = cfg.offset
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self.raw_actions)
            for index, dof_name in enumerate(dof_names):
                for re_expr, value in cfg.offset.items():
                    if re.fullmatch(re_expr, dof_name):
                        self._offset[:, index] = value

    def process_actions(self, actions):
        super().process_actions(actions)
        self.processed_actions = self.raw_actions * self._scale + self._offset


class JointPositionAction(JointAction):
    """Joint action that maps a vector of actions to a joint position configuration."""

    def __init__(self, cfg: JointPositionActionCfg, env: object) -> None:
        super().__init__(cfg, env)

        # use default dof positions as offset
        if cfg.offset_with_default:
            self._offset = self._asset.data.default_dof_pos
        else:
            self._offset = 0.0

    def apply_actions(self):
        self._asset.set_dof_position_targets(self.processed_actions, dof_ids=self._dof_ids)


class JointVelocityAction(JointAction):
    """Joint action that maps a vector of actions to a joint velocity configuration."""

    def __init__(self, cfg: JointVelocityActionCfg, env: object) -> None:
        super().__init__(cfg, env)

        # use default dof positions as offset
        if cfg.offset_with_default:
            self._offset = self._asset.data.default_dof_vel

    def apply_actions(self):
        self._asset.set_dof_velocity_targets(self.processed_actions, dof_ids=self._dof_ids)


@configclass
class JointActionCfg(ActionTermCfg):
    cls: type[ActionTerm] = JointAction
    """ Class of the action term."""
    joint_name_expr: list[str] = MISSING
    """Articulation's DOF name reqex that the action will be mapped to."""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions)."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions)."""


@configclass
class JointPositionActionCfg(JointActionCfg):
    cls: type[ActionTerm] = JointPositionAction
    """ Class of the action term."""
    offset_with_default: bool = True
    """Whether to use default dof positions as offset."""


@configclass
class JointVelocityActionCfg(JointActionCfg):
    cls: type[ActionTerm] = JointVelocityAction
    """ Class of the action term."""
    offset_with_default: bool = True
    """Whether to use default dof velocities as offset."""


# -- Non Holonomic Actions


class NonHolonomicAction(ActionTerm):
    """Non-holonomic action that maps a 2D action to a 3D velocity configuration.
    The action is a 2D vector (x and yaw velocities in base frame) that is mapped to the (world frame) velocity
    of the robot in the x, y and yaw directions."""

    _asset: RobotBase

    @property
    def action_dim(self) -> int:
        """Dimension of control actions."""
        return 2

    def __init__(self, cfg: NonHolonomicActionCfg, env: object) -> None:
        super().__init__(cfg, env)

        self.raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self.processed_actions = torch.zeros_like(self.raw_actions)

        self._dof_ids = []
        x_dof_id, x_dof_name = self._asset.find_dofs(cfg.x_dof_name)
        y_dof_id, y_dof_name = self._asset.find_dofs(cfg.y_dof_name)
        yaw_dof_id, yaw_dof_name = self._asset.find_dofs(cfg.yaw_dof_name)
        if len(x_dof_id) != 1 or len(y_dof_id) != 1 or len(yaw_dof_id) != 1:
            raise ValueError(
                f"x_dof_name, y_dof_name and yaw_dof_name must specify one dof each, found: x:{x_dof_name} y:{y_dof_name} yaw:{yaw_dof_name}"
            )
        self._dof_ids = [x_dof_id[0], y_dof_id[0], yaw_dof_id[0]]
        self.dof_vel = torch.zeros(self.num_envs, 3)

        self._body_idx, _ = self._asset.find_bodies(cfg.body_name)

        self._scale = torch.tensor(cfg.scale, device=self.device)
        self._offset = torch.tensor(cfg.scale, device=self.device)

    def process_actions(self, actions):
        super().process_actions(actions)
        scaled_actions = self.raw_actions * self._scale + self._offset
        # obtain current heading
        quat_w = self._asset.data.body_state_w[:, self._body_idx, 3:7]
        yaw_w = euler_xyz_from_quat(quat_w)[2]
        # compute dof velocities
        self.dof_vel[:, 0] = torch.cos(yaw_w) * scaled_actions[:, 0]  # x
        self.dof_vel[:, 1] = torch.sin(yaw_w) * scaled_actions[:, 0]  # y
        self.dof_vel[:, 2] = scaled_actions[:, 1]  # yaw
        self._asset.set_dof_velocity_targets(self.dof_vel, self._dof_ids)


@configclass
class NonHolonomicActionCfg:
    cls: type[ActionTerm] = NonHolonomicAction

    body_name: str = MISSING
    """ Name of the body with non-holonomic constraints"""
    x_dof_name: str = MISSING
    """X direction DOF name."""
    y_dof_name: str = MISSING
    """Y direction DOF name."""
    yaw_dof_name: str = MISSING
    """Yaw DOF name."""
    scale: tuple[float, float] = (1.0, 1.0)
    offset: tuple[float, float] = (1.0, 1.0)


# -- Gripper Action


class BinaryJointAction(ActionTerm):
    """Base class for binary joint actions.
    Defines two joint configurations, *open* and *close*, and maps a binary action to one of them."""

    _asset: RobotBase

    @property
    def action_dim(self) -> int:
        """Dimension of control actions."""
        return 1

    def __init__(self, cfg: BinaryJointActionCfg, env: object) -> None:
        super().__init__(cfg, env)

        self.raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self.processed_actions = torch.zeros_like(self.raw_actions)

        self._dof_ids, dof_names = self._asset.find_dofs(cfg.dof_names_expr)

        # process close/open commands
        self._command = torch.zeros(self.num_envs, len(self._dof_ids), device=self.device)
        self._open_command = torch.zeros(len(self._dof_ids), device=self.device)
        self._close_command = torch.zeros_like(self._open_command)
        for index, dof_name in enumerate(dof_names):
            found_open = False
            found_close = False
            for re_expr, value in cfg.open_command_expr.item():
                if re.fullmatch(re_expr, dof_name):
                    found_open = True
                    self._open_command[index] = value
            for re_expr, value in cfg.close_command_expr.item():
                if re.fullmatch(re_expr, dof_name):
                    found_close = True
                    self._close_command[index] = value
            if not found_open:
                raise ValueError(f"Could not find open command for dof {dof_name}")
            if not found_close:
                raise ValueError(f"Could not find close command for dof {dof_name}")

    def process_actions(self, actions) -> None:
        super().process_actions(actions)
        binary_mask = (actions == 0).unsqueeze(1)
        self._command = torch.where(binary_mask, self._close_command, self._open_command)


class BinaryJointPositionAction(BinaryJointAction):
    """Binary joint action that maps a binary action to a joint position configuration."""

    def apply_actions(self) -> None:
        self._asset.set_dof_position_targets(self._command)


class BinaryJointVelocityAction(BinaryJointAction):
    """Binary joint action that maps a binary action to a joint velocity configuration."""

    def apply_actions(self) -> None:
        self._asset.set_dof_velocity_targets(self._command)


@configclass
class BinaryJointActionCfg:
    cls: type[ActionTerm] = BinaryJointAction
    """ Class of the action term."""

    dof_names_expr: list[str] = MISSING
    """Articulation's DOF name reqex that the action will be mapped to."""

    open_command_expr: dict[str, float] = MISSING
    """The DOF command to move to *open* configuration.
    """

    close_command_expr: dict[str, float] = MISSING
    """The DOF command to move to *close* configuration.
    """


@configclass
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
    cls: type[ActionTerm] = BinaryJointPositionAction


@configclass
class BinaryJointVelocityActionCfg(BinaryJointActionCfg):
    cls: type[ActionTerm] = BinaryJointVelocityAction
