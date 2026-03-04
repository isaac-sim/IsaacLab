# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class NonHolonomicAction(ActionTerm):
    r"""Non-holonomic action that maps a two dimensional action to the velocity of the robot in
    the x, y and yaw directions.

    This action term helps model a skid-steer robot base. The action is a 2D vector which comprises of the
    forward velocity :math:`v_{B,x}` and the turning rate :\omega_{B,z}: in the base frame. Using the current
    base orientation, the commands are transformed into dummy joint velocity targets as:

    .. math::

        \dot{q}_{0, des} &= v_{B,x} \cos(\theta) \\
        \dot{q}_{1, des} &= v_{B,x} \sin(\theta) \\
        \dot{q}_{2, des} &= \omega_{B,z}

    where :math:`\theta` is the yaw of the 2-D base. Since the base is simulated as a dummy joint, the yaw is directly
    the value of the revolute joint along z, i.e., :math:`q_2 = \theta`.

    .. note::
        The current implementation assumes that the base is simulated with three dummy joints (prismatic joints along x
        and y, and revolute joint along z). This is because it is easier to consider the mobile base as a floating link
        controlled by three dummy joints, in comparison to simulating wheels which is at times is tricky because of
        friction settings.

        However, the action term can be extended to support other base configurations as well.

    .. tip::
        For velocity control of the base with dummy mechanism, we recommend setting high damping gains to the joints.
        This ensures that the base remains unperturbed from external disturbances, such as an arm mounted on the base.
    """

    cfg: actions_cfg.NonHolonomicActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, 2)."""
    _offset: torch.Tensor
    """The offset applied to the input action. Shape is (1, 2)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.NonHolonomicActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # parse the joint information
        # -- x joint
        x_joint_id, x_joint_name = self._asset.find_joints(self.cfg.x_joint_name)
        if len(x_joint_id) != 1:
            raise ValueError(
                f"Expected a single joint match for the x joint name: {self.cfg.x_joint_name}, got {len(x_joint_id)}"
            )
        # -- y joint
        y_joint_id, y_joint_name = self._asset.find_joints(self.cfg.y_joint_name)
        if len(y_joint_id) != 1:
            raise ValueError(f"Found more than one joint match for the y joint name: {self.cfg.y_joint_name}")
        # -- yaw joint
        yaw_joint_id, yaw_joint_name = self._asset.find_joints(self.cfg.yaw_joint_name)
        if len(yaw_joint_id) != 1:
            raise ValueError(f"Found more than one joint match for the yaw joint name: {self.cfg.yaw_joint_name}")
        # parse the body index
        self._body_idx, self._body_name = self._asset.find_bodies(self.cfg.body_name)
        if len(self._body_idx) != 1:
            raise ValueError(f"Found more than one body match for the body name: {self.cfg.body_name}")

        # process into a list of joint ids
        self._joint_ids = [x_joint_id[0], y_joint_id[0], yaw_joint_id[0]]
        self._joint_names = [x_joint_name[0], y_joint_name[0], yaw_joint_name[0]]
        # log info for debugging
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        logger.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel_command = torch.zeros(self.num_envs, 3, device=self.device)

        # save the scale and offset as tensors
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the non-holonomic action.
        It adds the following information to the base descriptor:
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.
        - body_name: The name of the body.
        - x_joint_name: The name of the x joint.
        - y_joint_name: The name of the y joint.
        - yaw_joint_name: The name of the yaw joint.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "non holonomic actions"
        self._IO_descriptor.scale = self._scale
        self._IO_descriptor.offset = self._offset
        self._IO_descriptor.clip = self._clip
        self._IO_descriptor.body_name = self._body_name
        self._IO_descriptor.x_joint_name = self._joint_names[0]
        self._IO_descriptor.y_joint_name = self._joint_names[1]
        self._IO_descriptor.yaw_joint_name = self._joint_names[2]
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        # obtain current heading
        quat_w = self._asset.data.body_quat_w[:, self._body_idx].view(self.num_envs, 4)
        yaw_w = euler_xyz_from_quat(quat_w)[2]
        # compute joint velocities targets
        self._joint_vel_command[:, 0] = torch.cos(yaw_w) * self.processed_actions[:, 0]  # x
        self._joint_vel_command[:, 1] = torch.sin(yaw_w) * self.processed_actions[:, 0]  # y
        self._joint_vel_command[:, 2] = self.processed_actions[:, 1]  # yaw
        # set the joint velocity targets
        self._asset.set_joint_velocity_target_index(target=self._joint_vel_command, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
