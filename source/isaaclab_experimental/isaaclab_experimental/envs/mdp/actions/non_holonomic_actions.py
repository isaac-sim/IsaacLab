# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.utils.warp.update_kernels import update_array2D_with_value_masked
from isaaclab_experimental.managers.action_manager import ActionTerm
from isaaclab_experimental.envs.mdp.kernels.action_kernels import process_non_holonomic_action, apply_non_holonomic_action

if TYPE_CHECKING:
    from isaaclab_experimental.envs import ManagerBasedEnvWarp
    from . import actions_cfg


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
    _scale: wp.array(dtype=wp.float32)
    """The scaling factor applied to the input action. Shape is (1, 2)."""
    _offset: wp.array(dtype=wp.float32)
    """The offset applied to the input action. Shape is (1, 2)."""
    _clip: wp.array(dtype=wp.vec2f)
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.NonHolonomicActionCfg, env: ManagerBasedEnvWarp):
        # initialize the action term
        super().__init__(cfg, env)

        # parse the joint information
        # -- x joint
        x_joint_mask, x_joint_name, x_joint_id = self._asset.find_joints(self.cfg.x_joint_name)
        if len(x_joint_id) != 1:
            raise ValueError(
                f"Expected a single joint match for the x joint name: {self.cfg.x_joint_name}, got {len(x_joint_id)}"
            )
        # -- y joint
        y_joint_mask, y_joint_name, y_joint_id = self._asset.find_joints(self.cfg.y_joint_name)
        if len(y_joint_id) != 1:
            raise ValueError(f"Found more than one joint match for the y joint name: {self.cfg.y_joint_name}")
        # -- yaw joint
        yaw_joint_mask, yaw_joint_name, yaw_joint_id = self._asset.find_joints(self.cfg.yaw_joint_name)
        if len(yaw_joint_id) != 1:
            raise ValueError(f"Found more than one joint match for the yaw joint name: {self.cfg.yaw_joint_name}")
        # parse the body index
        self._body_mask, self._body_name, self._body_idx = self._asset.find_bodies(self.cfg.body_name)
        if len(self._body_idx) != 1:
            raise ValueError(f"Found more than one body match for the body name: {self.cfg.body_name}")

        # process into a list of joint ids
        self._joint_ids = [x_joint_id[0], y_joint_id[0], yaw_joint_id[0]]
        self._joint_names = [x_joint_name[0], y_joint_name[0], yaw_joint_name[0]]
        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = wp.zeros((self.num_envs, self.action_dim), device=self.device, dtype=wp.float32)
        self._processed_actions = wp.zeros_like(self.raw_actions)
        self._joint_vel_command = wp.zeros((self.num_envs, 3), device=self.device, dtype=wp.float32)
        self._yaw_w = wp.zeros((self.num_envs,), device=self.device, dtype=wp.float32)

        # save the scale and offset as tensors
        self._scale = wp.array(self.cfg.scale, device=self.device, dtype=wp.float32)
        self._offset = wp.array(self.cfg.offset, device=self.device, dtype=wp.float32)
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = wp.zeros((self.num_envs, self.action_dim), device=self.device, dtype=wp.vec2f)
                wp.launch(
                    update_array2D_with_value,
                    dim=(self.num_envs, self.action_dim),
                    inputs=[
                        wp.vec2f(-float("inf"), float("inf")),
                        self._clip,
                    ],
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                wp.launch(
                    update_array2D_with_array1D_indexed,
                    dim=(self.num_envs, len(index_list)),
                    inputs=[
                        wp.array(value_list, dtype=wp.vec2f, device=self.device),
                        self._clip,
                        None,
                        wp.array(index_list, dtype=wp.int32, device=self.device),
                    ],
                )
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> wp.array:
        return self._raw_actions

    @property
    def processed_actions(self) -> wp.array:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions. NOTE: This is a reference, not a copy.
        self._raw_actions = actions
        wp.launch(
            process_non_holonomic_action,
            dim=(self.num_envs, self.action_dim),
            inputs=[
                self._raw_actions,
                self._processed_actions,
                self._scale,
                self._offset,
                self._clip,
            ],
        )

    def apply_actions(self):
        wp.launch(
            apply_non_holonomic_action,
            dim=(self.num_envs),
            inputs=[
                self._asset.data.body_pose_w,
                self._yaw_w,
                self._processed_actions,
                self._joint_vel_command,
                self._body_idx,
            ],
        )
        self._asset.set_joint_velocity_target(self._joint_vel_command, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None, mask: wp.array(dtype=wp.bool) | None = None) -> None:
        wp.launch(
            update_array2D_with_value_masked,
            dim=(self.num_envs, self.action_dim),
            inputs=[
                0.0,
                self._raw_actions,
                mask,
                None,
            ],
        )
