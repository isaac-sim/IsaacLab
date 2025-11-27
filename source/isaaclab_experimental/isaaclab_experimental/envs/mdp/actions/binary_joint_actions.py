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
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.warp.update_kernels import update_array2D_with_array1D_indexed, update_array2D_with_value, update_array2D_with_value_masked
from isaaclab.envs.mdp.kernels.action_kernels import where_array2D_float, where_array2D_binary, clip_array2D

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class BinaryJointAction(ActionTerm):
    """Base class for binary joint actions.

    This action term maps a binary action to the *open* or *close* joint configurations. These configurations are
    specified through the :class:`BinaryJointActionCfg` object. If the input action is a float vector, the action
    is considered binary based on the sign of the action values.

    Based on above, we follow the following convention for the binary action:

    1. Open action: 1 (bool) or positive values (float).
    2. Close action: 0 (bool) or negative values (float).

    The action term can mostly be used for gripper actions, where the gripper is either open or closed. This
    helps in devising a mimicking mechanism for the gripper, since in simulation it is often not possible to
    add such constraints to the gripper.
    """

    cfg: actions_cfg.BinaryJointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: wp.array(dtype=wp.vec2f)
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.BinaryJointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_masks, self._joint_names, self._joint_ids  = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = wp.zeros((self.num_envs, 1), device=self.device, dtype=wp.float32)
        self._processed_actions = wp.zeros((self.num_envs, self._num_joints), device=self.device, dtype=wp.float32)

        # parse open command
        self._open_command = wp.zeros((self._num_joints,), device=self.device, dtype=wp.float32)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        wp.launch(
            update_array2D_with_array1D_indexed,
            dim=(self.num_envs, len(index_list)),
            inputs=[
                wp.array(value_list, dtype=wp.float32, device=self.device),
                self._open_command,
                None,
                wp.array(index_list, dtype=wp.int32, device=self.device),
            ],
        )

        # parse close command
        self._close_command = wp.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        wp.launch(
            update_array2D_with_array1D_indexed,
            dim=(self.num_envs, self._num_joints),
            inputs=[
                wp.array(value_list, dtype=wp.float32, device=self.device),
                self._close_command,
                None,
                wp.array(index_list, dtype=wp.int32, device=self.device),
            ],
        )

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
                    dim=(self.num_envs, self.action_dim),
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
        return 1

    @property
    def raw_actions(self) -> wp.array:
        return self._raw_actions

    @property
    def processed_actions(self) -> wp.array:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: wp.array):
        # store the raw actions. NOTE: This is a reference, not a copy.
        self._raw_actions = actions
        # compute the binary mask
        if actions.dtype == wp.bool:
            # true: close, false: open
            wp.launch(
                where_array2D_binary,
                dim=(self.num_envs, self._num_joints),
                inputs=[
                    actions,
                    self._close_command,
                    self._open_command,
                    self._processed_actions,
                ],
            )
        else:
            # true: close, false: open
            wp.launch(
                where_array2D_float,
                dim=(self.num_envs, self._num_joints),
                inputs=[
                    actions,
                    0.0,
                    self._close_command,
                    self._open_command,
                    self._processed_actions,
                ],
            )
        # compute the command
        if self.cfg.clip is not None:
            wp.launch(
                clip_array2D,
                dim=(self.num_envs, self._num_joints),
                inputs=[
                    self._processed_actions,
                    self._clip,
                ],
            )

    def reset(self, env_ids: Sequence[int] | None = None, mask: wp.array(dtype=wp.bool) | None = None) -> None:
        if mask is not None:
            wp.launch(
                update_array2D_with_value_masked,
                dim=(self.num_envs, self._num_joints),
                inputs=[
                    0.0,
                    self._raw_actions,
                    mask,
                    None,
                ],
            )
        else:
            self._raw_actions.fill_(0.0)


class BinaryJointPositionAction(BinaryJointAction):
    """Binary joint action that sets the binary action into joint position targets."""

    cfg: actions_cfg.BinaryJointPositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


class BinaryJointVelocityAction(BinaryJointAction):
    """Binary joint action that sets the binary action into joint velocity targets."""

    cfg: actions_cfg.BinaryJointVelocityActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=self._joint_ids)
