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
from isaaclab.utils.warp.update_kernels import update_array2D_with_array1D_indexed, update_array2D_with_value_masked, update_array2D_with_value, update_array1D_with_array1D_indexed
from isaaclab.envs.mdp.kernels.action_kernels import process_joint_action, apply_relative_joint_position_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: wp.array(dtype=wp.float32)
    """The scaling factor applied to the input action."""
    _offset: wp.array(dtype=wp.float32)
    """The offset applied to the input action."""
    _clip: wp.array(dtype=wp.vec2f)
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_masks, self._joint_names, self._joint_ids = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = wp.zeros((self.num_envs, self.action_dim), device=self.device, dtype=wp.float32)
        self._processed_actions = wp.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = wp.zeros((self.action_dim), device=self.device, dtype=wp.float32)
            self._scale.fill_(float(cfg.scale))
        elif isinstance(cfg.scale, dict):
            self._scale = wp.zeros((self.action_dim), device=self.device, dtype=wp.float32)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            wp.launch(
                update_array1D_with_array1D_indexed,
                dim=(len(index_list)),
                inputs=[
                    wp.array(value_list, dtype=wp.float32, device=self.device),
                    self._scale,
                    None,
                    wp.array(index_list, dtype=wp.int32, device=self.device),
                ],
            )
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = wp.zeros((self.action_dim), device=self.device, dtype=wp.float32)
            self._offset.fill_(float(cfg.offset))
        elif isinstance(cfg.offset, dict):
            self._offset = wp.zeros((self.action_dim), device=self.device, dtype=wp.float32)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            wp.launch(
                update_array1D_with_array1D_indexed,
                dim=(len(index_list)),
                inputs=[
                    wp.array(value_list, dtype=wp.float32, device=self.device),
                    self._offset,
                    None,
                    wp.array(index_list, dtype=wp.int32, device=self.device),
                ],
            )
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
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
        else:
            self._clip = None

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

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
        print(self._clip)
        wp.launch(
            process_joint_action,
            dim=(self.num_envs, self.action_dim),
            inputs=[
                self._raw_actions,
                self._scale,
                self._offset,
                self._clip,
                self._processed_actions,
            ],
        )

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


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            wp.launch(
                update_array2D_with_array1D_indexed,
                dim=(self.num_envs, self.action_dim),
                inputs=[
                    self._asset.data.default_joint_pos[:, self._joint_ids],
                    self._offset,
                    None,
                    self._joint_ids,
                ],
            )

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class RelativeJointPositionAction(JointAction):
    r"""Joint action term that applies the processed actions to the articulation's joints as relative position commands.

    Unlike :class:`JointPositionAction`, this action term applies the processed actions as relative position commands.
    This means that the processed actions are added to the current joint positions of the articulation's joints
    before being sent as position commands.

    This means that the action applied at every step is:

    .. math::

         \text{applied action} = \text{current joint positions} + \text{processed actions}

    where :math:`\text{current joint positions}` are the current joint positions of the articulation's joints.
    """

    cfg: actions_cfg.RelativeJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.RelativeJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use zero offset for relative position
        if cfg.use_zero_offset:
            self._offset.fill_(0.0)

        self.current_actions = wp.zeros((self.num_envs, self.action_dim), device=self.device, dtype=wp.float32)

    def apply_actions(self):
        # add current joint positions to the processed actions
        wp.launch(
            apply_relative_joint_position_action,
            dim=(self.num_envs, self.action_dim),
            inputs=[
                self.processed_actions,
                self.current_actions,
                self._asset.data.joint_pos,
                self._joint_ids,
            ],
        )
        # set position targets
        self._asset.set_joint_position_target(self.current_actions, joint_ids=self._joint_ids)


class JointVelocityAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            wp.launch(
                update_array2D_with_array1D_indexed,
                dim=(self.num_envs, self.action_dim),
                inputs=[
                    self._asset.data.default_joint_vel[:, self._joint_ids],
                    self._offset,
                    None,
                    self._joint_ids,
                ],
            )

    def apply_actions(self):
        # set joint velocity targets
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)


class JointEffortAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

    cfg: actions_cfg.JointEffortActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint effort targets
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)
