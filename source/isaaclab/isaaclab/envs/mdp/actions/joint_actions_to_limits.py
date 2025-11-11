# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp
from typing import TYPE_CHECKING
from collections.abc import Sequence

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.warp.update_kernels import update_array2D_with_array1D_indexed, update_array2D_with_value, update_array2D_with_value_masked, update_array2D_with_array2D_masked
from isaaclab.envs.mdp.kernels.action_kernels import process_joint_position_to_limits_action, process_ema_joint_position_to_limits_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointPositionToLimitsAction(ActionTerm):
    """Joint position action term that scales the input actions to the joint limits and applies them to the
    articulation's joints.

    This class is similar to the :class:`JointPositionAction` class. However, it performs additional
    re-scaling of input actions to the actuator joint position limits.

    While processing the actions, it performs the following operations:

    1. Apply scaling to the raw actions based on :attr:`actions_cfg.JointPositionToLimitsActionCfg.scale`.
    2. Clip the scaled actions to the range [-1, 1] and re-scale them to the joint limits if
       :attr:`actions_cfg.JointPositionToLimitsActionCfg.rescale_to_limits` is set to True.

    The processed actions are then sent as position commands to the articulation's joints.
    """

    cfg: actions_cfg.JointPositionToLimitsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: wp.array(dtype=wp.float32)
    """The scaling factor applied to the input action."""
    _clip: wp.array(dtype=wp.vec2f)
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_masks, self._joint_names, self._joint_ids = self._asset.find_joints(self.cfg.joint_names)
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
            self._scale = wp.zeros((self.num_envs, self.action_dim), device=self.device, dtype=wp.float32)
            self._scale.fill_(float(cfg.scale))
        elif isinstance(cfg.scale, dict):
            self._scale = wp.ones((env.num_envs, self.action_dim), device=self.device, dtype=wp.float32)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            wp.launch(
                update_array2D_with_array1D_indexed,
                dim=(self.num_envs, len(index_list)),
                inputs=[
                    wp.array(value_list, dtype=wp.float32, device=self.device),
                    self._scale,
                    None,
                    wp.array(index_list, dtype=wp.int32, device=self.device),
                ],
            )
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
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

    # FIXME: Do we need to store the raw actions?
    def process_actions(self, actions: wp.array):
        # store the raw actions
        self._raw_actions.assign(actions)
        wp.launch(
            process_joint_position_to_limits_action,
            dim=(self.num_envs, self.action_dim),
            inputs=[
                self._raw_actions,
                self._scale,
            ],
        )



    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None, mask: wp.array(dtype=wp.bool) | None = None) -> None:
        if mask is not None:
            wp.launch(
                update_array2D_with_value_masked,
                dim=(self.num_envs, self.action_dim),
                inputs=[
                    wp.zeros((self.num_envs, self.action_dim), device=self.device, dtype=wp.float32),
                    self._raw_actions,
                    mask,
                    None,
                ],
            )
        else:
            self._raw_actions.fill_(0.0)


class EMAJointPositionToLimitsAction(JointPositionToLimitsAction):
    r"""Joint action term that applies exponential moving average (EMA) over the processed actions as the
    articulation's joints position commands.

    Exponential moving average (EMA) is a type of moving average that gives more weight to the most recent data points.
    This action term applies the processed actions as moving average position action commands.
    The moving average is computed as:

    .. math::

        \text{applied action} = \alpha \times \text{processed actions} + (1 - \alpha) \times \text{previous applied action}

    where :math:`\alpha` is the weight for the moving average, :math:`\text{processed actions}` are the
    processed actions, and :math:`\text{previous action}` is the previous action that was applied to the articulation's
    joints.

    In the trivial case where the weight is 1.0, the action term behaves exactly like
    the :class:`JointPositionToLimitsAction` class.

    On reset, the previous action is initialized to the current joint positions of the articulation's joints.
    """

    cfg: actions_cfg.EMAJointPositionToLimitsActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.EMAJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # parse and save the moving average weight
        if isinstance(cfg.alpha, float):
            # check that the weight is in the valid range
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {cfg.alpha}.")
            self._alpha = wp.zeros((self.num_envs, self.action_dim), device=self.device, dtype=wp.float32)
            self._alpha.fill_(cfg.alpha)
        elif isinstance(cfg.alpha, dict):
            self._alpha = wp.ones((env.num_envs, self.action_dim), device=self.device)
            # resolve the dictionary config
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
            # check that the weights are in the valid range
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Moving average weight must be in the range [0, 1]. Got {value} for joint {name}."
                    )
            wp.launch(
                update_array2D_with_array1D_indexed,
                dim=(env.num_envs, len(index_list)),
                inputs=[
                    wp.array(value_list, dtype=wp.float32, device=self.device),
                    self._alpha,
                    None,
                    wp.array(index_list, dtype=wp.int32, device=self.device),
                ],
            )
        else:
            raise ValueError(
                f"Unsupported moving average weight type: {type(cfg.alpha)}. Supported types are float and dict."
            )

        # initialize the previous targets
        self._prev_applied_actions = wp.zeros_like(self.processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None, mask: wp.array(dtype=wp.bool) | None = None) -> None:
        # check if specific environment ids are provided
        super().reset(env_ids, mask)
        # reset history to current joint positions
        wp.launch(
            update_array2D_with_array2D_masked,
            dim=(self.num_envs, self.action_dim),
            inputs=[
                self._asset.data.joint_pos,
                self._prev_applied_actions,
                mask,
                None,
            ],
        )

    def process_actions(self, actions: wp.array):
        # apply affine transformations
        super().process_actions(actions)
        wp.launch(
            process_ema_joint_position_to_limits_action,
            dim=(self.num_envs, self.action_dim),
            inputs=[
                self._processed_actions,
                self._alpha,
                self._clip,
            ],
        )