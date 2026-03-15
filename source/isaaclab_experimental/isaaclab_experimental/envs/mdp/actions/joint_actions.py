# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation

from isaaclab_experimental.managers.action_manager import ActionTerm
from isaaclab_experimental.utils.warp import resolve_1d_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


@wp.kernel
def _process_joint_actions_kernel(
    # input
    actions: wp.array(dtype=wp.float32, ndim=2),
    action_offset: int,
    # params
    scale: wp.array(dtype=wp.float32),
    offset: wp.array(dtype=wp.float32),
    clip: wp.array(dtype=wp.float32, ndim=2),
    # output
    raw_out: wp.array(dtype=wp.float32, ndim=2),
    processed_out: wp.array(dtype=wp.float32, ndim=2),
):
    env_id, j = wp.tid()
    col = action_offset + j

    a = actions[env_id, col]
    raw_out[env_id, j] = a

    x = a * scale[j] + offset[j]
    low = clip[j, 0]
    high = clip[j, 1]
    if x < low:
        x = low
    if x > high:
        x = high
    processed_out[env_id, j] = x


@wp.kernel
def _zero_masked_2d(mask: wp.array(dtype=wp.bool), values: wp.array(dtype=wp.float32, ndim=2)):
    env_id, j = wp.tid()
    if mask[env_id]:
        values[env_id, j] = 0.0


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
    _scale: wp.array
    """The scaling factor applied to the input action."""
    _offset: wp.array
    """The offset applied to the input action."""
    _clip: wp.array
    """The clip applied to the input action."""
    _joint_mask: wp.array
    """A persistent joint mask for capturable action application."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # FIXME: ArticulationData.resolve_joint_mask is not available on this branch.
        #  Port resolve_*_mask methods from dev/newton when articulation_data is aligned.
        _all_joint_mask = wp.ones((self._asset.num_joints,), dtype=wp.bool, device=self.device)
        _scratch_joint_mask = wp.zeros((self._asset.num_joints,), dtype=wp.bool, device=self.device)
        self._joint_mask = wp.clone(
            resolve_1d_mask(
                ids=self._joint_ids,
                mask=None,
                all_mask=_all_joint_mask,
                scratch_mask=_scratch_joint_mask,
                device=self.device,
            )
        )

        # create tensors for raw and processed actions (Warp)
        self._raw_actions = wp.zeros((self.num_envs, self.action_dim), dtype=wp.float32, device=self.device)
        self._processed_actions = wp.zeros_like(self.raw_actions)
        # FIXME: dev/newton set_joint_effort_target accepts partial data + joint_mask. Our branch
        #  has separate _index (partial data) and _mask (full data) variants. Pre-compute joint_ids
        #  as warp array for the _index variant.
        if self._joint_ids == slice(None):
            self._joint_ids_wp = None  # None means all joints
        else:
            self._joint_ids_wp = wp.array(list(self._joint_ids), dtype=wp.int32, device=self.device)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = wp.array([float(cfg.scale)] * self.action_dim, dtype=wp.float32, device=self.device)
        elif isinstance(cfg.scale, dict):
            scale_per_joint = [1.0] * self.action_dim
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            for idx, value in zip(index_list, value_list):
                scale_per_joint[idx] = float(value)
            self._scale = wp.array(scale_per_joint, dtype=wp.float32, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = wp.array([float(cfg.offset)] * self.action_dim, dtype=wp.float32, device=self.device)
        elif isinstance(cfg.offset, dict):
            offset_per_joint = [0.0] * self.action_dim
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            for idx, value in zip(index_list, value_list):
                offset_per_joint[idx] = float(value)
            self._offset = wp.array(offset_per_joint, dtype=wp.float32, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

        # parse clip
        clip_low = [-float("inf")] * self.action_dim
        clip_high = [float("inf")] * self.action_dim
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                for idx, value in zip(index_list, value_list):
                    clip_low[idx] = float(value[0])
                    clip_high[idx] = float(value[1])
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

        clip_np = np.column_stack([clip_low, clip_high]).astype(np.float32)
        self._clip = wp.array(clip_np, dtype=wp.float32, device=self.device)

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

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the joint action.
        It adds the following information to the base descriptor:
        - joint_names: The names of the joints.
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointAction"
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.scale = self._scale
        # This seems to be always [4xNum_joints] IDK why. Need to check.
        if isinstance(self._offset, wp.array):
            self._IO_descriptor.offset = self._offset.numpy().tolist()
        else:
            self._IO_descriptor.offset = None
        # FIXME: This is not correct. Add list support.
        if self.cfg.clip is not None:
            if isinstance(self._clip, wp.array):
                self._IO_descriptor.clip = self._clip.numpy().tolist()
            else:
                self._IO_descriptor.clip = None
        else:
            self._IO_descriptor.clip = None
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: wp.array, action_offset: int = 0):
        wp.launch(
            kernel=_process_joint_actions_kernel,
            dim=(self.num_envs, self.action_dim),
            inputs=[
                actions,
                int(action_offset),
                self._scale,
                self._offset,
                self._clip,
                self._raw_actions,
                self._processed_actions,
            ],
            device=self.device,
        )

    def reset(self, env_mask: wp.array | None = None) -> None:
        """Resets the action term (mask-based)."""
        if env_mask is None:
            self._raw_actions.fill_(0.0)
            return
        wp.launch(
            kernel=_zero_masked_2d,
            dim=(self.num_envs, self.action_dim),
            inputs=[env_mask, self._raw_actions],
            device=self.device,
        )


class JointEffortAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

    cfg: actions_cfg.JointEffortActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint effort targets
        # FIXME: dev/newton uses set_joint_effort_target(data, joint_mask=) which accepts
        #  partial data. Our branch uses the separate _index variant for partial data.
        self._asset.set_joint_effort_target_index(target=self.processed_actions, joint_ids=self._joint_ids_wp)
