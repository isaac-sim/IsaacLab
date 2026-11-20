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
from isaaclab.utils import index_fill_

if TYPE_CHECKING:
    from ... import ManagerBasedEnv
    from ...utils.io_descriptors import GenericActionIODescriptor
    from . import actions_cfg

logger = logging.getLogger(__name__)


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
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order, as_proxy=True
        )
        self._num_joints = len(joint_ids)
        self._joint_ids = joint_ids.torch
        # log the resolved joint names for debugging
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
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
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
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
        if isinstance(self._offset, torch.Tensor):
            self._IO_descriptor.offset = self._offset[0].detach().cpu().numpy().tolist()
        else:
            self._IO_descriptor.offset = self._offset
        # FIXME: This is not correct. Add list support.
        if self.cfg.clip is not None:
            if isinstance(self._clip, torch.Tensor):
                self._IO_descriptor.clip = self._clip[0].detach().cpu().numpy().tolist()
            else:
                self._IO_descriptor.clip = self._clip
        else:
            self._IO_descriptor.clip = None
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations in place: a recorded apply_actions reads this buffer
        self._processed_actions[:] = self._raw_actions * self._scale
        self._processed_actions.add_(self._offset)
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions.clamp_(min=self._clip[:, :, 0], max=self._clip[:, :, 1])

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        index_fill_(self._raw_actions, env_ids, 0.0)


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    is_graphable = True

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target_index(target=self.processed_actions, joint_ids=self._joint_ids)


class JointPositionChunkAction(JointPositionAction):
    """Joint position action that applies a chunk of position targets in order over the environment step.

    :class:`JointPositionAction` applies one target on all ``decimation`` physics steps of an environment step. This
    term instead takes a chunk of :attr:`~JointPositionChunkActionCfg.chunk_length` targets and applies them in order,
    each for ``decimation / chunk_length`` physics steps, so a policy that plans a chunk of future targets acts once
    per chunk. With :attr:`~JointPositionChunkActionCfg.interpolate`, the action is one target per joint instead,
    reached in equal increments from the previous chunk's last target, or from the joint positions after a reset.

    The action and :attr:`processed_actions` are ordered by target, then by joint. Scale, offset and clip apply to
    each target. The physics-step counter stays on the device, so a physics backend that records :meth:`apply_actions`
    once and replays it on every physics step applies the right target each time. A backend that runs the decimation
    loop itself must also run the actions, since the targets change between its physics steps.
    """

    is_graphable = True

    cfg: actions_cfg.JointPositionChunkActionCfg
    """The configuration of the action term."""

    _steps = 1
    """Targets per environment step; one while :class:`JointAction` sizes the per-joint scale, offset and clip."""

    def __init__(self, cfg: actions_cfg.JointPositionChunkActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        decimation = env.cfg.decimation
        steps = decimation if cfg.chunk_length is None else cfg.chunk_length
        if steps < 1 or decimation % steps:
            raise ValueError(
                f"chunk_length must be a positive divisor of the environment step's {decimation} physics steps"
                f" (decimation), so that each target holds for the same number of physics steps; got {steps}."
            )
        self._steps, self._hold = steps, decimation // steps
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # The chunk's targets, flat like every term's processed actions, and viewed one row per target.
        self._processed_actions = torch.zeros(self.num_envs, steps * self._num_joints, device=self.device)
        self._chunk = self._processed_actions.view(self.num_envs, steps, self._num_joints)
        # The physics step since the action, and the target that step applies.
        self._physics_step = torch.zeros(1, dtype=torch.long, device=self.device)
        self._index = torch.zeros_like(self._physics_step)
        self._target = torch.zeros(self.num_envs, 1, self._num_joints, device=self.device)
        # Interpolation: the increments, and which environments start from their joint positions after a reset.
        self._fractions = (torch.arange(1, steps + 1, device=self.device) / steps).view(1, -1, 1)
        self._restart = torch.ones(self.num_envs, 1, 1, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._num_joints * (1 if self.cfg.interpolate else self._steps)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        targets = self._raw_actions.view(self.num_envs, -1, self._num_joints)
        targets = targets * _per_target(self._scale) + _per_target(self._offset)
        if self.cfg.clip is not None:
            targets = targets.clamp(min=self._clip[:, None, :, 0], max=self._clip[:, None, :, 1])
        if self.cfg.interpolate:
            # From the previous chunk's last target, which the joints may still lag, so the command never steps back.
            joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids].unsqueeze(1)
            start = torch.where(self._restart, joint_pos, self._chunk[:, -1:])
            targets = start + self._fractions * (targets - start)
            self._restart.zero_()
        # in place: a recorded apply_actions reads this buffer
        self._chunk[:] = targets
        self._physics_step.zero_()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        index_fill_(self._restart, env_ids, True)

    def apply_actions(self):
        if getattr(self._env, "_physics_handles_decimation", False) and not self._env._physics_handles_actions:
            raise RuntimeError(
                f"{type(self).__name__} changes the joint targets between physics steps, but the physics backend runs"
                " the decimation loop without running the actions, so only the chunk's first target would apply."
                " Make every action term and asset graphable so that the backend runs the actions too."
            )
        torch.div(self._physics_step, self._hold, rounding_mode="floor", out=self._index)
        self._index.clamp_(max=self._steps - 1)
        torch.index_select(self._chunk, 1, self._index, out=self._target)
        self._physics_step.add_(1)
        self._asset.set_joint_position_target_index(target=self._target[:, 0], joint_ids=self._joint_ids)


def _per_target(value: torch.Tensor | float) -> torch.Tensor | float:
    """Broadcast a per-joint scale or offset ``(N, J)`` over the targets of a chunk ``(N, K, J)``."""
    return value.unsqueeze(1) if isinstance(value, torch.Tensor) else value


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
            self._offset = 0.0

    def apply_actions(self):
        # add current joint positions to the processed actions
        current_actions = self.processed_actions + self._asset.data.joint_pos.torch[:, self._joint_ids]
        # set position targets
        self._asset.set_joint_position_target_index(target=current_actions, joint_ids=self._joint_ids)


class JointVelocityAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    is_graphable = True

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel.torch[:, self._joint_ids].clone()

    def apply_actions(self):
        # set joint velocity targets
        self._asset.set_joint_velocity_target_index(target=self.processed_actions, joint_ids=self._joint_ids)


class JointEffortAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

    is_graphable = True

    cfg: actions_cfg.JointEffortActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint effort targets
        self._asset.set_joint_effort_target_index(target=self.processed_actions, joint_ids=self._joint_ids)
