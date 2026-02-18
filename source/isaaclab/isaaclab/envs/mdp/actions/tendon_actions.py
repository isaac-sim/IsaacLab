# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

logger = logging.getLogger(__name__)


class TendonAction(ActionTerm):
    """Action term that writes control values to tendon actuators.

    Tendon actuators in MuJoCo/Newton use CTRL_DIRECT mode and are controlled by
    writing to ``control.mujoco.ctrl``. The MuJoCo solver then applies the gain/bias
    model defined in the MJCF to produce joint forces through tendon coupling.

    This action term accepts continuous control values and writes them to the
    tendon actuator ctrl buffer after applying scale and offset transformations.
    """

    cfg: actions_cfg.TendonActionCfg
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.TendonActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        _, self._tendon_names, self._tendon_ids = self._asset.find_fixed_tendons(self.cfg.tendon_names)
        self._num_tendons = len(self._tendon_ids)
        logger.info(
            f"Resolved tendon names for the action term {self.__class__.__name__}:"
            f" {self._tendon_names} [{self._tendon_ids}]"
        )

        self._raw_actions = torch.zeros(self.num_envs, self._num_tendons, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # parse scale
        if isinstance(self.cfg.scale, dict):
            self._scale = torch.ones(self._num_tendons, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._tendon_names
            )
            self._scale[index_list] = torch.tensor(value_list, device=self.device)
        else:
            self._scale = self.cfg.scale

        # parse offset
        if isinstance(self.cfg.offset, dict):
            self._offset = torch.zeros(self._num_tendons, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.offset, self._tendon_names
            )
            self._offset[index_list] = torch.tensor(value_list, device=self.device)
        else:
            self._offset = self.cfg.offset

    @property
    def action_dim(self) -> int:
        return self._num_tendons

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "TendonAction"
        self._IO_descriptor.joint_names = self._tendon_names
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        torch.mul(self._raw_actions, self._scale, out=self._processed_actions)
        self._processed_actions.add_(self._offset)

    def apply_actions(self):
        self._asset.set_tendon_actuator_target(self._processed_actions, tendon_names=self._tendon_names)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class BinaryTendonAction(ActionTerm):
    """Binary action term for tendon actuators (e.g., gripper open/close).

    Maps a binary action (positive = open, negative/zero = close) to tendon
    actuator control values. This is analogous to :class:`BinaryJointPositionAction`
    but targets tendon actuators via ``control.mujoco.ctrl``.
    """

    cfg: actions_cfg.BinaryTendonActionCfg
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.BinaryTendonActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        _, self._tendon_names, self._tendon_ids = self._asset.find_fixed_tendons(self.cfg.tendon_names)
        self._num_tendons = len(self._tendon_ids)
        logger.info(
            f"Resolved tendon names for the action term {self.__class__.__name__}:"
            f" {self._tendon_names} [{self._tendon_ids}]"
        )

        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_tendons, device=self.device)

        # parse open command
        self._open_command = torch.zeros(self._num_tendons, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._tendon_names
        )
        if len(index_list) != self._num_tendons:
            raise ValueError(
                f"Could not resolve all tendons for the action term. Missing: {set(self._tendon_names) - set(name_list)}"
            )
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse close command
        self._close_command = torch.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._tendon_names
        )
        if len(index_list) != self._num_tendons:
            raise ValueError(
                f"Could not resolve all tendons for the action term. Missing: {set(self._tendon_names) - set(name_list)}"
            )
        self._close_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self._num_tendons, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._tendon_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "TendonAction"
        self._IO_descriptor.joint_names = self._tendon_names
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # positive/1 = open, negative/0 = close
        if actions.dtype == torch.bool:
            binary_mask = actions == 0
        else:
            binary_mask = actions < 0
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        self._asset.set_tendon_actuator_target(self._processed_actions, tendon_names=self._tendon_names)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
