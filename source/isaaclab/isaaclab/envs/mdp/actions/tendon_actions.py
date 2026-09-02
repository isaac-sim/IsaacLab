# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for articulations whose motors drive fixed tendons."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class FixedTendonPositionAction(ActionTerm):
    r"""Position targets for an articulation's fixed tendons.

    An underactuated hand has fewer motors than joints because some motors pull a tendon spanning
    several joints. Tendons are a separate entity from joints in the simulation, with their own
    index space, so a joint-position term cannot address one -- there is no joint to target.

    Combine this with a joint action term to cover a hand whose motors are of both kinds; the
    action manager concatenates the terms in the order the configuration declares them. The
    articulation decides how a tendon target reaches its solver, so this term is backend-neutral.
    """

    cfg: actions_cfg.FixedTendonPositionActionCfg
    """The configuration of the action term."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the processed action."""

    _asset: Articulation
    """The articulation asset on which the term is applied."""

    def __init__(self, cfg: actions_cfg.FixedTendonPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve as a proxy and keep the torch view: a plain list would be converted to a fresh
        # device array on every apply_actions, which is a per-step allocation on the control path.
        tendon_ids, self._tendon_names = self._asset.find_fixed_tendons(
            cfg.tendon_names, preserve_order=cfg.preserve_order, as_proxy=True
        )
        self._num_tendons = len(tendon_ids)
        self._tendon_ids = tendon_ids.torch
        # log the resolved tendon names for debugging
        logger.info(
            f"Resolved tendon names for the action term {self.__class__.__name__}:"
            f" {self._tendon_names} [{self._tendon_ids}]"
        )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._scale = cfg.scale
        self._offset = cfg.offset
        # parse clip
        if cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.clip, self._tendon_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

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
        """The IO descriptor of the action term.

        Adds the tendon names, scale, offset and clip to the base descriptor.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "FixedTendonPositionAction"
        self._IO_descriptor.tendon_names = self._tendon_names
        self._IO_descriptor.scale = self._scale
        self._IO_descriptor.offset = self._offset
        self._IO_descriptor.clip = self._clip[0].detach().cpu().numpy().tolist() if self.cfg.clip is not None else None
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # The raw action is clamped to [-1, 1] so scale and offset alone define the commanded span; a
        # Gaussian policy samples well past 1 early on, which would yank the fingers out of range.
        # Joint terms instead clip the processed action through ``cfg.clip``, honoured here as well.
        self._processed_actions[:] = self._raw_actions.clamp(-1.0, 1.0) * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions[:] = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        # The target is the tendon's own length coordinate. For one command to mean the same thing
        # on every backend, the asset must author each engine's tendon so their length coordinates
        # agree; that agreement belongs in the asset, not in a per-backend branch here.
        self._asset.set_fixed_tendon_position_target_index(
            target=self._processed_actions, fixed_tendon_ids=self._tendon_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
