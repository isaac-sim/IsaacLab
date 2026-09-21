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

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            # unmatched tendons keep scale 1, so a partial dictionary leaves them unscaled
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.scale, self._tendon_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.offset, self._tendon_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
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
        descriptor = super().IO_descriptor
        descriptor.shape = (self.action_dim,)
        descriptor.dtype = str(self.raw_actions.dtype)
        descriptor.action_type = "FixedTendonPositionAction"
        descriptor.tendon_names = self._tendon_names
        # a dictionary scale or offset resolves to a per-tendon tensor, which the descriptor
        # carries as plain values the way a joint term does
        for name in ("scale", "offset"):
            value = getattr(self, f"_{name}")
            if isinstance(value, torch.Tensor):
                value = value[0].detach().cpu().numpy().tolist()
            setattr(descriptor, name, value)
        descriptor.clip = self._clip[0].detach().cpu().numpy().tolist() if self.cfg.clip is not None else None
        return descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # Bounding the raw action would assume every caller sends normalized policy output, which is
        # the term's assumption about its users rather than a property of the tendon. The physical
        # bound is the task's to declare, through ``cfg.clip`` as every other action term does.
        self._processed_actions[:] = self._raw_actions * self._scale + self._offset
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
