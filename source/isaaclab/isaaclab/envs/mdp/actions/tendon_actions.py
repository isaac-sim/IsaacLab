# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for articulations whose motors drive fixed tendons."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import BaseArticulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


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

    _asset: BaseArticulation
    """The articulation asset on which the term is applied."""

    def __init__(self, cfg: actions_cfg.FixedTendonPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve as a proxy and keep the torch view: a plain list would be converted to a fresh
        # device array on every apply_actions, which is a per-step allocation on the control path.
        tendon_ids, self._tendon_names = self._asset.find_fixed_tendons(
            cfg.tendon_names, preserve_order=True, as_proxy=True
        )
        self._num_tendons = len(tendon_ids)
        self._tendon_ids = tendon_ids.torch

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._scale = float(cfg.scale)
        self._offset = float(cfg.offset)

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

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # Clip to [-1, 1] before mapping onto the tendon's span, exactly as the joint terms do.
        # A policy samples from an unbounded Gaussian, so early actions reach well past 1; without
        # this the command leaves the tendon's reachable range entirely and yanks the fingers.
        self._processed_actions[:] = self._raw_actions.clamp(-1.0, 1.0) * self._scale + self._offset

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
