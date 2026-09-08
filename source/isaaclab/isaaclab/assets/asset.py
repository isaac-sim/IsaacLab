# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_cfg import expand_env_regex_ns
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from pxr import Usd

    from .asset_base_cfg import AssetBaseCfg


class Asset:
    """An asset that is authored without creating a runtime simulation view."""

    cfg: AssetBaseCfg
    """A copy of the configuration used to author the asset."""

    def __init__(self, cfg: AssetBaseCfg):
        """Author an asset from its configuration.

        Args:
            cfg: Configuration for the asset.

        Raises:
            RuntimeError: If the configured spawner does not return a valid prim.
        """
        cfg.validate()
        cfg.prim_path = expand_env_regex_ns(cfg.prim_path)
        self.cfg = cfg.copy()
        self.stage: Usd.Stage = get_current_stage()
        self._prim: Usd.Prim | None = None

        if self.cfg.spawn is not None:
            spawn_path = self.cfg.spawn.spawn_path or self.cfg.prim_path
            self._prim = self.cfg.spawn.func(
                spawn_path,
                self.cfg.spawn,
                translation=self.cfg.init_state.pos,
                orientation=self.cfg.init_state.rot,
            )
            if not self._prim:
                raise RuntimeError(f"Could not spawn prim at path {spawn_path}.")
            self.cfg._post_spawn(self.stage)

    @property
    def prim(self) -> Usd.Prim | None:
        """The prim returned by the spawner, or ``None`` when no spawner is configured."""
        return self._prim

    def set_visibility(self, visible: bool, env_ids: Sequence[int] | None = None):
        """Set the visibility of the asset prims.

        Args:
            visible: Whether to make the prims visible.
            env_ids: Environment indices. Defaults to all instances.
        """
        if not hasattr(self, "_prims"):
            self._prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if env_ids is None:
            env_ids = range(len(self._prims))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        for env_id in env_ids:
            sim_utils.set_prim_visibility(self._prims[env_id], visible)
