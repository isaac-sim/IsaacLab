# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .cloner_strategies import random


@configclass
class CloneGroup:
    """Base class for legal clone-combination descriptors.

    Subclasses implement :meth:`resolve_assets` to decide which scene assets
    are active in one legal combination. The :attr:`weight` controls the
    relative sampling frequency of this combination.
    """

    weight: int = 1
    """Relative sampling weight for this clone combination."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        """Return the asset names included in this combination.

        Args:
            all_asset_names: All registered asset names in the scene.

        Returns:
            Subset of ``all_asset_names`` included in the combination.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement resolve_assets(). See CloneGroup docstring for an example."
        )


@configclass
class InclusionSet(CloneGroup):
    """Clone combination defined by explicitly listing active assets."""

    assets: list[str] = MISSING
    """Asset names active in this clone combination."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        """Return listed assets that are present in the scene config."""
        known = set(all_asset_names)
        return [asset_name for asset_name in self.assets if asset_name in known]


@configclass
class CloneCfg:
    """Configuration for environment replication.

    Holds the knobs :class:`~isaaclab.scene.InteractiveScene` forwards to
    :func:`~isaaclab.cloner.make_clone_plan` when building per-env layouts.
    """

    clone_strategy: Callable[..., object] = random
    """Function used to build prototype-to-environment mapping. Default is :func:`random`."""

    clone_combinations: list[CloneGroup] = []
    """Legal scene-asset combinations for heterogeneous clone planning.

    Each entry resolves to the asset names that are active in one legal
    combination. Assets not referenced by any entry are active in every
    combination. An empty list keeps the homogeneous/default behavior.
    """

    device: str = "cpu"
    """Torch device on which mapping buffers are allocated."""

    clone_regex: str = "/World/envs/env_.*"
    """Regex matching every replicated env prim. Used to expand ``{ENV_REGEX_NS}`` cfg macros."""
