# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import ClassVar, Optional, Tuple
from typing_extensions import Literal

from omni.isaac.orbit.sim import SpawnerCfg
from omni.isaac.orbit.utils import configclass


@configclass
class AssetBaseCfg:
    """Configuration parameters for an asset."""

    @configclass
    class InitialStateCfg:
        """Initial state of the asset."""

        # root position
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` of the root in simulation world frame.
        Defaults to (1.0, 0.0, 0.0, 0.0).
        """

    cls_name: ClassVar[str] = MISSING
    """Class name of the asset."""

    prim_path: str = MISSING
    """Prim path (or expression) to the asset.

    .. note::
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Robot`` will be replaced with ``/World/envs/env_.*/Robot``.
    """

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose."""

    spawn: Optional[SpawnerCfg] = MISSING
    """Spawn configuration for the asset.

    If :obj:`None`, then no prims are spawned by the asset class. Instead, it is assumed that the
    asset is already present in the scene.
    """

    collision_group: Literal[0, -1] = 0
    """Collision group of the asset. Defaults to ``0``.

    * ``-1``: global collision group (collides with all assets in the scene).
    * ``0``: local collision group (collides with other assets in the same environment).
    """

    debug_vis: bool = False
    """Whether to enable debug visualization for the asset. Defaults to ``False``."""
