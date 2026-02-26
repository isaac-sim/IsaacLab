# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Any, Literal

from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass

from .asset_base import AssetBase


@configclass
class AssetBaseCfg:
    """The base configuration class for an asset's parameters.

    Please see the :class:`AssetBase` class for more information on the asset class.
    """

    @configclass
    class InitialStateCfg:
        """Initial state of the asset.

        This defines the default initial state of the asset when it is spawned into the simulation, as
        well as the default state when the simulation is reset.

        After parsing the initial state, the asset class stores this information in the :attr:`data`
        attribute of the asset class. This can then be accessed by the user to modify the state of the asset
        during the simulation, for example, at resets.
        """

        # root position
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation (x, y, z, w) of the root in simulation world frame.
        Defaults to (0.0, 0.0, 0.0, 1.0).
        """

    class_type: type[AssetBase] = None
    """The associated asset class. Defaults to None, which means that the asset will be spawned
    but cannot be interacted with via the asset class.

    The class should inherit from :class:`isaaclab.assets.asset_base.AssetBase`.
    """

    model_parameter_override: dict[str, tuple[Any, str | None]] = {}
    """Allows to override Newton Model parameters from the asset configuration. Defaults to an empty dictionary.

    dict[str, tuple[Any, str | None]]:
     - The key, is the name of the Newton Model parameter,
     - The first value in the tuple, is the value to override the parameter with. It can only be a scalar.
     - The second value in the tuple is the expression to be used to resolve the value. If None, then we apply that
     parameter to all the elements in the selected model parameter.

     Example:
     # Changes the collision stiffness and damping of the asset
     {
        "shape_material_ke": (1000.0, "/World/envs/env_*/Cabinet/sektion/collisions/sektion_collision5_visual"),
        "shape_material_kd": (100.0, None),
     }
     """

    prim_path: str = MISSING
    """Prim path (or expression) to the asset.

    .. note::
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Robot`` will be replaced with ``/World/envs/env_.*/Robot``.
    """

    spawn: SpawnerCfg | None = None
    """Spawn configuration for the asset. Defaults to None.

    If None, then no prims are spawned by the asset class. Instead, it is assumed that the
    asset is already present in the scene.
    """

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose."""

    collision_group: Literal[0, -1] = 0
    """Collision group of the asset. Defaults to ``0``.

    * ``-1``: global collision group (collides with all assets in the scene).
    * ``0``: local collision group (collides with other assets in the same environment).
    """

    debug_vis: bool = False
    """Whether to enable debug visualization for the asset. Defaults to ``False``."""
