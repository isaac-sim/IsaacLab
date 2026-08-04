# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


@configclass
class AssetConverterBaseCfg:
    """The base configuration class for asset converters."""

    asset_path: str = MISSING
    """The absolute path to the asset file to convert into USD."""

    usd_dir: str | None = None
    """The output directory path to store the generated USD file. Defaults to None.

    If None, it is resolved as ``/tmp/IsaacLab/usd_{date}_{time}_{random}``, where
    the parameters in braces are runtime generated.
    """

    usd_file_name: str | None = None
    """The name of the generated usd file. Defaults to None.

    If None, it is resolved from the asset file name. For example, if the asset file
    name is ``"my_asset.urdf"``, then the generated USD file name is ``"my_asset.usd"``.

    If the providing file name does not end with ".usd" or ".usda", then the extension
    ".usd" is appended to the file name.
    """

    force_usd_conversion: bool = False
    """Force the conversion of the asset file to usd. Defaults to False.

    If True, then the USD file is always generated. It will overwrite the existing USD file if it exists.
    """

    make_instanceable: bool = True
    """Make the generated USD file instanceable. Defaults to True.

    Note:
        Instancing helps reduce the memory footprint of the asset when multiple copies of the asset are
        used in the scene. For more information, please check the USD documentation on
        `scene-graph instancing <https://openusd.org/dev/api/_usd__page__scenegraph_instancing.html>`_.
    """

    physics_variant: str = "physics"
    """The ``"Physics"`` variant to select on the generated USD file. Defaults to ``"physics"``.

    The URDF and MJCF importers emit physics as payloads behind a ``"Physics"`` variant set that they
    leave unselected, which composes the asset without joints, articulation roots, or mass properties.
    The converter authors this selection so that the asset carries physics on its own.

    The default ``"physics"`` variant holds the backend-portable description: standard ``UsdPhysics``
    joints, articulation roots, and mass, plus the Newton schemas. The ``"physx"`` and ``"mujoco"``
    variants sublayer it and add solver-specific tuning on top, so select one of those to author an
    asset for a specific backend. ``"none"`` converts without physics. Assets that do not provide the
    requested variant fall back to ``"physics"``.

    Every variant stays in the generated USD file, so this only decides what composes by default:
    :attr:`~isaaclab.sim.UsdFileCfg.variants` overrides the selection at spawn time.
    """
