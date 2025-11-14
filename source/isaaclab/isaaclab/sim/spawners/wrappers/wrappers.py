# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd

from isaaclab.sim.spawners.from_files import UsdFileCfg

if TYPE_CHECKING:
    from . import wrappers_cfg


def spawn_multi_asset(
    prim_path: str,
    cfg: wrappers_cfg.MultiAssetSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multiple assets based on the provided configurations.

    This function spawns multiple assets based on the provided configurations. The assets are spawned
    in the order they are provided in the list. If the :attr:`~MultiAssetSpawnerCfg.random_choice` parameter is
    set to True, a random asset configuration is selected for each spawn.

    Args:
        prim_path: The prim path to spawn the assets.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (w, x, y, z) order. Default is None.
        clone_in_fabric: Whether to clone in fabric. Default is False.
        replicate_physics: Whether to replicate physics. Default is False.

    Returns:
        The created prim at the first prim path.
    """
    # spawn everything first in a "Dataset" prim
    proto_prim_paths = list()
    for index, asset_cfg in enumerate(cfg.assets_cfg):
        # append semantic tags if specified
        if cfg.semantic_tags is not None:
            if asset_cfg.semantic_tags is None:
                asset_cfg.semantic_tags = cfg.semantic_tags
            else:
                asset_cfg.semantic_tags += cfg.semantic_tags
        # override settings for properties
        attr_names = ["mass_props", "rigid_props", "collision_props", "activate_contact_sensors", "deformable_props"]
        for attr_name in attr_names:
            attr_value = getattr(cfg, attr_name)
            if hasattr(asset_cfg, attr_name) and attr_value is not None:
                setattr(asset_cfg, attr_name, attr_value)
        # spawn single instance
        proto_prim_path = prim_path.replace(".*", f"{index}")
        asset_cfg.func(
            proto_prim_path,
            asset_cfg,
            translation=translation,
            orientation=orientation,
            clone_in_fabric=clone_in_fabric,
            replicate_physics=replicate_physics,
        )
        # append to proto prim paths
        proto_prim_paths.append(proto_prim_path)

    return prim_utils.get_prim_at_path(proto_prim_paths[0])


def spawn_multi_usd_file(
    prim_path: str,
    cfg: wrappers_cfg.MultiUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multiple USD files based on the provided configurations.

    This function creates configuration instances corresponding the individual USD files and
    calls the :meth:`spawn_multi_asset` method to spawn them into the scene.

    Args:
        prim_path: The prim path to spawn the assets.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (w, x, y, z) order. Default is None.
        clone_in_fabric: Whether to clone in fabric. Default is False.
        replicate_physics: Whether to replicate physics. Default is False.

    Returns:
        The created prim at the first prim path.
    """
    # needed here to avoid circular imports
    from .wrappers_cfg import MultiAssetSpawnerCfg

    # parse all the usd files
    if isinstance(cfg.usd_path, str):
        usd_paths = [cfg.usd_path]
    else:
        usd_paths = cfg.usd_path

    # make a template usd config
    usd_template_cfg = UsdFileCfg()
    for attr_name, attr_value in cfg.__dict__.items():
        # skip names we know are not present
        if attr_name in ["func", "usd_path", "random_choice"]:
            continue
        # set the attribute into the template
        setattr(usd_template_cfg, attr_name, attr_value)

    # create multi asset configuration of USD files
    multi_asset_cfg = MultiAssetSpawnerCfg(assets_cfg=[])
    for usd_path in usd_paths:
        usd_cfg = usd_template_cfg.replace(usd_path=usd_path)
        multi_asset_cfg.assets_cfg.append(usd_cfg)

    # propagate the contact sensor settings
    # note: the default value for activate_contact_sensors in MultiAssetSpawnerCfg is False.
    #  This ends up overwriting the usd-template-cfg's value when the `spawn_multi_asset`
    #  function is called. We hard-code the value to the usd-template-cfg's value to ensure
    #  that the contact sensor settings are propagated correctly.
    if hasattr(cfg, "activate_contact_sensors"):
        multi_asset_cfg.activate_contact_sensors = cfg.activate_contact_sensors

    # call the original function
    return spawn_multi_asset(prim_path, multi_asset_cfg, translation, orientation, clone_in_fabric, replicate_physics)
