# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import UsdFileCfg

if TYPE_CHECKING:
    from . import wrappers_cfg

logger = logging.getLogger(__name__)


def spawn_multi_asset(
    prim_path: str,
    cfg: wrappers_cfg.MultiAssetSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multiple assets into numbered prim paths derived from the provided configuration.

    Assets are created in the order they appear in ``cfg.assets_cfg`` using the base name in ``prim_path``,
    which must contain ``.*`` (for example, ``/World/Env_0/asset_.*`` spawns ``asset_0``, ``asset_1``, ...).
    The prefix portion of ``prim_path`` may also include ``.*`` (for example, ``/World/env_.*/asset_.*``);
    in this case, assets are spawned under the first match (``env_0``) and that structure is cloned to
    other matching environments by the scene's cloner.

    Args:
        prim_path: The prim path to spawn the assets.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (x, y, z, w) order. Default is None.
        clone_in_fabric: Whether to clone in fabric. Default is False.
        replicate_physics: Whether to replicate physics. Default is False.

    Returns:
        The created prim at the first prim path.
    """
    split_path = prim_path.split("/")
    prefix_path, base_name = "/".join(split_path[:-1]), split_path[-1]
    if ".*" not in base_name:
        raise ValueError(
            f" The base name '{base_name}' in the prim path '{prim_path}' must contain '.*' to indicate"
            " the path each individual multiple-asset to be spawned."
        )
    if cfg.random_choice:
        logger.warning(
            "`random_choice` parameter in `spawn_multi_asset` is deprecated, and nothing will happen. "
            "Use `isaaclab.scene.interactive_scene_cfg.InteractiveSceneCfg.random_heterogeneous_cloning` instead."
        )

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

        proto_prim_path = f"{prefix_path}/{base_name.replace('.*', str(index))}"
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

    return sim_utils.find_first_matching_prim(proto_prim_paths[0])


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
        orientation: The orientation of the spawned assets in (x, y, z, w) order. Default is None.
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
        if attr_name in ["func", "usd_path", "random_choice", "spawn_path"]:
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
