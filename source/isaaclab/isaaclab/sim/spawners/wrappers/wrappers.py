# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

import carb
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from pxr import Sdf, Usd

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import UsdFileCfg

if TYPE_CHECKING:
    from . import wrappers_cfg


def spawn_multi_asset(
    prim_path: str,
    cfg: wrappers_cfg.MultiAssetSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
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

    Returns:
        The created prim at the first prim path.
    """
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # find a free prim path to hold all the template prims
    template_prim_path = stage_utils.get_next_free_path("/World/Template")
    prim_utils.create_prim(template_prim_path, "Scope")

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
        proto_prim_path = f"{template_prim_path}/Asset_{index:04d}"
        asset_cfg.func(proto_prim_path, asset_cfg, translation=translation, orientation=orientation)
        # append to proto prim paths
        proto_prim_paths.append(proto_prim_path)

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]

    # acquire stage
    stage = stage_utils.get_current_stage()

    # manually clone prims if the source prim path is a regex expression
    # note: unlike in the cloner API from Isaac Sim, we do not "reset" xforms on the copied prims.
    #   This is because the "spawn" calls during the creation of the proto prims already handles this operation.
    with Sdf.ChangeBlock():
        for index, prim_path in enumerate(prim_paths):
            # spawn single instance
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            # randomly select an asset configuration
            if cfg.random_choice:
                proto_path = random.choice(proto_prim_paths)
            else:
                proto_path = proto_prim_paths[index % len(proto_prim_paths)]
            # copy the proto prim
            Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_path), env_spec.layer, Sdf.Path(prim_path))

    # delete the dataset prim after spawning
    prim_utils.delete_prim(template_prim_path)

    # set carb setting to indicate Isaac Lab's environments that different prims have been spawned
    # at varying prim paths. In this case, PhysX parser shouldn't optimize the stage parsing.
    # the flag is mainly used to inform the user that they should disable `InteractiveScene.replicate_physics`
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/isaaclab/spawn/multi_assets", True)

    # return the prim
    return prim_utils.get_prim_at_path(prim_paths[0])


def spawn_multi_usd_file(
    prim_path: str,
    cfg: wrappers_cfg.MultiUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn multiple USD files based on the provided configurations.

    This function creates configuration instances corresponding the individual USD files and
    calls the :meth:`spawn_multi_asset` method to spawn them into the scene.

    Args:
        prim_path: The prim path to spawn the assets.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (w, x, y, z) order. Default is None.

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
    # set random choice
    multi_asset_cfg.random_choice = cfg.random_choice

    # call the original function
    return spawn_multi_asset(prim_path, multi_asset_cfg, translation, orientation)
