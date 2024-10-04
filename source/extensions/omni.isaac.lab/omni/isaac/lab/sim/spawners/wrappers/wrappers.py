# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

import omni.isaac.core.utils.prims as prim_utils
import omni.usd
from pxr import Sdf, Usd

import omni.isaac.lab.sim as sim_utils

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
    in the order they are provided in the list. If the `random_choice` parameter is set to True, a random
    asset configuration is selected for each spawn.

    Args:
        prim_path: The prim path to spawn the assets.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets. Default is None.

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

    # spawn everything first in a "Dataset" prim
    prim_utils.create_prim("/World/Dataset", "Scope")
    proto_prim_paths = list()
    for index, asset_cfg in enumerate(cfg.assets_cfg):
        # spawn single instance
        proto_prim_path = f"/World/Dataset/Asset_{index:04d}"
        asset_cfg.func(proto_prim_path, asset_cfg, translation=translation, orientation=orientation)
        # append to proto prim paths
        proto_prim_paths.append(proto_prim_path)

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # acquire stage
    stage = omni.usd.get_context().get_stage()

    # manually clone prims if the source prim path is a regex expression
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
    prim_utils.delete_prim("/World/Dataset")

    # return the prim
    return prim_utils.get_prim_at_path(prim_paths[0])
