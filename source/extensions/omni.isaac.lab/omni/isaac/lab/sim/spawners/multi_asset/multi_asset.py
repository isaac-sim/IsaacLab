# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawns multiple objects randomly in the scene."""

from __future__ import annotations

import random
import re
import torch
from typing import TYPE_CHECKING

import omni.isaac.core.utils.prims as prim_utils
import omni.usd
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom, Vt

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim.spawners.multi_asset.asset_randomizer_cfg import AssetRandomizerCfg
from omni.isaac.lab.sim.spawners.spawner_cfg import SpawnerCfg

if TYPE_CHECKING:
    from .multi_asset_cfg import MultiAssetCfg


def spawn_multi_object_randomly_sdf(
    prim_path: str,
    cfg: MultiAssetCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawns multiple objects randomly in the scene.
    Args:
        prim_path: The path to the asset to spawn (e.g. "/World/env_*/Object")
        cfg: The configuration for the asset spawning
        translation: The translation to apply to the spawned asset
        orientation: The orientation to apply to the spawned asset
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

    object_id = 0
    for asset_cfg in cfg.assets_cfg:
        if isinstance(asset_cfg, AssetRandomizerCfg):
            for i in range(asset_cfg.num_random_assets):
                # set the seeds
                random.seed(asset_cfg.base_seed + i)
                proto_prim_paths.append(_spawn_dataset_objects(object_id, asset_cfg))
                object_id += 1
        else:
            proto_prim_paths.append(_spawn_dataset_objects(object_id, asset_cfg))
            object_id += 1

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # acquire stage
    stage = omni.usd.get_context().get_stage()

    if orientation is not None:
        # convert orientation ordering (wxyz to xyzw)
        orientation = (orientation[1], orientation[2], orientation[3], orientation[0])
        if isinstance(orientation[0], torch.Tensor):
            orientation = [x.cpu().item() for x in orientation]

    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for i, prim_path in enumerate(prim_paths):
            # spawn single instance
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            if cfg.randomize:
                # randomly select an asset configuration
                proto_path = random.choice(proto_prim_paths)
            else:
                proto_path = proto_prim_paths[i % len(proto_prim_paths)]

            Sdf.CopySpec(
                env_spec.layer,
                Sdf.Path(proto_path),
                env_spec.layer,
                Sdf.Path(prim_path),
            )

            # set the translation and orientation
            _ = UsdGeom.Xform(stage.GetPrimAtPath(proto_path)).GetPrim().GetPrimStack()

            translate_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
            if translate_spec is None:
                translate_spec = Sdf.AttributeSpec(env_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            if translation is not None:
                translate_spec.default = Gf.Vec3d(*translation)

            orient_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:orient")
            if orient_spec is None:
                orient_spec = Sdf.AttributeSpec(env_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)

            if orientation is not None:
                orient_spec.default = Gf.Quatd(*orientation)

            op_order_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
            if op_order_spec is None:
                op_order_spec = Sdf.AttributeSpec(env_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
            op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

            if cfg.postprocess_func is not None:
                cfg.postprocess_func(proto_path, prim_path, stage)

    # delete the dataset prim after spawning
    prim_utils.delete_prim("/World/Dataset")

    # return the prim
    return prim_utils.get_prim_at_path(prim_paths[0])


###
# Internal Helper functions
###


def _spawn_dataset_objects(
    object_id: int,
    asset_cfg: SpawnerCfg,
):
    """Helper function to spawn a single instance.
    This functions expects a prototype prim to be located at "/World/Dataset/Object_{object_id:02d}
    Args:
        object_id: The unique identifier for the object
        asset_cfg: The configuration for the asset to spawn
    """

    # spawn single instance
    proto_prim_path = f"/World/Dataset/Object_{object_id:02d}"

    prim = asset_cfg.func(proto_prim_path, asset_cfg)

    # set the prim visibility
    if hasattr(asset_cfg, "visible"):
        imageable = UsdGeom.Imageable(prim)
        if asset_cfg.visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    # set the semantic annotations
    if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
        # note: taken from replicator scripts.utils.utils.py
        for semantic_type, semantic_value in asset_cfg.semantic_tags:
            # deal with spaces by replacing them with underscores
            semantic_type_sanitized = semantic_type.replace(" ", "_")
            semantic_value_sanitized = semantic_value.replace(" ", "_")
            # set the semantic API for the instance
            instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
            sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
            # create semantic type and data attributes
            sem.CreateSemanticTypeAttr()
            sem.CreateSemanticDataAttr()
            sem.GetSemanticTypeAttr().Set(semantic_type)
            sem.GetSemanticDataAttr().Set(semantic_value)

    # activate rigid body contact sensors
    if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
        sim_utils.activate_contact_sensors(proto_prim_path, asset_cfg.activate_contact_sensors)

    return proto_prim_path
