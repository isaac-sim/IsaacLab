# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

import warp as wp
from newton import AxisType, ModelBuilder
from pxr import Sdf, Usd, UsdGeom, UsdUtils

CLONE = {
    "source": [],
    "destination": [],
    "mapping": torch.empty((0,), dtype=torch.bool),
}


def usd_replicate(stage: Usd.Stage, sources: list[str], destinations: list[str], mapping: torch.Tensor) -> None:
    root_layer = stage.GetRootLayer()
    with Sdf.ChangeBlock():
        for i, src in enumerate(sources):
            template = destinations[i]
            env_ids = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
            for w in env_ids:
                dest_path = template.format(int(w))
                # create the prim at destination then copy spec from src -> dest
                Sdf.CreatePrimInLayer(root_layer, dest_path)
                Sdf.CopySpec(root_layer, Sdf.Path(src), root_layer, Sdf.Path(dest_path))


def physx_replicate(
    stage: Usd.Stage,
    mapping: dict[str, list[str]],
    num_envs: int,
    *,
    use_env_ids: bool = True,
    use_fabric: bool = False,
) -> None:
    from omni.physx import get_physx_replicator_interface

    stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    current_dests: list[str] = []  # mutated per batch so rename_fn indexes the right list

    def attach_fn(_stage_id: int):
        return ["/World/envs", *list(mapping.keys())]

    def rename_fn(_replicate_path: str, i: int):
        return current_dests[i]

    def attach_end_fn(_stage_id: int):
        rep = get_physx_replicator_interface()
        for src, dests in mapping.items():
            if len(dests) > 0:
                current_dests[:] = dests
                rep.replicate(
                    _stage_id, src, len(dests), useEnvIds=len(dests) == num_envs, useFabricForReplication=use_fabric
                )
                if src not in dests:
                    stage.GetPrimAtPath(src).SetActive(False)
        # unregister only AFTER all replicate() calls completed
        rep.unregister_replicator(_stage_id)

    get_physx_replicator_interface().register_replicator(stage_id, attach_fn, attach_end_fn, rename_fn)


def newton_replicate(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    mapping: torch.Tensor,
    positions,
    orientations,
    up_axis: AxisType = "Z",
    simplify_meshes: bool = True,
):

    from isaaclab.sim._impl.newton_manager import NewtonManager

    # load everything except the prototype subtrees
    builder = ModelBuilder(up_axis=up_axis)
    stage_info = builder.add_usd(stage, ignore_paths=sources, load_non_physics_prims=False)

    # build a prototype for each source
    protos: dict[str, ModelBuilder] = {}
    for src_path in sources:
        p = ModelBuilder(up_axis=up_axis)
        p.add_usd(stage, root_path=src_path, load_non_physics_prims=False)
        if simplify_meshes:
            p.approximate_meshes("convex_hull")
        protos[src_path] = p

    # add by world, then by active sources in that world (column-wise)
    for w in range(mapping.size(1)):
        for j in torch.nonzero(mapping[:, w], as_tuple=True)[0]:
            builder.add_builder(protos[sources[j]], xform=wp.transform(positions[w], orientations[w]), world=w)

    # per-source, per-world renaming (strict prefix swap), compact style preserved
    for i, src_path in enumerate(sources):
        ol = len(src_path.rstrip("/"))
        swap = lambda name, new_root: new_root + name[ol:]  # noqa: E731
        worlds_list = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
        world_roots = {w: destinations[i].format(w) for w in worlds_list}

        for t in ("body", "joint", "shape", "articulation"):
            keys, worlds_arr = getattr(builder, f"{t}_key"), getattr(builder, f"{t}_world")
            for k, w in enumerate(worlds_arr):
                if w in world_roots and keys[k].startswith(src_path):
                    keys[k] = swap(keys[k], world_roots[w])

    NewtonManager.set_builder(builder)
    NewtonManager._num_envs = mapping.size(1)
    return builder, stage_info


def filter_collisions(
    stage: Usd.Stage,
    physicsscene_path: str,
    collision_root_path: str,
    prim_paths: list[str],
    global_paths: list[str] = [],
):
    """Filters collisions between clones. Clones will not collide with each other, but can collide with objects specified in global_paths.

    Args:
        physicsscene_path (str): Path to PhysicsScene object in stage.
        collision_root_path (str): Path to place collision groups under.
        prim_paths (List[str]): Paths of objects to filter out collision.
        global_paths (List[str]): Paths of objects to generate collision (e.g. ground plane).

    """
    from pxr import PhysxSchema

    physx_scene = PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(physicsscene_path))

    # We invert the collision group filters for more efficient collision filtering across environments
    physx_scene.CreateInvertCollisionGroupFilterAttr().Set(True)

    # Make sure we create the collision_scope in the RootLayer since the edit target may be a live layer in the case of Live Sync.
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        UsdGeom.Scope.Define(stage, collision_root_path)

    with Sdf.ChangeBlock():
        if len(global_paths) > 0:
            global_collision_group_path = collision_root_path + "/global_group"
            # add collision group prim
            global_collision_group = Sdf.PrimSpec(
                stage.GetRootLayer().GetPrimAtPath(collision_root_path),
                "global_group",
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            # prepend collision API schema
            global_collision_group.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))

            # expansion rule
            expansion_rule = Sdf.AttributeSpec(
                global_collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            # includes rel
            global_includes_rel = Sdf.RelationshipSpec(global_collision_group, "collection:colliders:includes", False)
            for global_path in global_paths:
                global_includes_rel.targetPathList.Append(global_path)

            # filteredGroups rel
            global_filtered_groups = Sdf.RelationshipSpec(global_collision_group, "physics:filteredGroups", False)
            # We are using inverted collision group filtering, which means objects by default don't collide across
            # groups. We need to add this group as a filtered group, so that objects within this group collide with
            # each other.
            global_filtered_groups.targetPathList.Append(global_collision_group_path)

        # set collision groups and filters
        for i, prim_path in enumerate(prim_paths):
            collision_group_path = collision_root_path + f"/group{i}"
            # add collision group prim
            collision_group = Sdf.PrimSpec(
                stage.GetRootLayer().GetPrimAtPath(collision_root_path),
                f"group{i}",
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            # prepend collision API schema
            collision_group.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))

            # expansion rule
            expansion_rule = Sdf.AttributeSpec(
                collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            # includes rel
            includes_rel = Sdf.RelationshipSpec(collision_group, "collection:colliders:includes", False)
            includes_rel.targetPathList.Append(prim_path)

            # filteredGroups rel
            filtered_groups = Sdf.RelationshipSpec(collision_group, "physics:filteredGroups", False)
            # We are using inverted collision group filtering, which means objects by default don't collide across
            # groups. We need to add this group as a filtered group, so that objects within this group collide with
            # each other.
            filtered_groups.targetPathList.Append(collision_group_path)
            if len(global_paths) > 0:
                filtered_groups.targetPathList.Append(global_collision_group_path)
                global_filtered_groups.targetPathList.Append(collision_group_path)


def grid_transforms(N: int, spacing: float = 1.0, up_axis: str = "z", device="cpu"):
    # rows/cols
    rows = int(math.ceil(math.sqrt(N)))
    cols = int(math.ceil(N / rows))

    idx = torch.arange(N, device=device)
    r = torch.div(idx, cols, rounding_mode="floor")
    c = idx % cols

    # centered grid coords
    x = (c - (cols - 1) * 0.5) * spacing
    y = ((rows - 1) * 0.5 - r) * spacing

    # place on plane based on up_axis
    z0 = torch.zeros_like(x)
    if up_axis.lower() == "z":
        pos = torch.stack([x, y, z0], dim=1)
    elif up_axis.lower() == "y":
        pos = torch.stack([x, z0, y], dim=1)
    else:  # up_axis == "x"
        pos = torch.stack([z0, x, y], dim=1)

    # identity orientations (w,x,y,z)
    ori = torch.zeros((N, 4), device=device)
    ori[:, 0] = 1.0
    return pos, ori
