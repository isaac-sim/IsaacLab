# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import math
import torch
from typing import TYPE_CHECKING

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

if TYPE_CHECKING:
    from .cloner_cfg import TemplateCloneCfg


def clone_from_template(stage: Usd.Stage, num_clones: int, template_clone_cfg: TemplateCloneCfg) -> None:
    """Clone assets from a template root into per-environment destinations.

    This utility discovers prototype prims under ``cfg.template_root`` whose names start with
    ``cfg.template_prototype_identifier``, builds a per-prototype mapping across
    ``num_clones`` environments (random or modulo), and then performs USD and/or PhysX replication
    according to the flags in ``cfg``.

    Args:
        stage: The USD stage to author into.
        num_clones: Number of environments to clone to (typically equals ``cfg.num_clones``).
        template_clone_cfg: Configuration describing template location, destination pattern,
            and replication/mapping behavior.
    """
    cfg: TemplateCloneCfg = template_clone_cfg
    world_indices = torch.arange(num_clones, device=cfg.device)
    clone_path_fmt = cfg.clone_regex.replace(".*", "{}")
    prototype_id = cfg.template_prototype_identifier
    prototypes = sim_utils.get_all_matching_child_prims(
        cfg.template_root,
        predicate=lambda prim: str(prim.GetPath()).split("/")[-1].startswith(prototype_id),
    )
    if len(prototypes) > 0:
        prototype_root_set = {"/".join(str(prototype.GetPath()).split("/")[:-1]) for prototype in prototypes}
        # discover prototypes per root then make a clone plan
        src: list[list[str]] = []
        dest: list[str] = []

        for prototype_root in prototype_root_set:
            protos = sim_utils.find_matching_prim_paths(f"{prototype_root}/.*")
            protos = [proto for proto in protos if proto.split("/")[-1].startswith(prototype_id)]
            src.append(protos)
            dest.append(prototype_root.replace(cfg.template_root, clone_path_fmt))

        src_paths, dest_paths, clone_masking = make_clone_plan(src, dest, num_clones, cfg.clone_strategy, cfg.device)

        # Spawn the first instance of clones from prototypes, then deactivate the prototypes, those first instances
        # will be served as sources for usd and physx replication.
        proto_idx = clone_masking.to(torch.int32).argmax(dim=1)
        proto_mask = torch.zeros_like(clone_masking)
        proto_mask.scatter_(1, proto_idx.view(-1, 1).to(torch.long), clone_masking.any(dim=1, keepdim=True))
        usd_replicate(stage, src_paths, dest_paths, world_indices, proto_mask)
        stage.GetPrimAtPath(cfg.template_root).SetActive(False)
        replicate_args = [clone_path_fmt.format(0)], [clone_path_fmt], world_indices, clone_masking[0].unsqueeze(0)
        get_pos = lambda path: stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Get()  # noqa: E731
        positions = torch.tensor([get_pos(clone_path_fmt.format(i)) for i in world_indices])
        # If all prototypes map to env_0, clone whole env_0 to all envs; else clone per-object
        if torch.all(proto_idx == 0):
            if cfg.clone_physics and cfg.physics_clone_fn is not None:
                cfg.physics_clone_fn(stage, *replicate_args, positions=positions)
            if cfg.clone_usd:
                # parse env_origins directly from clone_path
                usd_replicate(stage, *replicate_args, positions=positions)

        else:
            selected_src = [tpl.format(int(idx)) for tpl, idx in zip(dest_paths, proto_idx.tolist())]
            replicate_args = selected_src, dest_paths, world_indices, clone_masking
            if cfg.clone_physics and cfg.physics_clone_fn is not None:
                cfg.physics_clone_fn(stage, *replicate_args, positions=positions)
            if cfg.clone_usd:
                usd_replicate(stage, *replicate_args)


def make_clone_plan(
    sources: list[list[str]],
    destinations: list[str],
    num_clones: int,
    clone_strategy: callable,
    device: str = "cpu",
) -> tuple[list[str], list[str], torch.Tensor]:
    """Construct a cloning plan mapping prototype prims to per-environment destinations.

    The plan enumerates all combinations of prototypes, selects a combination per environment using ``clone_strategy``,
    and builds a boolean masking matrix indicating which prototype populates each environment slot.

    Args:
        sources: Prototype prim paths grouped by asset type (e.g., [[robot_a, robot_b], [obj_x]]).
        destinations: Destination path templates (one per group) with ``"{}"`` placeholder for env id.
        num_clones: Number of environments to populate.
        clone_strategy: Function that picks a prototype combo per environment; signature
            ``clone_strategy(combos: Tensor, num_clones: int, device: str) -> Tensor[num_clones, num_groups]``.
        device: Torch device for tensors in the plan. Defaults to ``"cpu"``.

    Returns:
        tuple: ``(src, dest, masking)`` where ``src`` and ``dest`` are flattened lists of prototype and
            destination paths, and ``masking`` is a ``[num_src, num_clones]`` boolean tensor with True
            when source ``src[i]`` is used for clone ``j``.
    """
    # 1) Flatten into src and dest lists
    src = [p for group in sources for p in group]
    dest = [dst for dst, group in zip(destinations, sources) for _ in group]
    group_sizes = [len(group) for group in sources]

    # 2) Enumerate all combinations of "one prototype per group"
    #    all_combos: list of tuples (g0_idx, g1_idx, ..., g_{G-1}_idx)
    all_combos = list(itertools.product(*[range(s) for s in group_sizes]))
    combos = torch.tensor(all_combos, dtype=torch.long, device=device)

    # 3) Assign a combination to each environment
    chosen = clone_strategy(combos, num_clones, device)

    # 4) Build masking: [num_src, num_clones] boolean
    #    For each env, for each group, mark exactly one prototype row as True.
    group_offsets = torch.tensor([0] + list(itertools.accumulate(group_sizes[:-1])), dtype=torch.long, device=device)
    rows = (chosen + group_offsets).view(-1)
    cols = torch.arange(num_clones, device=device).view(-1, 1).expand(-1, len(group_sizes)).reshape(-1)

    masking = torch.zeros((sum(group_sizes), num_clones), dtype=torch.bool, device=device)
    masking[rows, cols] = True
    return src, dest, masking


def usd_replicate(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
) -> None:
    """Replicate USD prims to per-environment destinations.

    Copies each source prim spec to destination templates for selected environments
    (``mask``). Optionally authors translate/orient from position/quaternion buffers.
    Replication runs in path-depth order (parents before children) for robust composition.

    Args:
        stage: USD stage.
        sources: Source prim paths.
        destinations: Destination formattable templates with ``"{}"`` for env index.
        env_ids: Environment indices.
        mask: Optional per-source or shared mask. ``None`` selects all.
        positions: Optional positions (``[E, 3]``) -> ``xformOp:translate``.
        quaternions: Optional orientations (``[E, 4]``) in ``xyzw`` -> ``xformOp:orient``.

    Returns:
        None
    """
    rl = stage.GetRootLayer()

    # Group replication by destination path depth so ancestors land before deeper paths.
    # This avoids composition issues for nested or interdependent specs.
    def dp_depth(template: str) -> int:
        dp = template.format(0)
        return Sdf.Path(dp).pathElementCount

    order = sorted(range(len(sources)), key=lambda i: dp_depth(destinations[i]))

    # Process in layers of equal depth, committing at each depth to stabilize composition
    depth_to_indices: dict[int, list[int]] = {}
    for i in order:
        d = dp_depth(destinations[i])
        depth_to_indices.setdefault(d, []).append(i)

    for depth in sorted(depth_to_indices.keys()):
        with Sdf.ChangeBlock():
            for i in depth_to_indices[depth]:
                src = sources[i]
                tmpl = destinations[i]
                # Select target environments for this source (supports None, [E], or [S, E])
                target_envs = env_ids if mask is None else env_ids[mask[i]]
                for wid in target_envs.tolist():
                    dp = tmpl.format(wid)
                    Sdf.CreatePrimInLayer(rl, dp)
                    Sdf.CopySpec(rl, Sdf.Path(src), rl, Sdf.Path(dp))

                    if positions is not None or quaternions is not None:
                        ps = rl.GetPrimAtPath(dp)
                        op_names = []
                        if positions is not None:
                            p = positions[wid]
                            t_attr = ps.GetAttributeAtPath(dp + ".xformOp:translate")
                            if t_attr is None:
                                t_attr = Sdf.AttributeSpec(ps, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                            t_attr.default = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                            op_names.append("xformOp:translate")
                        if quaternions is not None:
                            q = quaternions[wid]
                            o_attr = ps.GetAttributeAtPath(dp + ".xformOp:orient")
                            if o_attr is None:
                                o_attr = Sdf.AttributeSpec(ps, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
                            # xyzw convention: q[3] is w, q[0:3] is xyz
                            o_attr.default = Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))
                            op_names.append("xformOp:orient")
                        # Only author xformOpOrder for the ops we actually authored
                        if op_names:
                            op_order = ps.GetAttributeAtPath(dp + ".xformOpOrder") or Sdf.AttributeSpec(
                                ps, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                            )
                            op_order.default = Vt.TokenArray(op_names)


def filter_collisions(
    stage: Usd.Stage,
    physicsscene_path: str,
    collision_root_path: str,
    prim_paths: list[str],
    global_paths: list[str] = [],
) -> None:
    """Create inverted collision groups for clones.

    Creates one PhysicsCollisionGroup per prim under ``collision_root_path``, enabling
    inverted filtering so clones don't collide across groups. Optionally adds a global
    group that collides with all.

    Args:
        stage: USD stage.
        physicsscene_path: Path to PhysicsScene prim.
        collision_root_path: Root scope for collision groups.
        prim_paths: Per-clone prim paths.
        global_paths: Optional global-collider paths.

    Returns:
        None
    """
    # Invert collision group filters for more efficient cross-environment filtering
    safe_set_attribute_on_usd_prim(
        stage.GetPrimAtPath(physicsscene_path), "physxScene:invertCollisionGroupFilter", value=True, camel_case=False
    )

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
    """Create a centered grid of transforms for ``N`` instances.

    Computes ``(x, y)`` coordinates in a roughly square grid centered at the origin
    with the provided spacing, places the third coordinate according to ``up_axis``,
    and returns identity orientations. This matches the grid layout used by
    :class:`isaaclab.terrains.TerrainImporter` for consistent environment positioning.

    Args:
        N: Number of instances.
        spacing: Distance between neighboring grid positions.
        up_axis: Up axis for positions ("z", "y", or "x").
        device: Torch device for returned tensors.

    Returns:
        A tuple ``(pos, ori)`` where:
            - ``pos`` is a tensor of shape ``(N, 3)`` with positions.
            - ``ori`` is a tensor of shape ``(N, 4)`` with identity quaternions in ``(x, y, z, w)``.
    """
    # Match terrain_importer._compute_env_origins_grid layout for consistency
    num_rows = int(math.ceil(N / math.sqrt(N)))
    num_cols = int(math.ceil(N / num_rows))

    # Create meshgrid matching terrain's "ij" indexing
    ii, jj = torch.meshgrid(
        torch.arange(num_rows, device=device, dtype=torch.float32),
        torch.arange(num_cols, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Flatten and take first N elements
    ii = ii.flatten()[:N]
    jj = jj.flatten()[:N]

    # Match terrain's coordinate system: X from rows (negated), Y from cols
    x = -(ii - (num_rows - 1) / 2) * spacing
    y = (jj - (num_cols - 1) / 2) * spacing
    z0 = torch.zeros(N, device=device)

    # place on plane based on up_axis
    if up_axis.lower() == "z":
        pos = torch.stack([x, y, z0], dim=1)
    elif up_axis.lower() == "y":
        pos = torch.stack([x, z0, y], dim=1)
    else:  # up_axis == "x"
        pos = torch.stack([z0, x, y], dim=1)

    # identity orientations (x,y,z,w)
    ori = torch.zeros((N, 4), device=device)
    ori[:, 3] = 1.0  # w=1 for identity quaternion
    return pos, ori
