# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
import torch

from omni.physx import get_physx_replicator_interface
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdUtils, Vt

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


@configclass
class TemplateCloneCfg:
    """Configuration for template-based cloning.

    This configuration is consumed by :func:`~isaaclab.scene.cloner.clone_from_template` to
    replicate one or more "prototype" prims authored under a template root into multiple
    per-environment destinations. It supports both USD-spec replication and PhysX replication
    and allows choosing between random or round-robin prototype assignment across environments.

    The cloning flow is:

    1. Discover prototypes under :attr:`template_root` whose base name starts with
        :attr:`template_prototype_identifier` (for example, ``proto_asset_0``, ``proto_asset_1``).
    2. Build a per-prototype mapping to environments according to
        :attr:`random_heterogenous_cloning` (random) or modulo assignment (deterministic).
    3. Stamp the selected prototypes to destinations derived from :attr:`clone_regex`.
    4. Optionally perform PhysX replication for the same mapping.

    Example
    -------

    .. code-block:: python

        from isaaclab.scene.cloner import TemplateCloneCfg, clone_from_template
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        cfg = TemplateCloneCfg(
            num_clones=128,
            template_root="/World/template",
            template_prototype_identifier="proto_asset",
            clone_regex="/World/envs/env_.*",
            clone_usd=True,
            clone_physx=True,
            random_heterogenous_cloning=False,  # use round-robin mapping
            device="cpu",
        )

        clone_from_template(stage, num_clones=cfg.num_clones, template_clone_cfg=cfg)
    """

    template_root: str = "/World/template"
    """Root path under which template prototypes are authored."""

    template_prototype_identifier: str = "proto_asset"
    """Name prefix used to identify prototype prims under :attr:`template_root`."""

    clone_regex: str = "/World/envs/env_.*"
    """Destination template for per-environment paths.

    The substring ``".*"`` is replaced with ``"{}"`` internally and formatted with the
    environment index (e.g., ``/World/envs/env_0``, ``/World/envs/env_1``).
    """

    clone_usd: bool = True
    """Enable USD-spec replication to author cloned prims and optional transforms."""

    clone_physx: bool = True
    """Enable PhysX replication for the same mapping to speed up physics setup."""

    random_heterogenous_cloning: bool = True
    """Randomly assign prototypes to environments. Default is True.

    When enabled, each environment selects a prototype at random from the available prototypes
    under a given template root. When disabled, environments use a round‑robin assignment based
    on ``env_index % num_prototypes`` for deterministic distribution.
    """

    device: str = "cpu"
    """Torch device on which mapping buffers are allocated."""

    clone_in_fabric: bool = False
    """Enable/disable cloning in fabric for PhysX replication. Default is False."""


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
    cfg = template_clone_cfg
    world_indices = torch.arange(num_clones, device=cfg.device)
    clone_plan = {"src": [], "dest": [], "mapping": torch.empty((0,), dtype=torch.bool).to(cfg.device)}
    clone_path_fmt = cfg.clone_regex.replace(".*", "{}")
    prototype_id = cfg.template_prototype_identifier
    prototypes = sim_utils.get_all_matching_child_prims(
        cfg.template_root,
        predicate=lambda prim: str(prim.GetPath()).split("/")[-1].startswith(prototype_id),
    )
    prototype_root_set = {"/".join(str(prototype.GetPath()).split("/")[:-1]) for prototype in prototypes}
    for prototype_root in prototype_root_set:
        protos = sim_utils.find_matching_prim_paths(f"{prototype_root}/.*")
        protos = [proto for proto in protos if proto.split("/")[-1].startswith(prototype_id)]
        m = torch.zeros((len(protos), num_clones), dtype=torch.bool, device=cfg.device)
        # Optionally select prototypes randomly per environment; else round-robin by modulo
        if cfg.random_heterogenous_cloning:
            rand_idx = torch.randint(len(protos), (num_clones,), device=cfg.device)
            m[rand_idx, world_indices] = True
        else:
            m[world_indices % len(protos), world_indices] = True

        clone_plan["src"].extend(protos)
        clone_plan["dest"].extend([prototype_root.replace(cfg.template_root, clone_path_fmt)] * len(protos))
        clone_plan["mapping"] = torch.cat((clone_plan["mapping"].reshape(-1, m.size(1)), m), dim=0)

    proto_idx = clone_plan["mapping"].to(torch.int32).argmax(dim=1)
    proto_mask = torch.zeros_like(clone_plan["mapping"])
    proto_mask.scatter_(1, proto_idx.view(-1, 1).to(torch.long), clone_plan["mapping"].any(dim=1, keepdim=True))
    usd_replicate(stage, clone_plan["src"], clone_plan["dest"], world_indices, proto_mask)
    stage.GetPrimAtPath(cfg.template_root).SetActive(False)

    # If all prototypes map to env_0, clone whole env_0 to all envs; else clone per-object
    if torch.all(proto_idx == 0):
        replicate_args = [clone_path_fmt.format(0)], [clone_path_fmt], world_indices, clone_plan["mapping"]
        if cfg.clone_usd:
            # parse env_origins directly from clone_path
            get_translate = (
                lambda prim_path: stage.GetPrimAtPath(prim_path).GetAttribute("xformOp:translate").Get()
            )  # noqa: E731
            positions = torch.tensor([get_translate(clone_path_fmt.format(i)) for i in world_indices])
            usd_replicate(stage, *replicate_args, positions=positions)
    else:
        src = [tpl.format(int(idx)) for tpl, idx in zip(clone_plan["dest"], proto_idx.tolist())]
        replicate_args = src, clone_plan["dest"], world_indices, clone_plan["mapping"]
        if cfg.clone_usd:
            usd_replicate(stage, *replicate_args)

    if cfg.clone_physx:
        physx_replicate(stage, *replicate_args, use_fabric=cfg.clone_in_fabric)


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
        quaternions: Optional orientations (``[E, 4]``) in ``wxyz`` -> ``xformOp:orient``.

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
                            o_attr.default = Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])))
                            op_names.append("xformOp:orient")
                        # Only author xformOpOrder for the ops we actually authored
                        if op_names:
                            op_order = ps.GetAttributeAtPath(dp + ".xformOpOrder") or Sdf.AttributeSpec(
                                ps, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                            )
                            op_order.default = Vt.TokenArray(op_names)


def physx_replicate(
    stage: Usd.Stage,
    sources: list[str],  # e.g. ["/World/Template/A", "/World/Template/B"]
    destinations: list[str],  # e.g. ["/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"]
    env_ids: torch.Tensor,  # env_ids
    mapping: torch.Tensor,  # (num_sources, num_envs) bool; True -> place sources[i] into world=j
    use_fabric: bool = False,
) -> None:
    """Replicate prims via PhysX replicator with per-row mapping.

    Builds per-source destination lists from ``mapping`` and calls PhysX ``replicate``.
    Rows covering all environments use ``useEnvIds=True``; partial rows use ``False``.
    The replicator is registered for the call and then unregistered.

    Args:
        stage: USD stage.
        sources: Source prim paths (``S``).
        destinations: Destination templates (``S``) with ``"{}"`` for env index.
        env_ids: Environment indices (``[E]``).
        mapping: Bool/int mask (``[S, E]``) selecting envs per source.
        use_fabric: Use Fabric for replication.

    Returns:
        None
    """

    stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    current_worlds: list[int] = []
    current_template: str = ""
    num_envs = mapping.size(1)

    def attach_fn(_stage_id: int):
        return ["/World/envs", *sources]

    def rename_fn(_replicate_path: str, i: int):
        return current_template.format(current_worlds[i])

    def attach_end_fn(_stage_id: int):
        nonlocal current_template
        rep = get_physx_replicator_interface()
        for i, src in enumerate(sources):
            current_worlds[:] = env_ids[mapping[i]].tolist()
            current_template = destinations[i]
            rep.replicate(
                _stage_id,
                src,
                len(current_worlds),
                useEnvIds=len(current_worlds) == num_envs,
                useFabricForReplication=use_fabric,
            )
        # unregister only AFTER all replicate() calls completed
        rep.unregister_replicator(_stage_id)

    get_physx_replicator_interface().register_replicator(stage_id, attach_fn, attach_end_fn, rename_fn)


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
    """Create a centered grid of transforms for ``N`` instances.

    Computes ``(x, y)`` coordinates in a roughly square grid centered at the origin
    with the provided spacing, places the third coordinate according to ``up_axis``,
    and returns identity orientations (``wxyz``) for each instance.

    Args:
        N: Number of instances.
        spacing: Distance between neighboring grid positions.
        up_axis: Up axis for positions ("z", "y", or "x").
        device: Torch device for returned tensors.

    Returns:
        A tuple ``(pos, ori)`` where:
            - ``pos`` is a tensor of shape ``(N, 3)`` with positions.
            - ``ori`` is a tensor of shape ``(N, 4)`` with identity quaternions in ``(w, x, y, z)``.
    """
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
