# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import itertools
import logging
import math
import re
from collections.abc import Iterator, Sequence

import torch

from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, Vt

from . import _fabric_notices
from .clone_plan import ClonePlan

logger = logging.getLogger(__name__)


def get_suffix(path_expr: str, destination_template: str) -> str | None:
    """Return the part of ``path_expr`` below a destination template's env-instance root.

    The template's ``"{}"`` slot matches exactly one path segment (a concrete id like ``env_3``
    or a wildcard like ``env_.*``).

    Example:
        >>> tmpl = "/World/scenes/{}/Robot"
        >>> get_suffix("/World/scenes/env_3/Robot/base", tmpl)
        '/base'
        >>> get_suffix("/World/scenes/.*/Robot/base", tmpl)
        '/base'
        >>> get_suffix("/World/scenes/env_3/Robot", tmpl)
        ''
        >>> get_suffix("/World/scenes/env_3/Sensor", tmpl) is None
        True
        >>> get_suffix("/World/scenes/env_3/RobotArm", tmpl) is None
        True
        >>> get_suffix("/World/scenes/env_3/sub/Robot/base", tmpl) is None
        True
    """
    pattern = re.compile(r"[^/]+".join(re.escape(part) for part in destination_template.split("{}")))
    match = pattern.match(path_expr)
    if match is None:
        return None
    suffix = path_expr[match.end() :]
    return None if suffix and not suffix.startswith("/") else suffix


def resolve_clone_plan_source(path_expr: str, plan: ClonePlan) -> tuple[str, str, str] | None:
    """Resolve a destination path expression to its row's source path, destination glob, and asset suffix.

    Finds the rows whose destination template owns ``path_expr`` (same matching
    logic as :func:`iter_clone_plan_matches`), OR-merges their
    :attr:`~isaaclab.cloner.ClonePlan.clone_mask` rows, and splits the
    expression at the row's destination template so the asset-relative suffix is
    returned for downstream walks.

    Args:
        path_expr: Destination-side path expression (e.g., a sensor's ``prim_path``,
            with ``.*`` env wildcard).
        plan: Active clone plan.

    Returns:
        Three-tuple of ``(source_asset_path, dest_glob_prefix, asset_suffix)``. The
        ``asset_suffix`` is the part of ``path_expr`` beyond the matching row's
        destination template (empty when ``path_expr`` equals the row's template).
        Returns ``None`` when ``path_expr`` matches no row in the plan, letting
        callers fall back to direct stage resolution (e.g. for sensor frames
        mounted at the env root rather than under a planned asset).

    Raises:
        ValueError: When ``path_expr``'s matching rows span multiple distinct
            destination templates.
        NotImplementedError: When the union of matching rows' clone masks does not
            cover every env (partial-env heterogeneous coverage is unsupported).
    """
    matching_template: str | None = None
    matching_rows: list[int] = []
    matching_suffix: str | None = None
    for source_index, destination_template in enumerate(plan.destinations):
        if "{}" not in destination_template:
            continue
        suffix = get_suffix(path_expr, destination_template)
        if suffix is None:
            continue
        if matching_template is None:
            matching_template = destination_template
            matching_suffix = suffix
        elif destination_template != matching_template:
            raise ValueError(
                f"path_expr {path_expr!r}: matches multiple destination templates"
                f" {matching_template!r} and {destination_template!r}."
            )
        matching_rows.append(source_index)
    if matching_template is None:
        return None
    if not plan.clone_mask[matching_rows].any(dim=0).all():
        raise NotImplementedError(
            f"path_expr {path_expr!r}: partial-env heterogeneous coverage is unsupported;"
            " matching rows must collectively cover all envs."
        )
    return plan.sources[matching_rows[0]], matching_template.replace("{}", "*"), matching_suffix or ""


def iter_clone_plan_matches(plan: ClonePlan, path_expr: str) -> Iterator[tuple[str, str, str, tuple[int, ...]]]:
    """Yield clone-plan entries whose destinations own a path expression.

    Example:
        For an entry with source root ``"/World/source/Robot"``, destination
        template ``"/World/scenes/{}/Robot"``, and populated env ids
        ``(0, 2)``, querying ``"/World/scenes/.*/Robot/base"`` yields
        ``("/World/source/Robot", "/World/scenes/{}/Robot",
        "/World/source/Robot/base", (0, 2))``.

    Args:
        plan: Clone plan to query.
        path_expr: Destination prim path or path expression. Expressions are
            matched against each clone-plan destination template by treating
            the template's ``"{}"`` field as the populated environment slot.

    Yields:
        Tuples ``(source_root, destination_template, source_path, env_ids)``
        for the nearest matching destination root. Multiple source variants
        with the same destination root are preserved.
    """
    matches: list[tuple[str, str, str, tuple[int, ...]]] = []
    for source_index, (source_root, destination_template) in enumerate(zip(plan.sources, plan.destinations)):
        if "{}" not in destination_template:
            continue

        env_ids = tuple(int(i) for i in plan.clone_mask[source_index].nonzero(as_tuple=False).flatten().tolist())
        if not env_ids:
            continue

        source_root = source_root.rstrip("/") or "/"
        destination_template = destination_template.rstrip("/") or "/"

        suffix = get_suffix(path_expr, destination_template)
        if suffix is None:
            continue
        source_path = source_root + suffix if source_root != "/" else suffix or "/"

        matches.append((source_root, destination_template, source_path, env_ids))

    matches.sort(key=lambda match: len(match[1].format(match[3][0])), reverse=True)
    if matches:
        owner_length = len(matches[0][1].format(matches[0][3][0]))
        yield from (match for match in matches if len(match[1].format(match[3][0])) == owner_length)


@contextlib.contextmanager
def disabled_fabric_change_notifies(stage: Usd.Stage, *, restore: bool = True) -> Iterator[None]:
    """Suspend the ``IFabricUsd`` USD notice listener for the body of the ``with`` block.

    Targets the same handler that :meth:`isaacsim.core.cloner.Cloner.disable_change_listener`
    toggles, but goes through ``omni::fabric::IFabricUsd`` directly so we don't take an
    ``isaacsim.core.simulation_manager`` dependency.

    The listener is a global ``TfNotice`` registered when ``omni.fabric`` loads; it
    short-circuits via a soft flag (``IFabricUsd.cpp:739``). Toggling that flag is what
    skips the per-``Sdf.CopySpec`` Fabric sync that dominates cloning time on large scenes.

    When this provides a measurable speedup
    ----------------------------------------
    Bisection on the regression test (see ``test_cloner.py``) shows the listener cost is
    only on the critical path when **all** of these hold:

    1. The clone happens through the ``InteractiveScene`` path with ``replicate_physics=True``.
       Calling :func:`usd_replicate` directly on a stage produces no measurable gap; with
       ``replicate_physics=False`` the gap drops to ~1.19x. The PhysX replication path is
       what amplifies per-spec listener work.
    2. The cloned prims carry PhysX rigid-body schemas (e.g. ``UsdPhysics.RigidBodyAPI``,
       authored via ``rigid_props`` on a spawn cfg). Plain Xforms or geometry without
       physics schemas produce ~1.0x — the listener has no Fabric-tracked state to sync.
       ``mass_props`` and ``collision_props`` add nothing beyond ``rigid_props``.
    3. Total per-``Sdf.CopySpec`` firings reach ~32K — i.e. ``num_bodies × num_envs`` is
       large enough to dominate scene-init cost. Below this the speedup sinks into noise.

    Conditions outside this envelope (no PhysX schemas, single-env scenes, raw
    ``usd_replicate`` calls, ``replicate_physics=False``) won't see a perf win — the
    suspension is correct but its effect is lost in the rest of the work.

    Re-entrant: if the flag is already off on entry, ``__exit__`` leaves it off. Falls
    through to a no-op if the Carbonite interface can't be acquired (e.g. outside a live
    Kit application) — the caller never breaks, it just doesn't get the perf win.

    Args:
        stage: USD stage whose Fabric notice handler should be suspended.
        restore: When ``True`` (default), re-enable the handler on exit. Set to ``False``
            inside a known clone-then-``sim.reset`` window where the downstream Fabric
            resync happens anyway and re-enabling here would trigger a redundant
            ``forceMinimalPopulate`` batch — see ``PluginInterface.cpp:337``.

    Yields:
        None.
    """
    bindings = _fabric_notices.get_bindings()
    if bindings is None:
        yield
        return

    # usdrt only works with a live Kit app — defer import so module load stays cheap.
    import usdrt

    # Avoid leaking a strong reference into the global ``StageCache`` for stages we did not
    # author into the cache: ``Insert`` keeps the stage alive for the rest of the process.
    cache = UsdUtils.StageCache.Get()
    cached_id = cache.GetId(stage)
    stage_id = cached_id.ToLongInt() if cached_id.IsValid() else cache.Insert(stage).ToLongInt()
    # ``FabricId`` wraps a uint64; the C ABI needs the raw integer.
    fabric_id = usdrt.Usd.Stage.Attach(stage_id).GetFabricId().id
    # First-call ABI sanity check — if the toggle doesn't actually round-trip the flag
    # (e.g. Kit's vtable shifted), fall through to a no-op rather than corrupting state.
    if not bindings.validate_with(fabric_id):
        logger.warning("Fabric notice toggle failed round-trip check — suspension disabled")
        yield
        return
    was_enabled = bindings.is_enabled(fabric_id)
    if was_enabled:
        bindings.set_enable(fabric_id, False)
    try:
        yield
    finally:
        if restore and was_enabled:
            bindings.set_enable(fabric_id, True)


def make_clone_plan(
    sources: Sequence[Sequence[str]],
    destinations: Sequence[str],
    num_clones: int,
    clone_strategy: callable,
    device: str = "cpu",
) -> tuple[tuple[str, ...], tuple[str, ...], torch.Tensor]:
    """Compute the flat source/destination/mask components of a clone plan.

    Enumerates all combinations of prototypes, selects a combination per environment using
    ``clone_strategy``, and builds the boolean masking matrix that indicates which prototype
    populates each environment slot. The caller composes the returned tuple into a
    :class:`ClonePlan`.

    Args:
        sources: Prototype prim paths grouped by asset type (e.g., ``[[robot_a, robot_b], [obj_x]]``).
        destinations: Destination path templates (one per group) with ``"{}"`` placeholder for env id.
        num_clones: Number of environments to populate.
        clone_strategy: Function that picks a prototype combo per environment; signature
            ``clone_strategy(combos: Tensor, num_clones: int, device: str) -> Tensor[num_clones, num_groups]``.
        device: Torch device for the returned mask. Defaults to ``"cpu"``.

    Returns:
        A tuple ``(sources, destinations, clone_mask)`` where ``sources`` and ``destinations``
        are flattened per-source entries (one entry per prototype) and ``clone_mask`` is a
        ``[num_src, num_clones]`` boolean tensor on ``device``.
    """
    if len(sources) != len(destinations):
        raise ValueError(f"Expected one destination per source group, got {len(destinations)} and {len(sources)}.")
    if not sources:
        raise ValueError("Expected at least one source group.")
    group_sizes = [len(group) for group in sources]
    if any(size == 0 for size in group_sizes):
        raise ValueError("Source groups must not be empty.")

    # 1) Flatten into src and dest lists
    src = tuple(p for group in sources for p in group)
    dest = tuple(dst for dst, group in zip(destinations, sources) for _ in group)

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
    sources: Sequence[str],
    destinations: Sequence[str],
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

    """
    rl = stage.GetRootLayer()

    # Group replication by destination path depth so ancestors land before deeper paths.
    # This avoids composition issues for nested or interdependent specs.
    def dp_depth(template: str) -> int:
        """Return destination prim path depth for stable parent-first replication."""
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
                    if src == dp:
                        pass  # self-copy: CreatePrimInLayer already ensures it exists; CopySpec would be destructive
                    else:
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
    """Create inverted collision groups for clones (PhysX only).

    Sets PhysX scene attributes and collision groups on the prim at ``physicsscene_path``
    (no PhysxSchema import). Call only when the physics backend is PhysX; Newton uses
    its own collision/world handling and does not use USD PhysX collision groups.

    Creates one PhysicsCollisionGroup per prim under ``collision_root_path``, enabling
    inverted filtering so clones don't collide across groups. Optionally adds a global
    group that collides with all.

    Args:
        stage: USD stage.
        physicsscene_path: Path to PhysicsScene prim.
        collision_root_path: Root scope for collision groups.
        prim_paths: Per-clone prim paths.
        global_paths: Optional global-collider paths.

    """

    scene_prim = stage.GetPrimAtPath(physicsscene_path)
    # We invert the collision group filters for more efficient collision filtering across environments
    invert_attr = scene_prim.CreateAttribute("physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool)
    invert_attr.Set(True)

    # Make sure we create the collision_scope in the RootLayer since the edit target
    # may be a live layer in the case of Live Sync.
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
