# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import itertools
import logging
import math
import re
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

import torch

from pxr import Sdf, Usd, UsdGeom

from .clone_plan import ClonePlan
from .cloner_strategies import sequential

if TYPE_CHECKING:
    pass

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
        ValueError: When ``path_expr`` is owned by multiple distinct, equally
            specific destination templates (a genuine ambiguity). Nested
            templates do not conflict: the most specific (longest-matching) one
            wins, mirroring :func:`iter_clone_plan_matches`.
        NotImplementedError: When the union of matching rows' clone masks does not
            cover every env (partial-env heterogeneous coverage is unsupported).
    """
    # Collect every template that owns ``path_expr`` together with the suffix below it.
    # A shorter suffix means a longer matched prefix, i.e. a more specific (nearer) owner.
    candidates: list[tuple[str, str, int]] = []
    for source_index, destination_template in enumerate(plan.destinations):
        if "{}" not in destination_template:
            continue
        suffix = get_suffix(path_expr, destination_template)
        if suffix is None:
            continue
        candidates.append((destination_template, suffix, source_index))
    if not candidates:
        return None

    # The nearest owner is the one with the shortest suffix. Distinct templates that tie
    # at this minimal suffix length are a genuine ambiguity that callers cannot resolve.
    min_suffix_len = min(len(suffix) for _, suffix, _ in candidates)
    owning_templates = {template for template, suffix, _ in candidates if len(suffix) == min_suffix_len}
    if len(owning_templates) > 1:
        raise ValueError(f"path_expr {path_expr!r}: matches multiple destination templates {sorted(owning_templates)}.")
    matching_template = next(iter(owning_templates))
    matching_rows = [index for template, _, index in candidates if template == matching_template]
    matching_suffix = next(suffix for template, suffix, _ in candidates if template == matching_template)
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


def make_clone_plan(
    cfgs: Iterable[Any],
    num_clones: int,
    env_spacing: float,
    device: str,
    *,
    clone_strategy: Callable = sequential,
    valid_set: torch.Tensor | None = None,
) -> ClonePlan:
    """Build a :class:`~isaaclab.cloner.ClonePlan` from asset cfgs.

    Iterates ``cfgs``, identifies env-scoped cfgs with a spawn, expands
    :class:`~isaaclab.sim.MultiAssetSpawnerCfg` / :class:`~isaaclab.sim.MultiUsdFileCfg`
    into per-variant prototype rows, runs ``clone_strategy`` to assign prototypes to
    envs, and returns a self-contained :class:`ClonePlan` with ``cfg_rows`` populated.

    Each input cfg's ``spawn_path`` / ``spawn_paths`` is mutated so the subsequent
    asset constructor spawns the prototype into its first active environment. Cfgs
    whose ``prim_path`` is global (not under the env root ``/World/envs/``) or that
    lack a spawn are skipped — they do not appear in the plan and are not replicated.

    Args:
        cfgs: Asset cfgs with resolved ``prim_path`` (no ``{ENV_REGEX_NS}`` macros).
        num_clones: Number of target envs.
        env_spacing: Distance between neighboring grid env origins [m].
        device: Torch device for plan tensors.
        clone_strategy: Function that assigns prototype combinations to envs. Defaults
            to :func:`~isaaclab.cloner.sequential`.
        valid_set: Optional ``[num_combos, num_groups]`` long tensor of valid prototype
            combinations. ``None`` (default) uses the full cartesian product of every
            group's prototype indices.

    Returns:
        A :class:`ClonePlan` whose ``sources``/``destinations``/``clone_mask`` describe
        the flat prototype-to-env mapping and whose ``cfg_rows`` maps each cfg to the
        rows it owns.
    """
    import isaaclab.sim as sim_utils  # noqa: PLC0415

    def num_variants(spawn_cfg: Any) -> int:
        if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
            return len(spawn_cfg.assets_cfg)
        if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
            return 1 if isinstance(spawn_cfg.usd_path, str) else len(spawn_cfg.usd_path)
        return 1

    def set_spawn_paths(spawn_cfg: Any, paths: list[str | None]) -> None:
        if isinstance(spawn_cfg, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg)):
            spawn_cfg.spawn_paths = paths
        else:
            active = [p for p in paths if p is not None]
            if len(active) != 1:
                raise ValueError("Single spawner expects exactly one planned source path.")
            spawn_cfg.spawn_path = active[0]

    env_root_marker = "/World/envs/"
    env_template = "/World/envs/env_{}"

    # 1) Build per-group records: (cfg, spawn_cfg, destination_template, num_variants).
    groups: list[tuple[Any, Any, str, int]] = []
    for cfg in cfgs:
        if not hasattr(cfg, "prim_path") or not hasattr(cfg, "spawn") or cfg.spawn is None:
            continue
        prim_path = cfg.prim_path
        if env_root_marker not in prim_path:
            continue
        count = num_variants(cfg.spawn)
        if count <= 0:
            raise ValueError(f"Spawner at '{prim_path}' must have at least one variant.")
        destination = prim_path.replace(".*", "{}")
        groups.append((cfg, cfg.spawn, destination, count))

    env_ids = torch.arange(num_clones, dtype=torch.long, device=device)
    positions, _ = grid_transforms(num_clones, env_spacing, device=device)

    # 2) No env-scoped cfgs: emit an empty plan so the scene can still proceed.
    if not groups:
        empty_mask = torch.zeros((0, num_clones), dtype=torch.bool, device=device)
        return ClonePlan(
            sources=(),
            destinations=(),
            clone_mask=empty_mask,
            env_ids=env_ids,
            positions=positions,
            cfg_rows={},
        )

    # 3) Homogeneous (every cfg is single-variant): emit the simpler env-root plan.
    if all(count == 1 for _, _, _, count in groups):
        for cfg, spawn_cfg, destination, _ in groups:
            set_spawn_paths(spawn_cfg, [destination.format(0)])
        cfg_rows = {id(cfg): (0,) for cfg, _, _, _ in groups}
        return ClonePlan(
            sources=(env_template.format(0),),
            destinations=(env_template,),
            clone_mask=torch.ones((1, num_clones), dtype=torch.bool, device=device),
            env_ids=env_ids,
            positions=positions,
            cfg_rows=cfg_rows,
        )

    # 4) Heterogeneous: enumerate prototype combos, build per-row mask, mutate spawn paths.
    group_sizes = [count for _, _, _, count in groups]
    if valid_set is None:
        all_combos = list(itertools.product(*[range(s) for s in group_sizes]))
        combos = torch.tensor(all_combos, dtype=torch.long, device=device)
    else:
        combos = valid_set.to(device=device, dtype=torch.long)
    chosen = clone_strategy(combos, num_clones, device)

    group_offsets = torch.tensor([0] + list(itertools.accumulate(group_sizes[:-1])), dtype=torch.long, device=device)
    rows = (chosen + group_offsets).view(-1)
    cols = torch.arange(num_clones, device=device).view(-1, 1).expand(-1, len(group_sizes)).reshape(-1)

    num_rows = sum(group_sizes)
    clone_mask = torch.zeros((num_rows, num_clones), dtype=torch.bool, device=device)
    clone_mask[rows, cols] = True

    sources_list: list[str] = []
    destinations_list: list[str] = []
    cfg_rows: dict[int, tuple[int, ...]] = {}
    row = 0
    for cfg, spawn_cfg, destination, count in groups:
        cfg_rows[id(cfg)] = tuple(range(row, row + count))
        group_mask = clone_mask[row : row + count]
        env_ids_assigned = group_mask.to(torch.int).argmax(dim=1).tolist()
        active = group_mask.any(dim=1).tolist()
        paths = [
            destination.format(env_id) if is_active else None for env_id, is_active in zip(env_ids_assigned, active)
        ]
        for i, path in enumerate(paths):
            destinations_list.append(destination)
            # Inactive prototypes fall back to env-i so the source path stays valid even
            # when the variant has no active environment (matches the legacy behavior).
            sources_list.append(path if path is not None else destination.format(i))
        set_spawn_paths(spawn_cfg, paths)
        row += count

    return ClonePlan(
        sources=tuple(sources_list),
        destinations=tuple(destinations_list),
        clone_mask=clone_mask,
        env_ids=env_ids,
        positions=positions,
        cfg_rows=cfg_rows,
    )


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
