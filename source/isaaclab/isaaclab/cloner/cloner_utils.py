# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import itertools
import logging
import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pxr import Usd

from .clone_plan import ClonePlan
from .cloner_cfg import InclusionSet
from .cloner_strategies import sequential

logger = logging.getLogger(__name__)


def split_clone_template(destination_template: str) -> tuple[str, str]:
    """Split a clone destination template around its clone slot.

    The ``"{}"`` slot represents one concrete environment/instance path segment.

    Args:
        destination_template: Destination path template with ``"{}"`` for the instance id.

    Returns:
        The ``(prefix, suffix)`` around the clone slot.

    Raises:
        ValueError: If ``destination_template`` does not contain a clone slot.
    """
    destination_template = destination_template.rstrip("/") or "/"
    prefix, slot, suffix = destination_template.partition("{}")
    if slot != "{}":
        raise ValueError(f"Clone destination template must contain '{{}}': {destination_template!r}.")
    return prefix, suffix


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
    pattern = re.compile(r"[^/]+".join(re.escape(part) for part in split_clone_template(destination_template)))
    match = pattern.match(path_expr)
    if match is None:
        return None
    suffix = path_expr[match.end() :]
    return None if suffix and not suffix.startswith("/") else suffix


def replace_path_prefix(path: str, source_prefix: str, destination_prefix: str) -> str:
    """Replace ``source_prefix`` in ``path`` with ``destination_prefix`` on a path boundary."""
    source_prefix = source_prefix.rstrip("/") or "/"
    destination_prefix = destination_prefix.rstrip("/") or "/"
    if not path.startswith(source_prefix):
        return path
    suffix = path[len(source_prefix) :]
    if suffix and not suffix.startswith("/"):
        return path
    return destination_prefix + suffix


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
        Returns ``None`` when ``path_expr`` matches no row in the plan, or when the
        matching rows have no active env, letting callers fall back to direct stage
        resolution (e.g. for sensor frames mounted at the env root rather than under
        a planned asset).

        Partial-env coverage is supported: when the matching rows cover only a subset
        of envs (an asset present in some envs but not others, as in heterogeneous
        scenes), the returned destination glob resolves to just those envs.

    Raises:
        ValueError: When ``path_expr`` is owned by multiple distinct, equally
            specific destination templates (a genuine ambiguity). Nested
            templates do not conflict: the most specific (longest-matching) one
            wins, mirroring :func:`iter_clone_plan_matches`.
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
    # Partial-env coverage (the union of matching rows misses some envs) is expected for
    # heterogeneous scenes: an asset present in only a subset of envs (e.g. one robot type
    # per task group). The destination glob below resolves only to the envs that actually
    # received the asset, and callers (via the scene Selector) map those to global env ids.
    # Resolution must still walk a source that exists on stage, so prefer the first matching
    # row with at least one active env over an inactive fallback source.
    active_rows = [index for index in matching_rows if plan.clone_mask[index].any()]
    if not active_rows:
        return None
    return plan.sources[active_rows[0]], matching_template.replace("{}", "*"), matching_suffix or ""


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


def num_spawn_variants(spawn_cfg: Any) -> int:
    """Return the number of spawn variants declared by one spawner configuration.

    :class:`~isaaclab.sim.MultiAssetSpawnerCfg` declares one variant per asset
    configuration and :class:`~isaaclab.sim.MultiUsdFileCfg` one per USD path;
    every other spawner declares a single variant.

    Args:
        spawn_cfg: Spawner configuration to inspect.

    Returns:
        The number of spawn variants the configuration expands into.
    """
    import isaaclab.sim as sim_utils  # noqa: PLC0415

    if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
        return len(spawn_cfg.assets_cfg)
    if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
        return 1 if isinstance(spawn_cfg.usd_path, str) else len(spawn_cfg.usd_path)
    return 1


def make_valid_clone_combinations(
    asset_names: Sequence[str],
    variant_counts: Sequence[int],
    clone_combinations: Sequence[InclusionSet] | None = None,
    device: str = "cpu",
    *,
    all_asset_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Build the valid clone-combination variant tensor.

    Each combination contributes rows in proportion to its weight, split evenly
    across its spawn variants and interleaved round-robin, so any prefix of the
    tensor samples every combination.

    Args:
        asset_names: Clone-planned scene asset names, one per tensor column.
        variant_counts: Number of spawn variants per clone-planned asset.
        clone_combinations: Legal clone combinations; assets not mentioned by
            any combination are active in every row. ``None`` uses the full
            cartesian product of variants.
        device: Torch device for the output tensor. Defaults to ``"cpu"``.
        all_asset_names: Optional full scene asset-name list; combination
            entries may reference assets that are not clone-planned.

    Returns:
        A ``[num_valid_combinations, num_assets]`` tensor of source variant
        indices, ``-1`` where an asset is absent.

    Raises:
        ValueError: If the inputs are inconsistent or no valid rows result.
    """
    if len(asset_names) != len(variant_counts):
        raise ValueError(f"Expected one variant count per asset, got {len(variant_counts)} and {len(asset_names)}.")
    if not asset_names:
        raise ValueError("Expected at least one asset name.")
    if any(count <= 0 for count in variant_counts):
        raise ValueError("Variant counts must be positive.")

    if not clone_combinations:
        rows = itertools.product(*[range(count) for count in variant_counts])
        return torch.tensor(list(rows), dtype=torch.long, device=device)

    clone_asset_names = set(asset_names)
    known_assets = set(all_asset_names) if all_asset_names is not None else clone_asset_names
    combination_assets: list[set[str]] = []
    for combination in clone_combinations:
        if combination.weight < 0:
            raise ValueError("Clone combination weights must be non-negative.")
        unknown_assets = sorted(set(combination.assets) - known_assets)
        if unknown_assets:
            raise ValueError(f"Unknown assets in clone combination: {unknown_assets}.")
        combination_assets.append(set(combination.assets) & clone_asset_names)

    claimed_assets = set().union(*combination_assets) if combination_assets else set()

    expanded: list[tuple[int, list[tuple[int, ...]]]] = []
    for combination, active_assets in zip(clone_combinations, combination_assets):
        if combination.weight == 0:
            continue
        variant_ranges = []
        for asset_name, count in zip(asset_names, variant_counts):
            is_active = asset_name not in claimed_assets or asset_name in active_assets
            variant_ranges.append(range(count) if is_active else (-1,))
        expanded.append((combination.weight, list(itertools.product(*variant_ranges))))

    if not expanded:
        raise ValueError("Clone combinations produced no valid clone rows.")

    # A combination's share is its weight, split evenly across its spawn variants.
    # Integer multiplicities require a common denominator across variant counts.
    # Rows are emitted round-robin across combinations so a truncated prefix
    # (fewer environments than rows) still samples every combination.
    common_multiple = math.lcm(*[len(variants) for _, variants in expanded])
    rows = []
    cursors = [0] * len(expanded)
    for _ in range(common_multiple):
        for index, (weight, variants) in enumerate(expanded):
            for _ in range(weight):
                rows.append(variants[cursors[index] % len(variants)])
                cursors[index] += 1
    return torch.tensor(rows, dtype=torch.long, device=device)


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

    def set_spawn_paths(spawn_cfg: Any, paths: list[str | None]) -> None:
        if isinstance(spawn_cfg, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg)):
            spawn_cfg.spawn_paths = paths
        else:
            active = [p for p in paths if p is not None]
            if len(active) == 0:
                spawn_cfg.spawn_path = None
                return
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
        count = num_spawn_variants(cfg.spawn)
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
    if valid_set is None and all(count == 1 for _, _, _, count in groups):
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

    def validate_combo_tensor(combos: torch.Tensor, name: str, expected_rows: int | None = None) -> torch.Tensor:
        if combos.dtype == torch.bool or torch.is_floating_point(combos):
            raise ValueError(f"{name} must contain integer prototype indices.")
        combos = combos.to(device=device, dtype=torch.long)
        if combos.ndim != 2:
            raise ValueError(f"{name} must be a 2-D tensor, got shape {tuple(combos.shape)}.")
        if combos.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one row.")
        if expected_rows is not None and combos.shape[0] != expected_rows:
            raise ValueError(f"{name} must contain {expected_rows} rows, got {combos.shape[0]}.")
        if combos.shape[1] != len(group_sizes):
            raise ValueError(f"{name} must contain {len(group_sizes)} columns, got {combos.shape[1]}.")
        group_sizes_tensor = torch.tensor(group_sizes, dtype=torch.long, device=device).view(1, -1)
        invalid = (combos < -1) | ((combos >= group_sizes_tensor) & (combos != -1))
        if invalid.any():
            raise ValueError(f"{name} contains prototype indices outside [-1, group_size).")
        return combos

    if valid_set is None:
        all_combos = list(itertools.product(*[range(s) for s in group_sizes]))
        combos = torch.tensor(all_combos, dtype=torch.long, device=device)
    else:
        combos = validate_combo_tensor(valid_set, "valid_set")
    chosen = validate_combo_tensor(clone_strategy(combos, num_clones, device), "clone_strategy result", num_clones)

    group_offsets = torch.tensor([0] + list(itertools.accumulate(group_sizes[:-1])), dtype=torch.long, device=device)
    active = chosen >= 0
    rows = (chosen + group_offsets).view(-1)
    cols = torch.arange(num_clones, device=device).view(-1, 1).expand(-1, len(group_sizes)).reshape(-1)
    active_flat = active.view(-1)

    num_rows = sum(group_sizes)
    clone_mask = torch.zeros((num_rows, num_clones), dtype=torch.bool, device=device)
    if active_flat.any():
        clone_mask[rows[active_flat], cols[active_flat]] = True

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
    # Lazy: importing pxr from the kit-less usd-core wheel before Kit boots corrupts
    # Kit's own USD runtime; only this function needs pxr at runtime.
    from pxr import Sdf, Usd, UsdGeom  # noqa: PLC0415

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
