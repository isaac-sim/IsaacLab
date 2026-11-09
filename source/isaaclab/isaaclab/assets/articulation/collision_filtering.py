# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD authoring for articulation self-collision exclusions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pxr import Sdf, Usd, UsdPhysics

from ...cloner import path as cloner_path
from ...sim import MultiAssetSpawnerCfg, MultiUsdFileCfg, SimulationContext
from ...sim.utils import find_matching_prims
from ...utils.string import resolve_matching_names

if TYPE_CHECKING:
    from .articulation_cfg import ArticulationCfg


def _apply_self_collision_filters(cfg: ArticulationCfg, stage: Usd.Stage) -> None:
    """Add body exclusions on every source selected by the articulation configuration."""
    expressions = [cfg.prim_path]
    if cfg.spawn is not None:
        if cfg.spawn.spawn_path is not None:
            expressions.append(re.escape(cfg.spawn.spawn_path))
        if isinstance(cfg.spawn, (MultiAssetSpawnerCfg, MultiUsdFileCfg)) and cfg.spawn.spawn_paths is not None:
            expressions.extend(re.escape(path) for path in cfg.spawn.spawn_paths if path is not None)

    sim = SimulationContext.instance()
    plan = sim.get_clone_plan() if sim is not None else None
    if plan is not None:
        sources = cloner_path.get_asset_prototype_paths(plan)
        expressions.extend(
            re.escape(sources[index])
            for index in cloner_path.get_asset_prototypes(plan, cfg.prim_path)
            if sources[index] is not None
        )
        # A non-spawning articulation may be owned by a larger scene source.
        templates, _ = cloner_path.get_world_prototype_asset_templates(plan)
        for asset, template in zip(plan.topology.world_prototypes, templates, strict=True):
            source = sources[asset]
            if source is None or "{}" not in template:
                continue
            matched = cloner_path.match(cfg.prim_path, template)
            if matched is not None:
                expressions.append(re.escape(source) + matched.suffix)

    roots = {prim.GetPath() for expression in expressions for prim in find_matching_prims(expression, stage=stage)}
    if not roots:
        raise ValueError(f"No asset prims found for self-collision filters at {cfg.prim_path!r}.")

    # Joints and other links can be siblings of an asset path that selects its root body or joint.
    descriptors = UsdPhysics.UsdPhysicsLoadStageFromPrimRange(stage, [Sdf.Path.absoluteRootPath])
    paths, articulations = descriptors.get(UsdPhysics.ObjectType.Articulation, ([], []))
    pairs: set[tuple[Sdf.Path, Sdf.Path]] = set()
    for root_path in roots:
        selected = [
            desc
            for path, desc in zip(paths, articulations, strict=True)
            if (
                path.HasPrefix(root_path)
                if cfg.articulation_root_prim_path is None
                else str(path) == str(root_path) + cfg.articulation_root_prim_path
            )
        ]
        if len(selected) != 1 or not selected[0].isValid:
            raise ValueError(
                f"Self-collision filters require one valid articulation at {root_path}; found {len(selected)}."
                " Set articulation_root_prim_path to select an articulation explicitly."
            )
        body_paths = [path for path in selected[0].articulatedBodies if path != Sdf.Path.emptyPath]
        body_names = [path.name for path in body_paths]
        for left, right in cfg.self_collision_filter_pairs:
            left_ids = _resolve_filter_group(left, body_names, root_path)
            right_ids = _resolve_filter_group(right, body_names, root_path)
            for left_id in left_ids:
                for right_id in right_ids:
                    if left_id != right_id:
                        pairs.add(tuple(sorted((body_paths[left_id], body_paths[right_id]))))

    # Validate all sources before authoring, so a bad expression cannot apply a partial configuration.
    for pair in pairs:
        for path in pair:
            if stage.GetPrimAtPath(path).IsInstanceProxy():
                raise ValueError(f"Cannot author self-collision filters on instance-proxy body {path}.")
    for left, right in sorted(pairs):
        right_api = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(right))
        if left in right_api.GetFilteredPairsRel().GetTargets():
            continue
        api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(left))
        api.CreateFilteredPairsRel().AddTarget(right)


def _resolve_filter_group(selector: str | list[str], body_names: list[str], root_path: Sdf.Path) -> set[int]:
    """Resolve a union of expressions, requiring each expression to match a body."""
    expressions = [selector] if isinstance(selector, str) else selector
    if not isinstance(expressions, list) or not expressions or not all(isinstance(expr, str) for expr in expressions):
        raise ValueError(f"Self-collision filter groups must be a string or a non-empty list of strings: {selector!r}.")
    indices = set()
    for expression in expressions:
        try:
            matched, _ = resolve_matching_names(expression, body_names)
        except (ValueError, re.error) as exc:
            raise ValueError(f"Invalid self-collision filter {expression!r} at {root_path}: {exc}") from exc
        indices.update(matched)
    return indices
