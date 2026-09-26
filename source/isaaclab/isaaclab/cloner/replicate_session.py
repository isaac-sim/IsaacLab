# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author declared prototypes and dispatch their topology to clone backends."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from .. import sim as sim_utils
from ..sensors.camera.camera_cfg import CameraCfg
from ..sensors.sensor_base_cfg import SensorBaseCfg
from ..utils.string import string_to_callable
from ..utils.version import has_kit
from .clone_plan import ClonePlan, grid_transforms, make_clone_plan
from .cloner_cfg import DEFAULT_ENV_TEMPLATE, CloneCfg, InclusionSet, expand_env_regex_ns
from .cloner_strategies import sequential
from .path import match
from .query import get_world_prototypes
from .usd import UsdReplicateContext


def num_spawn_variants(spawn_cfg: Any) -> int:
    """Return the number of concrete prototypes declared by a spawner configuration."""
    if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
        return len(spawn_cfg.assets_cfg)
    if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
        return 1 if isinstance(spawn_cfg.usd_path, str) else len(spawn_cfg.usd_path)
    return 1


def make_valid_clone_combinations(
    asset_names: Sequence[str],
    variant_counts: Sequence[int],
    clone_combinations: Sequence[InclusionSet] | None = None,
    *,
    all_asset_names: Sequence[str] | None = None,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Expand named combinations into world memberships and relative weights.

    Repeated names create repeated instances. Each combination's weight is divided equally
    between its variant combinations; neither padding nor duplicate weighted worlds is needed.

    Args:
        asset_names: Names of the replicated asset declarations.
        variant_counts: Number of concrete prototypes for each declaration.
        clone_combinations: Allowed memberships; unnamed assets are present once in every world.
        all_asset_names: Scene names, including shared assets that do not enter world memberships.

    Returns:
        World prototypes indexing the flat variant library, and one weight per world prototype.
    """
    if len(asset_names) != len(variant_counts) or any(count <= 0 for count in variant_counts):
        raise ValueError("Each asset requires one positive variant count.")
    known = set(asset_names if all_asset_names is None else all_asset_names)
    combinations = clone_combinations or (InclusionSet(assets=list(dict.fromkeys(asset_names))),)
    claimed = set().union(*(set(combination.assets) for combination in combinations))
    offsets = np.cumsum([0, *variant_counts])
    worlds, weights = [], []
    for combination in combinations:
        unknown = set(combination.assets) - known
        if unknown:
            raise ValueError(f"Unknown assets in clone combination: {sorted(unknown)}.")
        choices = [
            range(int(offsets[index]), int(offsets[index + 1]))
            for index, name in enumerate(asset_names)
            for _ in range(combination.assets.count(name) if name in claimed else 1)
        ]
        variants = tuple(itertools.product(*choices))
        worlds.extend(variants)
        weights.extend([combination.weight / len(variants)] * len(variants))
    return tuple(worlds), np.asarray(weights, dtype=np.float64)


def replicate(plan: ClonePlan, *, replicate_physics: bool = True) -> None:
    """Execute one topology through its declared clone contexts.

    Args:
        plan: The active simulation's topology.
        replicate_physics: Whether the active physics context performs native replication.
    """
    sim = sim_utils.SimulationContext.instance()
    if sim.get_clone_plan() is not plan:
        raise ValueError("replicate() requires the active SimulationContext's ClonePlan.")
    routing = _context_asset_prototype_ids(plan, sim)
    for context_type in sorted(routing, key=lambda context_type: context_type.replicate_priority):
        if replicate_physics or context_type is not sim.physics_manager.clone_context_type:
            sim.clone_contexts[context_type].replicate(plan, routing[context_type])


def clone_plan_from_env_0(
    clone_cfg: CloneCfg,
    asset_cfgs: Iterable[Any],
    num_envs: int,
    env_spacing: float,
    *,
    positions: np.ndarray | None = None,
) -> ClonePlan:
    """Prepare one homogeneous topology and its USD authoring inputs.

    The plan retains only topology. Environment placement and prototype paths belong to
    the USD clone context; repeated instances inherit their source configuration's pose.

    Args:
        clone_cfg: Clone policy and USD environment namespace.
        asset_cfgs: Flat asset and sensor declarations, including shared assets.
        num_envs: Number of destination worlds.
        env_spacing: Grid spacing between world origins [m].
        positions: Optional world origins [m], shape [num_envs, 3].

    Returns:
        The simulation's published topology, ready for asset construction and replication.
    """
    asset_cfgs = tuple(asset_cfgs)
    if clone_cfg.clone_combinations or any(num_spawn_variants(getattr(cfg, "spawn", None)) != 1 for cfg in asset_cfgs):
        raise ValueError("clone_plan_from_env_0 requires homogeneous, single-variant declarations.")
    return _prepare_cloning(
        asset_cfgs,
        num_envs,
        env_spacing,
        env_template=clone_cfg.clone_template,
        positions=positions,
    )


class ReplicateSession:
    """Author prototypes before dispatching one shared world topology."""

    def __init__(
        self,
        cfgs: Iterable[Any],
        num_clones: int,
        env_spacing: float,
        *,
        clone_strategy: Callable[[np.ndarray, int], np.ndarray] = sequential,
        world_prototypes: Sequence[Sequence[int]] | None = None,
        weights: Sequence[float] | None = None,
        replicate_physics: bool = True,
        env_template: str = DEFAULT_ENV_TEMPLATE,
    ):
        """Capture prototype declarations, composition choices, and USD authoring inputs."""
        self._cfgs = cfgs
        self._replicate_physics = replicate_physics
        self._kwargs = dict(
            num_clones=num_clones,
            env_spacing=env_spacing,
            env_template=env_template,
            clone_strategy=clone_strategy,
            world_prototypes=world_prototypes,
            weights=weights,
        )
        self.plan: ClonePlan | None = None

    def __enter__(self) -> ReplicateSession:
        self.plan = _prepare_cloning(self._cfgs, **self._kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            replicate(self.plan, replicate_physics=self._replicate_physics)
        else:
            sim_utils.SimulationContext.instance().set_clone_plan(None)


def _prepare_cloning(
    cfgs: Iterable[Any],
    num_clones: int,
    env_spacing: float,
    *,
    clone_strategy: Callable[[np.ndarray, int], np.ndarray] = sequential,
    world_prototypes: Sequence[Sequence[int]] | None = None,
    weights: Sequence[float] | None = None,
    env_template: str = DEFAULT_ENV_TEMPLATE,
    positions: np.ndarray | None = None,
) -> ClonePlan:
    """Resolve concrete source definitions and prepare their USD authoring context."""
    sim = sim_utils.SimulationContext.instance()
    if sim is None or sim.get_clone_plan() is not None:
        raise RuntimeError("Clone preparation requires a simulation without an existing clone plan.")
    asset_prototypes, groups, shared, declarations = [], [], [], []
    for cfg in cfgs:
        cfg.prim_path = expand_env_regex_ns(cfg.prim_path, env_template)
        if isinstance(cfg, CameraCfg):
            sim.get_or_create_backend(cfg.renderer_cfg)
        spawn = getattr(cfg, "spawn", None)
        if spawn is not None:
            # These paths are planning outputs, including on reused configurations.
            spawn.spawn_path = None
        count = num_spawn_variants(spawn)
        indices = tuple(range(len(asset_prototypes), len(asset_prototypes) + count))
        declarations.append((cfg, indices))
        for variant in range(count):
            prototype = cfg
            if count > 1:
                prototype = copy.copy(cfg)
                prototype.spawn = copy.copy(spawn)
                if isinstance(spawn, sim_utils.MultiAssetSpawnerCfg):
                    prototype.spawn.assets_cfg = [spawn.assets_cfg[variant]]
                else:
                    prototype.spawn.usd_path = spawn.usd_path[variant]
            asset_prototypes.append(prototype)
        if isinstance(cfg, SensorBaseCfg) and spawn is None:
            continue
        if match(cfg.prim_path, env_template) is None:
            shared.extend(indices)
        else:
            groups.append(indices)
    worlds = tuple(itertools.product(*groups)) if world_prototypes is None else world_prototypes
    plan = make_clone_plan(
        asset_prototypes, worlds, num_clones, weights=weights, shared_assets=shared, clone_strategy=clone_strategy
    )
    usd = UsdReplicateContext(
        sim.stage,
        plan,
        env_template=env_template,
        positions=grid_transforms(num_clones, env_spacing)[0] if positions is None else positions,
    )
    source_paths = {index: path for index, path, _, world_ids in usd.instances if len(world_ids)}
    for cfg, indices in declarations:
        spawn = getattr(cfg, "spawn", None)
        if spawn is None:
            continue
        paths = [source_paths.get(index) for index in indices]
        if isinstance(spawn, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg)):
            spawn.spawn_path, spawn.spawn_paths = None, paths
            for index, path in zip(indices, paths, strict=True):
                plan.asset_prototypes[index].spawn.spawn_paths = [path]
        else:
            spawn.spawn_path = paths[0]
    sim.clone_contexts[UsdReplicateContext] = usd
    sim.set_clone_plan(plan)
    return plan


def _context_asset_prototype_ids(plan: ClonePlan, sim) -> dict[type, tuple[int, ...]]:
    """Resolve per-asset routing without adding execution policy to the topology."""
    physics_context = sim.physics_manager.clone_context_type
    render_contexts = {
        string_to_callable(context) if isinstance(context, str) else context
        for context in sim.render_context.clone_contexts
    }
    spawn_contexts = render_contexts - {physics_context}
    if has_kit():
        spawn_contexts.add(UsdReplicateContext)
    shared = set(map(int, plan.world_prototypes[: plan.world_prototype_starts[1]]))
    routing = {context: set() for context in render_contexts}
    if shared and physics_context is not None:
        routing[physics_context] = set()
    active = {int(index) for _, members, world_ids in get_world_prototypes(plan) if len(world_ids) for index in members}
    for index in sorted(active):
        cfg = plan.asset_prototypes[index]
        fields = vars(cfg)
        references = fields.get("cloning_contexts", ())
        contexts = (
            (() if physics_context is None else (physics_context,))
            if references is None
            else tuple(string_to_callable(value) if isinstance(value, str) else value for value in references)
        )
        if isinstance(fields.get("spawn"), sim_utils.SpawnerCfg):
            contexts += tuple(spawn_contexts)
        if index in shared:
            contexts += tuple(context for context in (physics_context, *render_contexts) if context is not None)
        for context_type in contexts:
            if not isinstance(context_type, type):
                raise TypeError(f"{type(cfg).__name__}.cloning_contexts must contain only context classes.")
            routing.setdefault(context_type, set()).add(index)
    for context_type in routing:
        if context_type not in sim.clone_contexts:
            sim.clone_contexts[context_type] = context_type(sim)
    return {context: tuple(sorted(indices)) for context, indices in routing.items()}
