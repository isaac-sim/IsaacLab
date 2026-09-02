# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import copy
import re
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

from isaaclab.physics import PhysicsManager
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors

from isaaclab_newton.cloner.newton_clone_utils import (
    _restore_visible_colliders_without_visual_shapes,
    build_source_builders,
    replicate_builder_mapping,
)
from isaaclab_newton.physics import NewtonCfg, NewtonManager
from isaaclab_newton.renderers.visual_material import import_builder_visual_material_paths

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.sim import SimulationContext


def copy_newton_clone_source(source_path: str, xform: wp.transform | None = None) -> ModelBuilder:
    """Copy a retained clone-source builder without sharing mutable shape geometry.

    Args:
        source_path: Clone-plan source prim path retained during Newton replication.
        xform: Optional transform applied while copying the source.

    Returns:
        An independent builder that is safe to finalize or extend.

    Raises:
        RuntimeError: If Newton replication did not retain the requested source.
    """
    source = NewtonManager._cl_protos.get(source_path)
    if source is None:
        raise RuntimeError(f"No retained Newton clone source for {source_path!r}.")
    builder = ModelBuilder(up_axis=source.up_axis)
    if xform is None:
        builder.add_builder(source)
    else:
        builder.add_builder(source, xform=xform)
    builder.shape_source = [
        value.copy() if callable(getattr(value, "copy", None)) else copy.copy(value) for value in builder.shape_source
    ]
    return builder


@contextlib.contextmanager
def newton_builder_world_hook(
    hook: Callable[[ModelBuilder, int, list[float], list[float]], None],
) -> Iterator[None]:
    """Temporarily extend every world built by Newton replication.

    The callback must not already be registered. On exit, the context removes
    only its callback and preserves hooks owned by other callers.

    Args:
        hook: Callback receiving the builder, world index, world position [m],
            and world orientation quaternion in xyzw order during replication.

    Yields:
        Control while the callback is registered.

    Raises:
        RuntimeError: If the callback is already registered.
    """
    hooks = NewtonManager._per_world_builder_hooks
    if hook in hooks:
        raise RuntimeError("Newton world-builder hook is already registered.")
    hooks.append(hook)
    try:
        yield
    finally:
        if hook in hooks:
            hooks.remove(hook)


def _build_newton_builder_from_mapping(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    up_axis: str = "Z",
    load_visual_shapes: bool = True,
    global_paths: tuple[str, ...] = (),
) -> tuple[ModelBuilder, object, dict, list, dict[str, ModelBuilder], list[tuple[str, int]]]:
    """Build a Newton model builder from clone mapping inputs and retain its source builders."""
    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    manager_cls = PhysicsManager._sim.physics_manager

    builder = manager_cls.create_builder(up_axis=up_axis)
    import_paths = (PhysicsManager._sim.cfg.physics_prim_path, *global_paths)
    hf_ignore_paths = manager_cls._inject_terrain_heightfields(stage, builder, root_paths=import_paths)
    import_results = []
    for root_path in import_paths:
        import_result = builder.add_usd(
            stage,
            root_path=root_path,
            ignore_paths=hf_ignore_paths,
            schema_resolvers=schema_resolvers,
            load_visual_shapes=load_visual_shapes,
        )
        _restore_visible_colliders_without_visual_shapes(
            builder, stage, import_result["path_shape_map"], load_visual_shapes
        )
        import_results.append(import_result)
    stage_info = import_results[0]
    replace_newton_builder_shape_colors(builder, stage)
    if load_visual_shapes:
        import_builder_visual_material_paths(builder, stage)

    # Deformable prim paths are handled by per_world_builder_hooks, not add_usd.
    # Resolve the regex prim_path patterns to concrete env_0 paths so add_usd
    # can skip them via ignore_paths.
    deformable_patterns = tuple(
        re.compile(entry.prim_path.replace(".*", "[^/]*")) for entry in NewtonManager._deformable_registry
    )
    deformable_ignore_paths = []
    if deformable_patterns:
        for source in sources:
            for child in Usd.PrimRange(stage.GetPrimAtPath(source)):
                child_path = str(child.GetPath())
                if any(pattern.fullmatch(child_path) for pattern in deformable_patterns):
                    deformable_ignore_paths.append(child_path)

    source_builders = build_source_builders(
        stage,
        sources,
        lambda: manager_cls.create_builder(up_axis=up_axis),
        schema_resolvers,
        ignore_paths=deformable_ignore_paths or None,
        load_visual_shapes=load_visual_shapes,
    )

    # Inject registered sites into source builders (and global sites into main builder).
    global_sites, source_sites, root_sites = NewtonManager._cl_inject_sites(builder, source_builders)

    replicate_args = (builder, sources, mapping, positions, quaternions, source_builders)
    local_site_map, world_xforms, fabric_body_bindings = replicate_builder_mapping(
        *replicate_args,
        destinations,
        env_ids,
        source_site_indices=source_sites,
        env_root_sites=root_sites,
        per_world_builder_hooks=NewtonManager._per_world_builder_hooks,
    )

    site_index_map = {label: (idx, None) for label, idx in global_sites.items()}
    site_index_map.update((label, (None, per_world)) for label, per_world in local_site_map.items())
    return builder, stage_info, site_index_map, world_xforms, source_builders, fabric_body_bindings


def _renderer_wants_visual_shapes() -> bool:
    """Whether anything in this run will draw the Newton model's visual-only shapes.

    Visual shapes are consumed by the viewers, offscreen ``rgb_array`` capture, and camera
    sensors on any renderer backend. A headless training run without cameras draws none of
    them, so importing them only costs USD parse time and memory.
    """
    sim = PhysicsManager._sim
    if sim is None:
        return True
    return bool(sim.is_rendering or sim.can_render_rgb_array() or sim.visual_shapes_required)


def _replicate_newton(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None,
    quaternions: torch.Tensor | None,
    up_axis: str,
    load_visual_shapes: bool,
    global_paths: tuple[str, ...],
) -> tuple[ModelBuilder, object, dict]:
    """Build one Newton model and publish its plan-derived lookup data."""
    builder, stage_info, site_index_map, world_xforms, source_builders, fabric_body_bindings = (
        _build_newton_builder_from_mapping(
            stage,
            sources,
            destinations,
            env_ids,
            mapping,
            positions,
            quaternions,
            up_axis,
            load_visual_shapes,
            global_paths,
        )
    )
    NewtonManager._cl_site_index_map = site_index_map
    NewtonManager._cl_fabric_body_bindings = fabric_body_bindings
    NewtonManager._world_xforms = world_xforms
    NewtonManager._cl_protos = source_builders
    NewtonManager.set_builder(builder)
    NewtonManager._num_envs = mapping.size(1)
    return builder, stage_info, site_index_map


class NewtonReplicateContext:
    """Build one Newton model from the rows routed to it in a clone plan."""

    replicate_priority = 0

    def __init__(self, sim_context: SimulationContext, *, up_axis: str = "Z"):
        """Initialize the context from its owning simulation."""
        self._sim = sim_context
        self.up_axis = up_axis

    def replicate(self, plan: ClonePlan) -> tuple[ModelBuilder, object, dict]:
        """Build and publish a Newton model from this context's plan rows."""
        if plan.env_ids is None:
            raise ValueError("ClonePlan.env_ids is required for replication.")
        rows = plan.context_rows[type(self)]
        cfg = self._sim.cfg.physics
        load_visual_shapes = cfg.load_visual_shapes if isinstance(cfg, NewtonCfg) else None
        return _replicate_newton(
            self._sim.stage,
            tuple(plan.sources[row] for row in rows),
            tuple(plan.destinations[row] for row in rows),
            plan.env_ids,
            plan.clone_mask[list(rows)],
            plan.positions,
            None,
            self.up_axis,
            _renderer_wants_visual_shapes() if load_visual_shapes is None else load_visual_shapes,
            plan.global_paths,
        )


def newton_physics_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    global_paths: tuple[str, ...] = (),
):
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model builder.
        up_axis: Up axis for the Newton model builder.
        global_paths: Shared scene-asset roots imported once. Defaults to none.

    Returns:
        Tuple of the populated Newton model builder and stage metadata.
    """
    del device
    cfg = PhysicsManager._cfg
    load_visual_shapes = cfg.load_visual_shapes if isinstance(cfg, NewtonCfg) else None
    builder, stage_info, _ = _replicate_newton(
        stage,
        sources,
        destinations,
        env_ids,
        mapping,
        positions,
        quaternions,
        up_axis,
        _renderer_wants_visual_shapes() if load_visual_shapes is None else load_visual_shapes,
        global_paths,
    )
    return builder, stage_info
