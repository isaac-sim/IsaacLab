# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import copy
import logging
import re
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from newton import Control, Model, ModelBuilder, State

from pxr import Usd

from isaaclab.physics import PhysicsManager
from isaaclab.scene_data import SceneDataFormat
from isaaclab.scene_data.deformable_vis_remap import (
    VolumeVisRemap,
    launch_batch_particle_slice_copy,
    launch_batch_volume_vis_remap,
)
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors

import isaaclab_newton.physics as newton_physics
from isaaclab_newton.cloner.newton_clone_utils import (
    _restore_visible_colliders_without_visual_shapes,
    build_source_builders,
    replicate_builder_mapping,
)
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
from isaaclab_newton.physics.visualization_builder import build_visualization_builder_from_plan
from isaaclab_newton.physics.visualization_deformables import populate_shadow_deformable_registry
from isaaclab_newton.renderers.visual_material import import_builder_visual_material_paths

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.scene_data import SceneDataProvider
    from isaaclab.sim import SimulationContext


logger = logging.getLogger(__name__)


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
    source = newton_physics.NewtonManager._cl_protos.get(source_path)
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
    hook: Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None],
) -> Iterator[None]:
    """Temporarily extend every world built by Newton replication.

    The callback must not already be registered. On exit, the context removes
    only its callback and preserves hooks owned by other callers.

    Args:
        hook: Callback receiving the builder, world index, world position [m] as a NumPy array,
            and world orientation quaternion in xyzw order as a NumPy array during replication.

    Yields:
        Control while the callback is registered.

    Raises:
        RuntimeError: If the callback is already registered.
    """
    hooks = newton_physics.NewtonManager._per_world_builder_hooks
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
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
    up_axis: str = "Z",
    load_visual_shapes: bool = True,
    global_paths: tuple[str, ...] = (),
) -> tuple[ModelBuilder, object, dict, list, dict[str, ModelBuilder], list[tuple[str, int]]]:
    """Build a Newton model builder from clone mapping inputs and retain its source builders."""
    if positions is None:
        positions = np.zeros((mapping.shape[1], 3), dtype=np.float32)
    if quaternions is None:
        quaternions = np.zeros((mapping.shape[1], 4), dtype=np.float32)
        quaternions[:, 3] = 1.0

    manager_cls = PhysicsManager._sim.physics_manager
    schema_resolvers = manager_cls._get_usd_import_schema_resolvers()

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
        re.compile(entry.prim_path.replace(".*", "[^/]*"))
        for entry in newton_physics.NewtonManager._deformable_registry
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
    global_sites, source_sites, root_sites = newton_physics.NewtonManager._cl_inject_sites(builder, source_builders)

    local_site_map, world_xforms, fabric_body_bindings = replicate_builder_mapping(
        builder=builder,
        sources=sources,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        source_builders=source_builders,
        destinations=destinations,
        env_ids=env_ids,
        source_site_indices=source_sites,
        env_root_sites=root_sites,
        per_world_builder_hooks=newton_physics.NewtonManager._per_world_builder_hooks,
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
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None,
    quaternions: np.ndarray | None,
    up_axis: str,
    load_visual_shapes: bool,
    global_paths: tuple[str, ...],
) -> tuple[ModelBuilder, object, dict]:
    """Build one Newton model and publish its plan-derived lookup data."""
    builder, stage_info, site_index_map, world_xforms, source_builders, fabric_body_bindings = (
        _build_newton_builder_from_mapping(
            stage=stage,
            sources=sources,
            destinations=destinations,
            env_ids=env_ids,
            mapping=mapping,
            positions=positions,
            quaternions=quaternions,
            up_axis=up_axis,
            load_visual_shapes=load_visual_shapes,
            global_paths=global_paths,
        )
    )
    newton_physics.NewtonManager._cl_site_index_map = site_index_map
    newton_physics.NewtonManager._cl_fabric_body_bindings = fabric_body_bindings
    newton_physics.NewtonManager._world_xforms = world_xforms
    newton_physics.NewtonManager._cl_protos = source_builders
    newton_physics.NewtonManager.set_builder(builder)
    newton_physics.NewtonManager._num_envs = mapping.shape[1]
    return builder, stage_info, site_index_map


class NewtonReplicateContext:
    """Own the Newton native model and scene-query state for one clone lifecycle."""

    replicate_priority = 0

    def __init__(self, sim_context: SimulationContext, *, up_axis: str = "Z"):
        self._sim = sim_context
        self.up_axis = up_axis
        cfg = sim_context.cfg.physics
        self.physics_cfg = cfg if isinstance(cfg, NewtonCfg) else None
        self.replicates_scene = self.physics_cfg is None
        self._capture_graph = None
        self.clear()

    def clear(self) -> None:
        """Release the native resource and all pointers derived from it."""
        self.builder: ModelBuilder | None = None
        self.model: Model | None = None
        self.state_0: State | None = None
        self.state_1: State | None = None
        self.control: Control | None = None
        self._transform_mapping = None
        self._transform_mapping_ready = False
        self._sdp_generation = None
        self._sensor_tasks: dict[str, Callable[[], None]] = {}
        self._sensor_eager_tasks: set[str] = set()
        self._sensor_state = None
        self._sensor_state_dirty = True
        self._invalidate_sensor_graph()
        self._deformable_registry = []
        self._shadow_deformable_entities = []
        self._scene_data_points = None
        self._scene_data_geometry_mapping = None
        self._sim_particle_q = None
        self._mapped_sim_particle_offsets = None
        self._shadow_deformable_sync_skip_warned = set()
        self._invalidate_shadow_deformable_batch_sync()

    def finalize(self) -> None:
        """Finalize the clone-built model and allocate its native state."""
        self.model = self.builder.finalize(device=self._sim.device)
        self.model.num_envs = self.model.world_count
        if self.physics_cfg is not None and self.physics_cfg.soft_contact_cfg is not None:
            self.model.soft_contact_ke = float(self.physics_cfg.soft_contact_cfg.soft_contact_ke)
            self.model.soft_contact_kd = float(self.physics_cfg.soft_contact_cfg.soft_contact_kd)
            self.model.soft_contact_mu = float(self.physics_cfg.soft_contact_cfg.soft_contact_mu)
        self.state_0 = self.model.state()
        if self.physics_cfg is not None:
            self._deformable_registry = newton_physics.NewtonManager._deformable_registry
            self.state_1 = self.model.state()
            self.control = self.model.control()
        self._transform_mapping_ready = False
        self._sdp_generation = None
        self._mark_sensor_state_dirty()

    def replicate(self, plan: ClonePlan) -> None:
        """Build this resource once from the source rows registered in the plan."""
        if self.physics_cfg is None:
            self.builder, (self._shadow_deformable_entities, groups) = build_visualization_builder_from_plan(
                self._sim.stage, plan, up_axis=self.up_axis, device=self._sim.device
            )
            populate_shadow_deformable_registry(self, groups)
            self.finalize()
            return
        rows = plan.context_rows[type(self)]
        self._deformable_registry = newton_physics.NewtonManager._deformable_registry
        load_visual_shapes = self.physics_cfg.load_visual_shapes
        _replicate_newton(
            stage=self._sim.stage,
            sources=tuple(plan.sources[row] for row in rows),
            destinations=tuple(plan.destinations[row] for row in rows),
            env_ids=plan.env_ids,
            mapping=plan.clone_mask[list(rows)],
            positions=plan.positions,
            quaternions=None,
            up_axis=self.up_axis,
            load_visual_shapes=_renderer_wants_visual_shapes() if load_visual_shapes is None else load_visual_shapes,
            global_paths=plan.global_paths,
        )

    def update_transforms(self) -> None:
        """Bind the SDP transform result directly to the native render state."""
        provider = self._sim.get_scene_data_provider()
        if not self._transform_mapping_ready:
            body_paths = set(self.model.body_label)
            if len(body_paths) != self.model.body_count or not body_paths.issubset(provider.backend.transform_paths):
                raise ValueError("Every Newton render body must have one unique SDP transform path.")
            self._transform_mapping = provider.create_mapping(list(self.model.body_label))
            self._transform_mapping_ready = True
        transforms = provider.request_transforms(
            SceneDataFormat.Transform, mapping=self._transform_mapping, count=self.model.body_count
        )
        generation = provider.transform_generation
        if transforms is not None and generation != self._sdp_generation:
            if self.state_0.body_q is not transforms.transforms:
                self._invalidate_sensor_graph()
            self.state_0.body_q = transforms.transforms
            self._sdp_generation = generation
            self._mark_sensor_state_dirty()
        if self.physics_cfg is not None or self.state_0.particle_q is None or provider.point_count == 0:
            return
        scene_data_provider = provider
        if self.state_0.particle_q is not None and scene_data_provider.point_count > 0:
            if self._scene_data_points is None:
                self._scene_data_points = SceneDataFormat.Points()

            # Recreate when ``None``. Identity layouts return ``None`` from
            # :meth:`create_geometry_mapping`, so this also refreshes mapped-offset and
            # batch-sync metadata each frame. Caching ``None`` via a ready-flag without
            # that refresh regresses Franka soft viz (stretched / missing deformables).
            if self._scene_data_geometry_mapping is None and self._shadow_deformable_entities:
                geometry_paths = [entity.root_path for entity in self._shadow_deformable_entities]
                geometry_offsets = [entity.sim_particle_offset for entity in self._shadow_deformable_entities]
                self._scene_data_geometry_mapping = scene_data_provider.create_geometry_mapping(
                    geometry_paths, geometry_offsets
                )

                # Invalidate the mapped-offset cache so that it can be rebuilt immediately after.
                self._mapped_sim_particle_offsets = None
                self._invalidate_shadow_deformable_batch_sync()

            if self._mapped_sim_particle_offsets is None:
                self._mapped_sim_particle_offsets = self._geometry_mapped_sim_offsets(scene_data_provider)

            if self._sim_particle_q is None:
                sim_total = sum(entity.sim_particle_count for entity in self._shadow_deformable_entities or [])
                if sim_total > 0:
                    device = self._sim.device or "cpu"
                    self._sim_particle_q = wp.zeros(sim_total, dtype=wp.vec3f, device=device)

            if self._sim_particle_q is not None:
                self._scene_data_points.points = self._sim_particle_q
                scene_data_provider.get_points(
                    self._scene_data_points,
                    mapping=self._scene_data_geometry_mapping,
                    allow_passthrough=False,
                )
                self._sync_render_particle_q_from_sim()
            else:
                self._scene_data_points.points = self.state_0.particle_q
                scene_data_provider.get_points(
                    self._scene_data_points,
                    mapping=self._scene_data_geometry_mapping,
                    allow_passthrough=False,
                )

        self._mark_sensor_state_dirty()

    def _register_sensor_task(self, name: str, update_fn: Callable[[], None], *, graph_capturable: bool = True) -> None:
        """Register a scene-query task.

        Args:
            name: Unique task name.
            update_fn: Callable run by :meth:`_update_sensor_tasks`.
            graph_capturable: Whether ``update_fn`` supports conditional CUDA graph capture.
        """
        if name in self._sensor_tasks:
            raise ValueError(f"Newton sensor task '{name}' is already registered.")
        model = self.model
        state = self.state_0
        if model is None or state is None:
            raise RuntimeError("Registering a Newton sensor task requires an initialized model and state.")
        if model.shape_count > 0 and model.bvh_shapes is None:
            model.bvh_build_shapes(state)
        if model.particle_count > 0 and model.bvh_particles is None:
            model.bvh_build_particles(state)
        self._sensor_tasks[name] = update_fn
        if not graph_capturable:
            self._sensor_eager_tasks.add(name)
        self._sensor_state = state
        self._sensor_state_dirty = True
        self._invalidate_sensor_graph()

    def _unregister_sensor_task(self, name: str) -> None:
        """Remove a scene-query task, ignoring unknown names."""
        if self._sensor_tasks.pop(name, None) is not None:
            self._sensor_eager_tasks.discard(name)
            self._invalidate_sensor_graph()

    def _update_sensor_tasks(self, *names: str) -> None:
        """Refresh derived state, refit the BVHs, and run the requested scene-query tasks."""
        for name in names:
            if name not in self._sensor_tasks:
                raise KeyError(f"Newton sensor task '{name}' is not registered.")

        # Resolve pending FK before entering the graph-capturable sensor pipeline.
        self.update_transforms()
        state = self.state_0
        if state is not self._sensor_state:
            self._sensor_state = state
            self._sensor_state_dirty = True
            self._invalidate_sensor_graph()
        if self._sensor_eager_tasks.intersection(names):
            if self._sensor_state_dirty:
                self._refit_sensor_bvh()
                self._sensor_state_dirty = False
            for name in names:
                self._sensor_tasks[name]()
            return
        use_cuda_graph = bool(getattr(self.physics_cfg, "use_cuda_graph", False)) and "cuda" in self._sim.device
        if use_cuda_graph and self._sensor_graph is None and not self._sensor_graph_capture_failed:
            self._capture_sensor_graph()
        if self._sensor_graph is None:
            if self._sensor_state_dirty:
                self._refit_sensor_bvh()
                self._sensor_state_dirty = False
            for name in names:
                self._sensor_tasks[name]()
            return

        assert self._sensor_flags_host is not None
        assert self._sensor_flags is not None
        self._sensor_flags_host.fill(0)
        self._sensor_flags_host[0] = int(self._sensor_state_dirty)
        task_names = tuple(name for name in self._sensor_tasks if name not in self._sensor_eager_tasks)
        for name in names:
            self._sensor_flags_host[1 + task_names.index(name)] = 1
        self._sensor_flags.assign(self._sensor_flags_host)
        wp.capture_launch(self._sensor_graph)
        self._sensor_state_dirty = False

    def _mark_sensor_state_dirty(self) -> None:
        """Invalidate acceleration structures after a native state publication."""
        if self.state_0 is not self._sensor_state:
            self._sensor_state = self.state_0
            self._invalidate_sensor_graph()
        self._sensor_state_dirty = True

    def _refit_sensor_bvh(self) -> None:
        """Refit the model shape and particle BVHs against the current state."""
        if self.model is None:
            return

        refit_shapes = self.model.shape_count > 0 and self.model.bvh_shapes is not None
        refit_particles = self.model.particle_count > 0 and self.model.bvh_particles is not None
        if not refit_shapes and not refit_particles:
            return

        if self._sensor_state is None:
            raise RuntimeError("Refitting Newton sensor BVHs requires an initialized sensor state.")

        if refit_shapes:
            self.model.bvh_refit_shapes(self._sensor_state)

        if refit_particles:
            self.model.bvh_refit_particles(self._sensor_state)

    def _invalidate_sensor_graph(self) -> None:
        """Discard captured scene-query graph resources."""
        self._sensor_graph = None
        self._sensor_flags = None
        self._sensor_flags_host = None
        self._sensor_graph_capture_failed = False

    def _capture_sensor_graph(self) -> None:
        """Capture BVH refit and scene-query tasks into a conditional graph."""
        graph_tasks = tuple(
            update_fn for name, update_fn in self._sensor_tasks.items() if name not in self._sensor_eager_tasks
        )
        with wp.ScopedDevice(self._sim.device):
            self._refit_sensor_bvh()
            for update_fn in graph_tasks:
                update_fn()

        self._sensor_flags = wp.zeros(1 + len(graph_tasks), dtype=wp.int32, device=self._sim.device)
        self._sensor_flags_host = np.zeros(1 + len(graph_tasks), dtype=np.int32)

        def pipeline() -> None:
            assert self._sensor_flags is not None
            wp.capture_if(self._sensor_flags[0:1], self._refit_sensor_bvh)
            for index, update_fn in enumerate(graph_tasks):
                wp.capture_if(self._sensor_flags[index + 1 : index + 2], update_fn)

        device = self._sim.device
        if self._capture_graph is not None:
            self._sensor_graph = self._capture_graph(device, capture_target=pipeline)
        else:
            try:
                with wp.ScopedCapture(device=device) as capture:
                    pipeline()
                self._sensor_graph = capture.graph
            except Exception:
                logger.exception("[NewtonManager] sensor CUDA graph capture failed")
                self._sensor_graph = None
        if self._sensor_graph is None:
            self._sensor_flags = None
            self._sensor_flags_host = None
            self._sensor_graph_capture_failed = True
            logger.warning("Newton sensor graph capture failed; falling back to eager execution.")
        else:
            logger.info("Captured Newton sensor graph with %d task(s).", len(graph_tasks))

    def _geometry_mapped_sim_offsets(self, scene_data_provider: SceneDataProvider) -> set[int]:
        """Return ``_sim_particle_q`` offsets filled by :meth:`get_points` for the cached mapping.

        Called once when the geometry mapping is created (or when that cache is
        invalidated). ``create_geometry_mapping`` indexes by backend entity and stores
        consumer destinations (or ``-1`` when a backend path has no consumer). The
        inverse case — a shadow entity whose path the backend never reports — never
        appears in that array, so its sim slice stays at the zero initialization.
        """
        mapping = self._scene_data_geometry_mapping
        if mapping is not None:
            return {int(value) for value in mapping.numpy() if int(value) >= 0}

        # Identity layout: backend entities land at sequential flat offsets.
        offsets: set[int] = set()
        flat_offset = 0
        for count in scene_data_provider.backend.geometry_counts:
            offsets.add(flat_offset)
            flat_offset += int(count)
        return offsets

    def _invalidate_shadow_deformable_batch_sync(self) -> None:
        """Drop cached batched remap/copy metadata."""
        self._shadow_deformable_remap_batches = None
        self._shadow_deformable_copy_batch = None
        self._shadow_deformable_batch_sync_key = None

    def _ensure_shadow_deformable_batch_sync(self) -> None:
        """Build batched remap/copy launch metadata for mapped shadow deformables."""
        entities = self._shadow_deformable_entities or []
        mapped_offsets = self._mapped_sim_particle_offsets
        sync_key = (
            id(entities),
            frozenset(mapped_offsets or ()),
            tuple(
                (
                    entity.root_path,
                    entity.sim_particle_offset,
                    entity.vis_particle_offset,
                    entity.sim_particle_count,
                    entity.vis_particle_count,
                    id(entity.volume_vis_remap),
                )
                for entity in entities
            ),
        )
        if self._shadow_deformable_batch_sync_key == sync_key:
            return

        self._shadow_deformable_batch_sync_key = sync_key
        self._shadow_deformable_remap_batches = []
        self._shadow_deformable_copy_batch = None

        if self._sim_particle_q is None or not entities:
            return

        device = str(self._sim_particle_q.device)
        copy_entity_ids: list[int] = []
        copy_src_offsets: list[int] = []
        copy_dst_offsets: list[int] = []
        copy_counts: list[int] = []
        remap_groups: dict[int, tuple[VolumeVisRemap, list]] = {}

        for entity_index, entity in enumerate(entities):
            if mapped_offsets is not None and entity.sim_particle_offset not in mapped_offsets:
                continue

            if entity.volume_vis_remap is not None:
                remap_key = id(entity.volume_vis_remap)
                if remap_key not in remap_groups:
                    remap_groups[remap_key] = (entity.volume_vis_remap, [])
                remap_groups[remap_key][1].append(entity)
            elif entity.vis_particle_count > 0 and entity.vis_particle_count == entity.sim_particle_count:
                copy_src_offsets.append(entity.sim_particle_offset)
                copy_dst_offsets.append(entity.vis_particle_offset)
                copy_counts.append(entity.vis_particle_count)

        if copy_counts:
            count_prefix = np.zeros(len(copy_counts), dtype=np.int32)
            running = 0
            for index, count in enumerate(copy_counts):
                count_prefix[index] = running
                for _ in range(int(count)):
                    copy_entity_ids.append(index)
                running += int(count)
            self._shadow_deformable_copy_batch = (
                wp.array(copy_entity_ids, dtype=wp.int32, device=device),
                wp.array(copy_src_offsets, dtype=wp.int32, device=device),
                wp.array(copy_dst_offsets, dtype=wp.int32, device=device),
                wp.array(np.asarray(copy_counts, dtype=np.int32), dtype=wp.int32, device=device),
                wp.array(count_prefix, dtype=wp.int32, device=device),
            )

        remap_batches: list[tuple] = []
        for remap, group_entities in remap_groups.values():
            entity_ids: list[int] = []
            sim_offsets: list[int] = []
            render_offsets: list[int] = []
            vis_counts: list[int] = []
            vis_prefix = np.zeros(len(group_entities), dtype=np.int32)
            running = 0
            for index, entity in enumerate(group_entities):
                vis_prefix[index] = running
                for _ in range(entity.vis_particle_count):
                    entity_ids.append(index)
                running += entity.vis_particle_count
                sim_offsets.append(entity.sim_particle_offset)
                render_offsets.append(entity.vis_particle_offset)
                vis_counts.append(entity.vis_particle_count)

            if not entity_ids:
                continue

            remap_batches.append(
                (
                    wp.array(entity_ids, dtype=wp.int32, device=device),
                    wp.array(np.asarray(sim_offsets, dtype=np.int32), dtype=wp.int32, device=device),
                    wp.array(np.asarray(render_offsets, dtype=wp.int32), dtype=wp.int32, device=device),
                    wp.array(np.asarray(vis_counts, dtype=wp.int32), dtype=wp.int32, device=device),
                    wp.array(vis_prefix, dtype=wp.int32, device=device),
                    remap,
                )
            )

        self._shadow_deformable_remap_batches = remap_batches

    def _sync_render_particle_q_from_sim(self) -> None:
        """Copy or remap sim nodal positions into shadow ``particle_q`` render slots."""
        if self.state_0 is None or self.state_0.particle_q is None or self._sim_particle_q is None:
            return
        if not self._shadow_deformable_entities:
            return

        mapped_offsets = self._mapped_sim_particle_offsets
        for entity in self._shadow_deformable_entities:
            if mapped_offsets is not None and entity.sim_particle_offset not in mapped_offsets:
                warned = self._shadow_deformable_sync_skip_warned
                if entity.root_path not in warned:
                    warned.add(entity.root_path)
                    logger.warning(
                        "Skipping particle sync for deformable '%s': no SceneData geometry "
                        "mapping resolved for sim_offset=%d; render slots stay at rest pose.",
                        entity.root_path,
                        entity.sim_particle_offset,
                    )
                continue

            if (
                entity.volume_vis_remap is None
                and entity.vis_particle_count != entity.sim_particle_count
                and entity.vis_particle_count > 0
            ):
                warned = self._shadow_deformable_sync_skip_warned
                if entity.root_path not in warned:
                    warned.add(entity.root_path)
                    logger.warning(
                        "Skipping particle sync for deformable '%s': vis_count=%d != sim_count=%d "
                        "and no volume remapping table is available; render slots stay at rest pose.",
                        entity.root_path,
                        entity.vis_particle_count,
                        entity.sim_particle_count,
                    )

        self._ensure_shadow_deformable_batch_sync()

        for (
            entity_ids,
            sim_offsets,
            render_offsets,
            vis_counts,
            vis_prefix,
            remap,
        ) in self._shadow_deformable_remap_batches or []:
            launch_batch_volume_vis_remap(
                self._sim_particle_q,
                self.state_0.particle_q,
                entity_ids,
                sim_offsets,
                render_offsets,
                vis_counts,
                vis_prefix,
                remap.tet_vertex_indices,
                remap.bary_weights,
            )

        copy_batch = self._shadow_deformable_copy_batch
        if copy_batch is not None:
            entity_ids, src_offsets, dst_offsets, counts, count_prefix = copy_batch
            launch_batch_particle_slice_copy(
                self._sim_particle_q,
                self.state_0.particle_q,
                entity_ids,
                src_offsets,
                dst_offsets,
                counts,
                count_prefix,
            )


def newton_physics_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
    up_axis: str = "Z",
    global_paths: tuple[str, ...] = (),
) -> tuple[ModelBuilder, dict[str, Any]]:
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        up_axis: Up axis for the Newton model builder.
        global_paths: Shared scene-asset roots imported once. Defaults to none.

    Returns:
        Tuple of the populated Newton model builder and stage metadata.
    """
    cfg = PhysicsManager._cfg
    load_visual_shapes = cfg.load_visual_shapes if isinstance(cfg, NewtonCfg) else None
    builder, stage_info, _ = _replicate_newton(
        stage=stage,
        sources=sources,
        destinations=destinations,
        env_ids=env_ids,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        up_axis=up_axis,
        load_visual_shapes=_renderer_wants_visual_shapes() if load_visual_shapes is None else load_visual_shapes,
        global_paths=global_paths,
    )
    return builder, stage_info
