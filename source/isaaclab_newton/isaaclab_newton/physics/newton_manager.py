# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import contextlib
import gc
import logging
import re
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from newton import Axis, Contacts, ModelBuilder, ModelFlags, State
from newton.selection import ArticulationView
from newton.sensors import SensorContact as NewtonContactSensor
from newton.sensors import SensorIMU as NewtonSensorIMU

from pxr import UsdGeom

import isaaclab.cloner as cloner
from isaaclab.physics import CallbackHandle, PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataProvider
from isaaclab.sim import SimulationContext
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import has_kit
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

from isaaclab_newton.physics.newton_backend_cfg import NewtonBackendCfg
from isaaclab_newton.physics.newton_builder_cfg import NewtonBuilderCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
from isaaclab_newton.renderers.visual_material import (
    VisualMaterialWriter,
    VisualShapeColorWriter,
)

from .newton_backend import NewtonBackend, NewtonCloneRecord, SiteEntry
from .newton_builder import create_builder
from .newton_scene_data import NewtonSceneDataBackend

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation
    from isaaclab.renderers.base_renderer import VisualMaterialBatch


def _compile_label_pattern(expr: str | list[str] | None) -> re.Pattern[str] | None:
    """Compile selector expressions for Newton's full label matching."""
    if not expr:
        return None
    return re.compile("|".join((expr,) if isinstance(expr, str) else expr))


logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _or_reset_masks_from_mask(
    env_mask: wp.array(dtype=wp.bool),
    articulation_ids: wp.array2d(dtype=int),
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
):
    """OR env_mask into world_mask and set corresponding articulation bits in fk_mask."""
    world, arti = wp.tid()
    if env_mask[world]:
        world_mask[world] = True
        fk_mask[articulation_ids[world, arti]] = True


@wp.kernel(enable_backward=False)
def _scatter_reset_masks_from_ids(
    env_ids: wp.array(dtype=Any),
    articulation_ids: wp.array2d(dtype=int),
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
):
    """Scatter-set world_mask and fk_mask from sparse env_ids."""
    i, arti = wp.tid()
    world = wp.int32(env_ids[i])
    world_mask[world] = True
    fk_mask[articulation_ids[world, arti]] = True


_SCATTER_RESET_MASKS_FROM_IDS_DISPATCHER = IndexKernelDispatcher(_scatter_reset_masks_from_ids, ("env_ids",))


def _scatter_reset_masks_from_ids_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the reset-mask writer matching the environment selector dtype."""
    return _SCATTER_RESET_MASKS_FROM_IDS_DISPATCHER.select(env_ids)


@wp.kernel(enable_backward=False)
def _or_world_reset_mask_from_mask(env_mask: wp.array(dtype=wp.bool), world_mask: wp.array(dtype=wp.bool)):
    """Mark masked worlds for solver reset without requesting FK."""
    world = wp.tid()
    if env_mask[world]:
        world_mask[world] = True


@wp.kernel(enable_backward=False)
def _scatter_world_reset_mask_from_ids(env_ids: wp.array(dtype=wp.int32), world_mask: wp.array(dtype=wp.bool)):
    """Mark selected worlds for solver reset without requesting FK."""
    world_mask[env_ids[wp.tid()]] = True


def create_newton_builder(cfg: NewtonBuilderCfg) -> ModelBuilder:
    """Construct the shared builder: solver-aware under Newton physics, plain otherwise.

    Args:
        cfg: The selected physics configuration.

    Returns:
        An empty builder.
    """
    if isinstance(cfg.physics_cfg, NewtonCfg):
        return create_builder(cfg.physics_cfg)
    return ModelBuilder()


class NewtonQueries:
    """Stateless query and capture functions operating on an explicit Newton resource."""

    @staticmethod
    def run_query(
        backend: NewtonBackend,
        timestamp: int,
        query: Callable[[], None],
        graph: tuple[tuple[int, ...], wp.Graph] | None,
        *,
        use_cuda_graph: bool = True,
    ) -> tuple[tuple[int, ...], wp.Graph] | None:
        """Refresh shared BVHs once per publication and run a consumer-owned query graph.

        Args:
            backend: Shared model, state, and acceleration structures.
            timestamp: Sum of the monotonic SDP transform and geometry timestamps for this state.
            query: Camera or ray-cast work using the supplied backend.
            graph: This consumer's previous captured query and pointer layout, or None.
            use_cuda_graph: Whether to capture GPU queries. CPU queries always run eagerly.

        Returns:
            The consumer's captured query and pointer layout, or None for eager execution.
        """
        model, state = backend.model, backend.state_0
        pointers = tuple(array.ptr if array is not None else 0 for array in (state.body_q, state.particle_q))
        cached = backend.bvh_refit
        if cached.timestamp != timestamp:
            if model.device.is_cuda and use_cuda_graph:
                if cached.data is None or cached.data[0] != pointers:
                    refit = partial(NewtonQueries._refit_bvh, backend)
                    cached.data = pointers, NewtonQueries.capture_graph(str(model.device), refit, relaxed=has_kit())
                wp.capture_launch(cached.data[1])
            else:
                NewtonQueries._refit_bvh(backend)
            cached.timestamp = timestamp
        if not model.device.is_cuda or not use_cuda_graph:
            query()
            return None
        if graph is None or graph[0] != pointers:
            graph = pointers, NewtonQueries.capture_graph(str(model.device), query, relaxed=has_kit())
        wp.capture_launch(graph[1])
        return graph

    @staticmethod
    def capture_graph(device: str, capture_target: Callable[[], None], *, relaxed: bool = False) -> wp.Graph:
        """Record work without an eager warmup or graph replay, using Warp's public capture API."""
        stream = wp.get_stream(device)
        mode = wp.CaptureMode.THREAD_LOCAL
        if relaxed:
            # RTX uses the legacy CUDA stream. A nonblocking stream avoids implicit synchronization.
            stream = wp.stream_from_torch(torch.cuda.Stream(device=device))
            mode = wp.CaptureMode.RELAXED
        with NewtonQueries._paused_gc(), wp.ScopedStream(stream):
            with wp.ScopedCapture(stream=stream, capture_mode=mode) as capture:
                capture_target()
        return capture.graph

    @staticmethod
    def _refit_bvh(backend: NewtonBackend) -> None:
        model, state = backend.model, backend.state_0
        if model.shape_count:
            model.bvh_refit_shapes(state)
        if model.particle_count:
            model.bvh_refit_particles(state)

    @staticmethod
    @contextlib.contextmanager
    def _paused_gc():
        """Keep collection-driven frees out of CUDA graph capture."""
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            yield
        finally:
            if was_enabled:
                gc.enable()
                gc.collect()


class NewtonManager(PhysicsManager):
    """Newton physics manager for Isaac Lab.

    Manager for physics lifecycle, solvers, contacts, sensors, replication, and CUDA graphs; the
    simulation context owns one instance as :attr:`~isaaclab.sim.SimulationContext.backend_manager`.
    Everything bound to the finalized model belongs to the registered :class:`NewtonBackend`. Solver-specific
    behavior lives in the stateless :class:`NewtonSolver` its configuration selects, which overrides only
    what differs.

    Lifecycle: ``initialize() -> reset() -> step()`` (repeated) ``-> close()``.
    """

    backend_name = "newton"

    def __init__(self, cfg: NewtonCfg):
        super().__init__(cfg)
        self._cfg: NewtonCfg = cfg
        self.builder: ModelBuilder | None = None
        """Shared construction builder, acquired at :meth:`initialize` and kept across hard resets."""
        self.backend: NewtonBackend | None = None
        """Data bound to the finalized model; replaced on every hard reset."""
        self.clear()
        self._scene_data_backend = NewtonSceneDataBackend(self)

    @property
    def shares_materials(self) -> bool:
        """Whether contacts read one material table shared by all environments instead of per-shape values."""
        solver_cfg = self._cfg.solver_cfg
        return solver_cfg.class_type.shares_materials(solver_cfg)

    @classmethod
    def provides_implicit_damping(cls) -> bool:
        # Newton's symplectic integrator has no implicit damping.
        return False

    @classmethod
    def setup_deformable_body(cls, prim: Any, deformable_type: str, sim_mesh_prim: Any, vis_mesh_prim: Any) -> None:
        """Apply Newton's token deformable anchor schemas and sync the visual mesh geometry."""
        sim_mesh_path = sim_mesh_prim.GetPath().pathString
        token = "PhysicsVolumeDeformableSimAPI" if deformable_type == "volume" else "PhysicsSurfaceDeformableSimAPI"
        if not sim_mesh_prim.AddAppliedSchema(token):
            raise RuntimeError(f"Failed to set {deformable_type} deformable sim API on prim '{sim_mesh_path}'.")
        # Newton renders the simulation mesh directly: overwrite the visual mesh geometry until
        # separate visual/simulation meshes are supported.
        vis_mesh = UsdGeom.Mesh(vis_mesh_prim)
        if deformable_type == "volume":
            tet_mesh = UsdGeom.TetMesh(sim_mesh_prim)
            surface_indices = tet_mesh.GetSurfaceFaceVertexIndicesAttr().Get()
            if surface_indices is None or len(surface_indices) == 0:
                raise ValueError(
                    f"Deformable body at '{prim.GetPath().pathString}' has no surface indices on its TetMesh"
                    " prim; cannot sync to visual mesh."
                )
            vis_mesh.GetPointsAttr().Set(tet_mesh.GetPointsAttr().Get())
            vis_mesh.GetFaceVertexIndicesAttr().Set(np.asarray(surface_indices).flatten())
            vis_mesh.GetFaceVertexCountsAttr().Set([3] * len(surface_indices))
        else:
            sim_mesh = UsdGeom.Mesh(sim_mesh_prim)
            vis_mesh.GetFaceVertexIndicesAttr().Set(sim_mesh.GetFaceVertexIndicesAttr().Get())
            vis_mesh.GetFaceVertexCountsAttr().Set(sim_mesh.GetFaceVertexCountsAttr().Get())
        if not prim.AddAppliedSchema("PhysicsDeformableBodyAPI"):
            raise RuntimeError(f"Failed to set deformable body API on prim '{prim.GetPath().pathString}'.")

    def initialize(self, sim_context: SimulationContext) -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        super().initialize(sim_context)
        self.builder = sim_context.get_or_create_backend(NewtonBuilderCfg(physics_cfg=self._cfg))

        # This context imports NewtonManager, so it can only be imported after this module initializes.
        from isaaclab_newton.cloner import NewtonReplicateContext  # noqa: PLC0415

        self.clone_context_type = NewtonReplicateContext

    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        A hard reset (``soft=False``) re-finalizes the Newton model, reallocating
        its device arrays. The cached collision pipeline, contacts and any
        captured CUDA graph reference the old buffers, so they are released here
        and rebuilt against the re-finalized model by :meth:`initialize_solver`.
        This avoids the illegal CUDA memory access (CUDA error 700) that would
        otherwise occur on the first step after a hard reset.

        A soft reset (``soft=True``) skips this full reinitialization and reuses
        the existing model, solver, collision pipeline and CUDA graph.

        Assets bind data once at construction. Hard resets recreate their views,
        data containers, and per-model step hooks through ``PHYSICS_READY``;
        do not register additional callbacks to rebind the discarded data.
        Reacquire ``asset.data`` and its arrays after a hard reset.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            self._close_backend()
            self.start_simulation()
            self.initialize_solver()

    def _close_backend(self, *, dispatch_stop: bool = True) -> None:
        """Stop consumers, drop their views and per-model hooks, and close the backend."""
        backend = self.backend
        if backend is not None:
            if dispatch_stop:
                self.dispatch_event(PhysicsEvent.STOP)
            # Assets recreate these views and hooks on PHYSICS_READY for the new model.
            self.views.clear()
            self._pre_actuator_callbacks.clear()
            self._post_actuator_callbacks.clear()
            self._post_step_callbacks.clear()
        # The CUDA graph points at the old model's freed buffers (CUDA 700 on the next step).
        self._graph = None
        self._graph_capture_pending = False
        self._use_newton_actuators_active = False
        if backend is not None:
            self.backend = None
            SimulationContext.instance().close_backend(backend)

    def forward(self) -> None:
        """Update articulation kinematics without stepping physics.

        Reset solver internals (:meth:`NewtonSolver.reset`) and update body poses from joint coordinates
        (:meth:`NewtonSolver.eval_fk`) for the articulations flagged in the backend's
        :attr:`~NewtonBackend.fk_mask` and the worlds flagged in :attr:`~NewtonBackend.world_mask` (see
        :meth:`invalidate_fk`). The masks are consumed (zeroed) afterwards so the next :meth:`step` does not
        redundantly re-solve them.

        Asset and scene-data reads share the same pending work.
        """
        backend = self.backend
        if backend is None or not (backend.kinematics_dirty or self.transforms_may_change_on_graph_replay):
            return
        if backend.solver is None:
            raise RuntimeError(
                "NewtonManager.initialize_solver() must run (via reset()) before forward()/step() can run forward"
                " kinematics."
            )
        solver_cfg, solver, state = self._cfg.solver_cfg, backend.solver, backend.state_0
        solver_cfg.class_type.reset(solver_cfg, solver, state, backend.world_mask)
        solver_cfg.class_type.eval_fk(solver_cfg, solver, state, backend.world_mask, backend.fk_mask)
        backend.fk_mask.zero_()
        backend.world_mask.zero_()
        backend.kinematics_dirty = False

    def video_capture_backend(self) -> str:
        """Newton GL headless perspective video capture."""
        return "newton_gl"

    def _mark_transforms_changed(self) -> None:
        """Publish authored rigid-body changes and invalidate cable geometry."""
        self._scene_data_backend.transforms_timestamp += 1
        self._scene_data_backend.geometry_timestamp += 1
        device = self._device
        if device is not None:
            device = wp.get_device(device)
            if device.is_cuda and device.stream.is_capturing:
                self.transforms_may_change_on_graph_replay = True

    def _mark_particles_dirty(self) -> None:
        """Invalidate SDP geometry after native particle writes."""
        self._scene_data_backend.geometry_timestamp += 1
        device = wp.get_device(self._device)
        if device.is_cuda and device.stream.is_capturing:
            self.transforms_may_change_on_graph_replay = True

    def step(self) -> None:
        """Step the physics simulation.

        The stepping logic follows one of two paths depending on whether
        **all** actuators are CUDA-graph-safe:

        **All-graphable path** (:meth:`_simulate_full`):

        Actuators and solver substeps are captured together in a single
        CUDA graph containing the full
        ``decimation x (actuators + solver substeps)`` loop.

        **Eager-actuator path** (fallback, some actuators not graph-safe):

        Actuators are stepped eagerly on the CPU timeline (outside the
        graph), then a graph containing only the solver substeps is
        launched via :meth:`_simulate_physics_only`.

        In both paths the sequence within one physics step is::

            zero actuated DOFs in control.joint_f
            -> actuator.step (computes effort, writes to control.joint_f)
            -> solver.step x num_substeps (integrates, reads control.joint_f)
            -> sensors.update
        """
        sim = self._sim
        if sim is None or not sim.is_playing():
            return
        backend = self.backend

        # Notify solver of model changes
        if self._model_changes:
            with wp.ScopedDevice(self._device):
                flags = 0
                ignored = self._cfg.solver_cfg.class_type.ignored_model_changes
                for change in self._model_changes:
                    if change in ignored and change not in self._warned_model_changes:
                        logger.warning(ignored[change])
                        self._warned_model_changes.add(change)
                    flags |= change
                # Refresh shared solver constants once after applying all pending property changes.
                backend.solver.notify_model_changed(flags)
                self._model_changes = set()

        # Reset-authored state and persistent solver resources must be ready before capture.
        # forward() also resets solver internals for the worlds flagged since the last boundary.
        self.forward()
        cfg = self._cfg
        device = self._device
        physics_dt = backend.dt
        use_graph = cfg is not None and cfg.use_cuda_graph and self._graph is not None and "cuda" in device  # type: ignore[union-attr]

        if self._is_all_graphable():
            # --- All actuators are graph-safe: actuators + solver in one graph ---
            if use_graph:
                wp.capture_launch(self._graph)
            else:
                with wp.ScopedDevice(device):
                    self._simulate_full()
            self._sim_time += physics_dt * self._decimation
        else:
            # --- Some actuators not graph-safe: step them eagerly, graph solver only ---
            for cb in self._pre_actuator_callbacks:
                cb()
            if backend.actuators is not None:
                backend.actuators.step(backend.state_0, backend.control, physics_dt)
            for cb in self._post_actuator_callbacks:
                cb()

            if use_graph:
                wp.capture_launch(self._graph)
            else:
                with wp.ScopedDevice(device):
                    self._simulate_physics_only()
            self._sim_time += physics_dt

        # Run the requested step eagerly before capture so lazy GPU allocations happen outside recording.
        if self._graph_capture_pending and cfg is not None and cfg.use_cuda_graph:
            simulate = self._simulate_full if self._is_all_graphable() else self._simulate_physics_only
            with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
                self._graph = self._capture_graph(simulate)
            self._graph_capture_pending = False

        self._mark_transforms_changed()

        captured = self._graph is not None
        cfg.solver_cfg.class_type.check_step(cfg.solver_cfg, backend.solver, captured, cfg.debug_mode)

    def close(self) -> None:
        """Clean up Newton physics resources."""
        super().close()
        self.clear()
        self.builder = None

    def get_scene_data_backend(self) -> SceneDataBackend | None:
        """Return the SceneDataBackend for the SceneDataProvider."""
        return self._scene_data_backend

    def register_callback(
        self,
        callback: Callable,
        event: PhysicsEvent,
        order: int = 0,
        name: str | None = None,
        wrap_weak_ref: bool = True,
    ) -> CallbackHandle:
        """Register a callback. Passes event to parent class."""
        return super().register_callback(callback, event, order, name, wrap_weak_ref)

    def get_physics_sim_view(self) -> list:
        """Return the registered articulation views."""
        return list(self.views.values())

    def is_fabric_enabled(self) -> bool:
        """Check if fabric interface is enabled (not applicable for Newton)."""
        return False

    def clear(self):
        """Clear all Newton-specific state (callbacks cleared by super().close())."""
        self._close_backend(dispatch_stop=False)
        self._warned_model_changes = set()
        self._deterministic_mode = wp.DeterministicMode.NOT_GUARANTEED
        self._solver_init_callbacks: list[Callable[[], None]] = []
        # In-graph step hooks, in registration order: before and after the actuator step, before every solver
        # substep, and after the last substep.
        self._pre_actuator_callbacks: list[Callable[[], None]] = []
        self._post_actuator_callbacks: list[Callable[[], None]] = []
        self._state_force_callbacks: list[Callable[[State], None]] = []
        self._post_step_callbacks: list[Callable[[], None]] = []
        # Substeps clear the input state's forces, so each one re-applies this copy of the staged body forces.
        self._staged_body_f: wp.array | None = None
        # Set by an articulation that took the ``use_newton_actuators=True``
        # branch in ``_process_actuators_cfg``.  Together with the adapter
        # check, this gates whether the decimation loop can be captured into
        # a CUDA graph (see :meth:`_is_all_graphable`).
        self._use_newton_actuators_active = False
        self._decimation = 1
        self._graph = None
        self._graph_capture_pending = False
        self.transforms_may_change_on_graph_replay = False
        self._model_changes: set[int] = set()
        # Site requests and replication outputs survive hard resets.
        # (body_pattern, per_world, xform floats) -> (label, transform)
        self._site_requests: dict[tuple[str | None, bool, tuple[float, ...]], tuple[str, wp.transform]] = {}
        self._site_index_map: dict[str, SiteEntry] = {}
        self._clone: NewtonCloneRecord | None = None
        self.views.clear()

    def register_site(self, body_pattern: str | None, xform: wp.transform, *, per_world: bool = False) -> str:
        """Request a site for injection into prototypes before replication.

        Sensors call this during ``__init__``. Identical ``(body_pattern, per_world, transform)`` requests share a site,
        and requests persist until :meth:`close`, so sites survive hard resets without being requested again.
        The pattern matches prototype-local body labels (e.g. ``"Robot/finger.*"``) during replication, and wildcard
        patterns create one site per matched body.

        Args:
            body_pattern: Regex matched against body labels, or ``None`` for a global site.
            xform: Site transform relative to the body.
            per_world: Create one bodyless site in each cloned world's frame. Requires ``body_pattern=None``.

        Returns:
            The assigned site label.
        """
        if per_world and body_pattern is not None:
            raise ValueError("per_world site registration requires body_pattern=None.")
        requests = self._site_requests
        key = (body_pattern, per_world, tuple(xform))
        if key not in requests:
            requests[key] = (f"ft_{len(requests)}", xform)
        return requests[key][0]

    def inject_sites(
        self, main_builder: ModelBuilder, source_builders: dict[str, ModelBuilder]
    ) -> tuple[dict[str, int], dict[int, dict[str, list[int]]], dict[str, wp.transform]]:
        """Add unresolved sites to source builders, or to the main builder for global and shared-asset sites.

        Args:
            main_builder: Top-level builder that receives global sites.
            source_builders: Source builders by source path.

        Returns:
            ``(global_site_indices, source_site_indices, env_root_sites)``: global site shape indices by label,
            builder-local site indices by ``id(builder)`` and label, and world-root site transforms by label.
        """
        global_sites: dict[str, int] = {}
        source_sites: dict[int, dict[str, list[int]]] = {}
        root_sites: dict[str, wp.transform] = {}
        for (body_pattern, per_world, _), (label, xform) in self._site_requests.items():
            if label in self._site_index_map:
                continue
            if per_world:
                root_sites[label] = xform
                continue
            if body_pattern is None:
                global_sites[label] = main_builder.add_site(body=-1, xform=xform, label=label)
                continue
            matched_any = False
            for builder in (*source_builders.values(), main_builder):
                if builder is main_builder and matched_any:
                    break
                indices, names = resolve_matching_names(body_pattern, builder.body_label, raise_when_no_match=False)
                if not indices:
                    continue
                matched_any = True
                sites = [
                    builder.add_site(body=body, xform=xform, label=f"{name}/{label}")
                    for body, name in zip(indices, names)
                ]
                source_sites.setdefault(id(builder), {})[label] = sites
            if not matched_any:
                raise ValueError(f"Site '{label}' with body_pattern '{body_pattern}' matched no builder bodies.")
        return global_sites, source_sites, root_sites

    def record_clone(self, record: NewtonCloneRecord, site_index_map: dict[str, SiteEntry]) -> None:
        """Record native replication outputs for model finalization and consumers.

        Args:
            record: Replication outputs.
            site_index_map: Sites resolved during replication, by label.
        """
        self._clone = record
        self._site_index_map.update(site_index_map)

    def get_site_index_map(self) -> dict[str, SiteEntry]:
        """Return resolved sites by label."""
        return self._site_index_map

    def get_world_xforms(self) -> list[wp.transform] | None:
        """Return the root transform of each cloned world, or ``None`` without replication."""
        clone = self._clone
        return None if clone is None else clone.world_xforms

    def get_clone_source_builders(self) -> dict[str, ModelBuilder]:
        """Return per-source builders retained from replication, keyed by clone-plan source path."""
        clone = self._clone
        return {} if clone is None else clone.source_builders

    def request_extended_state_attribute(self, attr: str) -> None:
        """Request an extended state attribute (e.g. ``"body_qdd"``) from the builder before finalization.

        Args:
            attr: State attribute name (must be in ``State.EXTENDED_ATTRIBUTES``).
        """
        self.builder.request_state_attributes(attr)

    def request_extended_contact_attribute(self, attr: str) -> None:
        """Request an extended contact attribute (e.g. ``"force"``) from the builder before finalization.

        Args:
            attr: Contact attribute name.
        """
        self.builder.request_contact_attributes(attr)

    def add_model_change(self, change: ModelFlags) -> None:
        """Register a model change to notify the solver."""
        self._model_changes.add(change)

    def invalidate_fk(
        self,
        env_mask: wp.array | None = None,
        env_ids: wp.array | None = None,
        articulation_ids: wp.array | None = None,
    ) -> None:
        """Mark environments as needing FK recomputation and solver reset.

        Called by asset write methods that modify joint coordinates or root
        transforms. The masks are consumed by the next forward, raw-state,
        rendering, or physics-step boundary.

        Args:
            env_mask: Boolean mask of dirtied environments. Shape ``(num_envs,)``.
                Used by ``_mask`` write methods.
            env_ids: Integer indices of dirtied environments.
                Used by ``_index`` write methods.
            articulation_ids: Mapping from ``(world, arti)`` to model articulation
                index. Shape ``(world_count, count_per_world)``. Obtained from
                ``ArticulationView.articulation_ids``.
        """
        self._mark_transforms_changed()

        backend = self.backend
        if backend is None:
            return
        backend.kinematics_dirty = True

        if articulation_ids is not None and env_mask is not None:
            wp.launch(
                _or_reset_masks_from_mask,
                dim=articulation_ids.shape,
                inputs=[env_mask, articulation_ids],
                outputs=[backend.world_mask, backend.fk_mask],
                device=self._device,
            )
        elif articulation_ids is not None and env_ids is not None:
            wp.launch(
                _scatter_reset_masks_from_ids_kernel(env_ids),
                dim=(env_ids.shape[0], articulation_ids.shape[1]),
                inputs=[env_ids, articulation_ids],
                outputs=[backend.world_mask, backend.fk_mask],
                device=self._device,
            )
        else:
            # Fallback: no topology info — mark everything dirty
            backend.world_mask[: backend.model.world_count].fill_(True)
            backend.fk_mask.fill_(True)

    def invalidate_body_state(
        self, env_ids: wp.array(dtype=wp.int32) | None = None, env_mask: wp.array(dtype=wp.bool) | None = None
    ) -> None:
        """Mark selected maximal-coordinate body state as changed without requesting FK.

        Args:
            env_ids: Integer indices of dirtied environments. Used by index write methods.
            env_mask: Boolean mask of dirtied environments. Used by mask write methods.
        """
        self._mark_transforms_changed()
        backend = self.backend
        if backend is None:
            return
        backend.kinematics_dirty = True
        if env_mask is not None:
            wp.launch(
                _or_world_reset_mask_from_mask,
                dim=env_mask.shape[0],
                inputs=[env_mask],
                outputs=[backend.world_mask],
                device=self._device,
            )
        elif env_ids is not None:
            wp.launch(
                _scatter_world_reset_mask_from_ids,
                dim=env_ids.shape[0],
                inputs=[env_ids],
                outputs=[backend.world_mask],
                device=self._device,
            )
        else:
            backend.world_mask[: backend.model.world_count].fill_(True)

    def _drain_stale_cuda_error(self) -> None:
        """Clear a stale CUDA error latched on the device before (re)initialization.

        Warp 1.15 leaves the per-thread CUDA error uncleared when
        ``wp_free_device_async`` fails to add a graph memory free node while a
        capture is still registered (its "capture ended" sibling branch clears
        the identical error as benign), and the next Warp array copy then
        surfaces that stale error as its own failure, aborting simulation
        initialization. Draining here keeps a prior simulation lifecycle's
        latched error from poisoning this one. Remove once the upstream Warp
        fix lands.
        """
        device = wp.get_device(str(self._device))
        if not device.is_cuda:
            return
        # Private Warp API: the drain primitives are not exposed publicly; getting the
        # device above guarantees the runtime is initialized. Guard the whole private
        # interaction so a future Warp internals reshuffle degrades to a skipped drain
        # rather than hard-failing simulation start.
        try:
            from warp._src.context import runtime as _wp_runtime

            core = _wp_runtime.core
            # wp_cuda_context_check drains via cudaGetLastError() but returns the
            # post-sync error state (0 once a non-sticky error was drained), and its
            # internal check_cuda() prints the drained error verbatim to stderr;
            # suppress the print and diff Warp's error buffer to report the drain.
            before = core.wp_get_error_string()
            was_enabled = bool(core.wp_is_error_output_enabled())
            core.wp_set_error_output_enabled(0)
            try:
                persistent = core.wp_cuda_context_check(device.context)
            finally:
                core.wp_set_error_output_enabled(1 if was_enabled else 0)
            after = core.wp_get_error_string()
        except (ImportError, AttributeError) as exc:
            logger.warning("Skipping stale CUDA error drain; Warp internals unavailable: %s", exc)
            return

        if persistent != 0:
            logger.error(
                "CUDA error %d persists after drain; the device context is likely unrecoverable: %s",
                persistent,
                after.decode(errors="replace"),
            )
        elif after != before:
            logger.warning(
                "Drained stale CUDA error latched by a prior lifecycle: %s (last Warp error recorded before drain: %s)",
                after.decode(errors="replace"),
                before.decode(errors="replace") or "<none>",
            )

    def start_simulation(self) -> None:
        """Start simulation by finalizing model and initializing state.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.
        """
        sim = self._sim
        builder = self.builder
        self._drain_stale_cuda_error()

        logger.info("Dispatching MODEL_INIT callbacks")
        self.dispatch_event(PhysicsEvent.MODEL_INIT)

        # The builder survives hard resets, so sites already added to it are skipped.
        global_sites, body_sites, root_sites = self.inject_sites(builder, {})
        site_map = self._site_index_map
        site_map.update((label, (index, None)) for label, index in global_sites.items())
        site_map.update((label, (None, [indices])) for label, indices in body_sites.get(id(builder), {}).items())
        for label, xform in root_sites.items():
            site_map[label] = (None, [[builder.add_site(body=-1, xform=xform, label=label)]])

        device = self._device
        logger.info(f"Finalizing model on device: {device}")
        builder.up_axis = Axis.Z
        if (solver_cfg := self._cfg.solver_cfg) is not None:
            solver_cfg.class_type.prepare_builder(solver_cfg, builder)
        with Timer(
            name="newton_finalize_builder",
            msg="Finalize builder took:",
            activity="Finalizing physics model",
            synchronize="both",
            device=device,
        ):
            cfg = NewtonBackendCfg(physics_cfg=self._cfg, device=sim.device)
            self.backend = sim.get_or_create_backend(cfg, dt=self.get_physics_dt())
            self.backend.model.set_gravity(sim.cfg.gravity)

        clone = self._clone
        if clone is not None:
            self.backend.particle_ranges = clone.particle_ranges
        self._scene_data_backend.initialize_geometry(
            [] if clone is None else clone.geometry_batches, {} if clone is None else clone.cable_bindings
        )
        logger.info("Dispatching PHYSICS_READY callbacks")
        self.dispatch_event(PhysicsEvent.PHYSICS_READY)

    def instantiate_builder_from_stage(self):
        """Import the explicitly declared clone plan into the Newton builder."""
        cloner.replicate(self._sim.get_clone_plan())

    def _initialize_contacts(self) -> None:
        """Allocate contacts from Newton's collision pipeline, or from the solver when it detects contacts itself."""
        backend = self.backend
        model, solver, solver_cfg = backend.model, backend.solver, self._cfg.solver_cfg
        solver_cls = solver_cfg.class_type
        if not solver_cls.uses_collision_pipeline(solver_cfg, solver):
            if self._cfg.collision_cfg is not None:
                raise ValueError(
                    f"NewtonCfg.collision_cfg cannot be set while {type(solver).__name__} does not use Newton's"
                    " collision pipeline. Remove collision_cfg, or configure the solver to use the pipeline (e.g."
                    " use_mujoco_contacts=False)."
                )
            backend.contacts = solver_cls.create_contacts(solver_cfg, solver)
        else:
            collision_cfg = self._cfg.collision_cfg
            args = collision_cfg.to_pipeline_args() if collision_cfg is not None else {"broad_phase": "explicit"}
            args["deterministic"] = self._deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED
            required = getattr(solver, "get_max_contact_count", lambda: 0)()
            if backend.collision_pipeline is None:
                backend.collision_pipeline = solver_cls.create_collision_pipeline(solver_cfg, solver, **args)
            backend.contacts = backend.collision_pipeline.contacts()
            # MuJoCo Warp can require more contacts than the pipeline estimates.
            if required > backend.contacts.rigid_contact_max:
                if args["deterministic"]:
                    # The deterministic sort buffer is sized at construction, so rebuild the pipeline to match.
                    args["rigid_contact_max"] = required
                    backend.collision_pipeline = solver_cls.create_collision_pipeline(solver_cfg, solver, **args)
                    backend.contacts = backend.collision_pipeline.contacts()
                else:
                    backend.contacts = Contacts(
                        rigid_contact_max=required,
                        soft_contact_max=0,
                        device=self._device,
                        requested_attributes=model.get_requested_contact_attributes(),
                    )
        if backend.contacts is not None:
            solver_cls.prepare_contacts(solver_cfg, solver, backend.contacts, backend.collision_pipeline)

    @staticmethod
    def _resolve_deterministic_mode(cfg: NewtonCfg) -> wp.DeterministicMode:
        """Translate the determinism request of a Newton configuration into a Warp mode.

        An explicit :attr:`NewtonCfg.deterministic_mode` wins over the generic
        :attr:`~isaaclab.physics.PhysicsCfg.deterministic` request; the selected solver then decides which mode
        applies (:meth:`NewtonSolver.applied_deterministic_mode`).

        Args:
            cfg: Newton physics configuration.

        Returns:
            The mode for the solver and collision pipeline.
        """
        if cfg.deterministic_mode != "not_guaranteed":
            mode = {"run_to_run": wp.DeterministicMode.RUN_TO_RUN, "gpu_to_gpu": wp.DeterministicMode.GPU_TO_GPU}[
                cfg.deterministic_mode
            ]
        else:
            mode = wp.DeterministicMode.RUN_TO_RUN if cfg.deterministic else wp.DeterministicMode.NOT_GUARANTEED
        return cfg.solver_cfg.class_type.applied_deterministic_mode(cfg.solver_cfg, mode)

    # ----- Lifecycle orchestration ----------------------------------------

    def initialize_solver(self) -> None:
        """Initialize the solver and collision pipeline.

        Construct the solver the solver configuration selects and its contacts, establish the initial body state, and
        schedule graph capture for the first step after the environment has authored its initial state.
        Initialization and capture do not advance physics.
        """
        cfg = self._cfg
        if cfg is None:
            return
        backend = self.backend

        with Timer(
            name="newton_initialize_solver",
            msg="Initialize solver took:",
            activity="Initializing solver",
            synchronize="both",
            device=self.get_device(),
        ):
            model = backend.model
            mode = self._deterministic_mode = self._resolve_deterministic_mode(cfg)
            solver_cfg = cfg.solver_cfg
            solver_cls = solver_cfg.class_type
            if mode != wp.DeterministicMode.NOT_GUARANTEED and not solver_cls.deterministic:
                raise ValueError(
                    f"Newton deterministic mode {mode.name} is not supported by {type(solver_cfg).__name__}."
                    " Use MJWarp on the GPU, XPBD, or Featherstone, or disable deterministic mode."
                )
            plan = self._sim.get_clone_plan() if self._sim is not None else None
            env_template = plan.env_template if plan is not None else None
            solver = backend.solver = solver_cls.build(solver_cfg, model, mode, env_template)
            if not solver_cls.single_state:
                solver_cls.init_state(solver_cfg, solver, backend.state_1)
            self._initialize_contacts()
            state_0 = backend.state_0
            self._staged_body_f = wp.zeros_like(state_0.body_f) if state_0.body_count else None

        pending_solver_init, self._solver_init_callbacks = self._solver_init_callbacks, []
        for cb in pending_solver_init:
            cb()

        # Picking callbacks must be registered after the solver exists, but before CUDA graph capture.
        sim = self._sim
        if solver_cls.applies_body_forces(solver_cfg, solver) and sim is not None:
            sim._prepare_newton_visualizer_for_capture()

        # Establish the initial kinematically-consistent body state, now that the solver exists.
        solver_cls.eval_fk(solver_cfg, solver, backend.state_0, None, None)
        self._mark_transforms_changed()

        self._invalidate_graph()

    def _invalidate_graph(self) -> None:
        """Schedule capture after reset and decimation setup, without advancing physics."""
        self._graph = None
        self._graph_capture_pending = False
        cfg = self._cfg
        device = self._device
        backend = self.backend
        if device is None or not cfg.use_cuda_graph or "cuda" not in device or backend.solver is None:
            return
        if not cfg.solver_cfg.class_type.supports_capture(cfg.solver_cfg, backend.solver):
            logger.warning(
                "%s cannot be captured in its current configuration; using eager execution.",
                type(backend.solver).__name__,
            )
            return
        self._graph_capture_pending = True

    def _capture_graph(self, capture_target: Callable[[], None]) -> wp.Graph:
        """Record physics or scene queries without an eager warmup or graph replay."""
        sim = self._sim
        relaxed = has_kit() and (sim.has_gui or sim.has_offscreen_render)
        return NewtonQueries.capture_graph(self._device, capture_target, relaxed=relaxed)

    # ------------------------------------------------------------------
    # Building blocks — used by _simulate_full / _simulate_physics_only
    # ------------------------------------------------------------------

    def _stage_body_forces(self) -> None:
        """Copy the body forces written to ``state_0`` before the step, so that every solver substep applies them."""
        if self._staged_body_f is not None:
            wp.copy(self._staged_body_f, self.backend.state_0.body_f)

    def _apply_staged_body_forces(self, state: State) -> None:
        """Write the staged body forces into the input state of a solver substep."""
        if self._staged_body_f is not None:
            wp.copy(state.body_f, self._staged_body_f)

    def _run_solver_substeps(self, contacts) -> None:
        """Run ``num_substeps`` solver iterations, handling double-buffered state swap.

        The staged body forces overwrite the input state's ``body_f``, so they are written before the state-force
        callbacks, which add on top of them.
        """
        backend = self.backend
        solver_cfg, solver = self._cfg.solver_cfg, backend.solver
        step = solver_cfg.class_type.step
        num_substeps = backend.num_substeps
        collide_every = self._cfg.collision_decimation
        # Last substep is skipped: its contact set would only feed the next tick's
        # top-of-loop collide(), not this one.
        collide_mid_loop = collide_every > 0 and backend.collision_pipeline is not None and contacts is not None

        if solver_cfg.class_type.single_state:
            for i in range(num_substeps):
                self._apply_staged_body_forces(backend.state_0)
                for callback in self._state_force_callbacks:
                    callback(backend.state_0)
                step(solver_cfg, solver, backend.state_0, backend.state_0, backend.control, contacts, backend.solver_dt)
                backend.state_0.clear_forces()
                if collide_mid_loop and (i + 1) % collide_every == 0 and i + 1 < num_substeps:
                    backend.collision_pipeline.collide(backend.state_0, contacts)
        else:
            need_copy_on_last = num_substeps % 2 == 1
            for i in range(num_substeps):
                self._apply_staged_body_forces(backend.state_0)
                for callback in self._state_force_callbacks:
                    callback(backend.state_0)
                step(solver_cfg, solver, backend.state_0, backend.state_1, backend.control, contacts, backend.solver_dt)
                if need_copy_on_last and i == num_substeps - 1:
                    backend.state_0.assign(backend.state_1)
                else:
                    backend.state_0, backend.state_1 = backend.state_1, backend.state_0
                backend.state_0.clear_forces()
                if collide_mid_loop and (i + 1) % collide_every == 0 and i + 1 < num_substeps:
                    backend.collision_pipeline.collide(backend.state_0, contacts)
        solver_cfg.class_type.post_step(solver_cfg, solver, backend.state_0)

    def _update_sensors(self, contacts) -> None:
        """Push latest state to registered IMU and contact sensors."""
        backend = self.backend
        for sensor in backend.imu_sensors:
            sensor.update(backend.state_0)
        if backend.contact_sensors:
            eval_contacts = contacts if contacts is not None else backend.contacts
            backend.solver.update_contacts(eval_contacts, backend.state_0)
            for sensor in backend.contact_sensors.values():
                sensor.update(backend.state_0, eval_contacts)

    # ------------------------------------------------------------------
    # Composite stepping routines
    # ------------------------------------------------------------------

    def _simulate_full(self) -> None:
        """Run ``decimation x (actuators + solver substeps)``, then sensors.

        Works for any decimation count (including 1).  All actuators must be
        graph-safe so the entire loop can be captured as a single CUDA graph.
        """
        backend = self.backend
        pipeline = backend.collision_pipeline
        contacts = backend.contacts if pipeline is not None else None

        solver_cfg = self._cfg.solver_cfg
        self._stage_body_forces()
        for i in range(self._decimation):
            solver_cfg.class_type.pre_step(solver_cfg, backend.solver, backend.state_0)
            if pipeline is not None:
                pipeline.collide(backend.state_0, contacts)

            for cb in self._pre_actuator_callbacks:
                cb()
            if backend.actuators is not None:
                backend.actuators.step(
                    backend.state_0,
                    backend.control,
                    backend.dt,
                    swap_state=self._decimation % 2 == 0 or i < self._decimation - 1,
                )
            for cb in self._post_actuator_callbacks:
                cb()

            self._run_solver_substeps(contacts)

        for cb in self._post_step_callbacks:
            cb()
        self._update_sensors(contacts)

    def _simulate_physics_only(self) -> None:
        """Collision + solver substeps + sensors (no actuators, no USD sync).

        Used when actuators are stepped eagerly outside the graph, or when
        there are no actuators at all.
        """
        backend = self.backend
        pipeline = backend.collision_pipeline
        contacts = backend.contacts if pipeline is not None else None
        solver_cfg = self._cfg.solver_cfg
        self._stage_body_forces()
        solver_cfg.class_type.pre_step(solver_cfg, backend.solver, backend.state_0)
        if pipeline is not None:
            pipeline.collide(backend.state_0, contacts)

        self._run_solver_substeps(contacts)
        for cb in self._post_step_callbacks:
            cb()
        self._update_sensors(contacts)

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Compile material-to-shape addresses for the active Newton model."""
        return self.backend.create_visual_material_writer(batches)

    def create_visual_shape_color_writer(
        self, asset: BaseArticulation, body_names: tuple[str, ...]
    ) -> VisualShapeColorWriter:
        """Compile selected articulation-body shape addresses for the active Newton model."""
        model = None if self.backend is None else self.backend.model
        view = asset.root_view
        if not isinstance(view, ArticulationView):
            root_expr = asset.cfg.prim_path
            root_expr += (
                "(?:/.*)?" if asset.cfg.articulation_root_prim_path is None else asset.cfg.articulation_root_prim_path
            )
            prim_paths = [path for path in model.articulation_label if re.fullmatch(root_expr, path)]
            view = ArticulationView(model, prim_paths, verbose=False)
        return VisualShapeColorWriter(model, view, body_names)

    def get_scene_data_provider(self) -> SceneDataProvider:
        """Return the active scene data provider."""
        return SimulationContext.instance().get_scene_data_provider()

    def _is_all_graphable(self) -> bool:
        """``True`` when the decimation loop can be captured into a CUDA graph.

        Requires:
          1. An articulation took the ``use_newton_actuators=True`` branch
             (signalled via :meth:`activate_newton_actuator_path`).
          2. Either no actuator adapter was needed (all-implicit) or every
             actuator in the adapter is CUDA-graph-safe.
        """
        if not self._use_newton_actuators_active:
            return False
        actuators = self.backend.actuators
        return actuators is None or actuators.is_all_graphable

    def activate_newton_actuator_path(self) -> None:
        """Opt an articulation into the Newton actuator fast path.

        Idempotent — called by every Newton-fast-path articulation's
        ``_process_actuators_cfg``:

        1. Sets :attr:`_use_newton_actuators_active`, which
           :meth:`_is_all_graphable` checks (adapter presence alone
           cannot distinguish the fast path from the standard Lab path).
        2. On first call, builds the single sim-level
           :class:`NewtonActuatorAdapter` over the full flat DOF layout as
           :attr:`NewtonBackend.actuators`; later calls reuse it.
        """
        self._use_newton_actuators_active = True

        backend = self.backend
        if backend is None or backend.actuators is not None or not backend.model.actuators:
            return
        from isaaclab.actuators.newton import NewtonActuatorAdapter  # noqa: PLC0415

        num_envs = backend.model.world_count
        backend.actuators = NewtonActuatorAdapter(
            actuators=list(backend.model.actuators),
            num_envs=num_envs,
            num_joints=backend.model.joint_dof_count // num_envs,
            dof_offset=0,
            device=self._device,
        )
        backend.actuators.finalize(backend.control)

    def register_solver_init_callback(self, callback: Callable[[], None]) -> None:
        """Append a one-shot hook invoked once the solver exists, before any CUDA graph capture.

        Assets initialize before the solver is created, so bindings that need it register here.
        """
        self._solver_init_callbacks.append(callback)

    def register_pre_actuator_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook invoked before the actuator step.

        Each callback runs inside the captured CUDA graph once per decimation iteration when
        :meth:`_is_all_graphable` is ``True``, and eagerly once per step otherwise.
        Hard resets discard these hooks; register them again when the new model is ready.
        """
        self._pre_actuator_callbacks.append(callback)

    def register_post_actuator_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook to the list invoked after the actuator step on every iteration.

        Each callback runs inside the captured CUDA graph (when
        :meth:`_is_all_graphable` is ``True``) right after
        :meth:`NewtonActuatorAdapter.step` and before the solver substeps,
        so kernel writes to ``state``/``control`` are visible to the
        integrator on the same iteration. Multiple articulations register
        their own implicit-DOF telemetry / FF-routing kernels here; all
        registered callbacks fire in registration order each step.
        Hard resets discard these hooks; register them again when the new model is ready.
        """
        self._post_actuator_callbacks.append(callback)

    def register_state_force_callback(self, callback: Callable[[State], None]) -> None:
        """Register a graph-safe callback that applies forces before every solver substep.

        Callbacks must be registered before solver initialization so they are
        included in CUDA graph capture.

        Args:
            callback: Function that adds forces [N, N·m] to the provided state.
        """
        if callback in self._state_force_callbacks:
            return
        self._state_force_callbacks.append(callback)

    def register_post_step_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook to the list invoked after the last solver substep on every step.

        Each callback runs inside the stepped (and, when
        :meth:`_is_all_graphable` is ``True``, captured) region right after the
        final solver substep of the decimation loop and before
        :meth:`_update_sensors`, so the launches it issues are recorded into
        every captured CUDA graph and replayed on each tick. The hook fires
        exactly once per :meth:`step` call, reflecting the state after all
        decimation iterations (and their solver substeps) have completed -- not
        once per substep and not once per decimation iteration. Callbacks must be
        graph-safe (fixed shapes, no host branching on device data) and must be
        registered before capture. Articulations with non-identity ordering
        register their backend-to-user state republish here; all registered
        callbacks fire in registration order each step.
        Hard resets discard these hooks; register them again when the new model is ready.
        """
        self._post_step_callbacks.append(callback)

    def unregister_post_step_callback(self, callback: Callable[[], None]) -> None:
        """Remove a previously registered post-step callback.

        Symmetric to :meth:`register_post_step_callback`, this lets an
        articulation deregister its republish hook when its callbacks are
        cleared so the bound method does not linger after the articulation is
        gone. Removing a callback that was never registered (or was already
        removed) is a safe no-op, matching the tolerant deregistration of other
        handles.
        """
        with contextlib.suppress(ValueError):
            self._post_step_callbacks.remove(callback)

    def set_decimation(self, decimation: int) -> None:
        """Set the decimation count and re-capture the CUDA graph.

        When all actuators are graphable the entire decimation loop
        (actuators + solver substeps, repeated *decimation* times)
        is captured as a single CUDA graph.

        Invalidate the existing graph when the loop changes. Its replacement is captured
        immediately after the next requested step runs eagerly.
        """
        self._decimation = max(1, decimation)
        if self._is_all_graphable():
            self._invalidate_graph()

    def handles_decimation(self) -> bool:
        """``True`` when :meth:`step` executes the full decimation loop internally.

        This is the case when all Newton actuators are CUDA-graph-safe.
        The full decimation loop (including the trivial ``decimation=1`` case)
        is folded into a single :meth:`step` call.
        """
        return self._is_all_graphable()

    def add_contact_sensor(
        self,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        verbose: bool = False,
    ) -> NewtonContactSensor:
        """Add a contact sensor for reporting contacts between bodies/shapes; identical requests share one sensor.

        Compiles the Isaac Lab regular expressions and delegates to
        :class:`newton.sensors.SensorContact`, which full-matches compiled patterns
        against model labels.

        Args:
            body_names_expr: Expression for body names to sense.
            shape_names_expr: Expression for shape names to sense.
            contact_partners_body_expr: Expression for contact partner body names.
            contact_partners_shape_expr: Expression for contact partner shape names.
            verbose: Print verbose information.

        Returns:
            The Newton contact sensor, updated at the end of every step.
        """
        solver_cfg = self._cfg.solver_cfg
        if solver_cfg is not None and not solver_cfg.class_type.contact_sensors:
            raise NotImplementedError(
                f"Newton contact sensors are not yet supported by {type(solver_cfg).__name__}:"
                " contact forces live in per-entry buffers."
            )
        if body_names_expr is None and shape_names_expr is None:
            raise ValueError("At least one of body_names_expr or shape_names_expr must be provided")
        if body_names_expr is not None and shape_names_expr is not None:
            raise ValueError("Only one of body_names_expr or shape_names_expr must be provided")
        if contact_partners_body_expr is not None and contact_partners_shape_expr is not None:
            raise ValueError("Only one of contact_partners_body_expr or contact_partners_shape_expr must be provided")

        sensor_target = body_names_expr or shape_names_expr
        partner_filter = contact_partners_body_expr or contact_partners_shape_expr or "all bodies/shapes"
        logger.info(f"Adding contact sensor for {sensor_target} with filter {partner_filter}")

        def _hashable_key(x):
            return tuple(x) if isinstance(x, list) else x

        sensor_key = (
            _hashable_key(body_names_expr),
            _hashable_key(shape_names_expr),
            _hashable_key(contact_partners_body_expr),
            _hashable_key(contact_partners_shape_expr),
        )
        backend = self.backend
        sensor = backend.contact_sensors.get(sensor_key)
        if sensor is not None:
            return sensor

        with Timer(
            name="newton_contact_sensor",
            msg="Contact sensor construction took:",
            synchronize="both",
            device=self.get_device(),
        ):
            sensor = NewtonContactSensor(
                backend.model,
                sensing_bodies=_compile_label_pattern(body_names_expr),
                sensing_shapes=_compile_label_pattern(shape_names_expr),
                counterpart_bodies=_compile_label_pattern(contact_partners_body_expr),
                counterpart_shapes=_compile_label_pattern(contact_partners_shape_expr),
                measure_total=True,
                verbose=verbose,
            )

        backend.contact_sensors[sensor_key] = sensor

        if backend.solver is not None and backend.contacts is not None and backend.contacts.force is None:
            self._initialize_contacts()

        return sensor

    def add_imu_sensor(self, sites: list[int]) -> NewtonSensorIMU:
        """Add an IMU sensor for measuring acceleration and angular velocity at sites.

        The ``body_qdd`` state attribute must be requested before finalization.

        Args:
            sites: Ordered list of site indices (one per environment).

        Returns:
            The Newton sensor, updated at the end of every step.
        """
        if self.backend is None:
            raise RuntimeError("add_imu_sensor called before model finalization (start_simulation).")
        sensor = NewtonSensorIMU(
            self.backend.model,
            sites=sites,
            request_state_attributes=False,  # Already requested via NewtonManager
        )
        self.backend.imu_sensors.append(sensor)
        logger.info(f"Added IMU sensor (sites={len(sites)})")
        return sensor
