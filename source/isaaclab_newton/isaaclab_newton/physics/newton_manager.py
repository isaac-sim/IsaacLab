# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import contextlib
import ctypes
import gc
import inspect
import logging
import re
from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp

# Load CUDA runtime for relaxed-mode graph capture (RTX-compatible).
# cudaStreamCaptureModeRelaxed (2) allows the RTX compositor's background
# CUDA stream to keep running during capture without invalidating it.
try:
    _cudart = ctypes.CDLL("libcudart.so.12")
except OSError:
    try:
        _cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        _cudart = None


@contextlib.contextmanager
def _paused_gc():
    """Pause Python garbage collection for the duration of a CUDA graph capture.

    A garbage-collection pass inside a capture window can drop the last
    reference to an array allocated earlier in the capture. While the capture
    is paused for a ``wp.capture_while``/``wp.capture_if`` conditional body,
    Warp then inserts the memory free node into the body graph with dependency
    nodes from the parent graph, which fails and latches a sticky CUDA error
    that poisons a later, unrelated copy. Reference-count-driven frees are
    deterministic solver behavior and remain allowed; only the collector is
    deferred, and a collection runs immediately after the capture window,
    where freeing graph-scoped allocations is handled correctly.

    Only generation 0 is collected afterwards: objects allocated inside the
    window are still in gen 0, so this reclaims the graph-scoped garbage the
    pause deferred, without the full-heap walk a ``gc.collect()`` would do
    (hundreds of milliseconds once the replicated model exists). Cycles that
    were already in an older generation before the window and became
    unreachable during it -- a previous ``wp.Graph``/``State`` released on a
    hard reset, for instance -- are left to the periodic collector.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
            gc.collect(0)


from newton import (
    Axis,
    CollisionPipeline,
    Contacts,
    Control,
    Heightfield,
    Model,
    ModelBuilder,
    ModelFlags,
    ShapeFlags,
    State,
    eval_fk,
)
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.sensors import SensorContact as NewtonContactSensor
from newton.sensors import SensorFrameTransform
from newton.sensors import SensorIMU as NewtonSensorIMU
from newton.solvers import SolverBase, SolverKamino

from pxr import Usd, UsdGeom

from isaaclab.physics import CallbackHandle, PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat, SceneDataProvider
from isaaclab.scene_data.deformable_vis_remap import (
    VolumeVisRemap,
    launch_batch_particle_slice_copy,
    launch_batch_volume_vis_remap,
)
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import checked_apply
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import has_kit
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

from isaaclab_newton.cloner.newton_clone_utils import (
    _restore_visible_colliders_without_visual_shapes,
    replicate_builder_mapping,
)
from isaaclab_newton.physics.visualization_builder import build_visualization_builder_from_stage_envs
from isaaclab_newton.physics.visualization_deformables import populate_shadow_deformable_registry

from .newton_manager_cfg import NewtonCfg, NewtonShapeCfg

if TYPE_CHECKING:
    from isaaclab_newton.actuators import NewtonActuatorAdapter

    from .newton_collision_cfg import NewtonCollisionPipelineCfg

logger = logging.getLogger(__name__)

# Tagged union for entries in _cl_site_index_map.
# _GlobalSite: (global_shape_idx, None)           — body_pattern was None
# _LocalSite:  (None, [[env0_idx, ...], ...])     — per-world site indices


@wp.kernel(enable_backward=False)
def _set_fabric_transforms(
    fabric_transforms: wp.fabricarray(dtype=wp.mat44d),
    newton_indices: wp.fabricarray(dtype=wp.uint32),
    newton_body_q: wp.array(ndim=1, dtype=wp.transformf),
):
    """Write Newton body transforms to Fabric world matrices.

    For each Fabric prim at thread ``i``, reads the Newton body transform at
    ``newton_body_q[newton_indices[i]]`` and stores it as a column-major
    ``mat44d`` in ``fabric_transforms[i]``.
    """
    i = int(wp.tid())
    idx = int(newton_indices[i])
    transform = newton_body_q[idx]
    fabric_transforms[i] = wp.transpose(wp.mat44d(wp.transform_to_matrix(transform)))


@wp.kernel(enable_backward=False)
def _sync_particle_points(
    fabric_points: wp.fabricarrayarray(dtype=wp.vec3f),
    fabric_world_matrices: wp.fabricarray(dtype=wp.mat44d),
    offsets: wp.fabricarray(dtype=wp.uint32),
    counts: wp.fabricarray(dtype=wp.uint32),
    particle_q: wp.array(dtype=wp.vec3f),
):
    """Write Newton particle positions into Fabric mesh point arrays as local-frame points.

    Newton stores particle positions in world space in ``state.particle_q``. The Fabric
    ``points`` attribute on a ``UsdGeom.Mesh`` is local-space -- Kit multiplies by the
    mesh prim's resolved ``omni:fabric:worldMatrix`` at render time.

    This kernel inverts the mesh prim's world matrix to convert each world-space particle
    position into local-space before writing.
    """
    i = wp.tid()
    offset = int(offsets[i])
    num_points = int(counts[i])

    # Un-transpose Fabric's stored matrix to get the standard homogeneous form
    world_matrix = wp.transpose(wp.mat44f(fabric_world_matrices[i]))
    inv_world_matrix = wp.inverse(world_matrix)

    for j in range(num_points):
        fabric_points[i][j] = wp.transform_point(inv_world_matrix, particle_q[offset + j])


@dataclass
class _ParticleVisualPrim:
    """A ``UsdGeom.Points`` prim mirroring a slice of Newton's particle state."""

    points_attr: Usd.Attribute
    offset: int
    count: int
    sync_frequency: int
    frames_since_sync: int


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


class NewtonSceneDataBackend(SceneDataBackend):
    """Scene data backend that reads rigid body transforms from Newton's simulation state.

    The backend reads ``body_q`` (an array of :class:`wp.transformf`) from
    Newton's current state and exposes it as :class:`SceneDataFormat.Transform`.
    Body paths come from the model's ``body_label`` attribute.
    """

    def __init__(self):
        self._scene_data = SceneDataFormat.Transform()

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        """Return the current Newton rigid body transforms as :class:`SceneDataFormat.Transform`."""
        self._scene_data.transforms = self.state.body_q
        return self._scene_data

    @property
    def transform_count(self) -> int:
        """Return the number of rigid body transforms in the Newton sim."""
        return self.model.body_count

    @property
    def transform_paths(self) -> list[str]:
        """Return the prim paths for each rigid body transform."""
        if self.model.body_label is not None:
            return list(self.model.body_label)
        return []

    @property
    def model(self) -> Model:
        return NewtonManager.get_model()

    @property
    def state(self) -> State:
        """Return Newton state after applying pending forward kinematics."""
        return NewtonManager.get_state()


def _eval_fk_unbound(world_reset_mask: wp.array | None, fk_mask: wp.array | None) -> None:
    """Default :attr:`NewtonManager._eval_fk` value before a solver is initialized.

    Raises so a stray ``forward()`` / ``step()`` before ``initialize_solver()`` fails loudly
    instead of silently running a wrong (or no) FK.
    """
    raise RuntimeError(
        "FK hook is not bound. NewtonManager.initialize_solver() must run "
        "(via reset()) before forward()/step() can run forward kinematics."
    )


def _reset_solver_internals_unbound(world_mask: wp.array | None) -> None:
    """Default reset-hook delegate value before a solver is initialized."""
    raise RuntimeError(
        "Solver reset hook is not bound. NewtonManager.initialize_solver() must run "
        "(via reset()) before forward()/step() can reset solver internals."
    )


class NewtonManager(PhysicsManager):
    """Abstract Newton physics manager for Isaac Lab.

    Class-level (singleton-like) manager that owns simulation lifecycle, model
    state, contacts/collision pipeline, sensors, replication, and CUDA-graph
    orchestration.
    Concrete subclasses (one per solver) implement :meth:`_build_solver` and
    may extend :meth:`_initialize_contacts`, :meth:`_prepare_builder_for_finalize`,
    :meth:`_step_solver`, :meth:`_supports_cuda_graph_capture`,
    :meth:`_reset_solver_internals`, :meth:`_solver_specific_clear`, and
    :meth:`_log_solver_debug`.

    Subclasses are selected via :attr:`NewtonSolverCfg.class_type`, which
    :meth:`NewtonCfg.__post_init__` propagates onto :attr:`NewtonCfg.class_type`
    so that ``SimulationContext`` resolves the matching subclass automatically.

    Lifecycle: ``initialize() -> reset() -> step()`` (repeated) ``-> close()``.

    .. note::
        Shared state lives on :class:`NewtonManager` (the base) by design — the
        framework imports ``NewtonManager`` directly and reads attributes such
        as ``_model`` / ``_state_0`` / ``_builder`` from many places.  Lifecycle
        methods therefore assign through the explicit base class
        (``NewtonManager._foo = ...``) rather than through ``cls`` so that the
        canonical state remains discoverable from external readers regardless of
        which subclass is active.
    """

    @classmethod
    def provides_implicit_damping(cls) -> bool:
        # Newton's symplectic integrator has no implicit damping.
        return False

    _solver_dt: float = 1.0 / 200.0
    _num_substeps: int = 1
    _decimation: int = 1
    _collision_decimation: int = 0
    _num_envs: int | None = None

    # Newton model and state
    _builder: ModelBuilder = None
    _model: Model = None
    _solver: SolverBase | None = None
    _use_single_state: bool | None = None
    """Use only one state for both input and output for solver stepping. Requires solver support."""
    _state_0: State = None
    _state_1: State = None
    _control: Control = None

    # Physics settings
    _gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81)
    _up_axis: str = "Z"

    # Collision and contacts
    _contacts: Contacts | None = None
    _needs_collision_pipeline: bool = False
    _collision_pipeline = None
    _collision_cfg: NewtonCollisionPipelineCfg | None = None
    _newton_contact_sensors: dict = {}  # Maps sensor_key to NewtonContactSensor
    _newton_frame_transform_sensors: list = []  # List of SensorFrameTransform
    _newton_imu_sensors: list = []  # List of NewtonSensorIMU
    _pending_extended_state_attributes: set[str] = set()
    _pending_extended_contact_attributes: set[str] = set()
    _report_contacts: bool = False
    _supports_contact_sensors: bool = True

    # Per-world reset masks (allocated in start_simulation, consumed in step/forward).
    _world_reset_mask: wp.array | None = None  # (num_envs,) wp.bool — for SolverKamino.reset(world_mask=...)
    _fk_reset_mask: wp.array | None = None  # (articulation_count,) wp.bool — for eval_fk(mask=...)
    # Solver-specialized FK delegate. Bound in initialize_solver() to the active subclass's choice of FK implementation.
    _eval_fk: Callable[[wp.array | None, wp.array | None], None] = _eval_fk_unbound
    # Solver-specialized reset delegate. Like _eval_fk, this must dispatch correctly through the base manager.
    _reset_solver_internals_delegate: Callable[[wp.array | None], None] = _reset_solver_internals_unbound

    # Newton actuator adapter (owns actuators and double-buffered states)
    _adapter: NewtonActuatorAdapter | None = None
    # In-graph hooks invoked after the actuator step and before the solver
    # substeps, in registration order. Multiple articulations register their
    # implicit-DOF telemetry / FF-routing kernels here.
    _post_actuator_callbacks: list[Callable[[], None]] = []
    # In-graph hooks invoked after the last solver substep and before sensors,
    # in registration order. Articulations with non-identity ordering register
    # their backend-to-user state republish kernels here so the reorders are
    # recorded into every captured graph.
    _post_step_callbacks: list[Callable[[], None]] = []

    # CUDA graphing
    _graph = None
    _graph_capture_pending: bool = False

    # Newton scene-query scheduling and graph execution.
    _sensor_tasks: dict[str, Callable[[], None]] = {}
    _sensor_graph: wp.Graph | None = None
    _sensor_flags: wp.array | None = None
    _sensor_flags_host: np.ndarray | None = None
    _sensor_state: State | None = None
    _sensor_state_dirty: bool = True
    _sensor_graph_capture_failed: bool = False
    _sensor_bvh_has_collision_shapes: bool = False  # set once a ray-cast sensor widens the shape BVH

    # USD/Fabric sync
    _newton_stage_path = None
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _clone_physics_only = False
    _transforms_dirty: bool = False
    _transforms_may_change_on_graph_replay: bool = False
    _particles_dirty: bool = False
    _newton_particle_offset_attr = "newton:particleOffset"
    _newton_particle_count_attr = "newton:particleCount"
    _particle_visual_prims: dict[str, _ParticleVisualPrim] = {}

    # cubric GPU transform hierarchy (replaces CPU update_world_xforms)
    _cubric = None
    _cubric_adapter: int | None = None
    _cubric_bound_fabric_id: int | None = None

    # Set to True after sync_transforms_to_usd() successfully writes body positions for
    # the first time in each simulation session.  Reset to False in clear().  Polled by
    # test drain helpers to know when the GPU has propagated the newton:index Fabric
    # attribute and body_q values are valid.
    _newton_fabric_ready: bool = False

    # Model changes (callbacks use unified system from PhysicsManager)
    _model_changes: set[int] = set()

    # Scene data backend
    _scene_data_backend: NewtonSceneDataBackend | None = None

    # Visualization-only state used when the sim backend is PhysX. Populated
    # lazily in :meth:`_ensure_visualization_model` and updated each render
    # frame in :meth:`update_visualization_state`.
    _scene_data: SceneDataFormat.Transform | None = None
    _scene_data_mapping: wp.array | None = None
    _scene_data_points: SceneDataFormat.Points | None = None
    _scene_data_geometry_mapping: wp.array | None = None
    _shadow_deformable_entities: list | None = None
    _sim_particle_q: wp.array | None = None
    _mapped_sim_particle_offsets: set[int] | None = None
    _shadow_deformable_sync_skip_warned: set[str] = set()
    _shadow_deformable_remap_batches: list | None = None
    _shadow_deformable_copy_batch: tuple | None = None
    _shadow_deformable_batch_sync_key: tuple | None = None

    # Views list for assets to register their views
    _views: list = []
    _mpm_object_registry: list = []

    # CL: Cloning / Replication logic
    # TODO: These attributes support cloning-specific logic and should be moved into a cloner class
    # Pending site requests from sensors.
    # Key: (body_pattern, per_world, xform_floats), Value: (label, wp.transform)
    # identical (body_pattern, per_world, transform) reuses the same site.
    _cl_pending_sites: dict[tuple[str | None, bool, tuple[float, ...]], tuple[str, wp.transform]] = {}

    # Maps each site label to its resolved global or local site entry.
    _GlobalSite = tuple[int, None]
    _LocalSite = tuple[None, list[list[int]]]
    _SiteEntry = _GlobalSite | _LocalSite
    _cl_site_index_map: dict[str, _SiteEntry] = {}
    _cl_fabric_body_bindings: list[tuple[str, int]] | None = None
    _world_xforms: list[wp.transform] | None = None
    # Per-source builders retained from replication, keyed by clone-plan source
    # path. Single-model consumers (e.g. batched Newton IK) finalize a single-env
    # model from these and resolve it via ``query.path_to_source``.
    _cl_protos: dict[str, ModelBuilder] = {}
    _deformable_registry: list = []
    _per_world_builder_hooks: list[Callable[[ModelBuilder, int, list[float], list[float]], None]] = []
    _post_replicate_hooks: list[Callable[[ModelBuilder], None]] = []

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        super().initialize(sim_context)

        # Newton-specific setup: get gravity from SimulationCfg (not physics manager cfg)
        sim = PhysicsManager._sim
        if sim is not None:
            NewtonManager._gravity_vector = sim.cfg.gravity  # type: ignore[union-attr]

            # USD/Fabric sync for Omniverse rendering (visualizer) or Newton+RTX (Kit cameras)
            try:
                requested = sim.resolve_visualizer_types()
            except Exception:
                requested = []
                viz_raw = sim.get_setting("/isaaclab/visualizer/types")
                if isinstance(viz_raw, str):
                    requested = [v for part in viz_raw.split(",") for v in part.split() if v]
            from isaaclab.app.settings_manager import get_settings_manager

            cameras_enabled = bool(get_settings_manager().get("/isaaclab/cameras_enabled", False))
            cls._clone_physics_only = not has_kit() or ("kit" not in requested and not cameras_enabled)

        cls._scene_data_backend = NewtonSceneDataBackend()

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset physics simulation.

        A hard reset (``soft=False``) re-finalizes the Newton model, reallocating
        its device arrays. The cached collision pipeline, contacts and any
        captured CUDA graph reference the old buffers, so they are released here
        and rebuilt against the re-finalized model by :meth:`initialize_solver`.
        This avoids the illegal CUDA memory access (CUDA error 700) that would
        otherwise occur on the first step after a hard reset.

        A soft reset (``soft=True``) skips this full reinitialization and reuses
        the existing model, solver, collision pipeline and CUDA graph.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            # Release the cached collision pipeline, contacts and CUDA graph;
            # they point at the old model's freed buffers (CUDA 700 on next step).
            NewtonManager._graph = None
            NewtonManager._graph_capture_pending = False
            NewtonManager._collision_pipeline = None
            NewtonManager._contacts = None

            cls.start_simulation()
            cls.initialize_solver()

    @classmethod
    def _eval_fk_impl(cls, world_reset_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Update body states from joint coordinates.

        Solver-specialized FK implementation. The base implementation runs Newton's generic
        ``eval_fk`` over the articulations selected by ``fk_mask``. Subclasses may override
        this method to use a solver-specific FK.

        Args:
            world_reset_mask: Per-world mask of environments to reset (``None`` means all).
                Unused by the base implementation; consumed by solver-specific overrides such as
                :meth:`NewtonKaminoManager._eval_fk_impl`.
            fk_mask: Per-articulation mask of articulations to update (``None`` means all).
        """
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, fk_mask)

    @classmethod
    def forward(cls) -> None:
        """Update articulation kinematics without stepping physics.

        Update body poses from joint coordinates via the solver-specialized FK delegate
        (:attr:`_eval_fk`, bound to the active subclass's :meth:`_eval_fk_impl` in
        :meth:`initialize_solver`). Only the articulations flagged dirty in
        :attr:`_fk_reset_mask` and :attr:`_world_reset_mask` (see :meth:`invalidate_fk`) are
        updated. The masks are consumed (zeroed) afterwards so the next :meth:`step` does not
        redundantly re-solve them.

        The delegate (rather than a direct ``cls._eval_fk_impl`` call) is required because the
        data layer invokes ``NewtonManager.forward()`` on the base class, where ``cls`` is the
        base ``NewtonManager``; the bound delegate dispatches to the concrete subclass override.
        """
        cls._reset_solver_internals_delegate(cls._world_reset_mask)
        cls._eval_fk(cls._world_reset_mask, cls._fk_reset_mask)
        if cls._fk_reset_mask is not None:
            cls._fk_reset_mask.zero_()
        if cls._world_reset_mask is not None:
            cls._world_reset_mask.zero_()
        cls._mark_sensor_state_dirty()

    @classmethod
    def pre_render(cls) -> None:
        """Refresh derived Newton state before cameras and visualizers read it."""
        if cls._fk_reset_mask is not None:
            cls.forward()
            if NewtonManager._transforms_may_change_on_graph_replay:
                NewtonManager._transforms_dirty = True
        cls.sync_transforms_to_usd()
        cls.sync_particles_to_usd()

    @classmethod
    def sync_transforms_to_usd(cls) -> None:
        """Write Newton body_q to USD Fabric world matrices for Kit viewport / RTX rendering.

        No-op when ``_usdrt_stage`` is None (i.e. Kit visualizer is not active)
        or when transforms have not changed since the last sync.

        Called at render cadence by :meth:`pre_render` (via
        :meth:`~isaaclab.sim.SimulationContext.render`).
        Physics stepping marks transforms dirty via :meth:`_mark_transforms_dirty`
        so that the expensive Fabric hierarchy update only runs once per render
        frame rather than after every physics step.

        Uses ``wp.fabricarray`` directly (no ``isaacsim.physics.newton`` extension needed).
        The Warp kernel reads ``state_0.body_q[newton_index[i]]`` and writes the
        corresponding ``mat44d`` to ``omni:fabric:worldMatrix`` for each prim.

        When cubric is available the method mirrors PhysX's ``DirectGpuHelper``
        pattern: pause Fabric change tracking, write transforms, resume tracking,
        then call ``IAdapter::compute`` on the GPU to propagate the hierarchy and
        notify the Fabric Scene Delegate.  Otherwise it falls back to the CPU
        ``update_world_xforms()`` path.
        """
        if cls._usdrt_stage is None or cls._model is None or cls._state_0 is None:
            return
        if not cls._transforms_dirty:
            return
        try:
            import usdrt

            # Lazy adapter creation: deferred from initialize_solver() to avoid
            # startup-ordering issues with the cubric plugin.
            if cls._cubric is not None and cls._cubric.available and cls._cubric_adapter is None:
                NewtonManager._cubric_adapter = cls._cubric.create_adapter()
                if cls._cubric_adapter is not None:
                    logger.info("cubric GPU transform hierarchy enabled")
                else:
                    logger.warning("cubric adapter creation failed; falling back to update_world_xforms()")
                    NewtonManager._cubric = None

            use_cubric = cls._cubric is not None and cls._cubric_adapter is not None

            fabric_hierarchy = None
            if hasattr(usdrt, "hierarchy"):
                fabric_hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
                    cls._usdrt_stage.GetFabricId(), cls._usdrt_stage.GetStageIdAsStageId()
                )

            # Pause hierarchy change tracking BEFORE SelectPrims.
            # SelectPrims with ReadWrite access calls getAttributeArrayGpu
            # internally, which marks Fabric buffers dirty.  If tracking is
            # still active at that point the hierarchy records the change and
            # Kit's updateWorldXforms will do an expensive connectivity
            # rebuild every frame.  PhysX avoids this via ScopedUSDRT which
            # pauses tracking before any Fabric writes.
            if use_cubric and fabric_hierarchy is not None:
                fabric_hierarchy.track_world_xform_changes(False)
                fabric_hierarchy.track_local_xform_changes(False)

            try:
                selection = cls._usdrt_stage.SelectPrims(
                    require_attrs=[
                        (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.ReadWrite),
                        (usdrt.Sdf.ValueTypeNames.UInt, cls._newton_index_attr, usdrt.Usd.Access.Read),
                    ],
                    device=str(PhysicsManager._device),
                )
                if selection.GetCount() == 0:
                    # The newton:index attribute is written CPU-side by start_simulation() but
                    # GPU propagation is deferred.  Keep _transforms_dirty=True so the next
                    # pre_render() retries once initialize_solver() has completed (FK delegate
                    # bound) and body_q holds valid values.
                    if cls._eval_fk is _eval_fk_unbound:
                        NewtonManager._transforms_dirty = False
                    return

                fabric_transforms = wp.fabricarray(selection, "omni:fabric:worldMatrix")
                newton_indices = wp.fabricarray(selection, cls._newton_index_attr)
                wp.launch(
                    _set_fabric_transforms,
                    dim=newton_indices.shape[0],
                    inputs=[fabric_transforms, newton_indices, cls._state_0.body_q],
                    device=PhysicsManager._device,
                )
                wp.synchronize_device(PhysicsManager._device)

                NewtonManager._newton_fabric_ready = True
                NewtonManager._transforms_dirty = False

                if use_cubric and fabric_hierarchy is not None:
                    fabric_id = cls._usdrt_stage.GetFabricId().id
                    if fabric_id != cls._cubric_bound_fabric_id:
                        cls._cubric.bind_to_stage(cls._cubric_adapter, fabric_id)
                        NewtonManager._cubric_bound_fabric_id = fabric_id
                    cls._cubric.compute(cls._cubric_adapter)
                elif fabric_hierarchy is not None:
                    fabric_hierarchy.update_world_xforms()
            finally:
                if use_cubric and fabric_hierarchy is not None:
                    fabric_hierarchy.track_world_xform_changes(True)
                    fabric_hierarchy.track_local_xform_changes(True)
        except Exception:
            logger.exception("[NewtonManager] sync_transforms_to_usd FAILED")

    @classmethod
    def sync_particles_to_usd(cls) -> None:
        """Write Newton particle positions to USD/Fabric for Kit viewport rendering.

        Two prim families are synced from ``state_0.particle_q``:

        * Fabric mesh prims tagged with ``newton:particleOffset`` /
          ``newton:particleCount`` (deformable visual meshes) receive
          local-frame points on the GPU via :meth:`_sync_fabric_mesh_particles`.
        * ``UsdGeom.Points`` prims registered through
          :meth:`register_particle_visual_prim` (MPM particle clouds) receive
          world-frame points via :meth:`_sync_particle_points_prims`.

        No-op when there is no particle state or nothing changed since the
        last sync.
        """
        if not cls._particles_dirty or cls._state_0 is None or cls._state_0.particle_q is None:
            return
        try:
            cls._sync_fabric_mesh_particles()
            NewtonManager._particles_dirty = cls._sync_particle_points_prims()
        except Exception:
            logger.exception("[NewtonManager] sync_particles_to_usd FAILED")

    @classmethod
    def _sync_fabric_mesh_particles(cls) -> None:
        """Write ``state_0.particle_q`` into Fabric mesh point arrays as local-frame points."""
        if cls._usdrt_stage is None:
            return
        import usdrt  # noqa: PLC0415

        selection = cls._usdrt_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.ReadWrite),
                (usdrt.Sdf.ValueTypeNames.UInt, cls._newton_particle_offset_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.UInt, cls._newton_particle_count_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
            ],
            device=str(PhysicsManager._device),
        )
        if selection.GetCount() == 0:
            return
        wp.launch(
            _sync_particle_points,
            dim=selection.GetCount(),
            inputs=[
                wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f),
                wp.fabricarray(data=selection, attrib="omni:fabric:worldMatrix"),
                wp.fabricarray(data=selection, attrib=cls._newton_particle_offset_attr),
                wp.fabricarray(data=selection, attrib=cls._newton_particle_count_attr),
                cls._state_0.particle_q,
            ],
            device=PhysicsManager._device,
        )

    @classmethod
    def _sync_particle_points_prims(cls) -> bool:
        """Write registered ``UsdGeom.Points`` prims; return ``True`` while throttled prims remain."""
        if not cls._particle_visual_prims:
            return False

        due = []
        for record in cls._particle_visual_prims.values():
            record.frames_since_sync += 1
            if record.frames_since_sync >= record.sync_frequency:
                record.frames_since_sync = 0
                due.append(record)
        if due:
            from pxr import Sdf, Vt  # noqa: PLC0415

            particle_q = cls._state_0.particle_q.numpy()
            with Sdf.ChangeBlock():
                for record in due:
                    points = particle_q[record.offset : record.offset + record.count]
                    record.points_attr.Set(Vt.Vec3fArray.FromNumpy(points))
        return len(due) < len(cls._particle_visual_prims)

    @classmethod
    def _mark_transforms_dirty(cls) -> None:
        """Flag that rigid-body transforms have changed and Fabric needs re-sync.

        The actual sync is deferred to :meth:`sync_transforms_to_usd`,
        which runs at render cadence via :meth:`pre_render`.
        """
        NewtonManager._transforms_dirty = True

        device = PhysicsManager._device
        if device is not None:
            device = wp.get_device(device)
            if device.is_cuda and device.stream.is_capturing:
                NewtonManager._transforms_may_change_on_graph_replay = True

    @classmethod
    def _mark_particles_dirty(cls) -> None:
        """Flag that particle positions have changed and Fabric needs re-sync.

        The actual sync is deferred to the particle sync callback (if registered),
        which runs at render cadence via :meth:`pre_render`.
        """
        NewtonManager._particles_dirty = True

    @classmethod
    def _mark_state_dirty(cls) -> None:
        """Flag that all physics state has changed and Fabric needs re-sync.

        Convenience method that marks both transforms and particles dirty.
        Called by :meth:`_simulate` after stepping.
        """
        cls._mark_transforms_dirty()
        cls._mark_particles_dirty()

    @classmethod
    def register_particle_visual_prim(
        cls, prim_path: str, particle_offset: int, particle_count: int, sync_frequency: int = 1
    ) -> None:
        """Register a ``UsdGeom.Points`` prim whose points mirror a slice of Newton's particle state.

        Args:
            prim_path: Stage path of an existing ``UsdGeom.Points`` prim.
            particle_offset: First index of the prim's slice in ``state.particle_q``.
            particle_count: Number of particles in the slice.
            sync_frequency: Sync the prim every N dirty render frames.
        """
        from pxr import UsdGeom  # noqa: PLC0415

        prim = get_current_stage().GetPrimAtPath(prim_path)
        NewtonManager._particle_visual_prims[prim_path] = _ParticleVisualPrim(
            points_attr=UsdGeom.Points(prim).GetPointsAttr(),
            offset=int(particle_offset),
            count=int(particle_count),
            sync_frequency=int(sync_frequency),
            frames_since_sync=int(sync_frequency),
        )

    @classmethod
    def step(cls) -> None:
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
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        cls._reset_solver_internals_delegate(cls._world_reset_mask)

        # Notify solver of model changes
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._solver.notify_model_changed(change)
                NewtonManager._model_changes = set()

        # Lazy CUDA graph capture
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        if cls._graph_capture_pending and cfg is not None and cfg.use_cuda_graph and "cuda" in device:  # type: ignore[union-attr]
            NewtonManager._graph_capture_pending = False
            NewtonManager._graph = cls._capture_relaxed_graph(device)
            if cls._graph is not None:
                # Kamino: StateKamino.from_newton() lazily allocates body_f_total,
                # joint_q_prev, and joint_lambdas via wp.clone/wp.zeros during the
                # first step() inside graph capture. Replay once to pin those
                # memory-pool addresses before any eager solver.reset() call.
                if isinstance(cls._solver, SolverKamino):
                    wp.capture_launch(cls._graph)
                logger.info("Newton CUDA graph captured (deferred relaxed mode, RTX-compatible)")
            else:
                logger.warning("Newton deferred CUDA graph capture failed; using eager execution")

        # Reconcile authored state after any mutating graph warmup and before the requested physics step.
        cls.forward()

        physics_dt = cls._solver_dt * cls._num_substeps
        use_graph = cfg is not None and cfg.use_cuda_graph and cls._graph is not None and "cuda" in device  # type: ignore[union-attr]

        if cls._is_all_graphable():
            # --- All actuators are graph-safe: actuators + solver in one graph ---
            if use_graph:
                wp.capture_launch(cls._graph)
            else:
                with wp.ScopedDevice(device):
                    cls._simulate_full()
            PhysicsManager._sim_time += physics_dt * cls._decimation
        else:
            # --- Some actuators not graph-safe: step them eagerly, graph solver only ---
            if cls._adapter is not None:
                cls._adapter.step(cls._state_0, cls._control, physics_dt)
            for cb in cls._post_actuator_callbacks:
                cb()

            if use_graph:
                wp.capture_launch(cls._graph)
            else:
                with wp.ScopedDevice(device):
                    cls._simulate_physics_only()
            PhysicsManager._sim_time += physics_dt

        if cls._usdrt_stage is not None:
            cls._mark_state_dirty()
        elif cls._particle_visual_prims:
            cls._mark_particles_dirty()
        cls._mark_sensor_state_dirty()

        # Launch solver-specific debug logging after stepping.
        cls._log_solver_debug()

    @classmethod
    def close(cls) -> None:
        """Clean up Newton physics resources."""
        super().close()
        cls.clear()

    @classmethod
    def get_scene_data_backend(cls) -> SceneDataBackend | None:
        """Return the SceneDataBackend for the SceneDataProvider."""
        return cls._scene_data_backend

    @classmethod
    def register_callback(
        cls,
        callback: Callable,
        event: PhysicsEvent,
        order: int = 0,
        name: str | None = None,
        wrap_weak_ref: bool = True,
    ) -> CallbackHandle:
        """Register a callback. Passes event to parent class."""
        return PhysicsManager.register_callback(callback, event, order, name, wrap_weak_ref)

    @classmethod
    def get_physics_sim_view(cls) -> list:
        """Get the list of registered views.

        Assets can append their views to this list, and sensors can access them.
        Returns a list that callers can append to.

        Returns:
            List of registered views (e.g., NewtonArticulationView instances).
        """
        return cls._views

    @classmethod
    def is_fabric_enabled(cls) -> bool:
        """Check if fabric interface is enabled (not applicable for Newton)."""
        return False

    @classmethod
    def clear(cls):
        """Clear all Newton-specific state (callbacks cleared by super().close())."""
        if cls._cubric is not None and cls._cubric_adapter is not None:
            cls._cubric.release_adapter(cls._cubric_adapter)
        NewtonManager._cubric = None
        NewtonManager._cubric_adapter = None
        NewtonManager._cubric_bound_fabric_id = None
        NewtonManager._newton_fabric_ready = False
        NewtonManager._builder = None
        NewtonManager._model = None
        NewtonManager._solver = None
        NewtonManager._use_single_state = None
        NewtonManager._state_0 = None
        NewtonManager._state_1 = None
        NewtonManager._control = None
        NewtonManager._contacts = None
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._eval_fk = _eval_fk_unbound
        NewtonManager._reset_solver_internals_delegate = _reset_solver_internals_unbound
        NewtonManager._collision_pipeline = None
        NewtonManager._collision_cfg = None
        NewtonManager._newton_contact_sensors = {}
        NewtonManager._newton_frame_transform_sensors = []
        NewtonManager._newton_imu_sensors = []
        NewtonManager._report_contacts = False
        NewtonManager._supports_contact_sensors = True
        NewtonManager._adapter = None
        NewtonManager._post_actuator_callbacks = []
        NewtonManager._post_step_callbacks = []
        # Set by an articulation that took the ``use_newton_actuators=True``
        # branch in ``_process_actuators_cfg``.  Together with the adapter
        # check, this gates whether the decimation loop can be captured into
        # a CUDA graph (see :meth:`_is_all_graphable`).
        NewtonManager._use_newton_actuators_active = False
        NewtonManager._decimation = 1
        # Per-world reset masks
        NewtonManager._world_reset_mask = None
        NewtonManager._fk_reset_mask = None
        NewtonManager._graph = None
        NewtonManager._graph_capture_pending = False
        NewtonManager._sensor_tasks = {}
        NewtonManager._invalidate_sensor_graph()
        NewtonManager._sensor_state = None
        NewtonManager._sensor_state_dirty = True
        NewtonManager._sensor_graph_capture_failed = False
        NewtonManager._sensor_bvh_has_collision_shapes = False
        NewtonManager._newton_stage_path = None
        NewtonManager._usdrt_stage = None
        NewtonManager._transforms_dirty = False
        NewtonManager._transforms_may_change_on_graph_replay = False
        NewtonManager._particles_dirty = False
        NewtonManager._particle_visual_prims = {}
        NewtonManager._mpm_object_registry = []
        NewtonManager._deformable_registry = []
        NewtonManager._per_world_builder_hooks = []
        NewtonManager._post_replicate_hooks = []
        NewtonManager._up_axis = "Z"
        NewtonManager._scene_data = None
        NewtonManager._scene_data_mapping = None
        NewtonManager._scene_data_points = None
        NewtonManager._scene_data_geometry_mapping = None
        NewtonManager._shadow_deformable_entities = None
        NewtonManager._sim_particle_q = None
        NewtonManager._mapped_sim_particle_offsets = None
        NewtonManager._shadow_deformable_sync_skip_warned = set()
        NewtonManager._shadow_deformable_remap_batches = None
        NewtonManager._shadow_deformable_copy_batch = None
        NewtonManager._shadow_deformable_batch_sync_key = None
        NewtonManager._model_changes = set()
        NewtonManager._scene_data_backend = None
        NewtonManager._cl_pending_sites = {}
        NewtonManager._cl_site_index_map = {}
        NewtonManager._cl_fabric_body_bindings = None
        NewtonManager._world_xforms = None
        NewtonManager._cl_protos = {}
        NewtonManager._pending_extended_state_attributes = set()
        NewtonManager._pending_extended_contact_attributes = set()
        NewtonManager._views = []
        cls._solver_specific_clear()

    @classmethod
    def set_builder(cls, builder: ModelBuilder) -> None:
        """Set the Newton model builder."""
        NewtonManager._builder = builder

    @classmethod
    def create_builder(cls, up_axis: str | None = None, **kwargs) -> ModelBuilder:
        """Create a :class:`ModelBuilder` configured with default settings.

        Forwards :class:`NewtonShapeCfg` defaults onto Newton's upstream
        ``ModelBuilder.default_shape_cfg`` via :func:`~isaaclab.utils.checked_apply`.
        Falls back to wrapper defaults when no Newton config is active so
        rough-terrain margin/gap still apply during early construction.

        Args:
            up_axis: Override for the up-axis. Defaults to ``None``, which uses
                the manager's ``_up_axis``.
            **kwargs: Forwarded to :class:`ModelBuilder`.

        Returns:
            New builder with up-axis and per-shape defaults (gap, margin) applied.
        """
        # Resolve which NewtonShapeCfg to apply: user override if active config
        # is NewtonCfg, else the wrapper's own defaults so callers from non-Newton
        # contexts (tests, early construction) still get the rough-terrain margin.
        cfg = PhysicsManager._cfg

        builder = ModelBuilder(up_axis=up_axis or cls._up_axis, **kwargs)
        builder.default_bvh_cfg = ModelBuilder.BvhConfig(
            mesh_constructor=cfg.bvh_constructor_geometry if isinstance(cfg, NewtonCfg) else None,
            gaussian_constructor=cfg.bvh_constructor_gaussian if isinstance(cfg, NewtonCfg) else None,
            shape_constructor=cfg.bvh_constructor_scene if isinstance(cfg, NewtonCfg) else None,
        )

        cls._register_builder_attributes(builder)
        shape_cfg = cfg.default_shape_cfg if isinstance(cfg, NewtonCfg) else NewtonShapeCfg()
        checked_apply(shape_cfg, builder.default_shape_cfg)
        return builder

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Subclass hook to register solver-specific custom attributes on *builder*.

        Override in solver subclasses (e.g. :class:`NewtonMPMManager`) that need
        Newton-side particle, shape, or body custom attributes registered before
        the builder is finalized. The default implementation is a no-op so
        solvers without custom attributes do not need to override it.

        Implementations should be **idempotent** — the same builder may be
        passed multiple times across :meth:`create_builder`,
        :meth:`instantiate_builder_from_stage`, and :meth:`start_simulation`.
        """

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Subclass hook to normalize *builder* before model finalization.

        Override in solver subclasses that need to adapt imported or replicated
        builder data before :meth:`ModelBuilder.finalize` allocates model arrays.
        The default implementation is a no-op.
        """

    @classmethod
    def cl_register_site(cls, body_pattern: str | None, xform: wp.transform, *, per_world: bool = False) -> str:
        """Register a site request for injection into prototypes before replication.

        Sensors call this during ``__init__``. Sites are injected into prototype
        builders by :meth:`_cl_inject_sites` (called from ``newton_replicate``)
        before ``add_builder``, so they replicate correctly per-world.

        Identical ``(body_pattern, per_world, transform)`` registrations share sites.

        The *body_pattern* is matched against prototype-local body labels
        (e.g. ``"Robot/link.*"``) when replication is active, or against the
        flat builder's body labels in the fallback path. Wildcard patterns
        that match multiple bodies create one site per matched body.

        Args:
            body_pattern: Regex pattern matched against body labels in the
                prototype builder (e.g. ``"Robot/link0"`` or ``"Robot/finger.*"``
                for multi-body wildcards), or ``None`` for global sites
                (world-origin reference, etc.).
            xform: Site transform relative to body.
            per_world: When ``True``, ``body_pattern`` must be ``None`` and one
                bodyless site is created in each cloned world's frame.

        Returns:
            Assigned site label suffix.
        """
        if per_world and body_pattern is not None:
            raise ValueError("per_world site registration requires body_pattern=None.")
        xform_key = tuple(xform)
        key = (body_pattern, per_world, xform_key)
        if key in cls._cl_pending_sites:
            return cls._cl_pending_sites[key][0]
        label = f"ft_{len(cls._cl_pending_sites)}"
        cls._cl_pending_sites[key] = (label, xform)
        return label

    @classmethod
    def request_extended_state_attribute(cls, attr: str) -> None:
        """Request an extended state attribute (e.g. ``"body_qdd"``).

        Sensors call this during ``__init__``, before model finalization.
        Attributes are forwarded to the builder in :meth:`start_simulation`
        so that subsequent ``model.state()`` calls allocate them.

        Args:
            attr: State attribute name (must be in ``State.EXTENDED_ATTRIBUTES``).
        """
        cls._pending_extended_state_attributes.add(attr)

    @classmethod
    def request_extended_contact_attribute(cls, attr: str) -> None:
        """Request an extended contact attribute (e.g. ``"force"``).

        Sensors call this during ``__init__``, before model finalization.
        Attributes are forwarded to the model in :meth:`start_simulation`
        so that subsequent ``Contacts`` creation includes them.

        Args:
            attr: Contact attribute name.
        """
        cls._pending_extended_contact_attributes.add(attr)

    @classmethod
    def _cl_inject_sites(
        cls,
        main_builder: ModelBuilder,
        source_builders: dict[str, ModelBuilder],
    ) -> tuple[dict[str, int], dict[int, dict[str, list[int]]], dict[str, wp.transform]]:
        """Inject registered sites into source builders before replication.

        Non-global sites are matched against source builder body labels using
        :func:`resolve_matching_names` (regex). Global sites
        (``body_pattern is None``) are added to *main_builder* with
        ``body=-1``.

        Returns source-builder-local shape indices so that ``newton_replicate`` can
        compute final indices during replication without a second pattern match.

        Pending requests are cleared after processing.

        Args:
            main_builder: Top-level builder that receives global sites.
            source_builders: ``{source_path: ModelBuilder}`` source builders.

        Returns:
            Tuple of ``(global_site_indices, source_site_indices, env_root_sites)`` where
            *global_site_indices* maps ``{label: main_builder_shape_idx}``,
            *source_site_indices* maps ``{id(source_builder): {label: [source_local_shape_idx, ...]}}``,
            and *env_root_sites* maps ``{label: env_root_relative_transform}``.
        """
        global_site_indices: dict[str, int] = {}
        source_site_indices: dict[int, dict[str, list[int]]] = {}

        env_root_sites: dict[str, wp.transform] = {}

        for (body_pattern, per_world, _xform_key), (label, xform) in cls._cl_pending_sites.items():
            if per_world:
                env_root_sites[label] = xform
                continue
            if body_pattern is None:
                site_idx = main_builder.add_site(body=-1, xform=xform, label=label)
                global_site_indices[label] = site_idx
                continue

            any_matched = False
            for _source_path, source_builder in source_builders.items():
                body_labels = list(source_builder.body_label)
                matched_indices, matched_names = resolve_matching_names(
                    body_pattern, body_labels, raise_when_no_match=False
                )
                if not matched_indices:  # Pattern has no matches in this source builder
                    continue

                any_matched = True
                source_builder_id = id(source_builder)
                site_indices: list[int] = []
                for body_idx, body_name in zip(matched_indices, matched_names):
                    site_label = f"{body_name}/{label}"
                    source_site_idx = source_builder.add_site(body=body_idx, xform=xform, label=site_label)
                    site_indices.append(source_site_idx)
                    logger.debug(f"Injected site '{site_label}' into source builder")
                source_site_indices.setdefault(source_builder_id, {})[label] = site_indices

            if not any_matched:
                raise ValueError(
                    f"Site '{label}' with body_pattern '{body_pattern}' matched no source-builder bodies "
                    f"across {len(source_builders)} source builder(s). "
                    f"Check that the pattern matches a body label in a source builder."
                )

        cls._cl_pending_sites.clear()
        return global_site_indices, source_site_indices, env_root_sites

    @classmethod
    def _cl_inject_sites_fallback(cls) -> None:
        """Inject pending sites into the flat builder (no-replication path).

        Populates :attr:`_cl_site_index_map` with the unified per-world structure:

        - Global sites (``body_pattern is None``): ``(shape_idx, None)``
        - Local and world sites: ``(None, [[idx, ...]])`` — one sublist for the single world.
        """
        builder = cls._builder
        body_labels = list(builder.body_label)

        for (body_pattern, per_world, _xform_key), (label, xform) in cls._cl_pending_sites.items():
            if per_world:
                site_idx = builder.add_site(body=-1, xform=xform, label=label)
                cls._cl_site_index_map[label] = (None, [[site_idx]])
                continue
            if body_pattern is None:
                site_idx = builder.add_site(body=-1, xform=xform, label=label)
                cls._cl_site_index_map[label] = (site_idx, None)
            else:
                try:
                    matched_indices, matched_names = resolve_matching_names(body_pattern, body_labels)
                except ValueError as e:
                    raise ValueError(
                        f"Site '{label}' with body_pattern '{body_pattern}' matched no bodies "
                        f"in the flat builder. Available body labels: {body_labels}."
                    ) from e

                site_indices: list[int] = []
                for body_idx in matched_indices:
                    site_label = f"{builder.body_label[body_idx]}/{label}"
                    site_idx = builder.add_site(body=body_idx, xform=xform, label=site_label)
                    site_indices.append(site_idx)

                # Single world (no replication): one-element outer list
                cls._cl_site_index_map[label] = (None, [site_indices])

        cls._cl_pending_sites.clear()

    @classmethod
    def add_model_change(cls, change: ModelFlags) -> None:
        """Register a model change to notify the solver."""
        cls._model_changes.add(change)

    @classmethod
    def invalidate_fk(
        cls,
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
        cls._mark_transforms_dirty()

        if cls._world_reset_mask is None or cls._fk_reset_mask is None:
            return

        if articulation_ids is not None and env_mask is not None:
            wp.launch(
                _or_reset_masks_from_mask,
                dim=articulation_ids.shape,
                inputs=[env_mask, articulation_ids],
                outputs=[NewtonManager._world_reset_mask, NewtonManager._fk_reset_mask],
                device=PhysicsManager._device,
            )
        elif articulation_ids is not None and env_ids is not None:
            wp.launch(
                _scatter_reset_masks_from_ids_kernel(env_ids),
                dim=(env_ids.shape[0], articulation_ids.shape[1]),
                inputs=[env_ids, articulation_ids],
                outputs=[NewtonManager._world_reset_mask, NewtonManager._fk_reset_mask],
                device=PhysicsManager._device,
            )
        else:
            # Fallback: no topology info — mark everything dirty
            NewtonManager._world_reset_mask.fill_(True)
            NewtonManager._fk_reset_mask.fill_(True)

    @classmethod
    def _drain_stale_cuda_error(cls) -> None:
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
        device = wp.get_device(str(PhysicsManager._device))
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

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation by finalizing model and initializing state.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.
        """
        logger.debug(f"Builder: {cls._builder}")

        cls._drain_stale_cuda_error()

        # Create builder from USD stage if not provided
        if cls._builder is None:
            cls.instantiate_builder_from_stage()
        cls._register_builder_attributes(cls._builder)

        logger.info("Dispatching MODEL_INIT callbacks")
        cls.dispatch_event(PhysicsEvent.MODEL_INIT)

        # Inject any pending site requests (no-replication fallback path).
        # In the replication path, _cl_inject_sites() already ran from newton_replicate.
        cls._cl_inject_sites_fallback()

        device = PhysicsManager._device
        logger.info(f"Finalizing model on device: {device}")
        cls._builder.up_axis = Axis.from_string(cls._up_axis)
        # Forward pending extended attribute requests to builder and clear them
        if cls._pending_extended_state_attributes:
            cls._builder.request_state_attributes(*cls._pending_extended_state_attributes)
            NewtonManager._pending_extended_state_attributes = set()
        cls._prepare_builder_for_finalize(cls._builder)
        with Timer(name="newton_finalize_builder", msg="Finalize builder took:"):
            NewtonManager._model = cls._builder.finalize(device=device)
            cls._model.set_gravity(cls._gravity_vector)
            cls._model.num_envs = cls._num_envs

        if cls._pending_extended_contact_attributes:
            cls._model.request_contact_attributes(*cls._pending_extended_contact_attributes)
            NewtonManager._pending_extended_contact_attributes = set()

        NewtonManager._state_0 = cls._model.state()
        NewtonManager._state_1 = cls._model.state()
        NewtonManager._control = cls._model.control()
        # The initial body-state update from joint coordinates is deferred to the tail of
        # initialize_solver(), where it runs through the solver-specialized FK delegate after the solver is initialized.

        # The single global actuator adapter is built lazily on the first
        # call to ``activate_newton_actuator_path`` from any Newton-fast-path
        # articulation after this point. Assign through the explicit base
        # class so external readers (which import ``NewtonManager`` directly)
        # observe the canonical state regardless of which subclass is active.
        NewtonManager._adapter = None
        NewtonManager._use_newton_actuators_active = False

        # Allocate per-world reset masks (used by all solvers for masked FK, and by Kamino for masked reset).
        NewtonManager._world_reset_mask = wp.zeros(cls._model.world_count, dtype=wp.bool, device=device)
        NewtonManager._fk_reset_mask = wp.zeros(cls._model.articulation_count, dtype=wp.bool, device=device)

        logger.info("Dispatching PHYSICS_READY callbacks")
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

        # Setup USD/Fabric sync for Kit viewport rendering
        if not cls._clone_physics_only:
            import usdrt

            body_paths = list(cls._model.body_label)
            NewtonManager._usdrt_stage = get_current_stage(fabric=True)
            body_bindings = NewtonManager._cl_fabric_body_bindings
            if body_bindings is None:
                # Non-replicated Newton stages do not pass through NewtonReplicateContext.
                body_bindings = [(body_path, i) for i, body_path in enumerate(body_paths)]

            fabric_hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
                cls._usdrt_stage.GetFabricId(), cls._usdrt_stage.GetStageIdAsStageId()
            )

            NewtonManager._initialize_fabric_body_prims(cls._usdrt_stage, fabric_hierarchy, usdrt, body_bindings)
            NewtonManager._initialize_fabric_particle_prims(
                cls._usdrt_stage,
                fabric_hierarchy,
                usdrt,
                NewtonManager._particle_visual_prims,
            )

            cls._mark_state_dirty()
            cls.sync_transforms_to_usd()
            cls.sync_particles_to_usd()

    @staticmethod
    def _initialize_fabric_body_prims(stage, fabric_hierarchy, usdrt, body_bindings: Sequence[tuple[str, int]]) -> None:
        """Initialize Fabric body prims used by Newton transform sync."""
        for prim_path, body_index in body_bindings:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                xformable_prim = usdrt.Rt.Xformable(prim)
                xformable_prim.SetWorldXformFromUsd()
            else:
                prim = stage.DefinePrim(prim_path, "Xform")
                xformable_prim = usdrt.Rt.Xformable(prim)
                xformable_prim.CreateFabricHierarchyWorldMatrixAttr()

            prim.CreateAttribute(NewtonManager._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, custom=True)
            prim.GetAttribute(NewtonManager._newton_index_attr).Set(body_index)
            # Tag with PhysicsRigidBodyAPI so cubric's eRigidBody mode applies
            # Inverse propagation (preserves Newton's world transforms and derives
            # local) instead of Forward.
            prim.AddAppliedSchema("PhysicsRigidBodyAPI")

        fabric_hierarchy.update_world_xforms()

    @staticmethod
    def _initialize_fabric_particle_prims(stage, fabric_hierarchy, usdrt, prim_paths: Iterable[str]) -> None:
        """Initialize Fabric world matrices for point prims used by particle sync."""
        prim_paths = tuple(prim_paths)
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                usdrt.Rt.Xformable(prim).SetWorldXformFromUsd()

        if prim_paths:
            fabric_hierarchy.update_world_xforms()

    @classmethod
    def _inject_terrain_heightfields(cls, stage: Usd.Stage, builder: ModelBuilder) -> list[str]:
        """Replace height-field-tagged terrain colliders with Newton heightfields.

        Scans the stage for prims carrying the ``newton:heightfield:resolution``
        attribute authored by :class:`~isaaclab.terrains.TerrainImporter`. For each,
        the collision mesh is rasterized into a :class:`newton.Heightfield` through
        :meth:`newton.Heightfield.create_from_mesh` and added to *builder* as a
        static heightfield shape. The tagged prim paths are returned so the caller
        can exclude them from ``add_usd`` -- otherwise the terrain would be imported
        twice (once as a mesh, once as a heightfield).

        Heightfields compile on the MuJoCo solver roughly two orders of magnitude
        faster than the equivalent multi-hundred-thousand-vertex terrain mesh while
        colliding identically at the same horizontal resolution.

        Args:
            stage: The USD stage being imported.
            builder: The Newton model builder receiving the heightfield shapes.

        Returns:
            Prim paths of terrain colliders that were converted to heightfields.
        """
        ignore_paths: list[str] = []
        xform_cache = UsdGeom.XformCache()
        for prim in stage.Traverse():
            attr = prim.GetAttribute("newton:heightfield:resolution")
            if not attr or not attr.HasAuthoredValue():
                continue
            resolution = float(attr.Get())
            # Locate the collision mesh under the tagged prim.
            if prim.IsA(UsdGeom.Mesh):
                mesh_prim = prim
            else:
                mesh_prim = next((p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)), None)
            if mesh_prim is None:
                continue
            mesh = UsdGeom.Mesh(mesh_prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            # Transform vertices into world frame (USD uses row-vector convention).
            mat = np.array(xform_cache.GetLocalToWorldTransform(mesh_prim), dtype=np.float64).reshape(4, 4)
            world = (points @ mat[:3, :3] + mat[3, :3]).astype(np.float32)
            device = str(PhysicsManager._device)
            wp_mesh = wp.Mesh(
                points=wp.array(world, dtype=wp.vec3, device=device),
                indices=wp.array(faces, dtype=wp.int32, device=device),
            )
            heightfield, xform = Heightfield.create_from_mesh(wp_mesh, resolution)
            builder.add_shape_heightfield(heightfield=heightfield, xform=xform)
            logger.info(
                "Converted terrain collider %s (%d faces) to a %dx%d heightfield.",
                prim.GetPath().pathString,
                faces.shape[0] // 3,
                heightfield.nrow,
                heightfield.ncol,
            )
            ignore_paths.append(prim.GetPath().pathString)
        return ignore_paths

    @classmethod
    def instantiate_builder_from_stage(cls):
        """Create builder from USD stage.

        Detects env Xforms (e.g. ``/World/Env_0``, ``/World/Env_1``) and builds
        each as a separate Newton world via ``begin_world``/``end_world``.
        Falls back to a flat ``add_usd`` when no env Xforms are found.

        """
        import re

        from pxr import UsdGeom

        stage = get_current_stage()
        up_axis = UsdGeom.GetStageUpAxis(stage)

        # Scan /World children for env-like Xforms (Env_0, env_1, ...)
        env_pattern = re.compile(r"^[Ee]nv_(\d+)$")
        world_prim = stage.GetPrimAtPath("/World")
        env_paths: list[tuple[int, str]] = []
        if world_prim and world_prim.IsValid():
            for child in world_prim.GetChildren():
                m = env_pattern.match(child.GetName())
                if m:
                    env_paths.append((int(m.group(1)), child.GetPath().pathString))
        env_paths.sort(key=lambda x: x[0])

        builder = cls.create_builder(up_axis=up_axis)

        schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

        # NOTE: None of the add_usd calls below pass joint_ordering or
        # bodies_follow_joint_ordering, so the live articulation's native
        # joint/body order comes from Newton's ModelBuilder.add_usd defaults
        # (joint_ordering="dfs", bodies_follow_joint_ordering=True).
        # isaaclab.assets.articulation.ordering_resolvers hardcodes matching
        # constants to emulate that same order for cross-backend name
        # resolution (see _get_mjwarp_names_from_newton_usd_builder). If
        # ordering arguments are ever passed here, update the resolver
        # constants in lockstep or MJWarp resolution will silently diverge
        # from the live backend.
        hf_ignore_paths = cls._inject_terrain_heightfields(stage, builder)

        if not env_paths:
            # No env Xforms — flat loading
            import_result = builder.add_usd(stage, ignore_paths=hf_ignore_paths, schema_resolvers=schema_resolvers)
            _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
            replace_newton_builder_shape_colors(builder, stage)
            NewtonManager._world_xforms = [wp.transform()]
            for hook in cls._per_world_builder_hooks:
                hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        else:
            # Load everything except the env subtrees (ground plane, lights, etc.)
            # and any terrain colliders already added as heightfields above.
            ignore_paths = [path for _, path in env_paths] + hf_ignore_paths
            import_result = builder.add_usd(stage, ignore_paths=ignore_paths, schema_resolvers=schema_resolvers)
            _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
            replace_newton_builder_shape_colors(builder, stage)

            _, proto_path = env_paths[0]
            source_builders = {proto_path: cls.create_builder(up_axis=up_axis)}
            import_result = source_builders[proto_path].add_usd(
                stage, root_path=proto_path, schema_resolvers=schema_resolvers
            )
            _restore_visible_colliders_without_visual_shapes(
                source_builders[proto_path], stage, import_result["path_shape_map"]
            )
            replace_newton_builder_shape_colors(source_builders[proto_path], stage)
            cls._cl_protos = source_builders

            global_site_indices, source_site_indices, env_root_sites = cls._cl_inject_sites(builder, source_builders)
            xform_cache = UsdGeom.XformCache()
            poses = []
            for _, env_path in env_paths:
                world_xform = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(env_path))
                translation = world_xform.ExtractTranslation()
                rotation = world_xform.ExtractRotationQuat()
                imag = rotation.GetImaginary()
                poses.append(
                    (
                        (translation[0], translation[1], translation[2]),
                        (imag[0], imag[1], imag[2], rotation.GetReal()),
                    )
                )

            positions = torch.tensor([pos for pos, _ in poses], dtype=torch.float32)
            quaternions = torch.tensor([quat for _, quat in poses], dtype=torch.float32)
            mapping = torch.ones((1, len(env_paths)), dtype=torch.bool)
            replicate_args = (builder, (proto_path,), mapping, positions, quaternions, source_builders)
            local_site_map, world_xforms = replicate_builder_mapping(
                *replicate_args,
                source_site_indices=source_site_indices,
                env_root_sites=env_root_sites,
                per_world_builder_hooks=cls._per_world_builder_hooks,
            )

            NewtonManager._cl_site_index_map = {label: (idx, None) for label, idx in global_site_indices.items()}
            NewtonManager._cl_site_index_map.update(
                (label, (None, per_world)) for label, per_world in local_site_map.items()
            )
            NewtonManager._world_xforms = world_xforms
            NewtonManager._num_envs = len(env_paths)

        cls.set_builder(builder)

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Initialize contacts using Newton's :class:`CollisionPipeline`.

        This default implementation handles solvers that rely on Newton's
        unified collision pipeline (XPBD, Featherstone, and MuJoCo with
        ``use_mujoco_contacts=False``).  Solver subclasses with internal
        contact handling (e.g. :class:`NewtonMJWarpManager` when
        ``use_mujoco_contacts=True``) override this method to allocate a
        :class:`Contacts` object sized to the solver's internal contact buffer.
        """
        if not cls._needs_collision_pipeline:
            return
        if cls._collision_pipeline is None:
            if cls._collision_cfg is not None:
                NewtonManager._collision_pipeline = CollisionPipeline(
                    cls._model, **cls._collision_cfg.to_pipeline_args()
                )
            else:
                NewtonManager._collision_pipeline = CollisionPipeline(cls._model, broad_phase="explicit")
        if cls._contacts is None:
            NewtonManager._contacts = cls._collision_pipeline.contacts()

    # ----- Solver construction (subclass contract) ------------------------

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg) -> SolverBase:
        """Construct a solver without changing the active manager state.

        Solver-manager subclasses override this hook so nested consumers can
        reuse their typed construction logic through ``solver_cfg.class_type``.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement solver construction.")

    @classmethod
    @abstractmethod
    def _build_solver(cls, model: Model, solver_cfg) -> None:
        """Construct the solver this manager owns and assign it onto the base class.

        Subclasses must populate the canonical :class:`NewtonManager` slots:

        * :attr:`NewtonManager._solver` — the constructed :class:`SolverBase`
          instance.
        * :attr:`NewtonManager._use_single_state` — ``True`` if the solver
          steps in-place on a single :class:`State` (e.g. MuJoCo); ``False``
          if it needs separate input/output states (e.g. XPBD, Featherstone,
          Kamino).
        * :attr:`NewtonManager._needs_collision_pipeline` — ``True`` if the
          manager owns Newton's :class:`CollisionPipeline` for contact
          generation; ``False`` if the solver runs internal collision
          detection (MuJoCo internal contacts, Kamino with its own detector).

        Writing through ``NewtonManager._foo`` (rather than ``cls._foo``)
        keeps the canonical state visible to external readers regardless of
        which subclass is active.

        Args:
            model: Finalized Newton model the solver should run on.
            solver_cfg: The manager-specific :class:`NewtonSolverCfg`
                subclass (i.e. the inner ``cfg.solver_cfg``, not the outer
                :class:`NewtonCfg`).
        """
        raise NotImplementedError("NewtonManager subclasses must implement _build_solver()")

    @staticmethod
    def _filter_solver_kwargs(solver_cls: type, solver_cfg) -> dict:
        """Return cfg fields that match ``solver_cls.__init__`` parameters.

        Drops keys that the solver constructor doesn't accept (e.g. cfg-only
        metadata like ``solver_type`` / ``class_type``). ``self`` and ``model``
        are always excluded — ``model`` is passed positionally at construction.
        """
        valid = set(inspect.signature(solver_cls.__init__).parameters) - {"self", "model"}
        return {k: v for k, v in solver_cfg.to_dict().items() if k in valid}

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run one solver substep.

        Default invokes :attr:`_solver` once.  Subclasses can override to
        batch multiple solvers within a single substep.
        """
        cls._solver.step(state_0, state_1, control, contacts, substep_dt)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Solver-specific cleanup hook called from :meth:`clear`.

        Default no-op.  Subclasses override to release sub-solver references
        or other solver-specific resources.
        """

    @classmethod
    def _log_solver_debug(cls) -> None:
        """Solver-specific debug logging after stepping.

        Default no-op.  Subclasses override to log solver-specific debug info
        (e.g. constraint violations, contact forces, etc.) after stepping.
        """

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Clear solver-internal state for environments reset since the last boundary.

        The hook runs immediately before reset masks are consumed by :meth:`step`
        and :meth:`forward`. The base implementation delegates to
        :meth:`SolverBase.reset` with ``flags=0``, preserving the joint state
        authored by Isaac Lab while clearing solver-owned buffers. Solvers with
        no reset implementation are unaffected.

        Args:
            world_mask: Per-world reset mask, or ``None`` when no simulation
                state is available.
        """
        if world_mask is None:
            return
        cls._solver.reset(cls._state_0, world_mask=world_mask, flags=0)

    # ----- Lifecycle orchestration ----------------------------------------

    @classmethod
    def initialize_solver(cls) -> None:
        """Initialize the solver and collision pipeline.

        Thin orchestrator: delegates solver construction to
        :meth:`_build_solver` (overridden by each solver subclass), allocates
        the collision pipeline (when applicable) via
        :meth:`_initialize_contacts`, then sets up cubric bindings and either
        captures the CUDA graph immediately or defers capture until the
        first :meth:`step` call (RTX-active path).

        .. warning::
            When using a CUDA-enabled device, the simulation is graphed.
            This means the function steps the simulation once to capture the
            graph, so it should only be called after everything else in the
            simulation is initialized.
        """
        cfg = PhysicsManager._cfg
        if cfg is None:
            return

        with Timer(name="newton_initialize_solver", msg="Initialize solver took:"):
            NewtonManager._num_substeps = cfg.num_substeps  # type: ignore[union-attr]
            NewtonManager._collision_decimation = cfg.collision_decimation  # type: ignore[union-attr]
            NewtonManager._solver_dt = cls.get_physics_dt() / cls._num_substeps
            NewtonManager._collision_cfg = cfg.collision_cfg  # type: ignore[union-attr]

            cls._build_solver(cls._model, cfg.solver_cfg)  # type: ignore[union-attr]
            if NewtonManager._solver is None:
                raise RuntimeError(
                    f"{cls.__name__}._build_solver did not assign NewtonManager._solver. "
                    "Subclasses of NewtonManager must populate NewtonManager._solver, "
                    "NewtonManager._use_single_state, and NewtonManager._needs_collision_pipeline."
                )
            cls._initialize_contacts()

        # Bind the solver-specialized FK delegate to the active subclass's _eval_fk_impl so
        # that forward()/step() dispatch correctly even when forward() is invoked through the
        # base class (the data layer imports NewtonManager directly). ``cls`` is the concrete
        # subclass here, since initialize_solver is reached via sim.physics_manager.reset().
        NewtonManager._eval_fk = cls._eval_fk_impl
        NewtonManager._reset_solver_internals_delegate = cls._reset_solver_internals

        # Establish the initial kinematically-consistent body state through the
        # solver-specialized FK delegate, now that the solver and the delegate both exist.
        # Runs before graph capture below so the capture warmup sees a valid body_q.
        cls._eval_fk(None, None)

        if cls._usdrt_stage is not None:
            cls._setup_cubric_bindings()

        # Skip the initial graph capture when the Newton actuator fast path is
        # active. Capturing here would use ``cls._decimation`` (still its default
        # of 1, because the env's ``set_decimation`` hasn't run yet); a second
        # capture from ``set_decimation`` then triggers an illegal-memory-access
        # CUDA fault inside the captured ``_simulate_full`` graph (back-to-back
        # captures of the contact + actuator pipeline don't survive re-capture
        # — root cause is in Newton's collision/actuator buffer handling, not
        # Lab code). For non-Newton-actuator paths this branch is unaffected:
        # ``set_decimation`` is a no-op for them (``_is_all_graphable`` is False),
        # so we still need the start-time capture below.
        if not cls._use_newton_actuators_active:
            cls._capture_or_defer_graph()

    @classmethod
    def _setup_cubric_bindings(cls) -> None:
        """Initialize cubric ctypes bindings when the Kit viewport is active.

        Adapter creation itself is deferred to the first
        :meth:`sync_transforms_to_usd` call to avoid startup-ordering issues
        with the cubric plugin.
        """
        from isaaclab_newton.physics._cubric import CubricBindings

        bindings = CubricBindings()
        if bindings.initialize():
            NewtonManager._cubric = bindings
            logger.info("cubric bindings ready (adapter deferred to first render)")
        else:
            NewtonManager._cubric = None
            logger.warning("cubric bindings init failed; falling back to update_world_xforms()")

    @classmethod
    def _capture_or_defer_graph(cls) -> None:
        """Capture (or schedule deferred capture of) the CUDA graph.

        Called by :meth:`start_simulation` and :meth:`set_decimation`
        whenever the graph needs to be (re-)captured.

        * **No USDRT / headless**: captures immediately via
          ``wp.ScopedCapture``.
        * **RTX active**: defers capture to the first :meth:`step` call
          via :meth:`_capture_relaxed_graph`, because RTX background
          streams are not yet idle during initialisation.
        * **CUDA graphs disabled**: clears the graph reference.
        """
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        if cfg is None or device is None:
            return

        use_cuda_graph = cfg.use_cuda_graph and "cuda" in device
        if use_cuda_graph and not cls._supports_cuda_graph_capture():
            NewtonManager._graph = None
            NewtonManager._graph_capture_pending = False
            logger.warning(
                "%s does not support CUDA graph capture for the current solver configuration; using eager execution.",
                cls.__name__,
            )
            return

        if use_cuda_graph:
            with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
                if cls._usdrt_stage is None:
                    simulate = cls._simulate_full if cls._is_all_graphable() else cls._simulate_physics_only
                    with _paused_gc(), wp.ScopedCapture(device=device) as capture:
                        simulate()
                    NewtonManager._graph = capture.graph
                    logger.info("Newton CUDA graph captured (standard Warp mode)")

                    # Kamino: StateKamino.from_newton() lazily allocates body_f_total,
                    # joint_q_prev, and joint_lambdas via wp.clone/wp.zeros during the
                    # first step() inside graph capture. Replay once to pin those
                    # memory-pool addresses before any eager solver.reset() call.
                    if isinstance(cls._solver, SolverKamino):
                        wp.capture_launch(cls._graph)
                else:
                    # RTX is active during initialization — cudaImportExternalMemory and other
                    # non-capturable RTX ops run on background CUDA streams right now.
                    # Defer capture to the first step() call, after RTX is fully initialized
                    # and idle between render frames (clean capture window).
                    NewtonManager._graph = None
                    NewtonManager._graph_capture_pending = True
                    logger.info("Newton CUDA graph capture deferred until first step() (RTX active)")
        else:
            NewtonManager._graph = None

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Return whether the active solver configuration supports CUDA graph capture."""
        return True

    @classmethod
    def _capture_relaxed_graph(cls, device: str, capture_target: Callable[[], None] | None = None):
        """Capture Newton physics (only) as a CUDA graph, RTX-compatible.

        Uses a hybrid approach to work around two conflicting requirements:

        1. RTX background threads use CUDA's legacy stream (stream 0) for async operations
           like ``cudaImportExternalMemory``.  A standard ``wp.ScopedCapture()`` uses
           ``cudaStreamCaptureModeThreadLocal`` on Warp's default stream (a blocking stream).
           A blocking stream synchronises implicitly with legacy stream 0, so RTX ops inside
           the capture window fail with error 906.

        2. ``mujoco_warp`` calls ``wp.capture_while`` inside ``solver.solve()``.
           ``wp.capture_while`` checks ``device.captures`` (populated by ``wp.capture_begin``)
           to decide whether to insert a conditional graph node (graph-capture path) or to run
           eagerly with ``wp.synchronize_stream`` (non-capture path).  Without an entry in
           ``device.captures``, it synchronises the capturing stream — which raises "Cannot
           synchronize stream while graph capture is active".

        Solution:

        - Create a **non-blocking** stream (``cudaStreamNonBlocking = 0x01``): no implicit sync
          with legacy stream 0, so RTX background threads are unaffected (avoids error 906).
        - Start the capture externally via ``cudaStreamBeginCapture`` with
          ``cudaStreamCaptureModeRelaxed`` so no other CUDA activity is disrupted.
        - Call ``wp.capture_begin(external=True, stream=fresh_stream)``:
          this registers the capture in Warp's ``device.captures`` *without* calling
          ``cudaStreamBeginCapture`` (already done) and *without* changing device-wide memory
          pool attributes (avoids error 900 in RTX's ``cudaMallocAsync``).
        - Run the simulate function inside ``ScopedStream(fresh_stream)``:
          kernels dispatch to ``fresh_stream`` and are captured; ``wp.capture_while`` finds the
          active capture and inserts a conditional graph node instead of synchronising.
        - Call ``wp.capture_end(stream=fresh_stream)`` to finalise the Warp-level capture.
        - Call ``cudaStreamEndCapture`` to close the CUDA stream capture and get the graph.

        Warmup run pre-allocates all solver scratch buffers so no ``cudaMalloc`` occurs during
        capture.  ``sync_transforms_to_usd`` (which calls ``wp.synchronize_device``) is
        excluded from the capture and runs eagerly in ``step()`` after ``wp.capture_launch``.

        When ``capture_target`` is provided it is captured instead of the physics simulate
        function (used for secondary graphs such as the sensor manager graph).

        Returns a ``wp.Graph`` on success, or ``None`` on failure.
        """
        if _cudart is None:
            logger.warning("libcudart not available; cannot use relaxed graph capture")
            return None

        # Warmup: pre-allocate all solver scratch buffers so the capture window has
        # no new cudaMalloc calls (which are forbidden inside graph capture).
        if capture_target is not None:
            simulate = capture_target
        else:
            simulate = cls._simulate_full if cls._is_all_graphable() else cls._simulate_physics_only
        with wp.ScopedDevice(device):
            simulate()
        wp.synchronize_stream(wp.get_stream(device))

        # Create a non-blocking stream (cudaStreamNonBlocking = 0x01).
        raw_handle = ctypes.c_void_p()
        ret = _cudart.cudaStreamCreateWithFlags(ctypes.byref(raw_handle), ctypes.c_uint(0x01))
        if ret != 0:
            logger.warning("cudaStreamCreateWithFlags(NonBlocking) failed (code %d)", ret)
            return None
        fresh_handle = raw_handle.value
        fresh_stream = wp.Stream(device, cuda_stream=fresh_handle, owner=False)

        with _paused_gc():
            # Start capture in relaxed mode BEFORE entering ScopedStream.
            ret = _cudart.cudaStreamBeginCapture(ctypes.c_void_p(fresh_handle), ctypes.c_int(2))
            if ret != 0:
                _cudart.cudaStreamDestroy(ctypes.c_void_p(fresh_handle))
                logger.warning("cudaStreamBeginCapture(relaxed) failed (code %d)", ret)
                return None

            try:
                wp.capture_begin(stream=fresh_stream, external=True)
            except Exception as exc:
                raw_graph = ctypes.c_void_p()
                _cudart.cudaStreamEndCapture(ctypes.c_void_p(fresh_handle), ctypes.byref(raw_graph))
                if raw_graph.value:
                    _cudart.cudaGraphDestroy(raw_graph)
                _cudart.cudaStreamDestroy(ctypes.c_void_p(fresh_handle))
                logger.warning("wp.capture_begin(external=True) failed: %s", exc)
                return None

            err_during_capture = None
            with wp.ScopedStream(fresh_stream, sync_enter=False):
                try:
                    simulate()
                except Exception as exc:
                    err_during_capture = exc

            if err_during_capture is None:
                try:
                    graph = wp.capture_end(stream=fresh_stream)
                except Exception as exc:
                    err_during_capture = exc
                    graph = None
            else:
                with contextlib.suppress(Exception):
                    wp.capture_end(stream=fresh_stream)
                graph = None

            raw_graph = ctypes.c_void_p()
            end_ret = _cudart.cudaStreamEndCapture(ctypes.c_void_p(fresh_handle), ctypes.byref(raw_graph))
            _cudart.cudaStreamDestroy(ctypes.c_void_p(fresh_handle))

        if err_during_capture is not None:
            if raw_graph.value:
                _cudart.cudaGraphDestroy(raw_graph)
            logger.warning("Newton graph capture aborted during simulate: %s", err_during_capture)
            return None

        if end_ret != 0 or not raw_graph.value:
            logger.warning("cudaStreamEndCapture failed (code %d)", end_ret)
            return None

        # Patch the Warp Graph object with the raw CUDA graph handle obtained
        # from our external cudaStreamEndCapture.  wp.capture_end(external=True)
        # returns a Graph with a stale handle; we overwrite it so that
        # wp.capture_launch() replays the correct graph.
        # NOTE: This relies on Warp internals (Graph.graph / Graph.graph_exec).
        # Setting graph_exec = None triggers lazy cudaGraphInstantiate on
        # the next capture_launch.  Replace with public API when available.
        graph.graph = raw_graph
        graph.graph_exec = None
        return graph

    # ------------------------------------------------------------------
    # Building blocks — used by _simulate_full / _simulate_physics_only
    # ------------------------------------------------------------------

    @classmethod
    def _run_solver_substeps(cls, contacts) -> None:
        """Run ``num_substeps`` solver iterations, handling double-buffered state swap."""
        collide_every = cls._collision_decimation
        # Last substep is skipped: its contact set would only feed the next tick's
        # top-of-loop collide(), not this one.
        collide_mid_loop = collide_every > 0 and cls._needs_collision_pipeline and contacts is not None

        if cls._use_single_state:
            for i in range(cls._num_substeps):
                cls._step_solver(cls._state_0, cls._state_0, cls._control, contacts, cls._solver_dt)
                cls._state_0.clear_forces()
                if collide_mid_loop and (i + 1) % collide_every == 0 and i + 1 < cls._num_substeps:
                    cls._collision_pipeline.collide(cls._state_0, contacts)
        else:
            cfg = PhysicsManager._cfg
            need_copy_on_last = cfg is not None and cls._num_substeps % 2 == 1
            for i in range(cls._num_substeps):
                cls._step_solver(cls._state_0, cls._state_1, cls._control, contacts, cls._solver_dt)
                if need_copy_on_last and i == cls._num_substeps - 1:
                    cls._state_0.assign(cls._state_1)
                else:
                    NewtonManager._state_0, NewtonManager._state_1 = cls._state_1, cls._state_0
                cls._state_0.clear_forces()
                if collide_mid_loop and (i + 1) % collide_every == 0 and i + 1 < cls._num_substeps:
                    cls._collision_pipeline.collide(cls._state_0, contacts)

    @classmethod
    def _update_sensors(cls, contacts) -> None:
        """Push latest state to all registered Newton sensors."""
        if cls._newton_frame_transform_sensors:
            for sensor in cls._newton_frame_transform_sensors:
                sensor.update(cls._state_0)
        if cls._newton_imu_sensors:
            for sensor in cls._newton_imu_sensors:
                sensor.update(cls._state_0)
        if cls._report_contacts:
            eval_contacts = contacts if contacts is not None else cls._contacts
            cls._solver.update_contacts(eval_contacts, cls._state_0)
            for sensor in cls._newton_contact_sensors.values():
                sensor.update(cls._state_0, eval_contacts)

    # ------------------------------------------------------------------
    # Composite stepping routines
    # ------------------------------------------------------------------

    @classmethod
    def _simulate_full(cls) -> None:
        """Run ``decimation x (actuators + solver substeps)``, then sensors.

        Works for any decimation count (including 1).  All actuators must be
        graph-safe so the entire loop can be captured as a single CUDA graph.
        """
        physics_dt = cls._solver_dt * cls._num_substeps
        contacts = cls._contacts if cls._needs_collision_pipeline else None

        for _ in range(cls._decimation):
            if cls._needs_collision_pipeline:
                cls._collision_pipeline.collide(cls._state_0, cls._contacts)

            if cls._adapter is not None:
                cls._adapter.step(cls._state_0, cls._control, physics_dt)
            for cb in cls._post_actuator_callbacks:
                cb()

            cls._run_solver_substeps(contacts)

        for cb in cls._post_step_callbacks:
            cb()
        cls._update_sensors(contacts)

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Collision + solver substeps + sensors (no actuators, no USD sync).

        Used when actuators are stepped eagerly outside the graph, or when
        there are no actuators at all.
        """
        if cls._needs_collision_pipeline:
            cls._collision_pipeline.collide(cls._state_0, cls._contacts)
            contacts = cls._contacts
        else:
            contacts = None

        cls._run_solver_substeps(contacts)
        for cb in cls._post_step_callbacks:
            cb()
        cls._update_sensors(contacts)

    # State accessors (used extensively by articulation/rigid object data)
    @classmethod
    def get_model(cls) -> Model:
        """Get the Newton model.

        When the active sim backend is Newton this returns the manager's own
        authoritative model. When the active sim backend is PhysX a shadow
        Newton model is built lazily (from the visualizer prebuilt artifact) so
        renderers/visualizers that operate on Newton ``Model`` and ``State`` can
        still drive a PhysX-simulated scene.
        """
        cls._ensure_visualization_model()
        return cls._model

    @classmethod
    def get_state_0(cls) -> State:
        """Get the current state."""
        cls._ensure_visualization_model()
        return cls._state_0

    @classmethod
    def get_state(cls, scene_data_provider: SceneDataProvider | None = None) -> State:
        """Get the current Newton state for visualization.

        Use this method from visualizers/renderers/video recorders that need a
        backend-agnostic Newton ``State``. When the sim backend is PhysX this
        refreshes the shadow ``_state_0.body_q`` from the live PhysX scene via
        :meth:`update_visualization_state` before returning, so callers never
        observe stale transforms. Under the Newton sim backend, pending
        forward kinematics is applied before returning the live state.
        """
        if cls._fk_reset_mask is not None and cls._backend_is_newton(scene_data_provider):
            cls.forward()
        cls.update_visualization_state(scene_data_provider)
        return cls.get_state_0()

    @classmethod
    def get_contacts(cls) -> Contacts | None:
        """Get the current Newton contact buffer, if the active solver exposes one."""
        return cls._contacts

    @classmethod
    def _register_sensor_task(
        cls, name: str, update_fn: Callable[[], None], *, include_collision_shapes: bool = False
    ) -> None:
        """Register a graph-capturable scene-query task.

        Args:
            name: Unique task name.
            update_fn: Graph-capturable callable run by :meth:`_update_sensor_tasks`.
            include_collision_shapes: Whether the task must see collision-only
                geometry. Newton builds the shape BVH over visible shapes, which is
                what renderers want; the first ray-cast sensor rebuilds it with
                collision shapes added, since those must be hit even when they carry
                no visual representation.
        """
        if name in cls._sensor_tasks:
            raise ValueError(f"Newton sensor task '{name}' is already registered.")
        model = cls.get_model()
        state = cls.get_state_0()
        if model is None or state is None:
            raise RuntimeError("Registering a Newton sensor task requires an initialized model and state.")
        if model.shape_count > 0:
            if include_collision_shapes and not cls._sensor_bvh_has_collision_shapes:
                model.bvh_build_shapes(state, shape_flags=ShapeFlags.VISIBLE | ShapeFlags.COLLIDE_SHAPES)
                NewtonManager._sensor_bvh_has_collision_shapes = True
            elif model.bvh_shapes is None:
                model.bvh_build_shapes(state)
        if model.particle_count > 0 and model.bvh_particles is None:
            model.bvh_build_particles(state)
        cls._sensor_tasks[name] = update_fn
        cls._sensor_state = state
        cls._sensor_state_dirty = True
        cls._invalidate_sensor_graph()

    @classmethod
    def _unregister_sensor_task(cls, name: str) -> None:
        """Remove a scene-query task, ignoring unknown names."""
        if cls._sensor_tasks.pop(name, None) is not None:
            cls._invalidate_sensor_graph()

    @classmethod
    def _update_sensor_tasks(cls, *names: str) -> None:
        """Refit the shape and particle BVHs and run the requested scene-query tasks."""
        for name in names:
            if name not in cls._sensor_tasks:
                raise KeyError(f"Newton sensor task '{name}' is not registered.")

        state = cls.get_state_0()
        if state is not cls._sensor_state:
            cls._sensor_state = state
            cls._sensor_state_dirty = True
            cls._invalidate_sensor_graph()
        cfg = PhysicsManager._cfg
        use_cuda_graph = bool(getattr(cfg, "use_cuda_graph", False)) and "cuda" in str(PhysicsManager._device)
        if use_cuda_graph and cls._sensor_graph is None and not cls._sensor_graph_capture_failed:
            cls._capture_sensor_graph()
        if cls._sensor_graph is None:
            if cls._sensor_state_dirty:
                cls._refit_sensor_bvh()
                cls._sensor_state_dirty = False
            for name in names:
                cls._sensor_tasks[name]()
            return

        assert cls._sensor_flags_host is not None
        assert cls._sensor_flags is not None
        cls._sensor_flags_host.fill(0)
        cls._sensor_flags_host[0] = int(cls._sensor_state_dirty)
        task_names = tuple(cls._sensor_tasks)
        for name in names:
            cls._sensor_flags_host[1 + task_names.index(name)] = 1
        cls._sensor_flags.assign(cls._sensor_flags_host)
        wp.capture_launch(cls._sensor_graph)
        cls._sensor_state_dirty = False

    @classmethod
    def _mark_sensor_state_dirty(cls) -> None:
        """Bind the current state and mark the shape and particle BVHs stale.

        Writes through :class:`NewtonManager` rather than ``cls`` because the
        sensor-task registry and its dirty flag are singleton state owned by the
        base class (sensors and renderers reach it via ``NewtonManager``). This
        method is invoked from the step loop where ``cls`` is the active solver
        subclass, so assigning through ``cls`` would shadow the base attribute
        and the per-step refit request would never reach the sensor tasks.
        """
        if NewtonManager._state_0 is None:
            return
        if NewtonManager._state_0 is not NewtonManager._sensor_state:
            NewtonManager._sensor_state = NewtonManager._state_0
            NewtonManager._invalidate_sensor_graph()
        NewtonManager._sensor_state_dirty = True

    @classmethod
    def _refit_sensor_bvh(cls) -> None:
        """Refit the model shape and particle BVHs against the current state."""
        if cls._model is None:
            return

        refit_shapes = cls._model.shape_count > 0 and cls._model.bvh_shapes is not None
        refit_particles = cls._model.particle_count > 0 and cls._model.bvh_particles is not None
        if not refit_shapes and not refit_particles:
            return

        if cls._sensor_state is None:
            raise RuntimeError("Refitting Newton sensor BVHs requires an initialized sensor state.")

        if refit_shapes:
            cls._model.bvh_refit_shapes(cls._sensor_state)

        if refit_particles:
            cls._model.bvh_refit_particles(cls._sensor_state)

    @classmethod
    def _invalidate_sensor_graph(cls) -> None:
        """Discard captured scene-query graph resources."""
        cls._sensor_graph = None
        cls._sensor_flags = None
        cls._sensor_flags_host = None
        cls._sensor_graph_capture_failed = False

    @classmethod
    def _capture_sensor_graph(cls) -> None:
        """Capture BVH refit and scene-query tasks into a conditional graph."""
        with wp.ScopedDevice(PhysicsManager._device):
            cls._refit_sensor_bvh()
            for update_fn in cls._sensor_tasks.values():
                update_fn()

        cls._sensor_flags = wp.zeros(1 + len(cls._sensor_tasks), dtype=wp.int32, device=PhysicsManager._device)
        cls._sensor_flags_host = np.zeros(1 + len(cls._sensor_tasks), dtype=np.int32)
        update_fns = tuple(cls._sensor_tasks.values())

        def pipeline() -> None:
            assert cls._sensor_flags is not None
            wp.capture_if(cls._sensor_flags[0:1], cls._refit_sensor_bvh)
            for index, update_fn in enumerate(update_fns):
                wp.capture_if(cls._sensor_flags[index + 1 : index + 2], update_fn)

        device = PhysicsManager._device
        if cls._usdrt_stage is not None:
            cls._sensor_graph = cls._capture_relaxed_graph(device, capture_target=pipeline)
        else:
            try:
                with wp.ScopedCapture(device=device) as capture:
                    pipeline()
                cls._sensor_graph = capture.graph
            except Exception:
                logger.exception("[NewtonManager] sensor CUDA graph capture failed")
                cls._sensor_graph = None
        if cls._sensor_graph is None:
            cls._sensor_flags = None
            cls._sensor_flags_host = None
            cls._sensor_graph_capture_failed = True
            logger.warning("Newton sensor graph capture failed; falling back to eager execution.")
        else:
            logger.info("Captured Newton sensor graph with %d task(s).", len(cls._sensor_tasks))

    @classmethod
    def get_num_envs(cls) -> int:
        return cls._num_envs

    @classmethod
    def _backend_is_newton(cls, scene_data_provider: SceneDataProvider | None = None) -> bool:
        """Return ``True`` when the active sim backend is Newton."""
        if scene_data_provider is not None:
            return isinstance(scene_data_provider.backend, NewtonSceneDataBackend)
        return isinstance(cls.get_scene_data_provider().backend, NewtonSceneDataBackend)

    @classmethod
    def _ensure_visualization_model(cls) -> None:
        """Build a shadow Newton model from the USD stage when the sim backend is PhysX.

        No-op when the sim backend is Newton (the manager's own ``_model`` /
        ``_state_0`` are authoritative) or when a shadow model has already been
        built. This is the entry point that makes :meth:`get_model` /
        :meth:`get_state` work uniformly across both sim backends.

        The shadow model is built by walking the USD stage and finalizing the resulting
        :class:`~newton.ModelBuilder`. Per-frame body transforms are pushed into
        ``_state_0.body_q`` by :meth:`update_visualization_state` using the new
        :class:`~isaaclab.scene_data.SceneDataProvider`.
        """

        if cls._model is not None and cls._state_0 is not None:
            return

        if cls._backend_is_newton():
            return

        stage = get_current_stage()
        if stage is None:
            logger.error(
                "[NewtonManager] No USD stage available; cannot build a Newton "
                "Model/State for visualization while the sim backend is PhysX."
            )
            return

        up_axis_token = UsdGeom.GetStageUpAxis(stage)
        up_axis = Axis.from_string(str(up_axis_token))

        env_paths = []
        envs_prim = stage.GetPrimAtPath("/World/envs")
        if envs_prim.IsValid():
            env_pattern = re.compile(r"^env_(\d+)$")
            env_paths = sorted(
                (int(match.group(1)), child.GetPath().pathString)
                for child in envs_prim.GetChildren()
                if (match := env_pattern.match(child.GetName()))
            )

        sim = SimulationContext.instance()
        assert sim is not None
        clone_plan = sim.get_clone_plan()
        if clone_plan is not None and not env_paths:
            logger.warning(
                "[NewtonManager] Clone plan is available but no environment prims were found; "
                "deferring visualization model creation."
            )
            return
        NewtonManager._num_envs = len(env_paths) if clone_plan is not None else 1
        builder, (shadow_entities, registry_groups) = build_visualization_builder_from_stage_envs(
            stage, env_paths, clone_plan, up_axis=up_axis, device=str(PhysicsManager._device or "cpu")
        )
        NewtonManager._shadow_deformable_entities = shadow_entities
        NewtonManager._scene_data_geometry_mapping = None
        NewtonManager._mapped_sim_particle_offsets = None
        cls._invalidate_shadow_deformable_batch_sync()
        sim_particle_total = sum(entity.sim_particle_count for entity in shadow_entities)
        if sim_particle_total > 0:
            device = PhysicsManager._device or "cpu"
            NewtonManager._sim_particle_q = wp.zeros(sim_particle_total, dtype=wp.vec3f, device=device)
        else:
            NewtonManager._sim_particle_q = None

        particle_count = getattr(builder, "particle_count", 0)
        if builder.body_count == 0 and particle_count == 0:
            if clone_plan is not None or env_paths:
                logger.error(
                    "[NewtonManager] USD stage walk produced no Newton bodies or particles; the shadow "
                    "Newton model for visualization will be empty. Common causes: the cloned "
                    "envs are not yet on the stage, or PhysX schemas could not be parsed by "
                    "Newton's add_usd. Check that /World/envs/env_<id> prims exist when the "
                    "renderer is initialized."
                )
                return
            logger.info(
                "[NewtonManager] USD stage walk produced no Newton bodies or particles; "
                "finalizing an empty visualization model."
            )
        elif builder.body_count == 0:
            log = logger.warning if clone_plan is not None or env_paths else logger.info
            log("[NewtonManager] USD stage walk produced no Newton bodies; finalizing an empty visualization model.")

        device = PhysicsManager._device or "cpu"
        try:
            NewtonManager._model = builder.finalize(device=device)
            NewtonManager._state_0 = cls._model.state()
            cls._model.num_envs = cls._num_envs
            NewtonManager._deformable_registry = []
            populate_shadow_deformable_registry(cls, registry_groups)

        except Exception:
            logger.exception(
                "[NewtonManager] Failed to finalize the shadow Newton ModelBuilder for "
                "visualization (sim backend is PhysX)."
            )
            NewtonManager._model = None
            NewtonManager._state_0 = None

    @classmethod
    def get_scene_data_provider(cls) -> SceneDataProvider:
        """Return the active scene data provider."""
        sim = SimulationContext.instance()
        assert sim is not None
        return sim.get_scene_data_provider()

    @classmethod
    def update_visualization_state(cls, scene_data_provider: SceneDataProvider | None = None) -> None:
        """Refresh visualization state for the active sim backend.

        Newton sim backend: no-op — ``_state_0`` is the live, authoritative state
        already advanced by :meth:`step` / forward kinematics.

        PhysX / OVPhysX sim backend: pull rigid-body transforms and deformable
        nodal positions from the :class:`~isaaclab.scene_data.SceneDataProvider`
        and write them into the shadow ``_state_0.body_q`` / ``particle_q`` so
        Newton-native consumers (Newton renderer, Newton/Rerun/Viser visualizers,
        OVRTX renderer, Newton GL video) see fresh poses and mesh points.

        Calls use ``allow_passthrough=False`` so identity mappings still copy into
        the pre-bound shadow buffers. Passthrough would rebind the temporary
        :class:`~isaaclab.scene_data.SceneDataFormat` fields away from
        ``_state_0``, leaving OVRTX and other ``get_state()`` consumers on stale
        rest-pose particle / body state.

        Invoked lazily from :meth:`get_state` so consumers do not need to
        coordinate the sync explicitly.
        """

        if scene_data_provider is None:
            scene_data_provider = cls.get_scene_data_provider()

        assert scene_data_provider is not None

        if cls._backend_is_newton(scene_data_provider):
            return

        cls._ensure_visualization_model()

        if cls._state_0 is None or cls._model is None:
            return

        if cls._state_0.body_q is not None:
            if cls._scene_data is None:
                cls._scene_data = SceneDataFormat.Transform()

            # Invalidate stale mapping when the model's body count changed (e.g. tiled → viewport
            # test within the same process where _model was rebuilt from a different stage).
            if cls._scene_data_mapping is not None and cls._scene_data_mapping.shape[0] != cls._model.body_count:
                cls._scene_data_mapping = None

            if cls._scene_data_mapping is None:
                body_labels = list(cls._model.body_label)
                body_paths = cls._resolve_scene_data_body_paths(body_labels, scene_data_provider.usd_stage)
                cls._scene_data_mapping = scene_data_provider.create_mapping(body_paths)

            cls._scene_data.transforms = cls._state_0.body_q
            scene_data_provider.get_transforms(
                cls._scene_data, mapping=cls._scene_data_mapping, allow_passthrough=False
            )

        if cls._state_0.particle_q is not None and scene_data_provider.point_count > 0:
            if cls._scene_data_points is None:
                cls._scene_data_points = SceneDataFormat.Points()

            # Recreate when ``None``. Identity layouts return ``None`` from
            # :meth:`create_geometry_mapping`, so this also refreshes mapped-offset and
            # batch-sync metadata each frame. Caching ``None`` via a ready-flag without
            # that refresh regresses Franka soft viz (stretched / missing deformables).
            if cls._scene_data_geometry_mapping is None and cls._shadow_deformable_entities:
                geometry_paths = [entity.root_path for entity in cls._shadow_deformable_entities]
                geometry_offsets = [entity.sim_particle_offset for entity in cls._shadow_deformable_entities]
                cls._scene_data_geometry_mapping = scene_data_provider.create_geometry_mapping(
                    geometry_paths, geometry_offsets
                )

                # Invalidate the mapped-offset cache so that it can be rebuilt immediately after.
                cls._mapped_sim_particle_offsets = None
                cls._invalidate_shadow_deformable_batch_sync()

            if cls._mapped_sim_particle_offsets is None:
                cls._mapped_sim_particle_offsets = cls._geometry_mapped_sim_offsets(scene_data_provider)

            if cls._sim_particle_q is None:
                sim_total = sum(entity.sim_particle_count for entity in cls._shadow_deformable_entities or [])
                if sim_total > 0:
                    device = PhysicsManager._device or "cpu"
                    cls._sim_particle_q = wp.zeros(sim_total, dtype=wp.vec3f, device=device)

            if cls._sim_particle_q is not None:
                cls._scene_data_points.points = cls._sim_particle_q
                scene_data_provider.get_points(
                    cls._scene_data_points,
                    mapping=cls._scene_data_geometry_mapping,
                    allow_passthrough=False,
                )
                cls._sync_render_particle_q_from_sim()
            else:
                cls._scene_data_points.points = cls._state_0.particle_q
                scene_data_provider.get_points(
                    cls._scene_data_points,
                    mapping=cls._scene_data_geometry_mapping,
                    allow_passthrough=False,
                )

        cls._mark_sensor_state_dirty()

    @classmethod
    def _geometry_mapped_sim_offsets(cls, scene_data_provider: SceneDataProvider) -> set[int]:
        """Return ``_sim_particle_q`` offsets filled by :meth:`get_points` for the cached mapping.

        Called once when the geometry mapping is created (or when that cache is
        invalidated). ``create_geometry_mapping`` indexes by backend entity and stores
        consumer destinations (or ``-1`` when a backend path has no consumer). The
        inverse case — a shadow entity whose path the backend never reports — never
        appears in that array, so its sim slice stays at the zero initialization.
        """
        mapping = cls._scene_data_geometry_mapping
        if mapping is not None:
            return {int(value) for value in mapping.numpy() if int(value) >= 0}

        # Identity layout: backend entities land at sequential flat offsets.
        offsets: set[int] = set()
        flat_offset = 0
        for count in scene_data_provider.backend.geometry_counts:
            offsets.add(flat_offset)
            flat_offset += int(count)
        return offsets

    @classmethod
    def _invalidate_shadow_deformable_batch_sync(cls) -> None:
        """Drop cached batched remap/copy metadata."""
        cls._shadow_deformable_remap_batches = None
        cls._shadow_deformable_copy_batch = None
        cls._shadow_deformable_batch_sync_key = None

    @classmethod
    def _ensure_shadow_deformable_batch_sync(cls) -> None:
        """Build batched remap/copy launch metadata for mapped shadow deformables."""
        entities = cls._shadow_deformable_entities or []
        mapped_offsets = cls._mapped_sim_particle_offsets
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
        if cls._shadow_deformable_batch_sync_key == sync_key:
            return

        cls._shadow_deformable_batch_sync_key = sync_key
        cls._shadow_deformable_remap_batches = []
        cls._shadow_deformable_copy_batch = None

        if cls._sim_particle_q is None or not entities:
            return

        device = str(cls._sim_particle_q.device)
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
            cls._shadow_deformable_copy_batch = (
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

        cls._shadow_deformable_remap_batches = remap_batches

    @classmethod
    def _sync_render_particle_q_from_sim(cls) -> None:
        """Copy or remap sim nodal positions into shadow ``particle_q`` render slots."""
        if cls._state_0 is None or cls._state_0.particle_q is None or cls._sim_particle_q is None:
            return
        if not cls._shadow_deformable_entities:
            return

        mapped_offsets = cls._mapped_sim_particle_offsets
        for entity in cls._shadow_deformable_entities:
            if mapped_offsets is not None and entity.sim_particle_offset not in mapped_offsets:
                warned = cls._shadow_deformable_sync_skip_warned
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
                warned = cls._shadow_deformable_sync_skip_warned
                if entity.root_path not in warned:
                    warned.add(entity.root_path)
                    logger.warning(
                        "Skipping particle sync for deformable '%s': vis_count=%d != sim_count=%d "
                        "and no volume remapping table is available; render slots stay at rest pose.",
                        entity.root_path,
                        entity.vis_particle_count,
                        entity.sim_particle_count,
                    )

        cls._ensure_shadow_deformable_batch_sync()

        for (
            entity_ids,
            sim_offsets,
            render_offsets,
            vis_counts,
            vis_prefix,
            remap,
        ) in cls._shadow_deformable_remap_batches or []:
            launch_batch_volume_vis_remap(
                cls._sim_particle_q,
                cls._state_0.particle_q,
                entity_ids,
                sim_offsets,
                render_offsets,
                vis_counts,
                vis_prefix,
                remap.tet_vertex_indices,
                remap.bary_weights,
            )

        copy_batch = cls._shadow_deformable_copy_batch
        if copy_batch is not None:
            entity_ids, src_offsets, dst_offsets, counts, count_prefix = copy_batch
            launch_batch_particle_slice_copy(
                cls._sim_particle_q,
                cls._state_0.particle_q,
                entity_ids,
                src_offsets,
                dst_offsets,
                counts,
                count_prefix,
            )

    @staticmethod
    def _resolve_scene_data_body_paths(body_paths: list[str | None], stage) -> list[str | None]:
        """Map Newton joint labels to their target rigid-body prim paths."""
        if stage is None:
            return body_paths

        from pxr import UsdPhysics

        def _joint_body_path(prim):
            joint = UsdPhysics.Joint(prim)
            for rel in (joint.GetBody1Rel(), joint.GetBody0Rel()):
                for target_path in rel.GetTargets():
                    target_prim = stage.GetPrimAtPath(target_path)
                    if target_prim.IsValid() and target_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        return target_path.pathString
            return None

        resolved_paths = body_paths.copy()
        for index, body_path in enumerate(body_paths):
            if body_path is None:
                continue
            prim = stage.GetPrimAtPath(body_path)
            if prim.IsValid() and prim.IsA(UsdPhysics.Joint):
                resolved_paths[index] = _joint_body_path(prim) or body_path
        return resolved_paths

    @classmethod
    def get_state_1(cls) -> State:
        """Get the next state."""
        return cls._state_1

    @classmethod
    def get_control(cls) -> Control:
        """Get the control object."""
        return cls._control

    @classmethod
    def get_dt(cls) -> float:
        """Get the physics timestep. Alias for get_physics_dt()."""
        return cls.get_physics_dt()

    @classmethod
    def get_solver_dt(cls) -> float:
        """Get the solver substep timestep."""
        return cls._solver_dt

    @classmethod
    def _is_all_graphable(cls) -> bool:
        """``True`` when the decimation loop can be captured into a CUDA graph.

        Requires:
          1. An articulation took the ``use_newton_actuators=True`` branch
             (signalled via :meth:`activate_newton_actuator_path`).
          2. Either no actuator adapter was needed (all-implicit) or every
             actuator in the adapter is CUDA-graph-safe.
        """
        if not cls._use_newton_actuators_active:
            return False
        return cls._adapter is None or cls._adapter.is_all_graphable

    @classmethod
    def activate_newton_actuator_path(cls) -> None:
        """Opt an articulation into the Newton actuator fast path.

        Idempotent — called by every Newton-fast-path articulation's
        ``_process_actuators_cfg``:

        1. Sets :attr:`_use_newton_actuators_active`, which
           :meth:`_is_all_graphable` checks (adapter presence alone
           cannot distinguish the fast path from the standard Lab path).
        2. On first call, builds the single sim-level
           :class:`NewtonActuatorAdapter` over the full flat DOF layout;
           later calls reuse it.
        """
        # Shared state lives on the base class so all readers (including
        # framework code that imports ``NewtonManager`` directly) see the
        # same flag regardless of which solver subclass is active.
        NewtonManager._use_newton_actuators_active = True

        if cls._adapter is not None:
            return
        if cls._model is None or not cls._model.actuators:
            return
        from isaaclab_newton.actuators import NewtonActuatorAdapter  # noqa: PLC0415

        dofs_per_env = cls._model.joint_dof_count // cls._num_envs
        NewtonManager._adapter = NewtonActuatorAdapter(
            actuators=list(cls._model.actuators),
            num_envs=cls._num_envs,
            num_joints=dofs_per_env,
            dof_offset=0,
            device=PhysicsManager._device,
        )
        cls._adapter.finalize(cls._control)

    @classmethod
    def register_post_actuator_callback(cls, callback: Callable[[], None]) -> None:
        """Append a hook to the list invoked after the actuator step on every iteration.

        Each callback runs inside the captured CUDA graph (when
        :meth:`_is_all_graphable` is ``True``) right after
        :meth:`NewtonActuatorAdapter.step` and before the solver substeps,
        so kernel writes to ``state``/``control`` are visible to the
        integrator on the same iteration. Multiple articulations register
        their own implicit-DOF telemetry / FF-routing kernels here; all
        registered callbacks fire in registration order each step.
        """
        cls._post_actuator_callbacks.append(callback)

    @classmethod
    def register_post_step_callback(cls, callback: Callable[[], None]) -> None:
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
        """
        cls._post_step_callbacks.append(callback)

    @classmethod
    def unregister_post_step_callback(cls, callback: Callable[[], None]) -> None:
        """Remove a previously registered post-step callback.

        Symmetric to :meth:`register_post_step_callback`, this lets an
        articulation deregister its republish hook when its callbacks are
        cleared so the bound method does not linger on the class-level list
        after the articulation is gone. Removing a callback that was never
        registered (or was already removed) is a safe no-op, matching the
        tolerant deregistration of other handles.
        """
        with contextlib.suppress(ValueError):
            cls._post_step_callbacks.remove(callback)

    @classmethod
    def set_decimation(cls, decimation: int) -> None:
        """Set the decimation count and re-capture the CUDA graph.

        When all actuators are graphable the entire decimation loop
        (actuators + solver substeps, repeated *decimation* times)
        is captured as a single CUDA graph.

        If a CUDA graph was previously captured, it is automatically
        re-captured with the new decimation count using the same
        strategy as :meth:`start_simulation`: standard
        ``wp.ScopedCapture`` when no USDRT stage is active, or
        deferred relaxed capture when RTX is running.
        """
        cls._decimation = max(1, decimation)
        if cls._is_all_graphable():
            cls._capture_or_defer_graph()

    @classmethod
    def handles_decimation(cls) -> bool:
        """``True`` when :meth:`step` executes the full decimation loop internally.

        This is the case when all Newton actuators are CUDA-graph-safe.
        The full decimation loop (including the trivial ``decimation=1`` case)
        is folded into a single :meth:`step` call.
        """
        return cls._is_all_graphable()

    @classmethod
    def add_contact_sensor(
        cls,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        verbose: bool = False,
    ) -> tuple[str | list[str] | None, str | list[str] | None, str | list[str] | None, str | list[str] | None]:
        """Add a contact sensor for reporting contacts between bodies/shapes.

        Converts Isaac Lab pattern conventions (``.*`` regex, full USD paths) to
        fnmatch globs and delegates to :class:`newton.sensors.SensorContact`.

        Args:
            body_names_expr: Expression for body names to sense.
            shape_names_expr: Expression for shape names to sense.
            contact_partners_body_expr: Expression for contact partner body names.
            contact_partners_shape_expr: Expression for contact partner shape names.
            verbose: Print verbose information.
        """
        if not NewtonManager._supports_contact_sensors:
            raise NotImplementedError(
                "Newton contact sensors are not yet supported by the active coupled solver because its "
                "contact forces live in per-entry buffers."
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

        def _to_fnmatch(expr: str | list[str] | None) -> str | list[str] | None:
            """Convert Isaac Lab regex expressions (``.*``) to fnmatch glob (``*``)."""
            if expr is None:
                return None
            if isinstance(expr, str):
                return expr.replace(".*", "*")
            return [p.replace(".*", "*") for p in expr]

        def _normalize_for_labels(expr: str | list[str] | None, labels: list[str]) -> str | list[str] | None:
            """Strip leading path components from *expr* when labels are bare names.

            Model labels may be full USD paths (``/World/envs/env_0/Robot/base``) or bare
            names (``base``).  When the labels are bare names but the user expression
            contains slashes, we strip everything up to the last ``/``.
            """
            if expr is None or not labels:
                return expr
            label_has_paths = any("/" in lbl for lbl in labels)
            items = [expr] if isinstance(expr, str) else list(expr)
            expr_uses_paths = any("/" in p for p in items)
            if label_has_paths or not expr_uses_paths:
                return expr
            normalized = [p.rsplit("/", 1)[-1] for p in items]
            return normalized[0] if isinstance(expr, str) else normalized

        sensor_key = (
            _hashable_key(body_names_expr),
            _hashable_key(shape_names_expr),
            _hashable_key(contact_partners_body_expr),
            _hashable_key(contact_partners_shape_expr),
        )

        body_labels = list(cls._model.body_label)
        shape_labels = list(cls._model.shape_label)

        with Timer(name="newton_contact_sensor", msg="Contact sensor construction took:"):
            sensor = NewtonContactSensor(
                cls._model,
                sensing_obj_bodies=_normalize_for_labels(_to_fnmatch(body_names_expr), body_labels),
                sensing_obj_shapes=_normalize_for_labels(_to_fnmatch(shape_names_expr), shape_labels),
                counterpart_bodies=_normalize_for_labels(_to_fnmatch(contact_partners_body_expr), body_labels),
                counterpart_shapes=_normalize_for_labels(_to_fnmatch(contact_partners_shape_expr), shape_labels),
                measure_total=True,
                verbose=verbose,
            )

        cls._newton_contact_sensors[sensor_key] = sensor
        NewtonManager._report_contacts = True

        if cls._solver is not None and cls._contacts is not None and cls._contacts.force is None:
            cls._initialize_contacts()

        return sensor_key

    @classmethod
    def add_frame_transform_sensor(cls, shapes: list[int], reference_sites: list[int]) -> int:
        """Add a frame transform sensor for measuring relative transforms.

        Creates a :class:`SensorFrameTransform` from pre-resolved shape and reference
        site indices, appends it to the internal list, and returns its index.

        Args:
            shapes: Ordered list of shape indices to measure.
            reference_sites: 1:1 list of reference site indices (same length as shapes).

        Returns:
            Index of the newly created sensor in :attr:`_newton_frame_transform_sensors`.
        """
        sensor = SensorFrameTransform(
            cls._model,
            shapes=shapes,
            reference_sites=reference_sites,
        )
        idx = len(cls._newton_frame_transform_sensors)
        cls._newton_frame_transform_sensors.append(sensor)
        logger.info(f"Added frame transform sensor (index={idx}, shapes={len(shapes)})")
        return idx

    @classmethod
    def add_imu_sensor(cls, sites: list[int]) -> int:
        """Add an IMU sensor for measuring acceleration and angular velocity at sites.

        Creates a ``newton.sensors.SensorIMU`` from pre-resolved site indices,
        appends it to the internal list, and returns its index.

        Args:
            sites: Ordered list of site indices (one per environment).

        Returns:
            Index of the newly created sensor in the internal IMU sensor list.
        """
        if cls._model is None:
            raise RuntimeError("add_imu_sensor called before model finalization (start_simulation).")
        sensor = NewtonSensorIMU(
            cls._model,
            sites=sites,
            request_state_attributes=False,  # Already requested via NewtonManager
        )
        idx = len(cls._newton_imu_sensors)
        cls._newton_imu_sensors.append(sensor)
        logger.info(f"Added IMU sensor (index={idx}, sites={len(sites)})")
        return idx
