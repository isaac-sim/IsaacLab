# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import contextlib
import ctypes
import logging
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

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
from newton import Axis, CollisionPipeline, Contacts, Control, Model, ModelBuilder, State, eval_fk
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.sensors import SensorContact as NewtonContactSensor
from newton.sensors import SensorFrameTransform
from newton.sensors import SensorIMU as NewtonSensorIMU
from newton.solvers import SolverBase, SolverKamino, SolverNotifyFlags

from isaaclab.physics import CallbackHandle, PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat, SceneDataProvider
from isaaclab.sim.utils.newton_model_utils import replace_newton_shape_colors
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import checked_apply
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.timer import Timer

from .newton_manager_cfg import NewtonCfg, NewtonShapeCfg

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

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


@wp.kernel(enable_backward=False)
def _or_reset_masks_from_mask(
    env_mask: wp.array(dtype=wp.bool),
    articulation_ids: wp.array2d(dtype=int),
    world_mask: wp.array(dtype=wp.int32),
    fk_mask: wp.array(dtype=wp.bool),
):
    """OR env_mask into world_mask and set corresponding articulation bits in fk_mask."""
    world, arti = wp.tid()
    if env_mask[world]:
        world_mask[world] = wp.int32(1)
        fk_mask[articulation_ids[world, arti]] = True


@wp.kernel(enable_backward=False)
def _scatter_reset_masks_from_ids(
    env_ids: wp.array(dtype=int),
    articulation_ids: wp.array2d(dtype=int),
    world_mask: wp.array(dtype=wp.int32),
    fk_mask: wp.array(dtype=wp.bool),
):
    """Scatter-set world_mask and fk_mask from sparse env_ids."""
    i, arti = wp.tid()
    world = env_ids[i]
    world_mask[world] = wp.int32(1)
    fk_mask[articulation_ids[world, arti]] = True


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
    def state(self) -> Model:
        return NewtonManager.get_state_0()


class NewtonManager(PhysicsManager):
    """Abstract Newton physics manager for Isaac Lab.

    Class-level (singleton-like) manager that owns simulation lifecycle, model
    state, contacts/collision pipeline, sensors, replication, and CUDA-graph
    orchestration.
    Concrete subclasses (one per solver) implement :meth:`_build_solver` and
    may extend :meth:`_initialize_contacts`, :meth:`_step_solver`,
    :meth:`_solver_specific_clear`, and :meth:`_log_solver_debug`.

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

    _solver_dt: float = 1.0 / 200.0
    _num_substeps: int = 1
    _decimation: int = 1
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
    # Per-world reset masks (allocated in start_simulation, consumed in step)
    _world_reset_mask: wp.array | None = None  # (num_envs,) wp.int32 — for SolverKamino.reset(world_mask=...)
    _fk_reset_mask: wp.array | None = None  # (articulation_count,) wp.bool — for eval_fk(mask=...)

    # Newton actuator adapter (owns actuators and double-buffered states)
    _adapter: NewtonActuatorAdapter | None = None
    # In-graph hooks invoked after the actuator step and before the solver
    # substeps, in registration order. Multiple articulations register their
    # implicit-DOF telemetry / FF-routing kernels here.
    _post_actuator_callbacks: list[Callable[[], None]] = []

    # CUDA graphing
    _graph = None
    _graph_capture_pending: bool = False

    # USD/Fabric sync
    _newton_stage_path = None
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _clone_physics_only = False
    _transforms_dirty: bool = False
    _particles_dirty: bool = False
    _newton_particle_offset_attr = "newton:particleOffset"
    _newton_particle_count_attr = "newton:particleCount"

    # cubric GPU transform hierarchy (replaces CPU update_world_xforms)
    _cubric = None
    _cubric_adapter: int | None = None
    _cubric_bound_fabric_id: int | None = None

    # Model changes (callbacks use unified system from PhysicsManager)
    _model_changes: set[int] = set()

    # Scene data backend
    _scene_data_backend: NewtonSceneDataBackend | None = None

    # Visualization-only state used when the sim backend is PhysX. Populated
    # lazily in :meth:`_ensure_visualization_model` and updated each render
    # frame in :meth:`update_visualization_state`.
    _scene_data: SceneDataFormat.Transform | None = None
    _scene_data_mapping: wp.array | None = None

    # Views list for assets to register their views
    _views: list = []

    # CL: Cloning / Replication logic
    # TODO: These attributes support cloning-specific logic and should be moved into a cloner class
    # Pending site requests from sensors.
    # Key: (body_pattern, xform_floats), Value: (label, wp.transform)
    # identical (body_pattern, transform) reuses the same site.
    _cl_pending_sites: dict[tuple[str | None, tuple[float, ...]], tuple[str, wp.transform]] = {}

    # Maps each site label to its resolved global or local site entry.
    _GlobalSite = tuple[int, None]
    _LocalSite = tuple[None, list[list[int]]]
    _SiteEntry = _GlobalSite | _LocalSite
    _cl_site_index_map: dict[str, _SiteEntry] = {}

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
            cls._clone_physics_only = "kit" not in requested and not cameras_enabled

        cls._scene_data_backend = NewtonSceneDataBackend()

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            cls.start_simulation()
            cls.initialize_solver()

    @classmethod
    def forward(cls) -> None:
        """Update articulation kinematics without stepping physics.

        Runs Newton's generic forward kinematics (``eval_fk``) over **all**
        articulations to compute body poses from joint coordinates. This is
        the full (unmasked) FK path used during initial setup. For incremental
        per-environment updates after resets, see :meth:`invalidate_fk` which
        accumulates masks consumed by :meth:`step`.
        """
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

    @classmethod
    def pre_render(cls) -> None:
        """Flush deferred Fabric writes before cameras/visualizers read the scene."""
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
        """Write Newton particle_q to Fabric mesh point arrays for Kit viewport rendering.

        For each deformable body whose mesh prim carries a ``newton:particleOffset``
        attribute, this function copies the corresponding slice of ``state_0.particle_q``
        into the Fabric ``points`` array so the Kit viewport reflects the current
        deformation.

        No-op when there is no ``_usdrt_stage``, no simulation state, or no
        deformable bodies registered.
        """
        if cls._usdrt_stage is None or cls._state_0 is None or cls._state_0.particle_q is None:
            return
        if not cls._particles_dirty:
            return
        pq = cls._state_0.particle_q
        try:
            import usdrt

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
            fabric_points = wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f)
            fabric_offsets = wp.fabricarray(data=selection, attrib=cls._newton_particle_offset_attr)
            fabric_counts = wp.fabricarray(data=selection, attrib=cls._newton_particle_count_attr)
            fabric_world_matrices = wp.fabricarray(data=selection, attrib="omni:fabric:worldMatrix")
            wp.launch(
                _sync_particle_points,
                dim=selection.GetCount(),
                inputs=[fabric_points, fabric_world_matrices, fabric_offsets, fabric_counts, pq],
                device=PhysicsManager._device,
            )
            NewtonManager._particles_dirty = False
        except Exception as exc:
            logger.debug("[sync_particles_to_usd] %s", exc)

    @classmethod
    def _mark_transforms_dirty(cls) -> None:
        """Flag that rigid-body transforms have changed and Fabric needs re-sync.

        The actual sync is deferred to :meth:`sync_transforms_to_usd`,
        which runs at render cadence via :meth:`pre_render`.
        """
        NewtonManager._transforms_dirty = True

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
                logger.info("Newton CUDA graph captured (deferred relaxed mode, RTX-compatible)")
            else:
                logger.warning("Newton deferred CUDA graph capture failed; using eager execution")

        # Ensure body_q is up-to-date before collision detection.
        # After env resets, joint_q is written but body_q (used by
        # broadphase/narrowphase) is stale until FK runs.
        # Only runs FK for dirtied articulations via the accumulated mask.
        if cls._needs_collision_pipeline:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, cls._fk_reset_mask)

        # Zero both masks after consumption
        NewtonManager._world_reset_mask.zero_()
        NewtonManager._fk_reset_mask.zero_()

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

        # Launch solver-specific debug logging after stepping.
        cls._log_solver_debug()

    @classmethod
    def close(cls) -> None:
        """Clean up Newton physics resources."""
        cls.clear()
        super().close()

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
        NewtonManager._builder = None
        NewtonManager._model = None
        NewtonManager._solver = None
        NewtonManager._use_single_state = None
        NewtonManager._state_0 = None
        NewtonManager._state_1 = None
        NewtonManager._control = None
        NewtonManager._contacts = None
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._collision_pipeline = None
        NewtonManager._collision_cfg = None
        NewtonManager._newton_contact_sensors = {}
        NewtonManager._newton_frame_transform_sensors = []
        NewtonManager._newton_imu_sensors = []
        NewtonManager._report_contacts = False
        NewtonManager._adapter = None
        NewtonManager._post_actuator_callbacks = []
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
        NewtonManager._newton_stage_path = None
        NewtonManager._usdrt_stage = None
        NewtonManager._transforms_dirty = False
        NewtonManager._particles_dirty = False
        NewtonManager._up_axis = "Z"
        NewtonManager._scene_data = None
        NewtonManager._scene_data_mapping = None
        NewtonManager._model_changes = set()
        NewtonManager._scene_data_backend = None
        NewtonManager._cl_pending_sites = {}
        NewtonManager._cl_site_index_map = {}
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
        builder = ModelBuilder(up_axis=up_axis or cls._up_axis, **kwargs)
        # Resolve which NewtonShapeCfg to apply: user override if active config
        # is NewtonCfg, else the wrapper's own defaults so callers from non-Newton
        # contexts (tests, early construction) still get the rough-terrain margin.
        cfg = PhysicsManager._cfg
        shape_cfg = cfg.default_shape_cfg if isinstance(cfg, NewtonCfg) else NewtonShapeCfg()
        checked_apply(shape_cfg, builder.default_shape_cfg)
        return builder

    @classmethod
    def cl_register_site(cls, body_pattern: str | None, xform: wp.transform) -> str:
        """Register a site request for injection into prototypes before replication.

        Sensors call this during ``__init__``. Sites are injected into prototype
        builders by :meth:`_cl_inject_sites` (called from ``newton_replicate``)
        before ``add_builder``, so they replicate correctly per-world.

        Identical ``(body_pattern, transform)`` registrations share sites.

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

        Returns:
            Assigned site label suffix.
        """
        xform_key = tuple(xform)
        key = (body_pattern, xform_key)
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
        proto_builders: dict[str, ModelBuilder],
    ) -> tuple[dict[str, int], dict[int, dict[str, list[int]]]]:
        """Inject registered sites into prototype builders before replication.

        Non-global sites are matched against prototype body labels using
        :func:`resolve_matching_names` (regex). Global sites
        (``body_pattern is None``) are added to *main_builder* with
        ``body=-1``.

        Returns proto-local shape indices so that ``newton_replicate`` can
        compute final indices during replication without a second pattern match.

        Pending requests are cleared after processing.

        Args:
            main_builder: Top-level builder that receives global sites.
            proto_builders: ``{src_path: ModelBuilder}`` prototype builders.

        Returns:
            Tuple of ``(global_sites, proto_sites)`` where *global_sites* maps
            ``{label: main_builder_shape_idx}`` and *proto_sites* maps
            ``{id(proto): {label: [proto_local_shape_idx, ...]}}``.
        """
        global_sites: dict[str, int] = {}
        proto_sites: dict[int, dict[str, list[int]]] = {}

        for (body_pattern, _xform_key), (label, xform) in cls._cl_pending_sites.items():
            if body_pattern is None:
                site_idx = main_builder.add_site(body=-1, xform=xform, label=label)
                global_sites[label] = site_idx
                continue

            any_matched = False
            for src_prefix, proto in proto_builders.items():
                body_labels = list(proto.body_label)
                matched_indices, matched_names = resolve_matching_names(
                    body_pattern, body_labels, raise_when_no_match=False
                )
                if not matched_indices:  # Pattern has no matches in this prototype
                    continue

                any_matched = True
                proto_id = id(proto)
                site_indices: list[int] = []
                for body_idx, body_name in zip(matched_indices, matched_names):
                    site_label = f"{body_name}/{label}"
                    proto_site_idx = proto.add_site(body=body_idx, xform=xform, label=site_label)
                    site_indices.append(proto_site_idx)
                    logger.debug(f"Injected site '{site_label}' into prototype")
                proto_sites.setdefault(proto_id, {})[label] = site_indices

            if not any_matched:
                raise ValueError(
                    f"Site '{label}' with body_pattern '{body_pattern}' matched no prototype bodies "
                    f"across {len(proto_builders)} prototype(s). "
                    f"Check that the pattern matches a body label in the prototype builder."
                )

        cls._cl_pending_sites.clear()
        return global_sites, proto_sites

    @classmethod
    def _cl_inject_sites_fallback(cls) -> None:
        """Inject pending sites into the flat builder (no-replication path).

        Populates :attr:`_cl_site_index_map` with the unified per-world structure:

        - Global sites (``body_pattern is None``): ``(shape_idx, None)``
        - Local sites: ``(None, [[idx, ...]])`` — one sublist for the single world.
        """
        builder = cls._builder
        body_labels = list(builder.body_label)

        for (body_pattern, _xform_key), (label, xform) in cls._cl_pending_sites.items():
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
    def add_model_change(cls, change: SolverNotifyFlags) -> None:
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
        transforms. The masks are consumed in :meth:`step` before physics
        stepping.

        Args:
            env_mask: Boolean mask of dirtied environments. Shape ``(num_envs,)``.
                Used by ``_mask`` write methods.
            env_ids: Integer indices of dirtied environments.
                Used by ``_index`` write methods.
            articulation_ids: Mapping from ``(world, arti)`` to model articulation
                index. Shape ``(world_count, count_per_world)``. Obtained from
                ``ArticulationView.articulation_ids``.
        """
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
                _scatter_reset_masks_from_ids,
                dim=(env_ids.shape[0], articulation_ids.shape[1]),
                inputs=[env_ids, articulation_ids],
                outputs=[NewtonManager._world_reset_mask, NewtonManager._fk_reset_mask],
                device=PhysicsManager._device,
            )
        else:
            # Fallback: no topology info — mark everything dirty
            NewtonManager._world_reset_mask.fill_(1)
            NewtonManager._fk_reset_mask.fill_(True)

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation by finalizing model and initializing state.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.
        """
        logger.debug(f"Builder: {cls._builder}")

        # Create builder from USD stage if not provided
        if cls._builder is None:
            cls.instantiate_builder_from_stage()

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
        with Timer(name="newton_finalize_builder", msg="Finalize builder took:"):
            NewtonManager._model = cls._builder.finalize(device=device)
            cls._model.set_gravity(cls._gravity_vector)
            cls._model.num_envs = cls._num_envs

            replace_newton_shape_colors(cls._model)

        if cls._pending_extended_contact_attributes:
            cls._model.request_contact_attributes(*cls._pending_extended_contact_attributes)
            NewtonManager._pending_extended_contact_attributes = set()

        NewtonManager._state_0 = cls._model.state()
        NewtonManager._state_1 = cls._model.state()
        NewtonManager._control = cls._model.control()
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

        # The single global actuator adapter is built lazily on the first
        # call to ``activate_newton_actuator_path`` from any Newton-fast-path
        # articulation after this point. Assign through the explicit base
        # class so external readers (which import ``NewtonManager`` directly)
        # observe the canonical state regardless of which subclass is active.
        NewtonManager._adapter = None
        NewtonManager._use_newton_actuators_active = False

        # Allocate per-world reset masks (used by all solvers for masked FK, and by Kamino for masked reset)
        NewtonManager._world_reset_mask = wp.zeros(cls._model.world_count, dtype=wp.int32, device=device)
        NewtonManager._fk_reset_mask = wp.zeros(cls._model.articulation_count, dtype=wp.bool, device=device)

        logger.info("Dispatching PHYSICS_READY callbacks")
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

        # Setup USD/Fabric sync for Kit viewport rendering
        if not cls._clone_physics_only:
            import usdrt

            body_paths = getattr(cls._model, "body_label", None) or getattr(cls._model, "body_key", None)
            if not body_paths:
                logger.warning(
                    "NewtonManager: model has no rigid bodies (body_label/body_key is empty). "
                    "USD/Fabric body sync for RTX is skipped. "
                    "Particle-only scenes (e.g. cloth) must register their own USD mesh update."
                )
                NewtonManager._usdrt_stage = None
            else:
                NewtonManager._usdrt_stage = get_current_stage(fabric=True)
                for i, prim_path in enumerate(body_paths):
                    prim = cls._usdrt_stage.GetPrimAtPath(prim_path)
                    prim.CreateAttribute(cls._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
                    prim.GetAttribute(cls._newton_index_attr).Set(i)
                    # Tag with PhysicsRigidBodyAPI so cubric's eRigidBody mode
                    # applies Inverse propagation (preserves Newton's world
                    # transforms and derives local) instead of Forward.
                    prim.AddAppliedSchema("PhysicsRigidBodyAPI")
                    xformable_prim = usdrt.Rt.Xformable(prim)
                    if not xformable_prim.HasWorldXform():
                        xformable_prim.SetWorldXformFromUsd()

                cls._mark_transforms_dirty()
                cls.sync_transforms_to_usd()

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

        builder = ModelBuilder(up_axis=up_axis)

        schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

        if not env_paths:
            # No env Xforms — flat loading
            builder.add_usd(stage, schema_resolvers=schema_resolvers)
        else:
            # Load everything except the env subtrees (ground plane, lights, etc.)
            ignore_paths = [path for _, path in env_paths]
            builder.add_usd(stage, ignore_paths=ignore_paths, schema_resolvers=schema_resolvers)

            # Build a prototype from the first env (all envs assumed identical)
            _, proto_path = env_paths[0]
            proto = ModelBuilder(up_axis=up_axis)
            proto.add_usd(
                stage,
                root_path=proto_path,
                schema_resolvers=schema_resolvers,
            )

            # Inject registered sites into the proto before replication
            global_sites, proto_sites = cls._cl_inject_sites(builder, {proto_path: proto})
            global_site_map: dict[str, tuple[int, None]] = {label: (idx, None) for label, idx in global_sites.items()}
            num_worlds = len(env_paths)
            local_site_map: dict[str, list[list[int]]] = {}
            site_entries = proto_sites.get(id(proto), {})

            # Add each env as a separate Newton world
            xform_cache = UsdGeom.XformCache()
            for col, (_, env_path) in enumerate(env_paths):
                builder.begin_world()
                offset = builder.shape_count
                world_xform = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(env_path))
                translation = world_xform.ExtractTranslation()
                rotation = world_xform.ExtractRotationQuat()
                pos = (translation[0], translation[1], translation[2])
                quat = (
                    rotation.GetImaginary()[0],
                    rotation.GetImaginary()[1],
                    rotation.GetImaginary()[2],
                    rotation.GetReal(),
                )
                builder.add_builder(proto, xform=wp.transform(pos, quat))
                for label, proto_shape_indices in site_entries.items():
                    if label not in local_site_map:
                        local_site_map[label] = [[] for _ in range(num_worlds)]
                    for proto_shape_idx in proto_shape_indices:
                        local_site_map[label][col].append(offset + proto_shape_idx)
                builder.end_world()

            NewtonManager._cl_site_index_map = {
                **global_site_map,
                **{label: (None, per_world) for label, per_world in local_site_map.items()},
            }
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
        if use_cuda_graph:
            with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
                if cls._usdrt_stage is None:
                    simulate = cls._simulate_full if cls._is_all_graphable() else cls._simulate_physics_only
                    with wp.ScopedCapture() as capture:
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
    def _capture_relaxed_graph(cls, device: str):
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

        Returns a ``wp.Graph`` on success, or ``None`` on failure.
        """
        if _cudart is None:
            logger.warning("libcudart not available; cannot use relaxed graph capture")
            return None

        # Warmup: pre-allocate all solver scratch buffers so the capture window has
        # no new cudaMalloc calls (which are forbidden inside graph capture).
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
        if cls._use_single_state:
            for _ in range(cls._num_substeps):
                cls._step_solver(cls._state_0, cls._state_0, cls._control, contacts, cls._solver_dt)
                cls._state_0.clear_forces()
        else:
            cfg = PhysicsManager._cfg
            need_copy_on_last = (cfg is not None and cfg.use_cuda_graph) and cls._num_substeps % 2 == 1  # type: ignore[union-attr]
            for i in range(cls._num_substeps):
                cls._step_solver(cls._state_0, cls._state_1, cls._control, contacts, cls._solver_dt)
                if need_copy_on_last and i == cls._num_substeps - 1:
                    cls._state_0.assign(cls._state_1)
                else:
                    NewtonManager._state_0, NewtonManager._state_1 = cls._state_1, cls._state_0
                cls._state_0.clear_forces()

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
        observe stale transforms. Under the Newton sim backend
        :meth:`update_visualization_state` is a no-op and this is equivalent to
        :meth:`get_state_0`.
        """
        cls.update_visualization_state(scene_data_provider)
        return cls.get_state_0()

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

        The shadow model is built by walking the USD stage via
        :meth:`_build_visualization_model_from_stage` and finalizing the resulting
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

        try:
            builder = cls._build_visualization_model_from_stage(stage)
        except Exception:
            logger.exception(
                "[NewtonManager] Failed to build a Newton ModelBuilder from the USD stage "
                "for visualization (sim backend is PhysX)."
            )
            return

        if builder is None or builder.body_count == 0:
            logger.error(
                "[NewtonManager] USD stage walk produced no Newton bodies; the shadow "
                "Newton model for visualization will be empty. Common causes: the cloned "
                "envs are not yet on the stage, or PhysX schemas could not be parsed by "
                "Newton's add_usd. Check that /World/envs/env_<id> prims exist when the "
                "renderer is initialized."
            )
            return

        device = PhysicsManager._device or "cpu"
        try:
            NewtonManager._model = builder.finalize(device=device)
            NewtonManager._state_0 = cls._model.state()
            cls._model.num_envs = cls._num_envs
            replace_newton_shape_colors(cls._model)

        except Exception:
            logger.exception(
                "[NewtonManager] Failed to finalize the shadow Newton ModelBuilder for "
                "visualization (sim backend is PhysX)."
            )
            NewtonManager._model = None
            NewtonManager._state_0 = None

    @classmethod
    def _build_visualization_model_from_stage(cls, stage) -> ModelBuilder | None:
        """Build a fresh Newton ``ModelBuilder`` from the USD stage for visualization.

        Walks IsaacLab's ``/World/envs/env_<id>`` convention and adds each env as
        its own Newton world. When the env subtree is identical across envs (the
        common cloned-scene case) a single env_0 prototype is built once and
        replicated via :meth:`ModelBuilder.add_builder`; otherwise each env is
        ingested independently with :meth:`ModelBuilder.add_usd`.

        This routine is intentionally independent of
        :meth:`instantiate_builder_from_stage` (which targets the live-sim path
        and uses a different naming convention and writes into ``cls._builder``
        and ``cls._cl_site_index_map``). The visualization shadow path must not
        pollute those live-sim slots. ``cls._num_envs`` is populated here too so
        :meth:`get_num_envs` returns the env count when the sim backend is PhysX
        (the live-sim path never runs in that configuration, so there is no slot
        to collide with).

        Args:
            stage: USD stage to inspect.

        Returns:
            A populated :class:`~newton.ModelBuilder`, or ``None`` when no
            ``/World/envs/env_<id>`` prims exist on the stage.
        """
        import re

        from pxr import UsdGeom

        up_axis_token = UsdGeom.GetStageUpAxis(stage)
        up_axis = Axis.from_string(str(up_axis_token))
        schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

        env_pattern = re.compile(r"^env_(\d+)$")
        env_paths: list[tuple[int, str]] = []
        envs_root = stage.GetPrimAtPath("/World/envs")
        if envs_root and envs_root.IsValid():
            for child in envs_root.GetChildren():
                if match := env_pattern.match(child.GetName()):
                    env_paths.append((int(match.group(1)), child.GetPath().pathString))
        env_paths.sort(key=lambda x: x[0])

        builder = ModelBuilder(up_axis=up_axis)

        if not env_paths:
            # Fallback: ingest the whole stage as a single world.
            builder.add_usd(stage, schema_resolvers=schema_resolvers)
            NewtonManager._num_envs = 1
            return builder

        NewtonManager._num_envs = len(env_paths)

        # Ingest stage-level (non-env) geometry into the global world (``current_world == -1``)
        # so visualization sees the ground plane, ceilings, fixed props, etc. The legacy
        # cloner-based prebuild did this via ``add_usd(stage, ignore_paths=["/World/envs"], ...)``
        # before adding the per-env worlds; without this, renderers/visualizers driven off the
        # shadow Newton model are missing every shape authored outside the env hierarchy.
        builder.add_usd(
            stage,
            ignore_paths=[r"/World/envs($|/.*)"],
            schema_resolvers=schema_resolvers,
        )

        # Build env_0 as a prototype, then replicate across envs.
        proto_env_path = env_paths[0][1]
        proto = ModelBuilder(up_axis=up_axis)
        proto.add_usd(
            stage,
            root_path=proto_env_path,
            schema_resolvers=schema_resolvers,
        )

        xform_cache = UsdGeom.XformCache()

        # ``add_builder`` copies the prototype's ``body_label`` (and sibling label arrays)
        # verbatim into each replicated world, so all worlds end up with prim paths under
        # the prototype env (e.g. ``/World/envs/env_0/...``). The visualization sync uses
        # these labels to map PhysX transforms (which carry distinct per-env paths) into
        # ``state.body_q``; without rewriting, ``paths.index()`` resolves every match to
        # world 0 and worlds 1..N never receive fresh poses. Rewrite the newly-added
        # labels after each ``add_builder`` so each world references its own env prim path.
        label_attrs = ("body_label", "articulation_label", "joint_label", "shape_label")
        label_starts = {attr: len(getattr(builder, attr)) for attr in label_attrs}

        # ``proto.add_usd`` ingests env_0's bodies at their absolute world positions
        # (``UsdPhysics.LoadUsdPhysicsFromRange`` reports world-space transforms), so
        # ``proto.body_q`` already encodes env_0's world transform. ``add_builder``
        # composes its ``xform`` onto every imported body, so passing each env's
        # absolute world transform here would double the offset; the correct xform is
        # the env's pose relative to the prototype (identity for env_0, env_X * env_0^-1
        # for the rest). Dynamic bodies are overwritten in ``update_visualization_state``
        # via the PhysX sync, but static bodies (e.g. the table) keep this initial pose
        # and render at the wrong position when env_0 is not at the world origin.
        proto_world_gf = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(proto_env_path))
        proto_translation = proto_world_gf.ExtractTranslation()
        proto_rotation = proto_world_gf.ExtractRotationQuat()
        proto_world_tf = wp.transform(
            (proto_translation[0], proto_translation[1], proto_translation[2]),
            (
                proto_rotation.GetImaginary()[0],
                proto_rotation.GetImaginary()[1],
                proto_rotation.GetImaginary()[2],
                proto_rotation.GetReal(),
            ),
        )
        proto_world_tf_inv = wp.transform_inverse(proto_world_tf)

        for _, env_path in env_paths:
            world_xform = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(env_path))
            translation = world_xform.ExtractTranslation()
            rotation = world_xform.ExtractRotationQuat()
            env_world_tf = wp.transform(
                (translation[0], translation[1], translation[2]),
                (
                    rotation.GetImaginary()[0],
                    rotation.GetImaginary()[1],
                    rotation.GetImaginary()[2],
                    rotation.GetReal(),
                ),
            )
            relative_tf = wp.transform_multiply(env_world_tf, proto_world_tf_inv)
            builder.begin_world()
            builder.add_builder(proto, xform=relative_tf)
            if env_path != proto_env_path:
                for attr in label_attrs:
                    labels = getattr(builder, attr)
                    for i in range(label_starts[attr], len(labels)):
                        labels[i] = labels[i].replace(proto_env_path, env_path, 1)
            for attr in label_attrs:
                label_starts[attr] = len(getattr(builder, attr))
            builder.end_world()

        return builder

    @classmethod
    def get_scene_data_provider(cls) -> SceneDataProvider:
        """Return the active scene data provider, or None if unavailable.

        Prefers ``PhysicsManager._sim`` when set; otherwise falls back to
        ``SimulationContext.instance()``.
        """
        sim = PhysicsManager._sim
        if sim is None:
            from isaaclab.sim import SimulationContext

            sim = SimulationContext.instance()

        assert sim is not None
        return sim.get_scene_data_provider()

    @classmethod
    def update_visualization_state(cls, scene_data_provider: SceneDataProvider | None = None) -> None:
        """Refresh visualization state for the active sim backend.

        Newton sim backend: no-op — ``_state_0`` is the live, authoritative state
        already advanced by :meth:`step` / forward kinematics.

        PhysX sim backend: pull rigid-body transforms from the
        :class:`~isaaclab.scene_data.SceneDataProvider` and write
        them into the shadow ``_state_0.body_q`` so Newton-native consumers
        (Newton renderer, Newton/Rerun/Viser visualizers, OVRTX renderer, Newton
        GL video) see fresh poses.

        Invoked lazily from :meth:`get_state` so consumers do not need to
        coordinate the sync explicitly.
        """

        if scene_data_provider is None:
            scene_data_provider = cls.get_scene_data_provider()

        assert scene_data_provider is not None

        if cls._backend_is_newton(scene_data_provider):
            return
        cls._ensure_visualization_model()
        if cls._state_0 is None or cls._model is None or cls._state_0.body_q is None:
            return

        if cls._scene_data is None:
            cls._scene_data = SceneDataFormat.Transform()
        if cls._scene_data_mapping is None:
            body_paths = list(getattr(cls._model, "body_label", None) or [])
            cls._scene_data_mapping = scene_data_provider.create_mapping(body_paths)

        cls._scene_data.transforms = cls._state_0.body_q
        scene_data_provider.get_transforms(cls._scene_data, mapping=cls._scene_data_mapping)

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
