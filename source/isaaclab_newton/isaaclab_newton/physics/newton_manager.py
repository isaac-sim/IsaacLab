# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import contextlib
import ctypes
import inspect
import logging
from typing import TYPE_CHECKING

import numpy as np
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
from newton.geometry import HydroelasticSDF
from newton.sensors import SensorContact as NewtonContactSensor
from newton.solvers import SolverBase, SolverFeatherstone, SolverMuJoCo, SolverNotifyFlags, SolverXPBD

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.timer import Timer

from .debug_state_buffer import DebugStateBuffer

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


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
    fabric_transforms[i] = wp.transpose(wp.mat44d(wp.math.transform_to_matrix(transform)))


class NewtonManager(PhysicsManager):
    """Newton physics manager for Isaac Lab.

    This is a class-level (singleton-like) manager for the Newton simulation.
    It handles solver configuration, physics stepping, and reset.

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    _solver_dt: float = 1.0 / 200.0
    _num_substeps: int = 1
    _num_envs: int | None = None

    # Newton model and state
    _builder: ModelBuilder = None
    _model: Model = None
    _solver: SolverBase = None
    _solver_type: str = "mujoco_warp"
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
    _collision_cfg = None  # CollisionPipelineCfg, set during initialize()
    _newton_contact_sensors: dict = {}  # Maps sensor_key to NewtonContactSensor
    _report_contacts: bool = False
    _fk_dirty: bool = False

    # CUDA graphing
    _graph = None
    _graph_capture_pending: bool = False

    # USD/Fabric sync
    _newton_stage_path = None
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _clone_physics_only = False
    _transforms_dirty: bool = False

    # cubric GPU transform hierarchy (replaces CPU update_world_xforms)
    _cubric = None
    _cubric_adapter: int | None = None
    _cubric_bound_fabric_id: int | None = None

    # Model changes (callbacks use unified system from PhysicsManager)
    _model_changes: set[int] = set()

    # Debug state buffer for NaN replay (None = disabled)
    _debug_state_buffer: DebugStateBuffer | None = None

    # Views list for assets to register their views
    _views: list = []

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
            cls._gravity_vector = sim.cfg.gravity  # type: ignore[union-attr]

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
        """Update articulation kinematics without stepping physics."""
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)
        cls._fk_dirty = False

    @classmethod
    def pre_render(cls) -> None:
        """Flush deferred Fabric writes before cameras/visualizers read the scene."""
        cls.sync_transforms_to_usd()

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
                cls._cubric_adapter = cls._cubric.create_adapter()
                if cls._cubric_adapter is not None:
                    logger.info("cubric GPU transform hierarchy enabled")
                else:
                    logger.warning("cubric adapter creation failed; falling back to update_world_xforms()")
                    cls._cubric = None

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
                    cls._transforms_dirty = False
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

                cls._transforms_dirty = False

                if use_cubric and fabric_hierarchy is not None:
                    fabric_id = cls._usdrt_stage.GetFabricId().id
                    if fabric_id != cls._cubric_bound_fabric_id:
                        cls._cubric.bind_to_stage(cls._cubric_adapter, fabric_id)
                        cls._cubric_bound_fabric_id = fabric_id
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
    def _mark_transforms_dirty(cls) -> None:
        """Flag that physics state has changed and Fabric needs re-sync.

        Called by :meth:`_simulate` after stepping. The actual sync is deferred
        to :meth:`sync_transforms_to_usd`, which runs at render cadence.
        """
        cls._transforms_dirty = True

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Notify solver of model changes
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._solver.notify_model_changed(change)
                cls._model_changes = set()

        # Lazy CUDA graph capture: deferred from initialize_solver() when RTX was active.
        # By the time step() is first called, RTX has fully initialized (all cudaImportExternalMemory
        # calls are done) and is idle between render frames — giving us a clean capture window.
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        if cls._graph_capture_pending and cfg is not None and cfg.use_cuda_graph and "cuda" in device:  # type: ignore[union-attr]
            cls._graph_capture_pending = False
            cls._graph = cls._capture_relaxed_graph(device)
            if cls._graph is not None:
                logger.info("Newton CUDA graph captured (deferred relaxed mode, RTX-compatible)")
            else:
                logger.warning("Newton deferred CUDA graph capture failed; using eager execution")

        # Snapshot state before stepping for NaN debug (captures post-reset state)
        if cls._debug_state_buffer is not None:
            cls._debug_state_buffer.snapshot_pre_step(cls._state_0)

        # Ensure body_q is up-to-date before collision detection.
        # After env resets, joint_q is written but body_q (used by
        # broadphase/narrowphase) is stale until FK runs.
        if cls._fk_dirty and cls._needs_collision_pipeline:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)
        cls._fk_dirty = False

        # Step simulation (graphed or not; _graph is None when capture is disabled or failed)
        if cfg is not None and cfg.use_cuda_graph and cls._graph is not None and "cuda" in device:  # type: ignore[union-attr]
            wp.capture_launch(cls._graph)
            if cls._usdrt_stage is not None:
                cls._mark_transforms_dirty()
        else:
            with wp.ScopedDevice(device):
                cls._simulate()

        # Debug convergence info
        if cfg is not None and cfg.debug_mode:  # type: ignore[union-attr]
            convergence_data = cls.get_solver_convergence_steps()
            logger.info(f"Solver convergence data: {convergence_data}")
            if convergence_data["max"] == cls._solver.mjw_model.opt.iterations:
                logger.warning(f"Solver didn't converge! max_iter={convergence_data['max']}")
        PhysicsManager._sim_time += cls._solver_dt * cls._num_substeps

        if cls._debug_state_buffer is not None:
            with wp.ScopedDevice(PhysicsManager._device):
                diagnostics = cls._collect_diagnostics()
                solver_data = None
                if isinstance(cls._solver, SolverMuJoCo):
                    mjw_opt = cls._solver.mjw_model.opt
                    mjw_data = cls._solver.mjw_data
                    collision_cfg = cls._collision_cfg
                    solver_cfg = getattr(PhysicsManager._cfg, "solver_cfg", None)
                    solver_cfg_dict = solver_cfg.to_dict() if solver_cfg is not None and hasattr(solver_cfg, "to_dict") else {}
                    solver_data = {
                        "mjw_data": mjw_data,
                        "max_iterations": mjw_opt.iterations,
                        "nv": int(cls._solver.mjw_model.nv),
                        "opt": {
                            "iterations": mjw_opt.iterations,
                            "tolerance": solver_cfg_dict.get("tolerance", getattr(mjw_opt, "tolerance", None)),
                            "integrator": solver_cfg_dict.get("integrator", getattr(mjw_opt, "integrator", None)),
                            "cone": solver_cfg_dict.get("cone", getattr(mjw_opt, "cone", None)),
                            "impratio": solver_cfg_dict.get("impratio", getattr(mjw_opt, "impratio", None)),
                            "njmax": solver_cfg_dict.get("njmax", 400),
                            "nconmax": solver_cfg_dict.get("nconmax", 200),
                            "use_mujoco_contacts": solver_cfg_dict.get("use_mujoco_contacts", not cls._needs_collision_pipeline),
                            "max_triangle_pairs": getattr(collision_cfg, "max_triangle_pairs", None) if collision_cfg else None,
                            "dt": cls._solver_dt,
                            "num_substeps": cls._num_substeps,
                        },
                    }
                cls._debug_state_buffer.step(cls._state_0, PhysicsManager._sim_time, diagnostics, solver_data)
            if cls._debug_state_buffer.nan_halt:
                nr = getattr(PhysicsManager._cfg, "nan_replay", None)
                export_dir = nr.export_path if nr is not None else "."
                raise RuntimeError(
                    f"NaN detected in physics state. Debug replay exported to {export_dir}. "
                    "Halting simulation."
                )

    @classmethod
    def close(cls) -> None:
        """Clean up Newton physics resources."""
        cls.clear()
        super().close()

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
    def set_debug_episode_length_buf(cls, buf: torch.Tensor) -> None:
        """Attach per-env episode length to the debug state buffer.

        When the buffer exports a NaN event it will include ``episode_step``
        for each affected environment so the user can tell whether the NaN
        happened right after reset or mid-episode.

        Args:
            buf: Episode length tensor, shape ``(num_envs,)``, dtype long.
                Typically ``env.episode_length_buf``.
        """
        if cls._debug_state_buffer is not None:
            cls._debug_state_buffer.episode_length_buf = buf

    @classmethod
    def is_fabric_enabled(cls) -> bool:
        """Check if fabric interface is enabled (not applicable for Newton)."""
        return False

    @classmethod
    def clear(cls):
        """Clear all Newton-specific state (callbacks cleared by super().close())."""
        if cls._cubric is not None and cls._cubric_adapter is not None:
            cls._cubric.release_adapter(cls._cubric_adapter)
        cls._cubric = None
        cls._cubric_adapter = None
        cls._cubric_bound_fabric_id = None
        cls._builder = None
        cls._model = None
        cls._solver = None
        cls._use_single_state = None
        cls._state_0 = None
        cls._state_1 = None
        cls._control = None
        cls._contacts = None
        cls._needs_collision_pipeline = False
        cls._collision_pipeline = None
        cls._newton_contact_sensors = {}
        cls._report_contacts = False
        cls._fk_dirty = False
        cls._graph = None
        cls._graph_capture_pending = False
        cls._newton_stage_path = None
        cls._usdrt_stage = None
        cls._transforms_dirty = False
        cls._up_axis = "Z"
        cls._model_changes = set()
        cls._views = []
        if cls._debug_state_buffer is not None:
            cls._debug_state_buffer.clear()
        cls._debug_state_buffer = None

    @staticmethod
    def _make_scene_exporter():
        """Build a callable that exports NaN'd env prim subtrees AND global collision geometry to USD.

        Copies all relevant prims into a single target layer in one pass to avoid
        multiple ``export_prim_to_file`` calls clobbering each other.

        Returns None if USD utilities are unavailable (headless / no-USD environments).
        """
        try:
            from pxr import Sdf, Usd, UsdGeom  # noqa: PLC0415

            from isaaclab.sim.utils.prims import resolve_paths  # noqa: PLC0415
        except ImportError:
            return None

        def _exporter(usd_path: str, env_ids: list[int]) -> None:
            stage = get_current_stage()
            if stage is None:
                return

            source_layer = stage.GetRootLayer()
            target_layer = Sdf.Layer.CreateNew(usd_path)
            target_stage = Usd.Stage.Open(target_layer)
            UsdGeom.SetStageUpAxis(target_stage, UsdGeom.GetStageUpAxis(stage))
            UsdGeom.SetStageMetersPerUnit(target_stage, UsdGeom.GetStageMetersPerUnit(stage))

            prim_paths_to_copy = []

            if not env_ids:
                prim_paths_to_copy.append("/World/envs/env_0")
            else:
                for eid in env_ids:
                    prim_paths_to_copy.append(f"/World/envs/env_{eid}")

            world_prim = stage.GetPrimAtPath("/World")
            if world_prim and world_prim.IsValid():
                for child in world_prim.GetChildren():
                    child_path = child.GetPath().pathString
                    if child_path == "/World/envs":
                        continue
                    if _prim_has_collision(child):
                        prim_paths_to_copy.append(child_path)

            for prim_path in prim_paths_to_copy:
                Sdf.CreatePrimInLayer(target_layer, prim_path)
                Sdf.CopySpec(source_layer, Sdf.Path(prim_path), target_layer, Sdf.Path(prim_path))

            if prim_paths_to_copy:
                target_layer.defaultPrim = Sdf.Path(prim_paths_to_copy[0]).name

            resolve_paths(source_layer.identifier, target_layer.identifier)
            target_layer.Save()

        def _prim_has_collision(prim) -> bool:
            from pxr import Usd, UsdPhysics  # noqa: PLC0415

            for p in Usd.PrimRange(prim):
                if p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.MeshCollisionAPI):
                    return True
            return False

        return _exporter

    @classmethod
    def set_builder(cls, builder: ModelBuilder) -> None:
        """Set the Newton model builder."""
        cls._builder = builder

    @classmethod
    def add_model_change(cls, change: SolverNotifyFlags) -> None:
        """Register a model change to notify the solver."""
        cls._model_changes.add(change)

    @classmethod
    def invalidate_fk(cls) -> None:
        """Mark forward kinematics as needing recomputation.

        Called by articulation write methods that modify ``joint_q`` or root
        transforms.  The flag is checked in :meth:`step` before collision
        detection to ensure ``body_q`` is up-to-date.
        """
        cls._fk_dirty = True

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

        device = PhysicsManager._device
        logger.info(f"Finalizing model on device: {device}")
        cls._builder.up_axis = Axis.from_string(cls._up_axis)

        with Timer(name="newton_finalize_builder", msg="Finalize builder took:"):
            cls._model = cls._builder.finalize(device=device)
            cls._model.set_gravity(cls._gravity_vector)
            cls._model.num_envs = cls._num_envs

        # #region agent log
        import os as _os
        _export_dir = _os.environ.get("NEWTON_SAVE_CLONER_STAGE", _os.environ.get("NEWTON_SAVE_MODEL", ""))
        if _export_dir:
            _os.makedirs(_export_dir, exist_ok=True)
            import numpy as _np
            _m = cls._model
            _data = {}
            for _attr in dir(_m):
                if _attr.startswith("_"):
                    continue
                _v = getattr(_m, _attr, None)
                if _v is None:
                    continue
                if hasattr(_v, "numpy"):
                    try:
                        _data[_attr] = _v.numpy()
                    except Exception:
                        pass
                elif isinstance(_v, (int, float)):
                    _data[_attr] = _np.array([_v])
            _np.savez_compressed(_os.path.join(_export_dir, "model.npz"), **_data)
            logger.info(f"Saved model ({len(_data)} arrays) to {_export_dir}/model.npz")
        # #endregion

        cls._state_0 = cls._model.state()
        cls._state_1 = cls._model.state()
        cls._control = cls._model.control()
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

        nan_replay_cfg = getattr(PhysicsManager._cfg, "nan_replay", None)
        if nan_replay_cfg is not None:
            cls._debug_state_buffer = DebugStateBuffer(
                cls._model,
                nan_replay_cfg.buffer_size,
                export_path=nan_replay_cfg.export_path,
                max_exports=nan_replay_cfg.max_exports,
                scene_exporter=cls._make_scene_exporter(),
            )
            logger.info("Debug state buffer enabled: %d snapshots", cls._debug_state_buffer.size)

        logger.info("Dispatching PHYSICS_READY callbacks")
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

        # Setup USD/Fabric sync for Kit viewport rendering
        if not cls._clone_physics_only:
            import usdrt

            body_paths = getattr(cls._model, "body_label", None) or getattr(cls._model, "body_key", None)
            if body_paths is None:
                raise RuntimeError("NewtonManager: model has no body_label/body_key, skipping USD/Fabric sync for RTX.")
            cls._usdrt_stage = get_current_stage(fabric=True)
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
            # #region agent log
            import json as _json, time as _time
            _log_path = "/home/zhengyuz/Projects/IsaacLab/.cursor/debug-dd2fc5.log"
            _entry = {"sessionId": "dd2fc5", "hypothesisId": "H_terrain_load",
                      "location": "newton_manager.py:instantiate_builder_from_stage",
                      "timestamp": int(_time.time() * 1000),
                      "message": "pre add_usd terrain",
                      "data": {"num_envs": len(env_paths), "ignore_paths_count": len(ignore_paths),
                               "stage_root_layer": str(stage.GetRootLayer().identifier),
                               "builder_shape_count_before": builder.shape_count}}
            with open(_log_path, "a") as _f:
                _f.write(_json.dumps(_entry) + "\n")
            # #endregion
            builder.add_usd(stage, ignore_paths=ignore_paths, schema_resolvers=schema_resolvers)
            # #region agent log
            _entry2 = {"sessionId": "dd2fc5", "hypothesisId": "H_terrain_load",
                       "location": "newton_manager.py:instantiate_builder_from_stage",
                       "timestamp": int(_time.time() * 1000),
                       "message": "post add_usd terrain",
                       "data": {"builder_shape_count_after": builder.shape_count,
                                "builder_body_count": builder.body_count,
                                "shape_labels": builder.shape_label[:10] if builder.shape_label else []}}
            with open(_log_path, "a") as _f:
                _f.write(_json.dumps(_entry2) + "\n")
            # #endregion

            # Build a prototype from the first env (all envs assumed identical)
            _, proto_path = env_paths[0]
            proto = ModelBuilder(up_axis=up_axis)
            proto.add_usd(
                stage,
                root_path=proto_path,
                schema_resolvers=schema_resolvers,
            )

            # Add each env as a separate Newton world
            xform_cache = UsdGeom.XformCache()
            for _, env_path in env_paths:
                builder.begin_world()
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
                builder.end_world()

            cls._num_envs = len(env_paths)

        # #region agent log
        import json as _json, time as _time
        _log_path = "/home/zhengyuz/Projects/IsaacLab/.cursor/debug-dd2fc5.log"
        _entry = {"sessionId": "dd2fc5", "hypothesisId": "H_terrain_load",
                  "location": "newton_manager.py:instantiate_builder_from_stage",
                  "timestamp": int(_time.time() * 1000),
                  "message": "final builder state",
                  "data": {"builder_shape_count": builder.shape_count,
                           "builder_body_count": builder.body_count,
                           "builder_joint_count": builder.joint_count,
                           "shape_labels": builder.shape_label[:20] if builder.shape_label else [],
                           "shape_geo_types": [int(builder.shape_geo_type[i]) for i in range(min(5, builder.shape_count))] if hasattr(builder, 'shape_geo_type') else []}}
        with open(_log_path, "a") as _f:
            _f.write(_json.dumps(_entry) + "\n")
        # #endregion

        cls.set_builder(builder)

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Unified method to initialize contacts and collision pipeline.

        This method handles both Newton collision pipeline and MuJoCo contact modes.
        It ensures contacts are properly initialized with force attributes if sensors are registered.
        """
        if cls._needs_collision_pipeline:
            # Newton collision pipeline: create pipeline and generate contacts
            if cls._collision_pipeline is None:
                # Convert config to dict, handling nested hydroelastic config
                cfg_dict = cls._collision_cfg.to_dict()

                # Convert HydroelasticSDFCfg to Newton's HydroelasticSDF.Config if present
                hydro_cfg = cfg_dict.pop("sdf_hydroelastic_config", None)
                cfg_dict["sdf_hydroelastic_config"] = HydroelasticSDF.Config(**hydro_cfg) if hydro_cfg else None

                cls._collision_pipeline = CollisionPipeline(cls._model, **cfg_dict)

            if cls._contacts is None:
                cls._contacts = cls._collision_pipeline.contacts()

        elif cls._solver is not None and isinstance(cls._solver, SolverMuJoCo):
            # MuJoCo contacts mode: create properly sized Contacts object
            # The solver's update_contacts() will populate this from MuJoCo data
            rigid_contact_max = cls._solver.get_max_contact_count()
            cls._contacts = Contacts(
                rigid_contact_max=rigid_contact_max,
                soft_contact_max=0,
                device=PhysicsManager._device,
                requested_attributes=cls._model.get_requested_contact_attributes(),
            )

    @classmethod
    def initialize_solver(cls) -> None:
        """Initialize the solver and collision pipeline.

        This function initializes the solver based on the specified solver type. Currently, only XPBD and MuJoCoWarp
        are supported. If the solver requires external collision detection (i.e., not using MuJoCo's internal
        contacts), a unified collision pipeline is created.

        The graphing of the simulation is performed in this function if the simulation is ran using
        a CUDA enabled device.

        .. warning::
            When using a CUDA enabled device, the simulation will be graphed. This means that this function steps the
            simulation once to capture the graph. Hence, this function should only be called after everything else in
            the simulation is initialized.
        """
        cfg = PhysicsManager._cfg
        if cfg is None:
            return

        with Timer(name="newton_initialize_solver", msg="Initialize solver took:"):
            cls._num_substeps = cfg.num_substeps  # type: ignore[union-attr]
            cls._solver_dt = cls.get_physics_dt() / cls._num_substeps

            # Create solver from config
            solver_cfg = cfg.solver_cfg  # type: ignore[union-attr]
            cfg_dict = solver_cfg.to_dict() if hasattr(solver_cfg, "to_dict") else {}
            cls._solver_type = cfg_dict.pop("solver_type", "mujoco_warp")

            if cls._solver_type == "mujoco_warp":
                # SolverMuJoCo does not require distinct input & output states
                cls._use_single_state = True
                solver_sig = inspect.signature(SolverMuJoCo.__init__)
                valid_solver_args = set(solver_sig.parameters.keys()) - {"self", "model"}
                cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_solver_args}
                cls._solver = SolverMuJoCo(cls._model, **cfg_dict)
            elif cls._solver_type == "xpbd":
                cls._use_single_state = False
                cls._solver = SolverXPBD(cls._model, **cfg_dict)
            elif cls._solver_type == "featherstone":
                cls._use_single_state = False
                cls._solver = SolverFeatherstone(cls._model, **cfg_dict)
            else:
                raise ValueError(f"Invalid solver type: {cls._solver_type}")

            # Determine if we need external collision detection
            # - SolverMuJoCo with use_mujoco_contacts=True: uses internal MuJoCo collision detection
            # - SolverMuJoCo with use_mujoco_contacts=False: needs Newton's unified collision pipeline
            # - Other solvers (XPBD, Featherstone): always need Newton's unified collision pipeline
            if isinstance(cls._solver, SolverMuJoCo):
                cls._needs_collision_pipeline = not solver_cfg.use_mujoco_contacts
                if cls._collision_cfg is not None:
                    raise ValueError("NewtonManager: collision_cfg is not supported with use_mujoco_contacts=True")
            else:
                cls._needs_collision_pipeline = True

            # Store collision pipeline config
            cls._collision_cfg = getattr(cfg, "collision_cfg", None)

            # Initialize contacts and collision pipeline
            cls._initialize_contacts()

        # Prepare cubric ctypes bindings (acquires IAdapter from carb framework).
        # Adapter creation is deferred to the first sync_transforms_to_usd() call
        # at render time to avoid any startup-ordering issues with the cubric
        # plugin initialisation.
        if cls._usdrt_stage is not None:
            from isaaclab_newton.physics._cubric import CubricBindings

            cls._cubric = CubricBindings()
            if cls._cubric.initialize():
                logger.info("cubric bindings ready (adapter deferred to first render)")
            else:
                logger.warning("cubric bindings init failed; falling back to update_world_xforms()")
                cls._cubric = None

        device = PhysicsManager._device

        use_cuda_graph = cfg.use_cuda_graph and "cuda" in device  # type: ignore[union-attr]

        with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
            if use_cuda_graph:
                if cls._usdrt_stage is None:
                    # No RTX active — use standard Warp capture (cudaStreamCaptureModeGlobal).
                    with wp.ScopedCapture() as capture:
                        cls._simulate()
                    cls._graph = capture.graph
                    logger.info("Newton CUDA graph captured (standard Warp mode)")
                else:
                    # RTX is active during initialization — cudaImportExternalMemory and other
                    # non-capturable RTX ops run on background CUDA streams right now.
                    # Defer capture to the first step() call, after RTX is fully initialized
                    # and idle between render frames (clean capture window).
                    cls._graph = None
                    cls._graph_capture_pending = True
                    logger.info("Newton CUDA graph capture deferred until first step() (RTX active)")
            else:
                cls._graph = None

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
        - Run ``_simulate_physics_only()`` inside ``ScopedStream(fresh_stream)``:
          kernels dispatch to ``fresh_stream`` and are captured; ``wp.capture_while`` finds the
          active capture and inserts a conditional graph node instead of synchronising.
        - Call ``wp.capture_end(stream=fresh_stream)`` to finalise the Warp-level capture.
        - Call ``cudaStreamEndCapture`` to close the CUDA stream capture and get the graph.

        Warmup run pre-allocates all MuJoCo-Warp scratch buffers so no ``cudaMalloc`` occurs during
        capture.  ``sync_transforms_to_usd`` (which calls ``wp.synchronize_device``) is
        excluded from the capture and runs eagerly in ``step()`` after ``wp.capture_launch``.

        Returns a ``wp.Graph`` on success, or ``None`` on failure.
        """
        if _cudart is None:
            logger.warning("libcudart not available; cannot use relaxed graph capture")
            return None

        # Warmup: pre-allocate all MuJoCo-Warp scratch buffers so the capture window has
        # no new cudaMalloc calls (which are forbidden inside graph capture).
        with wp.ScopedDevice(device):
            cls._simulate_physics_only()
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
                cls._simulate_physics_only()
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

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Run one physics step without Fabric/USD sync — safe for CUDA graph capture.

        Used by :meth:`_capture_relaxed_graph` to capture only the pure physics kernels.
        ``sync_transforms_to_usd`` is excluded because it calls ``wp.synchronize_device``
        (forbidden inside graph capture) and ``wp.fabricarray`` (device-wide allocation).
        The caller (``step()``) is responsible for calling ``sync_transforms_to_usd()``
        eagerly after ``wp.capture_launch``.
        """
        if cls._needs_collision_pipeline:
            cls._collision_pipeline.collide(cls._state_0, cls._contacts)
            contacts = cls._contacts
        else:
            contacts = None

        def step_fn(state_0, state_1):
            cls._solver.step(state_0, state_1, cls._control, contacts, cls._solver_dt)

        if cls._use_single_state:
            for i in range(cls._num_substeps):
                step_fn(cls._state_0, cls._state_0)
                cls._state_0.clear_forces()
        else:
            cfg = PhysicsManager._cfg
            need_copy_on_last_substep = (cfg is not None and cfg.use_cuda_graph) and cls._num_substeps % 2 == 1  # type: ignore[union-attr]
            for i in range(cls._num_substeps):
                step_fn(cls._state_0, cls._state_1)
                if need_copy_on_last_substep and i == cls._num_substeps - 1:
                    cls._state_0.assign(cls._state_1)
                else:
                    cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                cls._state_0.clear_forces()

        if cls._report_contacts:
            eval_contacts = contacts if contacts is not None else cls._contacts
            cls._solver.update_contacts(eval_contacts, cls._state_0)
            for sensor in cls._newton_contact_sensors.values():
                sensor.update(cls._state_0, eval_contacts)

    @classmethod
    def _simulate(cls) -> None:
        """Run one simulation step with substeps and USD sync.

        Delegates physics work to :meth:`_simulate_physics_only` and then
        marks transforms dirty for the next render-cadence sync.
        """
        cls._simulate_physics_only()

        if cls._usdrt_stage is not None:
            cls._mark_transforms_dirty()

    @classmethod
    def get_solver_convergence_steps(cls) -> dict[str, float | int]:
        """Get solver convergence statistics."""
        niter = cls._solver.mjw_data.solver_niter.numpy()
        return {
            "max": np.max(niter),
            "mean": np.mean(niter),
            "min": np.min(niter),
            "std": np.std(niter),
        }

    @classmethod
    def _collect_diagnostics(cls) -> dict | None:
        """Gather per-step solver diagnostics for the debug state buffer.

        Returns None if the solver doesn't expose MuJoCo data (non-MJWarp solvers).
        All values stay on GPU; the debug buffer copies to CPU only on NaN export.
        """
        if not isinstance(cls._solver, SolverMuJoCo):
            return None

        import torch  # noqa: PLC0415

        mjd = cls._solver.mjw_data
        diag: dict = {}

        # Scalar per world -- cheap
        diag["solver_niter"] = mjd.solver_niter

        # Per-dof arrays -- wp.to_torch is zero-copy, clone happens in buffer
        diag["qfrc_applied"] = mjd.qfrc_applied
        diag["qfrc_constraint"] = mjd.qfrc_constraint
        diag["qfrc_actuator"] = mjd.qfrc_actuator
        diag["qacc"] = mjd.qacc

        # Mass matrix diagonal min -- qM is (num_worlds, nv_pad, nv_pad)
        # nv_pad may be larger than nv (actual DOFs), with zero padding.
        # Only check the first nv diagonal elements.
        nv = int(cls._solver.mjw_model.nv)
        qM = wp.to_torch(mjd.qM)
        if qM.dim() == 3:
            diag_vals = torch.diagonal(qM, dim1=-2, dim2=-1)[:, :nv]
            diag["qM_diag_min"] = diag_vals.min(dim=-1).values
        elif qM.dim() == 2:
            diag["qM_diag_min"] = torch.diag(qM)[:nv].min().unsqueeze(0)

        # Contact penetration -- vectorized scatter_reduce, no Python loop
        contact = getattr(mjd, "contact", None)
        if contact is not None:
            dist = wp.to_torch(contact.dist)
            if dist.numel() > 0:
                worldid = wp.to_torch(contact.worldid).long()
                nw = getattr(cls._model, "world_count", 0) or 1
                diag["contact_dist_min"] = torch.zeros(nw, device=dist.device).scatter_reduce(
                    0, worldid, dist, reduce="amin", include_self=True
                )

        return diag

    # State accessors (used extensively by articulation/rigid object data)
    @classmethod
    def get_model(cls) -> Model:
        """Get the Newton model."""
        return cls._model

    @classmethod
    def get_state_0(cls) -> State:
        """Get the current state."""
        return cls._state_0

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
                include_total=True,
                verbose=verbose,
            )

        cls._newton_contact_sensors[sensor_key] = sensor
        cls._report_contacts = True

        if cls._solver is not None and cls._contacts is not None and cls._contacts.force is None:
            cls._initialize_contacts()

        return sensor_key
