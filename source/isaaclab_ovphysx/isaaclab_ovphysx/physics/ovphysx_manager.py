# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX Manager for Isaac Lab.

This module manages an ovphysx-based physics simulation lifecycle without Kit dependencies.
It serializes the current USD stage in memory, attaches it to ovphysx through OVStage, and
steps the simulation using the ovphysx C/Python API.
"""

from __future__ import annotations

import atexit
import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar

import warp as wp

from pxr import UsdPhysics

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat

from isaaclab_ovphysx._runtime import import_ovphysx

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

    from .ovphysx_manager_cfg import OvPhysxCfg

__all__ = ["OvPhysxManager", "OvPhysxSceneDataBackend"]

logger = logging.getLogger(__name__)


class OvPhysxSceneDataBackend(SceneDataBackend):
    """Scene-data backend for the OVPhysX physics manager.

    Mirrors the contract of ``PhysxSceneDataBackend`` but adapts to the
    ovphysx wheel's one-pattern-per-binding API: each distinct env-wildcard
    rigid-body prim path produces its own ``TT.RIGID_BODY_POSE`` binding.
    :attr:`transforms` reads each binding into its pre-allocated float32
    staging buffer and concatenates them into a single ``wp.transformf``
    array.

    The merged-buffer + staging-buffer separation is required because the
    wheel's ``TensorBinding.read(dst)`` writes into ``dst`` only when
    ``dst.shape == binding.shape``, so we cannot read directly into a slice
    of the merged buffer.

    Unlike PhysX -- which receives a live :class:`omni.physics.tensors.SimulationView`
    via a ``simulation_view`` property setter and discovers prims lazily --
    OVPhysX wires bindings through an explicit :meth:`setup` call that
    takes the live ``ovphysx.PhysX`` handle and the USD stage. The wheel
    exposes a ``physx + stage`` pair rather than a single ``SimulationView``,
    so a property setter would have to either bundle the two or fire on the
    second assignment; the explicit call keeps the lifecycle obvious.
    """

    def __init__(self):
        self._physx = None
        # Each entry: ``{"pattern": str, "pose": TensorBinding,
        # "pose_buf": wp.array (float32, (N, 7)),
        # "pose_buf_transformf": wp.array (transformf, (N,)),
        # "row_offset": int, "row_count": int}``.
        # The ``pose_buf_transformf`` view aliases ``pose_buf`` via zero-copy
        # ``wp.array(ptr=...)``; cached at setup time so per-step reads in
        # :attr:`transforms` don't churn Python allocations.
        self._rigid_bindings: list[dict[str, Any]] = []
        self._merged_transforms: wp.array | None = None
        self._scene_data = SceneDataFormat.Transform()

    @property
    def transform_count(self) -> int:
        """Sum of per-binding row counts."""
        return sum(int(entry["row_count"]) for entry in self._rigid_bindings)

    @property
    def transform_paths(self) -> list[str]:
        """Concatenated ``prim_paths`` across all bindings, in registration order."""
        paths: list[str] = []
        for entry in self._rigid_bindings:
            paths.extend(list(entry["pose"].prim_paths))
        return paths

    def setup(self, physx, stage, device: str) -> None:
        """Discover RigidBodyAPI prims, dedup by env-wildcard form, create one binding per pattern.

        Args:
            physx: Live ``ovphysx.PhysX`` instance (the wheel handle).
            stage: USD stage to traverse for RigidBodyAPI prims.
            device: Warp device string used to allocate the staging and merged buffers.
        """
        from isaaclab_ovphysx import tensor_types as TT  # local: keep heavy ovphysx out of module load
        from isaaclab_ovphysx.sim.views.ovphysx_view import OvPhysxView

        self._physx = physx
        self._rigid_bindings = []
        self._merged_transforms = None

        if stage is None:
            return

        # Discover RigidBodyAPI prims, dedup by env-wildcard form.
        patterns: set[str] = set()
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                patterns.add(re.sub(r"/World/envs/env_\d+", "/World/envs/env_*", prim.GetPath().pathString))

        if not patterns:
            return

        # One pose binding per distinct pattern.
        total_count = 0
        for pattern in sorted(patterns):
            try:
                view = OvPhysxView(physx, pattern=pattern, device=device)
                pose_binding = view.binding_for(TT.RIGID_BODY_POSE)
            except Exception as exc:
                logger.warning("Failed to create RIGID_BODY_POSE binding for %s: %s", pattern, exc)
                continue
            row_count = int(pose_binding.shape[0])
            if row_count == 0:
                logger.debug("Pattern %s matched 0 rigid bodies; skipping.", pattern)
                continue
            pose_buf = wp.zeros(pose_binding.shape, dtype=wp.float32, device=device)
            # Zero-copy reinterpret of the (N, 7) float32 staging buffer as (N,) wp.transformf.
            # Same pointer + layout; transformf is 7 float32s (pos.xyz + quat.xyzw). Cached
            # so per-step ``transforms`` reads don't reallocate the view object.
            pose_buf_transformf = wp.array(
                ptr=pose_buf.ptr,
                shape=(row_count,),
                dtype=wp.transformf,
                device=str(pose_buf.device),
                copy=False,
            )
            self._rigid_bindings.append(
                {
                    "pattern": pattern,
                    "view": view,
                    "pose": pose_binding,
                    "pose_buf": pose_buf,
                    "pose_buf_transformf": pose_buf_transformf,
                    "row_offset": total_count,
                    "row_count": row_count,
                }
            )
            total_count += row_count

        if total_count > 0:
            self._merged_transforms = wp.zeros((total_count,), dtype=wp.transformf, device=device)

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        """Read all bindings into the merged buffer; return as ``SceneDataFormat.Transform``.

        Each binding's float32 ``(N, 7)`` read buffer is reinterpreted as ``(N,)`` of
        ``wp.transformf`` (zero-copy via ``wp.array(ptr=..., dtype=wp.transformf)``,
        cached on the entry at setup time) and copied into the merged buffer at the
        binding's ``row_offset``.

        Returns:
            ``SceneDataFormat.Transform`` whose ``transforms`` field is a
            ``wp.array(dtype=wp.transformf)`` of length :attr:`transform_count`.
            Each ``wp.transformf`` row carries position [m] followed by
            quaternion (xyzw, unit). ``transforms`` is ``None`` when no
            bindings are wired.
        """
        if self._merged_transforms is None or not self._rigid_bindings:
            self._scene_data.transforms = self._merged_transforms
            return self._scene_data

        for entry in self._rigid_bindings:
            try:
                entry["view"].read_into("rigid_body_pose", entry["pose_buf"])
            except Exception as exc:
                logger.warning("RIGID_BODY_POSE read failed for %s: %s", entry["pattern"], exc)
                continue
            wp.copy(
                self._merged_transforms,
                entry["pose_buf_transformf"],
                dest_offset=int(entry["row_offset"]),
                src_offset=0,
                count=int(entry["row_count"]),
            )

        self._scene_data.transforms = self._merged_transforms
        return self._scene_data


class OvPhysxManager(PhysicsManager):
    """Manages an ovphysx-backed physics simulation lifecycle.

    Unlike PhysxManager, this manager does not depend on a host Kit or
    Carbonite runtime, or on the Omniverse timeline. It drives the simulation
    through the OVPhysX Python wheel and its packaged runtime.

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    _cfg: ClassVar[OvPhysxCfg | None] = None
    _physx: ClassVar[Any] = None  # ovphysx.PhysX (lazy import)
    _ovstage: ClassVar[Any] = None
    _stage_usda: ClassVar[str | None] = None
    _warmup_done: ClassVar[bool] = False
    # First device selected by IsaacLab for this process. OVPhysX 0.5.9 makes
    # CPU-only mode sticky but does not symmetrically lock GPU mode. IsaacLab
    # nevertheless pins the first choice so all contexts follow one predictable
    # process lifecycle and a CPU-first process is never asked to switch modes.
    _locked_device: ClassVar[str | None] = None
    # Pending (source, targets, parent_positions) triples queued by
    # replication is queued before the PhysX instance exists.  Replayed via
    # physx.clone() in _warmup_and_load().
    # parent_positions is a list of (x, y, z) tuples — one per target.
    _pending_clones: ClassVar[list[tuple[str, list[str], list[tuple[float, float, float]]]]] = []
    _atexit_registered: ClassVar[bool] = False
    _scene_data_backend: ClassVar[OvPhysxSceneDataBackend | None] = None

    @classmethod
    def get_dt(cls) -> float:
        """Get the physics timestep. Alias for get_physics_dt()."""
        return cls.get_physics_dt()

    @classmethod
    def fix_articulation_root(cls, articulation_prim: Any, stage: Any = None) -> Any:
        """Fix and normalize an articulation root for the OVPhysX parser."""
        root = super().fix_articulation_root(articulation_prim, stage)
        if root.HasAPI(UsdPhysics.RigidBodyAPI):
            return cls._relocate_articulation_root(
                root,
                companion_schema="PhysxArticulationAPI",
                companion_namespace="physxArticulation",
            )
        return root

    @classmethod
    def register_clone(
        cls, source: str, targets: list[str], parent_positions: list[tuple[float, float, float]] | None = None
    ) -> None:
        """Register a (source, targets, parent_positions) triple for replay via physx.clone().

        Called by :func:`~isaaclab_ovphysx.cloner.ovphysx_replicate` during
        scene setup, before the PhysX instance exists.  The clone operations
        are executed in :meth:`_warmup_and_load` immediately after
        attaching the populated OVStage.

        Args:
            source: Source prim path (env_0 articulation root).
            targets: Target prim paths for env_1..N.
            parent_positions: World positions (x, y, z) for each target's parent
                Xform prim (e.g. /World/envs/env_N).  When provided the clone
                plugin sets those transforms in Fabric so all environments start
                at their correct grid locations, preventing solver divergence
                during the warmup step.
        """
        cls._pending_clones.append((source, targets, parent_positions or []))

    _physx_schemas_registered: ClassVar[bool] = False

    @classmethod
    def _ensure_physx_schemas_registered(cls) -> None:
        """Register the codeless USD plugins published by the OVPhysX wheel.

        The wheel's public paths include both ``PhysxSchema`` and
        ``OmniUsdPhysicsDeformableSchema``. USD's plugin registry makes repeated
        registration idempotent by plugin identity, so registering all published
        paths also preserves any provider that Kit registered first.
        """
        if cls._physx_schemas_registered:
            return
        try:
            import ovphysx  # noqa: PLC0415

            from pxr import Plug  # noqa: PLC0415
        except ImportError:
            return
        schema_paths = [str(path) for path in ovphysx.codeless_schema_paths()]
        Plug.Registry().RegisterPlugins(schema_paths)
        cls._physx_schemas_registered = True

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the physics manager with simulation context.

        This stores the config and device but does not load the USD stage yet --
        the stage may not be fully populated at this point.  The actual load
        happens lazily in :meth:`reset`.

        ``cls._physx`` is intentionally not cleared here: if the current
        :class:`SimulationContext` already constructed it and has not been
        closed, the manager reuses that instance. ``cls._locked_device`` carries
        IsaacLab's conservative first-device policy for this process.
        """
        super().initialize(sim_context)
        cls._ensure_physx_schemas_registered()
        cls._warmup_done = False
        cls._stage_usda = None
        cls._pending_clones = []
        # Construct the SceneDataBackend eagerly so :class:`SimulationContext`
        # captures a real instance (not ``None``) when it builds the central
        # :class:`~isaaclab.scene.scene_data_provider.SceneDataProvider` in
        # its own ``__init__``. Bindings stay empty until :meth:`_warmup_and_load`
        # calls :meth:`OvPhysxSceneDataBackend.setup`, at which point the wheel
        # and the USD stage are live. Matches PhysX's pattern of constructing
        # the backend during ``initialize()``.
        cls._scene_data_backend = OvPhysxSceneDataBackend()

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset physics simulation.

        On the first (non-soft) reset the method:
        - Serializes the current USD stage in memory
        - Creates the ovphysx.PhysX instance
        - Populates and attaches an OVStage
        - Warms up GPU buffers (if on CUDA)
        - Dispatches PHYSICS_READY
        """
        if not soft:
            if not cls._warmup_done:
                cls._warmup_and_load()
            cls.dispatch_event(PhysicsEvent.PHYSICS_READY, payload={})

    @classmethod
    def forward(cls) -> None:
        """No-op -- ovphysx does not have a fabric/rendering pipeline."""
        pass

    @classmethod
    def step(cls) -> None:
        """Step the simulation by one physics timestep."""
        if cls._physx is None:
            return
        dt = cls.get_physics_dt()
        cls._step_physx(cls._physx, dt=dt)
        cls._physx.update_articulations_kinematic()
        PhysicsManager._sim_time += dt

    @staticmethod
    def _step_physx(physx: Any, dt: float) -> None:
        """Step the pinned OVPhysX runtime synchronously."""
        physx.step_sync(dt=dt)

    @staticmethod
    def _reset_physx_stage(physx: Any) -> None:
        """Clear the loaded stage through the pinned OVPhysX runtime API."""
        operation = physx.reset_stage()
        physx.wait_op(operation)

    @classmethod
    def close(cls) -> None:
        """Release ovphysx resources and clean up."""
        # Dispatch STOP while the runtime is still live. Asset and sensor callbacks
        # invalidate raw native handles before the view registry drains the remaining
        # binding caches and the runtime is released.
        try:
            super().close()
        finally:
            try:
                cls._release_physx()
            finally:
                cls._stage_usda = None
                cls._warmup_done = False
                # Drop the SceneDataBackend singleton: its cached bindings and buffers
                # belong to the runtime instance just released. The next
                # SimulationContext re-creates it in initialize().
                cls._scene_data_backend = None

    @classmethod
    def _release_physx(cls) -> None:
        """Release the OVPhysX runtime instance and its owned OVStage.

        Safe to call multiple times. ``_locked_device`` intentionally survives
        release so later IsaacLab contexts keep the process's first device
        choice; this is required for CPU-first processes and conservative for
        GPU-first processes.
        """
        physx = cls._physx
        cls._physx = None
        try:
            if physx is not None:
                try:
                    cls._close_physx_views(physx)
                finally:
                    try:
                        cls._reset_physx_stage(physx)
                    finally:
                        physx.release()
        finally:
            cls._destroy_ovstage()

    @classmethod
    def _attach_ovstage(cls, stage_usda: str) -> None:
        """Populate an OVStage from USDA text and attach it to the runtime."""
        import ovstage  # noqa: PLC0415

        stage = ovstage.Stage("isaaclab")
        try:
            ovstage.population.open_usd_from_string(
                stage,
                stage_usda,
                ordinal=1,
                # FIXME: Use PHYSICS once OVStage includes native-instance collider
                # dependencies in physics-only population.
                domains=ovstage.PopulationDomain.ALL,
            )
            cls._physx.attach_ovstage(stage, read_ordinal=1)
        except Exception:
            stage.destroy()
            raise
        cls._ovstage = stage

    @classmethod
    def _destroy_ovstage(cls) -> None:
        """Destroy the attached OVStage after PhysX has released its stage."""
        if cls._ovstage is not None:
            cls._ovstage.destroy()
            cls._ovstage = None

    @staticmethod
    def _close_physx_views(physx: Any) -> None:
        """Destroy every cached :class:`OvPhysxView` binding for ``physx``."""
        from isaaclab_ovphysx.sim.views.ovphysx_view import OvPhysxView  # noqa: PLC0415

        OvPhysxView._close_all_for(physx)

    @classmethod
    def _prepare_physx_for_stage_reuse(cls) -> None:
        """Invalidate stage-bound handles before resetting a cached runtime."""
        physx = cls._physx
        if physx is None:
            return
        cls.dispatch_event(PhysicsEvent.STOP, payload={})
        cls._close_physx_views(physx)
        cls._reset_physx_stage(physx)
        cls._destroy_ovstage()

    @classmethod
    def get_physx_instance(cls) -> Any:
        """Return the underlying ovphysx.PhysX instance (or None if not yet created)."""
        return cls._physx

    @classmethod
    def get_gravity(cls) -> tuple[float, float, float]:
        """Return the world-frame gravity vector [m/s^2] from the active simulation cfg.

        Mirrors PhysX's ``SimulationView.get_gravity()`` so backend-agnostic sensor code
        can read gravity through one classmethod.

        Raises:
            RuntimeError: If no simulation is active. Call :meth:`initialize` first.
        """
        if cls._sim is None or not hasattr(cls._sim, "cfg"):
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        return cls._sim.cfg.gravity

    @classmethod
    def get_scene_data_backend(cls) -> SceneDataBackend:
        """Return the SceneDataBackend for the central SceneDataProvider.

        Constructed eagerly in :meth:`initialize` so :class:`SimulationContext`
        captures a real instance (not ``None``) when wiring up the central
        :class:`~isaaclab.scene.scene_data_provider.SceneDataProvider`. Bindings
        are empty until :meth:`_warmup_and_load` calls
        :meth:`OvPhysxSceneDataBackend.setup` against the live ovphysx ``PhysX``
        and USD stage; reads against an unsetup backend return empty data
        rather than raising.
        """
        return cls._scene_data_backend

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_env0_only_stage(sim_stage: Any) -> str:
        """Serialize the simulation stage in memory with cloned environments stripped.

        The returned USDA contains every prim from the composed live stage except
        ``/World/envs/env_<i>`` for ``i != 0``. This keeps global prims and the
        env-0 clone source without mutating the live stage.

        Args:
            sim_stage: Live USD stage held by ``SimulationContext``.

        Returns:
            Env-0-scoped flattened USDA content.
        """
        layer = sim_stage.Flatten()
        envs_spec = layer.GetPrimAtPath("/World/envs")
        if envs_spec is None or not envs_spec:
            logger.debug("OvPhysxManager: no /World/envs prim — serialized stage as-is.")
            return layer.ExportToString()

        env_name_re = re.compile(r"^env_(\d+)$")
        names_to_remove = [
            child_name
            for child_name in list(envs_spec.nameChildren.keys())
            if (match := env_name_re.match(child_name)) and match.group(1) != "0"
        ]
        for child_name in names_to_remove:
            del envs_spec.nameChildren[child_name]
        if names_to_remove:
            logger.info(
                "OvPhysxManager: stripped %d env_<i!=0> subtrees from in-memory USD (kept env_0 + globals)",
                len(names_to_remove),
            )
        return layer.ExportToString()

    @classmethod
    def _warmup_and_load(cls) -> None:
        """Serialize the USD stage and attach it to the ovphysx runtime.

        When no runtime is active, constructs a new :class:`ovphysx.PhysX`
        instance. The first construction also records IsaacLab's process device
        choice. On subsequent calls before :meth:`close`, it invalidates existing
        handles, reuses the active instance, attaches the new USD through OVStage,
        replays pending clones, and (on GPU) re-runs ``warmup_gpu`` so the new
        stage's bodies are resident.

        Raises:
            RuntimeError: If ``SimulationContext`` is not set, or if a device
                different from IsaacLab's first device choice is requested.
                OVPhysX CPU-only mode is process-wide and cannot be reversed;
                IsaacLab applies the same conservative policy in both
                directions for a predictable lifecycle.
        """
        sim = PhysicsManager._sim
        if sim is None:
            raise RuntimeError("OvPhysxManager: SimulationContext is not set.")

        device_str = PhysicsManager._device
        if "cuda" in device_str:
            parts = device_str.split(":")
            gpu_index = int(parts[1]) if len(parts) > 1 else 0
            ovphysx_device = "gpu"
        else:
            gpu_index = 0
            ovphysx_device = "cpu"

        if cls._locked_device is not None and ovphysx_device != cls._locked_device:
            raise RuntimeError(
                f"OvPhysxManager is locked to device {cls._locked_device!r} for the lifetime of this process; "
                f"cannot switch to {ovphysx_device!r}. IsaacLab pins the first OVPhysX device choice because "
                "CPU-only mode cannot be reversed; restart the process to use a different device."
            )

        scene_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        if scene_prim.IsValid():
            cls._configure_physx_scene_prim(scene_prim, PhysicsManager._cfg, ovphysx_device)

        # Flatten the current USD stage to USDA text so OVStage can populate it
        # without an intermediate file.
        #
        # When ``InteractiveScene`` runs with ``clone_usd=True``, the live USD
        # stage carries env_0..N's full asset subtrees as authored copies.
        # Handing that whole stage to OVStage would make the wheel ingest
        # all 4096 envs as independent USD-defined bodies, defeating the
        # ``physx.clone()`` fast path and turning every subsequent
        # ``create_tensor_binding`` call into an O(N) USD enumeration -- the
        # hang you'd see at large env counts.
        #
        # The workaround: strip ``/World/envs/env_<i>`` for i != 0 from the
        # flattened layer before handing it to the wheel. Sensors that read
        # USD directly (RayCaster, Camera, ContactSensor discovery) still see
        # the full N-env stage; only the wheel-side physics ingestion is
        # scoped to env_0, and ``physx.clone()`` re-populates env_1..N in
        # the physics runtime with proper clone lineage (which is what the
        # binding fast path expects).
        stage_usda = cls._serialize_env0_only_stage(sim.stage)
        cls._stage_usda = stage_usda
        logger.info("OvPhysxManager: serialized env_0-scoped USD stage in memory")

        if cls._physx is None:
            cls._construct_physx(ovphysx_device, gpu_index)
            cls._locked_device = ovphysx_device
        else:
            # Bindings are tied to the realized objects of one stage. Invalidate
            # asset/sensor handles and drain generic views before resetting the
            # cached runtime; PHYSICS_READY after this method rebuilds them.
            cls._prepare_physx_for_stage_reuse()

        cls._attach_ovstage(stage_usda)
        logger.info("OvPhysxManager: attached OVStage to ovphysx (device=%s)", ovphysx_device)

        # Replay pending physics clones registered by ovphysx_replicate().
        # The USD stage contains only env_0's physics; env_1..N are empty
        # Xform containers.  physx.clone() creates the remaining environments
        # in the physics runtime without modifying the attached OVStage.
        if cls._pending_clones:
            # The cfg-level OvPhysX replicator registers pending clones for physics
            # regardless of whether USD copies were also queued for rendering. Execute
            # unconditionally — no USD content heuristic is needed.
            for source, targets, parent_positions in cls._pending_clones:
                logger.info(
                    "OvPhysxManager: cloning %s -> %d targets (%s ... %s)",
                    source,
                    len(targets),
                    targets[0],
                    targets[-1],
                )
                if parent_positions:
                    transforms = [(x, y, z, 0.0, 0.0, 0.0, 1.0) for x, y, z in parent_positions]
                else:
                    transforms = None
                op_idx = cls._physx.clone(source, targets, transforms)
                cls._physx.wait_op(op_idx)
            cls._pending_clones = []

        # GPU bodies must be re-warmed after every OVStage attachment: the cached PhysX
        # instance carries its old buffer layout from the previous stage.
        if ovphysx_device == "gpu":
            cls._physx.warmup_gpu()

        # Initialize the SceneDataBackend now that the wheel's PhysX is live and
        # the OVStage is attached. The central
        # ``isaaclab.scene.scene_data_provider.SceneDataProvider`` consumes this
        # via :meth:`get_scene_data_backend`.
        if cls._scene_data_backend is None:
            cls._scene_data_backend = OvPhysxSceneDataBackend()
        cls._scene_data_backend.setup(cls._physx, sim.stage, PhysicsManager._device)

        cls.dispatch_event(PhysicsEvent.MODEL_INIT, payload={})
        cls._warmup_done = True

    @classmethod
    def _construct_physx(cls, ovphysx_device: str, gpu_index: int) -> None:
        """Bootstrap the ``ovphysx`` wheel and create the :class:`ovphysx.PhysX` instance.

        Configures worker threads, stores the result on ``cls._physx``, and
        registers normal process-exit cleanup once.
        """
        ovphysx = import_ovphysx()
        ovphysx.bootstrap()
        cls._physx = cls._create_physx_instance(ovphysx, ovphysx_device, gpu_index)
        if not cls._atexit_registered:
            # Globally retained environments may otherwise keep TensorBinding DLPack
            # caches alive until Python module finalization. Normal atexit cleanup runs
            # before that phase and preserves the process's real exit status.
            atexit.register(cls.close)
            cls._atexit_registered = True

    @staticmethod
    def _create_physx_instance(ovphysx: Any, ovphysx_device: str, gpu_index: int) -> Any:
        """Create a PhysX instance through the pinned OVPhysX runtime API.

        Args:
            ovphysx: Imported OVPhysX runtime module.
            ovphysx_device: Physics device, either ``"cpu"`` or ``"gpu"``.
            gpu_index: CUDA device ordinal selected for GPU physics.

        Returns:
            The configured ``ovphysx.PhysX`` instance.
        """

        carbonite_overrides = {
            "/physics/physxDispatcher": True,
            "/physics/updateToUsd": False,
            "/physics/updateVelocitiesToUsd": False,
            "/physics/updateParticlesToUsd": False,
        }
        if ovphysx_device == "gpu":
            carbonite_overrides.update(
                {
                    "/physics/suppressReadback": True,
                    "/physics/suppressFabricUpdate": True,
                }
            )
        ovphysx.PhysX.set_cpu_mode(ovphysx_device == "cpu")
        physx_kwargs = {
            "config": ovphysx.PhysXConfig(num_threads=8, carbonite_overrides=carbonite_overrides),
        }
        if ovphysx_device == "gpu":
            physx_kwargs["active_cuda_gpus"] = str(gpu_index)
        return ovphysx.PhysX(**physx_kwargs)

    @staticmethod
    def _configure_physx_scene_prim(scene_prim, cfg, device: str) -> None:
        """Apply PhysxSceneAPI schema and device-specific scene attributes to the
        scene prim.

        The PhysxSchema USD plugin may not be loaded in standalone ovphysx mode,
        so we write the apiSchemas list entry and scene attributes directly via
        raw Sdf metadata manipulation instead of using the high-level USD API.

        The schema, scene-query-support, and solver-determinism/accuracy attributes are applied
        regardless of device. The GPU-specific dynamics/broadphase/capacity attributes are
        applied only when ``device == "gpu"`` — without them PhysX defaults to
        CPU broadphase even when OVPhysX is configured for GPU execution.

        Args:
            scene_prim: The /World/PhysicsScene prim to configure.
            cfg: The :class:`OvPhysxCfg` carrying solver-determinism flags and GPU buffer-capacity
                values. The GPU buffer-capacity values are only consulted when ``device == "gpu"``.
            device: Resolved physics device — one of ``"cpu"`` or ``"gpu"``.
        """
        from pxr import Sdf

        schemas = Sdf.TokenListOp()
        current = scene_prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
        items = list(current.prependedItems) if current.prependedItems else []
        if "PhysxSceneAPI" not in items:
            items.append("PhysxSceneAPI")
        schemas.prependedItems = items
        scene_prim.SetMetadata("apiSchemas", schemas)

        # Propagate scene query support from SimulationCfg so omni.physx creates
        # the scene with the correct query mode.  OvPhysxCfg does not carry this field.
        sim_cfg = PhysicsManager._sim.cfg if PhysicsManager._sim is not None else None
        enable_sq = getattr(sim_cfg, "enable_scene_query_support", False)
        scene_prim.CreateAttribute("physxScene:enableSceneQuerySupport", Sdf.ValueTypeNames.Bool).Set(enable_sq)

        if cfg is not None:
            scene_prim.CreateAttribute("physxScene:enableEnhancedDeterminism", Sdf.ValueTypeNames.Bool).Set(
                cfg.enable_enhanced_determinism
            )
            scene_prim.CreateAttribute("physxScene:enableExternalForcesEveryIteration", Sdf.ValueTypeNames.Bool).Set(
                cfg.enable_external_forces_every_iteration
            )

        if device == "gpu":
            scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
            scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.String).Set("GPU")

            if cfg is not None:
                for attr, val in [
                    ("gpuMaxRigidContactCount", cfg.gpu_max_rigid_contact_count),
                    ("gpuMaxRigidPatchCount", cfg.gpu_max_rigid_patch_count),
                    ("gpuFoundLostPairsCapacity", cfg.gpu_found_lost_pairs_capacity),
                    ("gpuFoundLostAggregatePairsCapacity", cfg.gpu_found_lost_aggregate_pairs_capacity),
                    ("gpuTotalAggregatePairsCapacity", cfg.gpu_total_aggregate_pairs_capacity),
                    ("gpuCollisionStackSize", cfg.gpu_collision_stack_size),
                ]:
                    scene_prim.CreateAttribute(f"physxScene:{attr}", Sdf.ValueTypeNames.UInt).Set(val)
