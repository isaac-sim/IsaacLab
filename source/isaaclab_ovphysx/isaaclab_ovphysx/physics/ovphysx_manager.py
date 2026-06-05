# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX Manager for Isaac Lab.

This module manages an ovphysx-based physics simulation lifecycle without Kit dependencies.
It exports the current USD stage to disk, loads it into ovphysx, and steps the simulation
using the ovphysx C/Python API.
"""

from __future__ import annotations

import atexit
import inspect
import logging
import os
import re
import tempfile
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
                pose_binding = physx.create_tensor_binding(pattern=pattern, tensor_type=TT.RIGID_BODY_POSE)
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
                entry["pose"].read(entry["pose_buf"])
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

    Unlike PhysxManager, this manager does not depend on Kit, Carbonite, or the
    Omniverse timeline.  It drives the simulation entirely through the ovphysx
    Python wheel.

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    _cfg: ClassVar[OvPhysxCfg | None] = None
    _physx: ClassVar[Any] = None  # ovphysx.PhysX (lazy import)
    _usd_handle: ClassVar[Any] = None
    _stage_path: ClassVar[str | None] = None
    _warmup_done: ClassVar[bool] = False
    _tmp_dir: ClassVar[tempfile.TemporaryDirectory | None] = None
    # Device the process is locked to once :meth:`_warmup_and_load` constructs the
    # ``ovphysx.PhysX`` instance for the first time.  ``ovphysx<=0.3.7`` enforces
    # a process-global device-mode lock at the C++ layer (see HACK note on
    # :meth:`_release_physx`); we mirror it here so a clear Python error is raised
    # if a later :class:`~isaaclab.sim.SimulationContext` requests a different device.
    _locked_device: ClassVar[str | None] = None
    # Pending (source, targets, parent_positions) triples registered by
    # ovphysx_replicate() before the PhysX instance exists.  Replayed via
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
    def register_clone(
        cls, source: str, targets: list[str], parent_positions: list[tuple[float, float, float]] | None = None
    ) -> None:
        """Register a (source, targets, parent_positions) triple for replay via physx.clone().

        Called by :func:`~isaaclab_ovphysx.cloner.ovphysx_replicate` during
        scene setup, before the PhysX instance exists.  The clone operations
        are executed in :meth:`_warmup_and_load` immediately after
        ``physx.add_usd()``.

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
        """Register the ``PhysxSchema`` USD plugin shipped with the ovphysx wheel.

        In Kit-based runs ``omni.physx`` registers the schema; in kitless
        runs it must be registered manually before the wheel can match
        ``PhysxContactReportAPI`` and friends on the stage.  The wheel
        bundles the plugin under ``ovphysx/plugins/usd/PhysxSchema``.  This
        method is idempotent — :meth:`pxr.Plug.Registry.RegisterPlugins`
        is a no-op once the plugin is registered.
        """
        if cls._physx_schemas_registered:
            return
        try:
            import os  # noqa: PLC0415

            import ovphysx  # noqa: PLC0415

            from pxr import Plug  # noqa: PLC0415
        except Exception:
            return
        plugin_root = os.path.join(os.path.dirname(ovphysx.__file__), "plugins", "usd")
        for sub in ("PhysxSchema/resources", "PhysxSchemaAddition/resources"):
            path = os.path.join(plugin_root, sub)
            if os.path.isdir(path):
                Plug.Registry().RegisterPlugins(path)
        cls._physx_schemas_registered = True

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the physics manager with simulation context.

        This stores the config and device but does not load the USD stage yet --
        the stage may not be fully populated at this point.  The actual load
        happens lazily in :meth:`reset`.

        ``cls._physx`` is intentionally not cleared here: the ovphysx C++ instance
        is process-global (see HACK on :meth:`_release_physx`).  When a previous
        :class:`SimulationContext` has already constructed it, we reuse it rather
        than dropping the only Python reference (which would trigger the
        destructor race) or re-constructing (which would hit the wheel's
        device-mode lock).  ``cls._locked_device`` carries the device the cached
        instance is bound to.
        """
        super().initialize(sim_context)
        cls._ensure_physx_schemas_registered()
        cls._warmup_done = False
        cls._usd_handle = None
        cls._stage_path = None
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
        - Exports the current USD stage to a temp file
        - Creates the ovphysx.PhysX instance
        - Loads the exported USD
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
        sim_time = PhysicsManager._sim_time
        cls._physx.step_sync(dt=dt, sim_time=sim_time)
        cls._physx.update_articulations_kinematic()
        PhysicsManager._sim_time += dt

    @classmethod
    def close(cls) -> None:
        """Release ovphysx resources and clean up."""
        cls._release_physx()

        cls._usd_handle = None
        cls._stage_path = None
        cls._warmup_done = False
        # Drop the SceneDataBackend singleton: its cached ``TensorBinding`` handles
        # point into the wheel's prior scene which we just ``physx.reset()``-ed.
        # The next :class:`SimulationContext` re-creates the backend in
        # :meth:`initialize`. Matches Newton's lifecycle.
        cls._scene_data_backend = None

        if cls._tmp_dir is not None:
            cls._tmp_dir.cleanup()
            cls._tmp_dir = None

        super().close()

    @classmethod
    def _release_physx(cls) -> None:
        """Soft-reset the ovphysx runtime stage; keep the C++ instance alive.

        Calls ``physx.reset()`` to clear the loaded scene, but does **not** drop
        the Python reference.  The cached :class:`ovphysx.PhysX` is reused by the
        next :class:`~isaaclab.sim.SimulationContext` via the reuse path in
        :meth:`_warmup_and_load`.  Safe to call multiple times.

        HACK(ovphysx<=0.3.7): the wheel's bundled libcarb.so and Kit's libcarb.so
        coexist in the same process whenever ``import pxr`` runs (Kit USD plugins
        on ``LD_LIBRARY_PATH`` pull in Kit's Carbonite).  Both register C++ static
        destructors that race at process exit -- and crucially, also race when
        ``ovphysx.PhysX``'s Python destructor fires mid-process via refcount drop.
        So we must never let the only Python reference go to zero while the
        process is alive.  ``os._exit(0)`` (registered via ``atexit`` in
        :meth:`_warmup_and_load`) sidesteps the static-destructor phase entirely
        at process exit.  Remove this workaround once the wheel ships a
        namespace-isolated Carbonite (different soname / hidden visibility).
        """
        if cls._physx is not None:
            op = cls._physx.reset()
            cls._physx.wait_op(op)

    @classmethod
    def get_physx_instance(cls) -> Any:
        """Return the underlying ovphysx.PhysX instance (or None if not yet created)."""
        return cls._physx

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
    def _export_env0_only_stage(sim_stage: Any, target_file: str) -> None:
        """Export the simulation stage to ``target_file`` with env_1..N stripped.

        Writes a USD file containing every prim under the live stage **except**
        ``/World/envs/env_<i>`` for ``i != 0``. Globals (``/physicsScene``,
        ``/World/ground``, lights, materials, etc.) and ``/World/envs/env_0`` are
        retained.  ``physx.clone()`` is then expected to repopulate env_1..N at
        the physics layer with proper clone lineage so that subsequent
        ``create_tensor_binding`` calls hit the wheel's fast path.

        Implementation: export the full stage to disk, then re-open the result
        as an :class:`Sdf.Layer` and delete env_1..N prim specs in place.  This
        avoids mutating the live stage (which other consumers -- sensors,
        visualizers -- still see in its full N-env form).

        Limitations:
            * **Homogeneous-env assumption.** Every env is treated as an
              identical copy of env_0 from the physics runtime's point of view.
              Anything authored *only* under ``/World/envs/env_<i>`` for
              ``i != 0`` (per-env mass overrides, per-env friction, per-env
              collision filters, etc.) is dropped from the file handed to
              ``physx.add_usd`` and therefore not seen by PhysX. Sensors and
              visualizers still see those overrides in USD (the live stage is
              unmodified), so a divergence is possible.  Per-env physics state
              must instead be written via the runtime APIs
              (``RigidObject.write_root_state_to_sim_index``, etc.).
            * **Global path convention.** Any physics-relevant prim that lives
              under ``/World/envs/env_<i!=0>/`` (e.g. an asset-specific
              ``PhysicsScene``, a per-env material) gets stripped. Globals must
              live outside ``/World/envs`` (or under ``/World/envs/env_0``) to
              survive the export.
            * **Static topology.** Envs added or removed at runtime after
              warmup are not supported by ``physx.clone()`` lineage and would
              require a re-warmup with a re-exported stage.

        Args:
            sim_stage: Live USD stage held by ``SimulationContext``.
            target_file: Output ``.usda`` file path.  Overwritten if it exists.
        """
        from pxr import Sdf  # noqa: PLC0415

        # Step 1: full flatten-export of the live stage.  We pass the full file
        # to ``Sdf.Layer.OpenAsAnonymous`` so the edits below don't write back
        # to the source layer on disk.
        sim_stage.Export(target_file)

        # Step 2: open the exported file as an editable Sdf layer and delete
        # ``/World/envs/env_<digits>`` children for digits != 0.  Walking the
        # ``/World/envs`` ``PrimSpec``'s ``nameChildren`` keeps us scoped to
        # the env-namespace and leaves the rest of the stage untouched.
        layer = Sdf.Layer.FindOrOpen(target_file)
        if layer is None:
            raise RuntimeError(
                f"OvPhysxManager: failed to re-open exported USD layer at {target_file!r} for env-scoping."
            )
        envs_spec = layer.GetPrimAtPath("/World/envs")
        if envs_spec is None or not envs_spec:
            # No /World/envs in the stage (single-env or non-IsaacLab scene); nothing to scope.
            logger.debug("OvPhysxManager: no /World/envs prim — exported stage as-is.")
            return

        env_name_re = re.compile(r"^env_(\d+)$")
        names_to_remove = [
            child_name
            for child_name in list(envs_spec.nameChildren.keys())
            if (match := env_name_re.match(child_name)) and match.group(1) != "0"
        ]
        for child_name in names_to_remove:
            del envs_spec.nameChildren[child_name]

        if names_to_remove:
            layer.Export(target_file)
            logger.info(
                "OvPhysxManager: stripped %d env_<i!=0> subtrees from exported USD (kept env_0 + globals)",
                len(names_to_remove),
            )

    @classmethod
    def _warmup_and_load(cls) -> None:
        """Export the USD stage and load it into the ovphysx runtime.

        On the first call per process, constructs the :class:`ovphysx.PhysX`
        instance, registers the ``atexit`` handler, and locks the process to
        the resolved device.  On subsequent calls, reuses the cached instance
        (see HACK on :meth:`_release_physx`) -- exporting the new USD,
        re-attaching it via ``add_usd``, replaying pending clones, and (on GPU)
        re-running ``warmup_gpu`` so the new stage's bodies are resident.

        Raises:
            RuntimeError: if ``SimulationContext`` is not set, or if a device
                different from the process-locked one is requested.  The wheel
                enforces a process-global device-mode lock at the C++ layer;
                we surface it here as a clear Python error before the wheel
                would raise :exc:`ovphysx.types.PhysXDeviceError`.
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
                f"cannot switch to {ovphysx_device!r}.  ovphysx<=0.3.7 binds device mode at the C++ layer on the "
                "first ovphysx.PhysX(...) construction and it cannot be changed without restarting the process."
            )

        scene_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        if scene_prim.IsValid():
            cls._configure_physx_scene_prim(scene_prim, PhysicsManager._cfg, ovphysx_device)

        # Export the current USD stage to a temporary file so ovphysx can load it.
        #
        # When ``InteractiveScene`` runs with ``clone_usd=True``, the live USD
        # stage carries env_0..N's full asset subtrees as authored copies.
        # Handing that stage to ``physx.add_usd`` would make the wheel ingest
        # all 4096 envs as independent USD-defined bodies, defeating the
        # ``physx.clone()`` fast path and turning every subsequent
        # ``create_tensor_binding`` call into an O(N) USD enumeration -- the
        # hang you'd see at large env counts.
        #
        # The workaround: strip ``/World/envs/env_<i>`` for i != 0 from the
        # exported file before handing it to the wheel.  Sensors that read
        # USD directly (RayCaster, Camera, ContactSensor discovery) still see
        # the full N-env stage; only the wheel-side physics ingestion is
        # scoped to env_0, and ``physx.clone()`` re-populates env_1..N in
        # the physics runtime with proper clone lineage (which is what the
        # binding fast path expects).
        cls._tmp_dir = tempfile.TemporaryDirectory(prefix="isaaclab_ovphysx_")
        stage_file = os.path.join(cls._tmp_dir.name, "scene.usda")
        cls._export_env0_only_stage(sim.stage, stage_file)
        cls._stage_path = stage_file
        logger.info("OvPhysxManager: exported env_0-scoped USD stage to %s", stage_file)

        if cls._physx is None:
            cls._construct_physx(ovphysx_device, gpu_index)
            cls._locked_device = ovphysx_device
        else:
            # Reuse path: the cached PhysX may still hold the prior stage (the
            # wheel allows only one loaded USD at a time).  ``physx.reset()`` is
            # idempotent on an already-cleared stage and required when this is
            # a second :meth:`_warmup_and_load` within the same SimulationContext
            # (e.g. when a caller manually clears ``_warmup_done`` to force a
            # re-warmup).
            op = cls._physx.reset()
            cls._physx.wait_op(op)

        usd_handle, op_idx = cls._physx.add_usd(stage_file)
        cls._physx.wait_op(op_idx)
        cls._usd_handle = usd_handle
        logger.info("OvPhysxManager: loaded USD into ovphysx (device=%s)", ovphysx_device)

        # Replay pending physics clones registered by ovphysx_replicate().
        # The USD stage contains only env_0's physics; env_1..N are empty
        # Xform containers.  physx.clone() creates the remaining environments
        # in the physics runtime without modifying the USD file.
        if cls._pending_clones:
            # ovphysx_replicate() only registers pending clones when clone_usd=False,
            # meaning the USD contains only env_0 physics and physx.clone() is required
            # to populate env_1..N in the physics runtime.  Execute unconditionally —
            # no USD content heuristic is needed.
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

        # GPU bodies must be re-warmed after every add_usd: the cached PhysX
        # instance carries its old buffer layout from the previous stage.
        if ovphysx_device == "gpu":
            cls._physx.warmup_gpu()

        # Initialize the SceneDataBackend now that the wheel's PhysX is live and
        # the USD is loaded. The central
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

        Runs once per process.  Configures worker threads, registers the
        process-exit ``os._exit(0)`` handler, and stores the result on
        ``cls._physx``.  See HACK on :meth:`_release_physx` for why the
        instance must outlive every individual :class:`SimulationContext`.
        """
        # HACK (temporary): hide pxr from sys.modules during ovphysx bootstrap.
        # IsaacSim's pxr reports version 0.25.5 (pip convention) while ovphysx
        # expects 25.11 (OpenUSD release convention).  Hiding pxr causes
        # ovphysx.check_usd_compatibility() to skip the Python-side version
        # check.  This should go away once ovphysx ships a namespaced USD
        # copy with isolated symbols (same "import pxr" API, no collision).
        import sys as _sys

        _hidden_pxr = {k: _sys.modules.pop(k) for k in list(_sys.modules) if k == "pxr" or k.startswith("pxr.")}
        try:
            _ovphysx_bootstrap = import_ovphysx()
            _ovphysx_bootstrap.bootstrap()
        finally:
            _sys.modules.update(_hidden_pxr)

        ovphysx = import_ovphysx()

        physx_kwargs = {"device": ovphysx_device}
        physx_signature = inspect.signature(ovphysx.PhysX)
        physx_parameters = physx_signature.parameters
        if "active_cuda_gpus" in physx_parameters:
            if ovphysx_device == "gpu":
                # ovphysx 0.4 accepts a comma-separated CUDA ordinal string; IsaacLab selects one GPU.
                physx_kwargs["active_cuda_gpus"] = str(gpu_index)
                physx_kwargs["config"] = ovphysx.PhysXConfig(
                    carbonite_overrides={
                        "/physics/suppressReadback": True,
                        "/physics/suppressFabricUpdate": True,
                    }
                )
        elif "gpu_index" in physx_parameters:
            physx_kwargs["gpu_index"] = gpu_index

        cls._physx = ovphysx.PhysX(**physx_kwargs)

        # Without worker threads the stepper runs simulate()+fetchResults()
        # synchronously, blocking the calling thread for the full GPU step time.
        #
        # COMPAT(ovphysx<=0.3.7): The public 0.3.7 wheel exposes typed config
        # setters (set_config_int32 etc.) rather than the Carbonite-settings-based
        # set_setting() added in newer internal builds.  This guard keeps both
        # working.  REVERT once the public wheel ships set_setting().
        if hasattr(cls._physx, "set_setting"):
            cls._physx.set_setting("/persistent/physics/numThreads", "8")
            cls._physx.set_setting("/physics/physxDispatcher", "true")
            cls._physx.set_setting("/physics/updateToUsd", "false")
            cls._physx.set_setting("/physics/updateVelocitiesToUsd", "false")
            cls._physx.set_setting("/physics/updateParticlesToUsd", "false")
        else:
            cls._physx.set_config_int32(ovphysx.ConfigInt32.NUM_THREADS, 8)

        # FIXME(malesiani): re-evaluate this when carbonite ships an isolated copy.
        # At process exit, two Carbonite instances are in memory:
        #   1. ovphysx's bundled libcarb.so  (RPATH $ORIGIN/../plugins/)
        #   2. kit's libcarb.so              (pulled in via LD_LIBRARY_PATH by Fabric/usdrt plugins)
        #
        # Why does kit's libcarb end up here even though we skip AppLauncher?
        # Note: AppLauncher always starts the full Kit runtime — even headless=True
        # still loads Kit.  "Kitless" in IsaacLab means AppLauncher is not used at all.
        # But we still import `pxr` from IsaacSim's Kit USD build.  The moment `import pxr` runs, the Kit USD
        # runtime loads Fabric infrastructure (omni.physx.fabric.plugin, usdrt.population.plugin)
        # from kit's plugin directories, which are on LD_LIBRARY_PATH via setup_python_env.sh.
        # Those plugins link against kit's libcarb.so, so kit's Carbonite lands in memory
        # purely from `import pxr`, regardless of whether the Kit App is launched.
        #
        # Both Carbonite instances register C++ static destructors.  At process exit those
        # destructors race and segfault.  The workaround is to release ovphysx cleanly
        # (so GPU resources are freed) and then call os._exit() to skip the static destructor
        # phase entirely.  os._exit() terminates the process without running C++ atexit
        # handlers or static destructors, sidestepping the conflict.
        #
        # Proper long-term fix: ovphysx ships a fully namespace-isolated Carbonite
        # (different soname / hidden visibility) so its symbols never collide with kit's.
        if not cls._atexit_registered:

            def _atexit_release_and_exit():
                # Skip physx.release() -- it deadlocks due to dual-Carbonite
                # static destructor races (ovphysx's bundled libcarb vs Kit's).
                # GPU resources are reclaimed by the driver at process exit.
                os._exit(0)

            atexit.register(_atexit_release_and_exit)
            cls._atexit_registered = True

    @staticmethod
    def _configure_physx_scene_prim(scene_prim, cfg, device: str) -> None:
        """Apply PhysxSceneAPI schema and device-specific scene attributes to the
        scene prim.

        The PhysxSchema USD plugin may not be loaded in standalone ovphysx mode,
        so we write the apiSchemas list entry and scene attributes directly via
        raw Sdf metadata manipulation instead of using the high-level USD API.

        The schema and scene-query-support attribute are applied regardless of
        device. The GPU-specific dynamics/broadphase/capacity attributes are
        applied only when ``device == "gpu"`` — without them PhysX defaults to
        CPU broadphase even when ovphysx is created with ``device="gpu"``.

        Args:
            scene_prim: The /World/PhysicsScene prim to configure.
            cfg: The :class:`OvPhysxCfg` carrying GPU buffer-capacity values.
                Only consulted when ``device == "gpu"``.
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
