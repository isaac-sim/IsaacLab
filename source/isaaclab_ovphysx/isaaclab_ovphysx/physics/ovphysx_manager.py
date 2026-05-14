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
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar

from isaaclab.physics import PhysicsEvent, PhysicsManager

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

    from .ovphysx_manager_cfg import OvPhysxCfg

__all__ = ["OvPhysxManager"]

logger = logging.getLogger(__name__)


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
        cls._warmup_done = False
        cls._usd_handle = None
        cls._stage_path = None
        cls._pending_clones = []

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
        PhysicsManager._sim_time += dt

    @classmethod
    def close(cls) -> None:
        """Release ovphysx resources and clean up."""
        cls._release_physx()

        cls._usd_handle = None
        cls._stage_path = None
        cls._warmup_done = False

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        cls._tmp_dir = tempfile.TemporaryDirectory(prefix="isaaclab_ovphysx_")
        stage_file = os.path.join(cls._tmp_dir.name, "scene.usda")
        sim.stage.Export(stage_file)
        cls._stage_path = stage_file
        logger.info("OvPhysxManager: exported USD stage to %s", stage_file)

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
            import ovphysx as _ovphysx_bootstrap

            _ovphysx_bootstrap.bootstrap()
        finally:
            _sys.modules.update(_hidden_pxr)

        import ovphysx

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
