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
    # Pending (source, targets) pairs registered by ovphysx_replicate() before
    # the PhysX instance exists.  Replayed via physx.clone() in _warmup_and_load().
    _pending_clones: ClassVar[list[tuple[str, list[str]]]] = []

    @classmethod
    def register_clone(cls, source: str, targets: list[str]) -> None:
        """Register a (source, targets) pair to be replayed via physx.clone().

        Called by :func:`~isaaclab_ovphysx.cloner.ovphysx_replicate` during
        scene setup, before the PhysX instance exists.  The clone operations
        are executed in :meth:`_warmup_and_load` immediately after
        ``physx.add_usd()``.
        """
        cls._pending_clones.append((source, targets))

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the physics manager with simulation context.

        This stores the config and device but does not create the ovphysx
        instance yet -- the USD stage may not be fully populated at this point.
        The actual creation happens lazily in :meth:`reset`.
        """
        super().initialize(sim_context)
        cls._warmup_done = False
        cls._physx = None
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
        op_idx = cls._physx.step(dt=dt, sim_time=sim_time)
        cls._physx.wait_op(op_idx)
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
        """Release the ovphysx instance if it exists.  Safe to call multiple times."""
        if cls._physx is not None:
            try:
                cls._physx.release()
            except Exception:
                pass
            cls._physx = None

    @classmethod
    def get_physx_instance(cls) -> Any:
        """Return the underlying ovphysx.PhysX instance (or None if not yet created)."""
        return cls._physx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _warmup_and_load(cls) -> None:
        """Export the USD stage, create the ovphysx instance, and load the scene."""
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

        # Configure GPU dynamics on the PhysicsScene prim before export.
        # Without this, PhysX defaults to CPU broadphase even when
        # ovphysx is created with device="gpu".
        # We write the apiSchemas list entry and attributes directly because
        # the PhysxSchema USD plugin may not be loaded in standalone mode.
        from pxr import Sdf
        scene_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        if scene_prim.IsValid() and ovphysx_device == "gpu":
            schemas = Sdf.TokenListOp()
            current = scene_prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
            items = list(current.prependedItems) if current.prependedItems else []
            if "PhysxSceneAPI" not in items:
                items.append("PhysxSceneAPI")
            schemas.prependedItems = items
            scene_prim.SetMetadata("apiSchemas", schemas)
            scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
            scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.String).Set("GPU")
            # Apply GPU buffer capacities from cfg.  PhysX defaults (1024 for aggregate
            # pairs) are far too small for multi-env articulated simulations and produce
            # "needs to increase ... capacity" errors with missed interactions.
            cfg = PhysicsManager._cfg
            if cfg is not None:
                scene_prim.CreateAttribute(
                    "physxScene:gpuMaxRigidContactCount", Sdf.ValueTypeNames.UInt
                ).Set(cfg.gpu_max_rigid_contact_count)
                scene_prim.CreateAttribute(
                    "physxScene:gpuMaxRigidPatchCount", Sdf.ValueTypeNames.UInt
                ).Set(cfg.gpu_max_rigid_patch_count)
                scene_prim.CreateAttribute(
                    "physxScene:gpuFoundLostPairsCapacity", Sdf.ValueTypeNames.UInt
                ).Set(cfg.gpu_found_lost_pairs_capacity)
                scene_prim.CreateAttribute(
                    "physxScene:gpuFoundLostAggregatePairsCapacity", Sdf.ValueTypeNames.UInt
                ).Set(cfg.gpu_found_lost_aggregate_pairs_capacity)
                scene_prim.CreateAttribute(
                    "physxScene:gpuTotalAggregatePairsCapacity", Sdf.ValueTypeNames.UInt
                ).Set(cfg.gpu_total_aggregate_pairs_capacity)
                scene_prim.CreateAttribute(
                    "physxScene:gpuCollisionStackSize", Sdf.ValueTypeNames.UInt
                ).Set(cfg.gpu_collision_stack_size)

        # Export the current USD stage to a temporary file so ovphysx can load it.
        cls._tmp_dir = tempfile.TemporaryDirectory(prefix="isaaclab_ovphysx_")
        stage_file = os.path.join(cls._tmp_dir.name, "scene.usda")
        sim.stage.Export(stage_file)
        cls._stage_path = stage_file
        logger.info("OvPhysxManager: exported USD stage to %s", stage_file)

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
        cls._physx = ovphysx.PhysX(device=ovphysx_device, gpu_index=gpu_index)

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
        def _atexit_release_and_exit():
            cls._release_physx()
            os._exit(0)

        atexit.register(_atexit_release_and_exit)

        usd_handle, op_idx = cls._physx.add_usd(stage_file)
        cls._physx.wait_op(op_idx)
        cls._usd_handle = usd_handle
        logger.info("OvPhysxManager: loaded USD into ovphysx (device=%s)", ovphysx_device)

        # Replay pending physics clones registered by ovphysx_replicate().
        # The USD stage contains only env_0's physics; env_1..N are empty
        # Xform containers.  physx.clone() creates the remaining environments
        # in the physics runtime without modifying the USD file.
        if cls._pending_clones:
            for source, targets in cls._pending_clones:
                logger.info(
                    "OvPhysxManager: cloning %s -> %d targets (%s ... %s)",
                    source, len(targets), targets[0], targets[-1],
                )
                op_idx = cls._physx.clone(source, targets)
                cls._physx.wait_op(op_idx)
            cls._pending_clones = []

        if ovphysx_device == "gpu":
            cls._physx.warmup_gpu()

        cls.dispatch_event(PhysicsEvent.MODEL_INIT, payload={})
        cls._warmup_done = True
