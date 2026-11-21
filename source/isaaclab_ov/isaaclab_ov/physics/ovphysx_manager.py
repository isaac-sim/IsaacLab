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
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from pxr import Sdf, Usd, UsdPhysics

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend
from isaaclab.scene_data.deformable_discovery import (
    deformable_prototypes,
    expand_deformable_entries,
)
from isaaclab.sim.schemas.schemas import _setup_omniphysics_deformable_body
from isaaclab.sim.simulation_context import SimulationContext

from isaaclab_ov._clone import CloneRecipe, clone_transforms_from_positions
from isaaclab_ov.cloner import OvPhysxReplicateContext
from isaaclab_ov.cloner.replicate import _serialize_stage
from isaaclab_ov.stage import OvstageBackendCfg

from . import ovphysx_backend
from .ovphysx_backend import OvPhysxBackend
from .ovphysx_backend_cfg import OvPhysxBackendCfg
from .ovphysx_scene_data import OvPhysxSceneDataBackend

if TYPE_CHECKING:
    from .ovphysx_manager_cfg import OvPhysxCfg

__all__ = ["OvPhysxManager", "OvPhysxSceneDataBackend"]


def _set_scene_dynamics_device(scene_prim: Usd.Prim, device: str) -> None:
    """Select CPU or GPU dynamics, with the matching broadphase, on a physics scene prim.

    The PhysxSchema USD plugin may not be loaded in standalone ovphysx mode, so the ``PhysxSceneAPI``
    entry and the scene attributes are written through raw metadata and attributes.

    Args:
        scene_prim: A ``UsdPhysics.Scene`` prim.
        device: Resolved physics device, either ``"cpu"`` or ``"gpu"``.
    """
    schemas = Sdf.TokenListOp()
    current = scene_prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    items = list(current.prependedItems) if current.prependedItems else []
    if "PhysxSceneAPI" not in items:
        items.append("PhysxSceneAPI")
    schemas.prependedItems = items
    scene_prim.SetMetadata("apiSchemas", schemas)

    # The schema default enables GPU dynamics, so CPU scenes must author it explicitly.
    use_gpu = device == "gpu"
    scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(use_gpu)
    scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.String).Set("GPU" if use_gpu else "MBP")


logger = logging.getLogger(__name__)


class _OvPhysxEvent(Enum):
    """Internal binding phase after native cloning and before render expansion.

    It also initializes late consumers on ordinary resets, independently of MODEL_INIT's
    construction callbacks, which run only during warmup or rebuild.
    """

    BINDINGS_READY = "ovphysx_bindings_ready"


class OvPhysxManager(PhysicsManager):
    """Manages an ovphysx-backed physics simulation lifecycle.

    Unlike PhysxManager, this manager does not depend on a host Kit or
    Carbonite runtime, or on the Omniverse timeline. It drives the simulation
    through the OVPhysX Python wheel and its packaged runtime.

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    def __init__(self, cfg: OvPhysxCfg):
        super().__init__(cfg)
        self._cfg: OvPhysxCfg = cfg
        self.backend: OvPhysxBackend | None = None
        """Native resource borrowed from the context registry after warmup."""
        self._stage_usda: str | None = None
        self._warmup_done: bool = False
        self._requires_full_stage: bool = False
        self._clone_recipes: list[CloneRecipe] = []
        self._atexit_registered: bool = False
        self._scene_data_backend: OvPhysxSceneDataBackend | None = None

    backend_name = "ovphysx"
    clone_context_type = OvPhysxReplicateContext

    binding_init_event = _OvPhysxEvent.BINDINGS_READY
    """Binding phase before render cloning and public readiness when a shared stage is borrowed.

    Otherwise, its subscribers run with PHYSICS_READY subscribers in callback priority order.
    """

    def dispatch_event(self, event: PhysicsEvent, payload: Any = None) -> None:
        """Resolve binding readiness using the stage ownership selected after callback registration.

        Args:
            event: Lifecycle event to dispatch.
            payload: Optional data to pass to callbacks.
        """
        if event == PhysicsEvent.PHYSICS_READY and (self.backend is None or self.backend.owns_scene):
            self._dispatch_events((self.binding_init_event, event), payload)
        else:
            super().dispatch_event(event, payload)

    @property
    def kinematics_dirty(self) -> bool:
        """Whether the native resource needs an articulation kinematics update."""
        return self.backend is not None and self.backend.kinematics_dirty

    @kinematics_dirty.setter
    def kinematics_dirty(self, value: bool) -> None:
        if self.backend is not None:
            self.backend.kinematics_dirty = value

    def get_dt(self) -> float:
        """Get the physics timestep. Alias for get_physics_dt()."""
        return self.get_physics_dt()

    def require_full_stage(self) -> None:
        """Load every authored environment during the next stage warmup."""
        self._requires_full_stage = True

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
    def setup_deformable_body(cls, prim: Any, deformable_type: str, sim_mesh_prim: Any, vis_mesh_prim: Any) -> None:
        """Apply the OmniPhysics deformable anchor APIs, rest state, and bind pose.

        OVPhysX reads the same OmniPhysics deformable schemas as the Kit PhysX backend, so both
        backends share one authoring path.
        """
        _setup_omniphysics_deformable_body(prim, deformable_type, sim_mesh_prim, vis_mesh_prim)

    def register_clone(
        self, source: str, targets: list[str], parent_positions: list[tuple[float, float, float]] | None = None
    ) -> None:
        """Queue clones at the given world positions with identity rotations.

        Args:
            source: Source prim path (env_0 articulation root).
            targets: Target prim paths for env_1..N.
            parent_positions: Final world positions (x, y, z) [m] for whole-environment
                target roots. Each position uses an identity rotation.
        """
        target_transforms = clone_transforms_from_positions(parent_positions or [])
        self._clone_recipes.append((source, targets, target_transforms, None, 0))

    @classmethod
    def _prepare_stage_creation(cls) -> None:
        """Register OvPhysX USD schemas before creating the selected backend's stage."""
        cls._ensure_physx_schemas_registered()

    @classmethod
    def _ensure_physx_schemas_registered(cls) -> None:
        """Register the USD schemas consumed by the OVPhysX runtime."""
        ovphysx_backend._ensure_physx_schemas_registered()

    def initialize(self, sim_context: SimulationContext) -> None:
        """Initialize the physics manager with simulation context.

        This stores the config and device but does not load the USD stage yet --
        the stage may not be fully populated at this point.  The actual load
        happens lazily in :meth:`reset`.

        The simulation registry retains its native resource across reinitialization.
        """
        super().initialize(sim_context)
        self._ensure_physx_schemas_registered()
        self._warmup_done = False
        self._requires_full_stage = False
        self._stage_usda = None
        self._clone_recipes = []
        # SimulationContext retains this backend; native bindings remain deferred until the first SDP read.
        self._scene_data_backend = OvPhysxSceneDataBackend()
        self.kinematics_dirty = False

    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        On the first (non-soft) reset the method:
        - Serializes the current USD stage in memory
        - Creates the ovphysx.PhysX instance
        - Populates and attaches an OVStage
        - Runs the runtime warmup entry point (on CPU and CUDA)
        - Dispatches PHYSICS_READY

        A forced re-warm dispatches :attr:`~isaaclab.physics.PhysicsEvent.STOP`
        before replacing the attached stage so listeners discard stale bindings.
        """
        if not soft:
            if not self._warmup_done:
                if self.backend is not None and self.backend.stage is not None:
                    self.dispatch_event(PhysicsEvent.STOP, payload={})
                self._warmup_and_load()
            elif not self.backend.owns_scene:
                # Consumers added since the previous reset still need their binding callback.
                self.dispatch_event(self.binding_init_event, payload={})
            self.dispatch_event(PhysicsEvent.PHYSICS_READY, payload={})
        self.kinematics_dirty = True
        self._scene_data_backend.invalidate()

    def forward(self) -> None:
        """Evaluate and publish state changes made without stepping physics."""
        if self.backend is not None:
            self.backend.forward()

    def pre_render(self) -> None:
        """Finish native kinematics before SDP publishes manually written joint poses."""
        self.update_kinematics()

    def update_kinematics(self) -> None:
        """Update dirty articulation kinematics without publishing or rendering."""
        if self.backend is not None:
            self.backend.update_kinematics()

    def step(self) -> None:
        """Step the simulation by one physics timestep."""
        if self.backend is None or self.backend.physx is None:
            return
        dt = self.get_physics_dt()
        self.backend.step(dt)
        self._sim_time += dt

    def close(self) -> None:
        """Release ovphysx resources and clean up."""
        sim = self._sim
        # Dispatch STOP while the runtime is still live. Asset and sensor callbacks
        # invalidate raw native handles before the view registry drains the remaining
        # binding caches and the runtime is released.
        try:
            super().close()
        finally:
            try:
                if self.backend is not None:
                    try:
                        sim.close_backend(self.backend)
                    except Exception:
                        # Keep the registry owner available for a retry after native teardown fails.
                        self._sim = sim
                        raise
                    self.backend = None
                    # Keep the exit hook while a retry is possible; drop it so closed managers are not retained.
                    if self._atexit_registered:
                        atexit.unregister(self._close_at_exit)
                        self._atexit_registered = False
            finally:
                self._stage_usda = None
                self._warmup_done = False
                self._requires_full_stage = False
                self._clone_recipes = []
                # Drop the SceneDataBackend: its cached bindings and buffers
                # belong to the runtime instance just released. The next
                # SimulationContext re-creates it in initialize().
                self._scene_data_backend = None
                self.kinematics_dirty = False

    def get_physx_instance(self) -> Any:
        """Return the underlying ovphysx.PhysX instance (or None if not yet created)."""
        return None if self.backend is None else self.backend.physx

    def get_gravity(self) -> tuple[float, float, float]:
        """Return the last successfully applied world-frame gravity vector [m/s^2].

        Mirrors PhysX's ``SimulationView.get_gravity()`` so backend-agnostic sensor code
        can read gravity from their owning manager. The value tracks :meth:`set_gravity`,
        falling back to the simulation cfg until the first live update.

        Raises:
            RuntimeError: If no simulation is active. Call :meth:`initialize` first.
        """
        if self._sim is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        if self.backend is not None:
            return self.backend.gravity.get(self._sim.cfg.physics_prim_path, tuple(self._sim.cfg.gravity))
        return tuple(self._sim.cfg.gravity)

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """Set the scene-wide gravity vector through OvStage [m/s^2].

        The OvPhysX runtime accepts live scene changes only as sealed OvStage
        control updates. This method authors the scene's gravity direction and
        magnitude at the next control ordinal, seals it, and applies that
        single ordinal to the running simulation. Repeated values already
        applied successfully to the current stage do not create control updates.

        Args:
            gravity: World-frame gravity vector [m/s^2].

        Raises:
            RuntimeError: If the OVPhysX simulation has not been initialized.
            ValueError: If gravity does not contain three finite values.
        """
        if self._sim is None or self.get_physx_instance() is None or self.backend.stage is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")

        self.backend.set_gravity(gravity, physics_prim_path=self._sim.cfg.physics_prim_path)

    def get_scene_data_backend(self) -> SceneDataBackend:
        """Return the SDP backend created during initialization; native bindings are acquired lazily."""
        return self._scene_data_backend

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warmup_and_load(self) -> None:
        """Serialize the USD stage and attach it to the ovphysx runtime.

        When no runtime is active, constructs a new :class:`ovphysx.PhysX` instance for the
        simulation device and registers process-exit cleanup. CPU and CUDA simulations may follow
        each other in one process. On a forced re-warm before :meth:`close`, it reuses the active
        instance, attaches the new USD through OVStage, rebuilds active clone recipes through
        full-stage materialization or runtime replay, and re-runs the supported warmup entry point
        so the new stage's bodies are resident.

        Raises:
            RuntimeError: If ``SimulationContext`` is not set.
        """
        sim = self._sim
        if sim is None:
            raise RuntimeError("OvPhysxManager: SimulationContext is not set.")
        shared_scene = next(
            (scene for scene in sim.get_backends(OvstageBackendCfg) if scene.cfg.scene_key is None), None
        )
        if shared_scene is not None and self._requires_full_stage:
            raise ValueError(
                "Shared OVStage cloning does not support full-stage physics imports. "
                "Disable ISAAC_LAB_SHARE_OVSTAGE for this scene."
            )
        plan = sim.get_clone_plan()
        entries = None
        if plan is not None:
            env_ids = np.arange(len(plan.topology.world_prototype_layout))
            entries = expand_deformable_entries(deformable_prototypes(sim.stage, plan), plan, env_ids, plan.positions)

        ovphysx_device = "gpu" if "cuda" in self._device else "cpu"
        full_stage = self._requires_full_stage or ovphysx_device == "cpu"

        self._configure_physics_scenes(sim.stage, sim.cfg.physics_prim_path, self._cfg, ovphysx_device)

        stage_usda, native_clones = _serialize_stage(
            sim.stage,
            self._clone_recipes,
            full_stage,
            plan,
            friction_replacements=self._cfg.legacy_joint_friction_replacements,
            filter_render_only=shared_scene is not None,
            preserve_physics_ancestors=shared_scene is not None,
        )
        self._stage_usda = stage_usda

        self.backend = sim.get_or_create_backend(
            OvPhysxBackendCfg(
                device=self._device,
                cooked_collider_cache_dir=self._cfg.cooked_collider_cache_dir,
            ),
            scene=shared_scene,
        )
        # The context already holds this manager's publication, so the backend publishes into it.
        self.backend.scene_data = self._scene_data_backend
        if not self._atexit_registered:
            atexit.register(self._close_at_exit)
            self._atexit_registered = True
        # Scene-data bindings stay deferred until requested; headless training never consumes them.
        self.backend.load(stage_usda, deformable_entries=entries)
        logger.info("OvPhysxManager: attached OVStage to ovphysx (device=%s)", ovphysx_device)

        for source, targets, transforms, env_ids, _ in native_clones:
            self.backend.physx.wait_task(self.backend.physx.clone(source, targets, transforms or None, env_ids=env_ids))

        # Native metadata and bindings must see the newly attached bodies, including on CPU.
        self.backend.warmup()

        self.dispatch_event(PhysicsEvent.MODEL_INIT, payload={})
        if not self.backend.owns_scene:
            self.dispatch_event(self.binding_init_event, payload={})
        if shared_scene is not None and shared_scene.render_copies_pending:
            # Acquire publication bindings while the stage still contains only originals.
            # Complete the reserved ordinal before public callbacks can author gravity or cameras.
            self._scene_data_backend._ensure_setup()
            shared_scene.complete_render_copies()
        self._warmup_done = True

    def _close_at_exit(self) -> None:
        """Release a live OVPhysX runtime without leaking an atexit exception."""
        if self.backend is None or self.backend.physx is None:
            return
        try:
            self.close()
        except Exception:
            logger.exception("Failed to close OVPhysX during process exit.")

    def _configure_physics_scenes(
        self, stage: Usd.Stage, physics_prim_path: str, cfg: OvPhysxCfg | None, device: str
    ) -> None:
        """Configure every physics scene in a stage for the simulation device.

        The simulation's physics scene receives the full OVPhysX scene configuration. OVPhysX also
        simulates every other ``UsdPhysics.Scene`` in the stage, such as one authored by an asset, so
        each of those selects the simulation device's dynamics and broadphase.

        Args:
            stage: The USD stage to configure.
            physics_prim_path: Path of the simulation's physics scene prim.
            cfg: The configuration of the simulation's physics scene.
            device: Resolved physics device, either ``"cpu"`` or ``"gpu"``.
        """
        scene_prim = stage.GetPrimAtPath(physics_prim_path)
        if scene_prim.IsValid():
            self._configure_physx_scene_prim(scene_prim, cfg, device)
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene) and prim != scene_prim:
                _set_scene_dynamics_device(prim, device)

    def _configure_physx_scene_prim(self, scene_prim, cfg, device: str) -> None:
        """Apply the OVPhysX scene configuration to the simulation's physics scene prim.

        The schema, dynamics device and broadphase (see :func:`_set_scene_dynamics_device`),
        scene-query-support, solver-determinism/accuracy and selected contact attributes are applied
        regardless of device. The GPU buffer-capacity attributes are applied only to GPU scenes.

        Args:
            scene_prim: The /World/PhysicsScene prim to configure.
            cfg: The :class:`OvPhysxCfg` carrying solver-determinism flags and GPU buffer-capacity
                values. The GPU buffer-capacity values are only consulted when ``device == "gpu"``.
            device: Resolved physics device — one of ``"cpu"`` or ``"gpu"``.
        """
        _set_scene_dynamics_device(scene_prim, device)

        # PhysX uses the declared timestep to derive automatic collision contact offsets.
        sim_cfg = self._sim.cfg
        scene_prim.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.Int).Set(int(1.0 / sim_cfg.dt))
        scene_prim.CreateAttribute("physxScene:enableSceneQuerySupport", Sdf.ValueTypeNames.Bool).Set(
            sim_cfg.enable_scene_query_support
        )

        if cfg is not None:
            # OvPhysX answers the backend-agnostic determinism request with enhanced determinism.
            # This is best-effort: reproducibility is not verified end to end.
            scene_prim.CreateAttribute("physxScene:enableEnhancedDeterminism", Sdf.ValueTypeNames.Bool).Set(
                cfg.enable_enhanced_determinism or cfg.deterministic
            )
            scene_prim.CreateAttribute("physxScene:enableExternalForcesEveryIteration", Sdf.ValueTypeNames.Bool).Set(
                cfg.enable_external_forces_every_iteration
            )
            for attr, value_type, value in (
                ("bounceThreshold", Sdf.ValueTypeNames.Float, cfg.bounce_threshold_velocity),
                ("frictionCorrelationDistance", Sdf.ValueTypeNames.Float, cfg.friction_correlation_distance),
                ("solveArticulationContactLast", Sdf.ValueTypeNames.Bool, cfg.solve_articulation_contact_last),
            ):
                if value is not None:
                    scene_prim.CreateAttribute(f"physxScene:{attr}", value_type).Set(value)

        if device == "gpu" and cfg is not None:
            for attr, val in [
                ("gpuMaxRigidContactCount", cfg.gpu_max_rigid_contact_count),
                ("gpuMaxRigidPatchCount", cfg.gpu_max_rigid_patch_count),
                ("gpuFoundLostPairsCapacity", cfg.gpu_found_lost_pairs_capacity),
                ("gpuFoundLostAggregatePairsCapacity", cfg.gpu_found_lost_aggregate_pairs_capacity),
                ("gpuTotalAggregatePairsCapacity", cfg.gpu_total_aggregate_pairs_capacity),
                ("gpuCollisionStackSize", cfg.gpu_collision_stack_size),
            ]:
                scene_prim.CreateAttribute(f"physxScene:{attr}", Sdf.ValueTypeNames.UInt).Set(val)
