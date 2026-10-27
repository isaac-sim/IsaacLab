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
import importlib.util
import logging
import math
import os
import stat
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import ovphysx
import ovstage
import warp as wp

from pxr import Plug, Sdf, Usd, UsdPhysics

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat
from isaaclab.scene_data.deformable_discovery import (
    deformable_geometry_batches,
    deformable_prototypes,
    expand_deformable_entries,
)
from isaaclab.sim.schemas.schemas import _setup_omniphysics_deformable_body
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils import instantiate
from isaaclab.utils.buffers import TimestampedBuffer

from isaaclab_ov._clone import CloneRecipe, clone_transforms_from_positions
from isaaclab_ov.cloner import OvPhysxReplicateContext
from isaaclab_ov.cloner.replicate import _convert_legacy_joint_properties, _serialize_stage
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView
from isaaclab_ov.stage import OvstageBackend, OvstageBackendCfg, export_stage_to_string

from . import ovphysx_compat
from .ovphysx_compat import OVPHYSX_LIFECYCLE_ENTRY_POINTS
from .ovphysx_manager_cfg import DEFAULT_COOKED_COLLIDER_CACHE_DIR, OvPhysxBackendCfg

if TYPE_CHECKING:
    from isaaclab.scene_data.deformable_discovery import DeformableStageEntry

    from .ovphysx_manager_cfg import OvPhysxCfg

__all__ = ["OvPhysxManager", "OvPhysxSceneDataBackend"]


def _prepare_default_cache_dir(cache_dir: str) -> str:
    """Create the default cooked-collider cache directory and refuse one this user does not own.

    The default sits in the shared temporary directory, so any local user can pre-create the path;
    OVPhysX follows a symlink there and writes through it. A directory the caller configured is
    their own choice and is passed through untouched.

    Args:
        cache_dir: Default cache directory to create or validate.

    Returns:
        The validated directory.

    Raises:
        RuntimeError: If the path exists as a symlink, a non-directory, or another user's directory.
    """
    try:
        os.makedirs(cache_dir, mode=0o700)
        return cache_dir
    except FileExistsError:
        pass
    entry = os.lstat(cache_dir)
    if stat.S_ISLNK(entry.st_mode):
        raise RuntimeError(f"OVPhysX cache directory '{cache_dir}' is a symlink; refusing to write through it.")
    if not stat.S_ISDIR(entry.st_mode):
        raise RuntimeError(f"OVPhysX cache directory '{cache_dir}' exists and is not a directory.")
    if hasattr(os, "getuid") and entry.st_uid != os.getuid():
        raise RuntimeError(f"OVPhysX cache directory '{cache_dir}' is owned by another user; refusing to use it.")
    return cache_dir


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


class OvPhysxSceneDataBackend(SceneDataBackend):
    """Scene-data backend for the OVPhysX physics manager.

    The runtime's rigid-body view fills one pose buffer directly. Deformable views
    fill their native slices without an intermediate packing pass.
    """

    def __init__(self, manager):
        self._manager = manager
        self.backend: OvPhysxBackend | None = None
        """Native resource borrowed from the context registry after warmup."""
        self._transforms = TimestampedBuffer(SceneDataFormat.Transform())
        self.transforms_timestamp = 0
        self.geometry_timestamp = 0
        self._geometry = TimestampedBuffer()
        self._deformable_bindings: list[tuple[OvPhysxView, Any, wp.array]] = []
        self._pending_setup: tuple[OvPhysxBackend, str, Sequence[DeformableStageEntry] | None] | None = None

    def _defer_setup(
        self, backend: OvPhysxBackend, device: str, entries: Sequence[DeformableStageEntry] | None = None
    ) -> None:
        """Defer renderer-only bindings until scene data is requested."""
        self.backend = None
        self._transforms.data.transforms = None
        self.transforms_timestamp += 1
        self._deformable_bindings = []
        self._geometry = TimestampedBuffer()
        self.geometry_timestamp += 1
        self._pending_setup = (backend, device, entries)

    def _ensure_setup(self) -> None:
        if self._pending_setup is not None:
            backend, device, entries = self._pending_setup
            # Native output metadata names the actual bodies, not articulation-root aliases.
            # Keep these renderer-only bindings out of headless simulation startup.
            from ovphysx.types import SimObjectType

            kinds = (SimObjectType.RIGID_BODY, SimObjectType.ARTICULATION_LINK)
            body_paths = []
            if ovphysx_compat.OVPHYSX_USES_TASKS:
                for kind in kinds:
                    with backend.physx.read_sync(kind, ["mass"]) as bodies:
                        for group in bodies.groups:
                            body_paths.extend(bodies.dictionary.get_path_strings(group.prim_list))
            else:
                from ovstage import PathDictionary

                with PathDictionary() as paths:
                    for kind in kinds:
                        with backend.physx.read(kind, ["mass"]) as bodies:
                            for group in bodies.groups:
                                body_paths.extend(paths.get_path_strings(group.prim_list))
            if body_paths:
                backend.rigid_body_view = backend.physx.create_tensor_binding(prim_paths=body_paths)
            self.setup(backend, device, entries)
            self._pending_setup = None

    @property
    def transform_count(self) -> int:
        """Number of poses in the native publication."""
        self._ensure_setup()
        poses = self._transforms.data.transforms
        return 0 if poses is None else len(poses)

    @property
    def transform_paths(self) -> list[str]:
        """Native body paths in the same order as the published poses."""
        self._ensure_setup()
        view = None if self.backend is None else self.backend.rigid_body_view
        return [] if view is None else view.prim_paths

    def setup(
        self, backend: OvPhysxBackend, device: str, entries: Sequence[DeformableStageEntry] | None = None
    ) -> None:
        """Bind the native body table and declared deformable geometry.

        Args:
            backend: Initialized native physics resource.
            device: Warp device string used to allocate the published buffers.
            entries: Declared deformables captured before native stage import. ``None`` leaves
                scene geometry uninitialized; an empty sequence declares no deformables.
        """
        self.backend = backend
        self._transforms.data.transforms = None
        self.transforms_timestamp += 1
        self._deformable_bindings = []
        self.geometry_timestamp += 1
        self._geometry = TimestampedBuffer()

        if backend.rigid_body_view is not None and backend.rigid_body_view.count:
            self._transforms.data.transforms = wp.empty(
                backend.rigid_body_view.count, dtype=wp.transformf, device=device
            )

        if entries is not None:
            self._setup_deformable_bindings(backend.physx, entries, device)

    def _setup_deformable_bindings(self, physx, entries: Sequence[DeformableStageEntry], device: str) -> None:
        """Bind exact planned deformables directly into one flat publication buffer."""
        from isaaclab_ov import tensor_types as TT

        groups = {}
        for entry in entries:
            groups.setdefault((entry.deformable_type, entry.vertex_count), []).append(entry)

        views, native_entries = [], []
        for (deformable_type, count), entries in groups.items():
            tensor_type = (
                TT.DEFORMABLE_SIM_NODAL_POSITION if deformable_type == "volume" else TT.SURFACE_DEFORMABLE_SIM_POSITION
            )
            view = OvPhysxView(
                physx,
                prim_paths=[entry.root_path for entry in entries],
                device=device,
                tensor_types=[tensor_type],
                eager=True,
            )
            by_path = {path: entry for entry in entries for path in (entry.root_path, entry.sim_mesh_path)}
            ordered = [by_path[path] for path in view.prim_paths]
            if sorted(entry.root_path for entry in ordered) != sorted(entry.root_path for entry in entries):
                raise RuntimeError("OVPhysX deformable binding does not cover the declared clone-plan entries.")
            if tuple(view.binding_for(tensor_type).shape) != (len(entries), count, 3):
                raise RuntimeError("OVPhysX deformable node counts disagree with the clone plan.")
            native_entries.extend(ordered)
            views.append((view, tensor_type, count))

        if not views:
            self._geometry.data = []
            return
        counts = [entry.vertex_count for entry in native_entries]
        points = wp.empty(sum(counts), dtype=wp.vec3f, device=device)
        offset = 0
        for view, tensor_type, count in views:
            buffer = wp.array(
                ptr=points.ptr + offset * wp.types.type_size_in_bytes(wp.vec3f),
                shape=(view.count, count),
                dtype=wp.vec3f,
                device=device,
                copy=False,
            )
            self._deformable_bindings.append((view, tensor_type, buffer))
            offset += view.count * count
        offsets = np.cumsum(np.r_[0, counts[:-1]])
        self._geometry.data = deformable_geometry_batches(native_entries, offsets, device=device)
        for publication, _ in self._geometry.data:
            publication.points = points

    def get_geometry_batches(self, output_format: Any = SceneDataFormat.Points) -> list:
        """Publish native positions with exact visual paths and interpolation metadata.

        Raises:
            RuntimeError: If scene geometry was not initialized from a clone plan.
        """
        self._ensure_setup()
        if self._geometry.data is None:
            raise RuntimeError("Declare and replicate a ClonePlan before requesting scene geometry.")
        if self._geometry.timestamp != self.geometry_timestamp:
            for view, tensor_type, buffer in self._deformable_bindings:
                view.read_into(tensor_type, buffer)
            self._geometry.timestamp = self.geometry_timestamp
        return self._geometry.data

    @property
    def native_geometry_formats(self) -> tuple[Any, ...]:
        """Return the native geometry formats compiled from the declared prototypes.

        Raises:
            RuntimeError: If scene geometry was not initialized from a clone plan.
        """
        self._ensure_setup()
        if self._geometry.data is None:
            raise RuntimeError("Declare and replicate a ClonePlan before requesting scene geometry.")
        return tuple(dict.fromkeys(publication._cls for publication, _ in self._geometry.data))

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        """Publish native rigid-body poses [m, xyzw]."""
        self._ensure_setup()
        if self._transforms.timestamp != self.transforms_timestamp:
            self._manager.pre_render()
            if self._transforms.data.transforms is not None:
                self.backend.rigid_body_view.read(self._transforms.data.transforms)
            self._transforms.timestamp = self.transforms_timestamp
        return self._transforms.data


class OvPhysxBackend:
    """Own the native OVPhysX runtime and either own or borrow its stage."""

    def __init__(self, cfg: OvPhysxBackendCfg, *, scene: OvstageBackend | None = None):
        ovphysx.bootstrap()
        is_gpu = cfg.device.startswith("cuda:")
        cache_dir = cfg.cooked_collider_cache_dir
        if cache_dir == DEFAULT_COOKED_COLLIDER_CACHE_DIR:
            cache_dir = _prepare_default_cache_dir(cache_dir)
        carbonite_overrides = {
            "/physics/physxDispatcher": True,
            "/physics/updateToUsd": False,
            "/physics/updateVelocitiesToUsd": False,
            "/physics/updateParticlesToUsd": False,
            "/ovphysx/clone/useEnvIds": is_gpu,
        }
        if is_gpu:
            carbonite_overrides.update({"/physics/suppressReadback": True, "/physics/suppressFabricUpdate": True})
        # CPU scenes select CPU dynamics through their scene attributes (see _set_scene_dynamics_device).
        # OVPhysX's process-wide CPU-only mode is irreversible and would block later CUDA scenes.
        self.physx = ovphysx.PhysX(
            config=ovphysx.PhysXConfig(
                num_threads=8, cooked_collider_cache_dir=cache_dir, carbonite_overrides=carbonite_overrides
            ),
            active_cuda_gpus=cfg.device.removeprefix("cuda:") if is_gpu else None,
        )
        self.scene = scene
        self.owns_scene = scene is None
        self.stage: Any = None
        self.rigid_body_view: Any = None

    def load(self, stage_usda: str) -> None:
        """Replace the attached scene after draining its native bindings.

        Args:
            stage_usda: Exported USD scene to populate before physics attaches.
        """
        if self.stage is not None:
            if self.rigid_body_view is not None:
                self.rigid_body_view.destroy()
                self.rigid_body_view = None
            OvPhysxView._close_all_for(self.physx)
            ovphysx_compat.detach_stage(self.physx)
            self.scene.reset()
            self.stage = None
        if self.scene is None:
            # PHYSICS population currently omits native-instance collider dependencies.
            self.scene = instantiate(OvstageBackendCfg(population_domains=ovstage.PopulationDomain.ALL))
        try:
            self.scene.populate(stage_usda)
            operation = self.physx.attach_ovstage(self.scene.stage, read_ordinal=self.scene.ordinal - 1)
        except Exception:
            if self.owns_scene:
                self.scene.close()
                self.scene = None
            raise
        self.stage = self.scene.stage
        # Accepted attach failures can retain native ownership. Keep the stage for cleanup.
        ovphysx_compat.finish_stage_operation(self.physx, operation)

    def close(self) -> None:
        """Release bindings, the runtime, and its stage in native teardown order."""
        physx = self.physx
        if physx is None:
            if self.owns_scene and self.scene is not None:
                self.scene.close()
            self.stage = None
            return

        # Legacy release errors are terminal; current destroy errors may be retryable.
        destroy_entry_point = OVPHYSX_LIFECYCLE_ENTRY_POINTS["destroy"]
        release_owners = destroy_entry_point == "release"
        try:
            try:
                if self.rigid_body_view is not None:
                    self.rigid_body_view.destroy()
                    self.rigid_body_view = None
                OvPhysxView._close_all_for(physx)
            finally:
                try:
                    ovphysx_compat.detach_stage(physx)
                finally:
                    try:
                        getattr(physx, destroy_entry_point)()
                    except Exception:
                        if destroy_entry_point == "destroy":
                            # Keep both owners if native teardown did not reach its terminal state.
                            try:
                                physx.handle
                            except RuntimeError:
                                release_owners = True
                            except Exception:
                                release_owners = False
                        raise
                    else:
                        release_owners = True
        finally:
            if release_owners:
                self.physx = None
                if self.owns_scene and self.scene is not None:
                    self.scene.close()
                self.stage = None


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
        self.kinematics_dirty: bool = False
        self._gravity: tuple[float, float, float] | None = None

    # USD plugin registries are process-wide; simulation state remains instance-owned.
    _physx_schemas_registered = False

    backend_name = "ovphysx"
    clone_context_type = OvPhysxReplicateContext

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
        """Register the USD schemas consumed by the OVPhysX runtime.

        OVStage maintains its own USD schema registry, so register the selected
        PhysX provider and the separately packaged Newton schemas there even when
        the host USD runtime already provides the same plugins. For the host USD
        registry, only register providers that are not already available from a
        compiled plugin.
        """
        if cls._physx_schemas_registered:
            return
        schema_paths = ovphysx_compat.physx_schema_paths(ovphysx)
        ovstage.population.register_usd_schemas([str(path) for path in schema_paths])
        # Locate the required schema package without importing it into the host USD registry.
        ovstage.population.register_usd_schemas(os.path.dirname(importlib.util.find_spec("newton_usd_schemas").origin))
        registry = Plug.Registry()
        registered_names = {plugin.name.casefold() for plugin in registry.GetAllPlugins()}
        # The wheel documents ``<module>/resources`` as its stable layout and its
        # bundled plugin names match those module directory names case-insensitively.
        schema_paths = [str(path) for path in schema_paths if path.parent.name.casefold() not in registered_names]
        if schema_paths:
            registry.RegisterPlugins(schema_paths)
        cls._physx_schemas_registered = True

    def initialize(self, sim_context: SimulationContext) -> None:
        """Initialize the physics manager with simulation context.

        This stores the config and device but does not load the USD stage yet --
        the stage may not be fully populated at this point.  The actual load
        happens lazily in :meth:`reset`.

        The simulation registry retains its native resource across reinitialization.
        """
        super().initialize(sim_context)
        self._ensure_physx_schemas_registered()
        self._gravity = tuple(sim_context.cfg.gravity)
        self._warmup_done = False
        self._requires_full_stage = False
        self._stage_usda = None
        self._clone_recipes = []
        # SimulationContext retains this backend; native bindings remain deferred until the first SDP read.
        self._scene_data_backend = OvPhysxSceneDataBackend(self)
        self.kinematics_dirty = False

    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        On the first (non-soft) reset the method:
        - Serializes the current USD stage in memory
        - Creates the ovphysx.PhysX instance
        - Populates and attaches an OVStage
        - Warms up GPU buffers (if on CUDA)
        - Dispatches PHYSICS_READY

        A forced re-warm dispatches :attr:`~isaaclab.physics.PhysicsEvent.STOP`
        before replacing the attached stage so listeners discard stale bindings.
        """
        if not soft:
            if not self._warmup_done:
                if self.backend is not None and self.backend.stage is not None:
                    self.dispatch_event(PhysicsEvent.STOP, payload={})
                self._warmup_and_load()
            self.dispatch_event(PhysicsEvent.PHYSICS_READY, payload={})
        self.kinematics_dirty = True
        self._scene_data_backend.transforms_timestamp += 1
        self._scene_data_backend.geometry_timestamp += 1

    def forward(self) -> None:
        """Evaluate and publish state changes made without stepping physics."""
        self.update_kinematics()
        self._scene_data_backend.transforms_timestamp += 1
        self._scene_data_backend.geometry_timestamp += 1

    def pre_render(self) -> None:
        """Finish native kinematics before SDP publishes manually written joint poses."""
        self.update_kinematics()

    def update_kinematics(self) -> None:
        """Update dirty articulation kinematics without publishing or rendering."""
        if not self.kinematics_dirty:
            return
        if self.backend is not None and self.backend.physx is not None:
            self.backend.physx.update_articulations_kinematic()
            self.kinematics_dirty = False

    def step(self) -> None:
        """Step the simulation by one physics timestep."""
        if self.backend is None or self.backend.physx is None:
            return
        dt = self.get_physics_dt()
        self.backend.physx.step_sync(dt=dt)
        self.kinematics_dirty = True
        self.update_kinematics()
        self._scene_data_backend.transforms_timestamp += 1
        self._scene_data_backend.geometry_timestamp += 1
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
        if self._gravity is None:
            return tuple(self._sim.cfg.gravity)
        return self._gravity

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """Set the scene-wide gravity vector through OvStage [m/s^2].

        The OvPhysX runtime accepts live scene changes only as sealed OvStage
        control updates. This method authors the scene's gravity direction and
        magnitude at the next control ordinal, seals it, and applies that
        single ordinal to the running simulation.

        Args:
            gravity: World-frame gravity vector [m/s^2].

        Raises:
            RuntimeError: If the OVPhysX simulation has not been initialized.
            ValueError: If gravity does not contain three finite values.
        """
        if self._sim is None or self.get_physx_instance() is None or self.backend.stage is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")

        gravity_array = np.asarray(gravity, dtype=np.float32)
        if gravity_array.shape != (3,) or not np.all(np.isfinite(gravity_array)):
            raise ValueError("Gravity must contain three finite values.")

        magnitude = float(np.linalg.norm(gravity_array))
        if math.isclose(magnitude, 0.0):
            direction = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        else:
            direction = (gravity_array / magnitude).reshape(1, 3)
        scene = self.backend.scene
        # Physics must consume only this control edit, not pending rendering/output writes.
        scene.commit()
        ordinal = scene.ordinal
        with scene.query([self._sim.cfg.physics_prim_path]) as query:
            scene.stage.write_attribute(query, "physics:gravityDirection", ordinal, direction, is_array=False).wait()
            scene.stage.write_attribute(
                query, "physics:gravityMagnitude", ordinal, np.array([magnitude], dtype=np.float32), is_array=False
            ).wait()
            scene.commit()
            operation = self.backend.physx.update_from_ovstage(ordinal, ordinal)
            ovphysx_compat.finish_stage_operation(self.backend.physx, operation)

        # Publish only after successful completion; a failed native update need not be atomic.
        self._gravity = (float(gravity_array[0]), float(gravity_array[1]), float(gravity_array[2]))

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

        scene_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        if scene_prim.IsValid() and self._clone_recipes:
            scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)
        self._configure_physics_scenes(sim.stage, sim.cfg.physics_prim_path, self._cfg, ovphysx_device)

        if shared_scene is not None:
            stage_usda = export_stage_to_string(
                sim.stage,
                shared_scene.clone_env_paths,
                source_paths=tuple(source for source, _ in shared_scene.clone_copies),
                keep_env_roots=False,
                env_paths=shared_scene.population_env_paths,
            )
            if ovphysx_compat.OVPHYSX_USES_TASKS:
                layer = Sdf.Layer.CreateAnonymous()
                layer.ImportFromString(stage_usda)
                _convert_legacy_joint_properties(layer, self._cfg.legacy_joint_friction_replacements)
                stage_usda = layer.ExportToString()
            native_clones = ()
        else:
            full_stage = self._requires_full_stage or ovphysx_device == "cpu"
            stage_usda, native_clones = _serialize_stage(
                sim.stage,
                self._clone_recipes,
                full_stage,
                plan,
                friction_replacements=self._cfg.legacy_joint_friction_replacements,
            )
        self._stage_usda = stage_usda

        self.backend = sim.get_or_create_backend(
            OvPhysxBackendCfg(
                device=self._device,
                cooked_collider_cache_dir=self._cfg.cooked_collider_cache_dir,
            ),
            scene=shared_scene,
        )
        if not self._atexit_registered:
            atexit.register(self._close_at_exit)
            self._atexit_registered = True
        self.backend.load(stage_usda)
        logger.info("OvPhysxManager: attached OVStage to ovphysx (device=%s)", ovphysx_device)

        for source, targets, transforms, env_ids, _ in native_clones:
            ovphysx_compat.wait_for_operation(
                self.backend.physx, self.backend.physx.clone(source, targets, transforms or None, env_ids=env_ids)
            )

        # Native metadata and bindings must see the newly attached bodies, including on CPU.
        getattr(self.backend.physx, OVPHYSX_LIFECYCLE_ENTRY_POINTS["warmup"])()

        # The central SceneDataProvider can request these bindings later. Headless
        # training never consumes them, so avoid binding every rigid link here.
        self._scene_data_backend._defer_setup(self.backend, self._device, entries)

        self.dispatch_event(PhysicsEvent.MODEL_INIT, payload={})
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
        scene-query-support, and solver-determinism/accuracy attributes are applied regardless of
        device. The GPU buffer-capacity attributes are applied only to GPU scenes.

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
