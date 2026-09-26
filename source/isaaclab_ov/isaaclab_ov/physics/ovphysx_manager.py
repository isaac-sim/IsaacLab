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
import contextlib
import importlib.util
import logging
import math
import os
import re
import stat
from collections.abc import Sequence
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat
from isaaclab.scene_data.deformable_discovery import (
    deformable_geometry_batches,
    deformable_prototypes,
    expand_deformable_entries,
)
from isaaclab.sim.simulation_context import SimulationContext

from isaaclab_ov._clone import CloneRecipe, CloneTransform, clone_transforms_from_positions, ordered_clone_paths
from isaaclab_ov._runtime import import_ovphysx
from isaaclab_ov.cloner import OvPhysxReplicateContext
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView
from isaaclab_ov.stage import create_ovstage

from .ovphysx_compat import OVPHYSX_LIFECYCLE_ENTRY_POINTS, OVPHYSX_VERSION, clone_physics, supports_clone_env_ids
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


logger = logging.getLogger(__name__)

# Large articulated clone calls scale poorly in OVPhysX 0.6.3; keep each operation bounded.
_MAX_CLONE_TARGETS_PER_CALL = 512


def _newton_schema_root() -> str | None:
    """Return the installed Newton USD schema plugin root, if available."""
    spec = importlib.util.find_spec("newton_usd_schemas")
    if spec is None or spec.origin is None:
        return None
    schema_root = os.path.dirname(spec.origin)
    if not os.path.isfile(os.path.join(schema_root, "plugInfo.json")):
        return None
    return schema_root


class OvPhysxSceneDataBackend(SceneDataBackend):
    """Scene-data backend for the OVPhysX physics manager.

    Each rigid-body binding reads directly into its portion of one native pose
    buffer. Pointer aliases preserve the binding shape without staging or merging.
    """

    def __init__(self):
        self._rigid_bindings: list[tuple[OvPhysxView, wp.array]] = []
        self._transforms = SceneDataFormat.Transform()
        self.transforms_version = 0
        self._transforms_version_last_update = -1
        self.geometry_timestamp = 0
        self._geometry_timestamp_last_update = -1
        self._geometry_batches: list | None = None
        self._deformable_bindings: list[tuple[OvPhysxView, Any, wp.array]] = []
        self._pending_setup: tuple[Any, Any, str, Sequence[DeformableStageEntry] | None] | None = None

    def _defer_setup(
        self, physx: Any, stage: Any, device: str, entries: Sequence[DeformableStageEntry] | None = None
    ) -> None:
        """Defer renderer-only bindings until scene data is requested."""
        self._rigid_bindings = []
        self._transforms.transforms = None
        self.transforms_version += 1
        self._deformable_bindings = []
        self._geometry_batches = None
        self.geometry_timestamp += 1
        self._pending_setup = (physx, stage, device, entries)

    def _ensure_setup(self) -> None:
        if self._pending_setup is not None:
            self.setup(*self._pending_setup)
            self._pending_setup = None

    @property
    def transform_count(self) -> int:
        """Number of poses in the native publication."""
        self._ensure_setup()
        poses = self._transforms.transforms
        return 0 if poses is None else len(poses)

    @property
    def transform_paths(self) -> list[str]:
        """Concatenated ``prim_paths`` across all bindings, in registration order."""
        self._ensure_setup()
        return [path for view, _ in self._rigid_bindings for path in view.prim_paths]

    def setup(self, physx, stage, device: str, entries: Sequence[DeformableStageEntry] | None = None) -> None:
        """Discover RigidBodyAPI prims, dedup by env-wildcard form, create one binding per pattern.

        Args:
            physx: Live ``ovphysx.PhysX`` instance (the wheel handle).
            stage: USD stage to traverse for RigidBodyAPI prims.
            device: Warp device string used to allocate the published buffers.
            entries: Declared deformables captured before native stage import. ``None`` leaves
                scene geometry uninitialized; an empty sequence declares no deformables.
        """
        from isaaclab_ov import tensor_types as TT  # local: keep heavy ovphysx out of module load

        self._rigid_bindings = []
        self._transforms.transforms = None
        self.transforms_version += 1
        self._deformable_bindings = []
        self.geometry_timestamp += 1
        self._geometry_batches = None

        if stage is None:
            return

        # Discover RigidBodyAPI prims, dedup by env-wildcard form.
        patterns: set[str] = set()
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                patterns.add(re.sub(r"/World/envs/env_\d+", "/World/envs/env_*", prim.GetPath().pathString))

        views = []
        for pattern in sorted(patterns):
            view = OvPhysxView(physx, pattern=pattern, device=device)
            view.binding_for(TT.RIGID_BODY_POSE)
            if view.count == 0:
                logger.debug("Pattern %s matched 0 rigid bodies; skipping.", pattern)
                view.close()
                continue
            views.append(view)

        if views:
            poses = wp.empty(sum(view.count for view in views), dtype=wp.transformf, device=device)
            self._transforms.transforms = poses
            offset = 0
            for view in views:
                buffer = wp.array(
                    ptr=poses.ptr + offset * wp.types.type_size_in_bytes(wp.transformf),
                    shape=(view.count,),
                    dtype=wp.transformf,
                    device=device,
                    copy=False,
                )
                self._rigid_bindings.append((view, buffer))
                offset += view.count

        if entries is not None:
            self._setup_deformable_bindings(physx, entries, device)

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
            self._geometry_batches = []
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
        self._geometry_batches = deformable_geometry_batches(native_entries, points, offsets)

    def get_geometry_batches(self, output_format: Any = SceneDataFormat.Points) -> list:
        """Publish native positions with exact visual paths and interpolation metadata.

        Raises:
            RuntimeError: If scene geometry was not initialized from a clone plan.
        """
        self._ensure_setup()
        if self._geometry_batches is None:
            raise RuntimeError("Declare and replicate a ClonePlan before requesting scene geometry.")
        if self._geometry_timestamp_last_update != self.geometry_timestamp:
            for view, tensor_type, buffer in self._deformable_bindings:
                view.read_into(tensor_type, buffer)
            self._geometry_timestamp_last_update = self.geometry_timestamp
        return self._geometry_batches

    @property
    def native_geometry_formats(self) -> tuple[Any, ...]:
        """Return the native geometry formats compiled from the declared prototypes.

        Raises:
            RuntimeError: If scene geometry was not initialized from a clone plan.
        """
        self._ensure_setup()
        if self._geometry_batches is None:
            raise RuntimeError("Declare and replicate a ClonePlan before requesting scene geometry.")
        return tuple(dict.fromkeys(publication._cls for publication, _ in self._geometry_batches))

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        """Publish native rigid-body poses [m, xyzw]."""
        self._ensure_setup()
        if self._transforms_version_last_update != self.transforms_version:
            OvPhysxManager.pre_render()
            for view, buffer in self._rigid_bindings:
                view.read_into("rigid_body_pose", buffer)
            self._transforms_version_last_update = self.transforms_version
        return self._transforms


class OvPhysxBackend:
    """Own the native OVPhysX runtime and its attached OVStage for one simulation."""

    def __init__(self, cfg: OvPhysxBackendCfg):
        ovphysx = import_ovphysx()
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
            # Retained heterogeneous sources use USD collision groups instead.
            "/ovphysx/clone/useEnvIds": cfg.use_env_ids,
        }
        if is_gpu:
            carbonite_overrides.update({"/physics/suppressReadback": True, "/physics/suppressFabricUpdate": True})
        ovphysx.PhysX.set_cpu_mode(not is_gpu)
        self.physx = ovphysx.PhysX(
            config=ovphysx.PhysXConfig(
                num_threads=8, cooked_collider_cache_dir=cache_dir, carbonite_overrides=carbonite_overrides
            ),
            active_cuda_gpus=cfg.device.removeprefix("cuda:") if is_gpu else None,
        )
        self.stage: Any = None

    def close(self) -> None:
        """Release bindings, the runtime, and its stage in native teardown order."""
        physx = self.physx
        if physx is None:
            if self.stage is not None:
                self.stage.destroy()
                self.stage = None
            return

        # Legacy release errors are terminal; current destroy errors may be retryable.
        destroy_entry_point = OVPHYSX_LIFECYCLE_ENTRY_POINTS["destroy"]
        release_owners = destroy_entry_point == "release"
        try:
            try:
                OvPhysxView._close_all_for(physx)
            finally:
                try:
                    physx.wait_op(physx.reset_stage())
                finally:
                    try:
                        destroy = getattr(physx, destroy_entry_point, None)
                        if destroy is None:
                            raise AttributeError(
                                f"OVPhysX does not expose the selected {destroy_entry_point}() lifecycle entry point"
                            )
                        destroy()
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
                if self.stage is not None:
                    self.stage.destroy()
                    self.stage = None


class OvPhysxManager(PhysicsManager):
    """Manages an ovphysx-backed physics simulation lifecycle.

    Unlike PhysxManager, this manager does not depend on a host Kit or
    Carbonite runtime, or on the Omniverse timeline. It drives the simulation
    through the OVPhysX Python wheel and its packaged runtime.

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    clone_context_type = OvPhysxReplicateContext

    _cfg: ClassVar[OvPhysxCfg | None] = None
    backend: ClassVar[OvPhysxBackend | None] = None
    """Native runtime borrowed from the simulation registry after warmup; the registry owns its lifetime."""
    _stage_usda: ClassVar[str | None] = None
    _warmup_done: ClassVar[bool] = False
    _next_control_ordinal: ClassVar[int] = 2
    _requires_full_stage: ClassVar[bool] = False
    # Device mode is process-wide; later contexts must reuse the first selected device.
    _locked_device: ClassVar[str | None] = None
    # Active clone recipes survive the consumable pending queue so a forced
    # re-warmup can rebuild serialized-stage or runtime-only clones.
    _active_clone_recipes: ClassVar[list[CloneRecipe]] = []
    # Consumable snapshot of the active recipes. Full-stage warmup materializes
    # these into serialized USDA; env-0-only warmup replays them with physx.clone().
    _pending_clones: ClassVar[list[CloneRecipe]] = []
    _atexit_registered: ClassVar[bool] = False
    _scene_data_backend: ClassVar[OvPhysxSceneDataBackend | None] = None
    _kinematics_dirty: ClassVar[bool] = False
    # Gravity currently applied to the running scene [m/s^2]. Seeded from ``SimulationCfg.gravity``
    # in :meth:`initialize` and refreshed by :meth:`set_gravity`. ``cfg.gravity`` stays the nominal
    # value that randomization terms resample from, so live updates must not be written back to it.
    _gravity: ClassVar[tuple[float, float, float] | None] = None

    @classmethod
    def get_dt(cls) -> float:
        """Get the physics timestep. Alias for get_physics_dt()."""
        return cls.get_physics_dt()

    @classmethod
    def require_full_stage(cls) -> None:
        """Load every authored environment during the next stage warmup."""
        cls._requires_full_stage = True

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
        """Queue clones at the given world positions with identity rotations.

        Args:
            source: Source prim path (env_0 articulation root).
            targets: Target prim paths for env_1..N.
            parent_positions: Final world positions (x, y, z) [m] for whole-environment
                target roots. Each position uses an identity rotation.
        """
        target_transforms = clone_transforms_from_positions(parent_positions or [])
        cls._register_clone_transforms(source, targets, target_transforms, None)

    @classmethod
    def _register_clone_transforms(
        cls,
        source: str,
        targets: list[str],
        target_transforms: list[CloneTransform],
        target_env_ids: list[int] | None = None,
    ) -> None:
        """Register final target-root world poses for the current simulation context."""
        recipe = (
            source,
            list(targets),
            list(target_transforms),
            None if target_env_ids is None else list(target_env_ids),
        )
        cls._active_clone_recipes.append(recipe)
        cls._pending_clones.append(recipe)

    @classmethod
    def _resolved_clone_paths(cls, pattern: str, source_path: str) -> list[str] | None:
        """Return exact cloned prim paths when the recipes cover every environment."""
        sim = PhysicsManager._sim
        plan = sim.get_clone_plan() if sim is not None else None
        if plan is None or plan.env_ids is None:
            return None
        anchors = [
            source
            for source, _, _, _ in cls._active_clone_recipes
            if source_path == source or source_path.startswith(source + "/")
        ]
        if not anchors:
            return None
        suffix = source_path[len(max(anchors, key=len)) :]
        paths: list[str] = []
        for source, targets, _, _ in cls._active_clone_recipes:
            candidate = source + suffix
            if fnmatchcase(candidate, pattern) and sim.stage.GetPrimAtPath(candidate).IsValid():
                paths.extend([candidate, *(target + suffix for target in targets)])
        paths = list(dict.fromkeys(paths))
        if len(paths) != len(plan.env_ids):
            return None
        return ordered_clone_paths(paths, [pattern])

    @classmethod
    def _rearm_pending_clones(cls) -> None:
        """Refresh the consumable clone queue from active context recipes."""
        cls._pending_clones = [
            (
                source,
                list(targets),
                list(target_transforms),
                None if target_env_ids is None else list(target_env_ids),
            )
            for source, targets, target_transforms, target_env_ids in cls._active_clone_recipes
        ]

    _physx_schemas_registered: ClassVar[bool] = False

    @classmethod
    def _prepare_stage_creation(cls) -> None:
        """Register OvPhysX USD schemas before creating the selected backend's stage."""
        cls._ensure_physx_schemas_registered()

    @classmethod
    def _ensure_physx_schemas_registered(cls) -> None:
        """Register the USD schemas consumed by the OVPhysX runtime.

        OVStage maintains its own USD schema registry, so register the wheel's
        schema root and the separately packaged Newton schemas there even when
        the host USD runtime already provides the same plugins. For the host USD
        registry, only register providers that are not already available from a
        compiled plugin.
        """
        if cls._physx_schemas_registered:
            return
        try:
            import ovphysx  # noqa: PLC0415

            from pxr import Plug  # noqa: PLC0415
        except ImportError:
            return
        try:
            import ovstage  # noqa: PLC0415
        except ImportError:
            pass  # Host USD schemas can still be registered without OVStage.
        else:
            schema_root = getattr(ovphysx, "codeless_schema_root", None)
            register_ovstage_schemas = getattr(getattr(ovstage, "population", None), "register_usd_schemas", None)
            if callable(register_ovstage_schemas):
                if callable(schema_root):
                    register_ovstage_schemas(str(schema_root()))
                if (newton_schema_root := _newton_schema_root()) is not None:
                    register_ovstage_schemas(newton_schema_root)
        registry = Plug.Registry()
        registered_names = {plugin.name.casefold() for plugin in registry.GetAllPlugins()}
        # The wheel documents ``<module>/resources`` as its stable layout and its
        # bundled plugin names match those module directory names case-insensitively.
        schema_paths = [
            str(path) for path in ovphysx.codeless_schema_paths() if path.parent.name.casefold() not in registered_names
        ]
        if schema_paths:
            registry.RegisterPlugins(schema_paths)
        cls._physx_schemas_registered = True

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the physics manager with simulation context.

        This stores the config and device but does not load the USD stage yet --
        the stage may not be fully populated at this point.  The actual load
        happens lazily in :meth:`reset`.

        The simulation registry retains its native resource across reinitialization.
        ``cls._locked_device`` carries the process-wide first-device policy.
        """
        super().initialize(sim_context)
        sim_context.clone_contexts[cls.clone_context_type] = cls.clone_context_type(sim_context)
        cls._ensure_physx_schemas_registered()
        cls._gravity = tuple(sim_context.cfg.gravity)
        cls._warmup_done = False
        cls._requires_full_stage = False
        cls._stage_usda = None
        cls._pending_clones = []
        cls._active_clone_recipes = []
        # Construct the SceneDataBackend eagerly so :class:`SimulationContext`
        # captures a real instance (not ``None``) when it builds the central
        # :class:`~isaaclab.scene.scene_data_provider.SceneDataProvider` in
        # its own ``__init__``. Bindings stay empty until :meth:`_warmup_and_load`
        # calls :meth:`OvPhysxSceneDataBackend.setup`, at which point the wheel
        # and the USD stage are live. Matches PhysX's pattern of constructing
        # the backend during ``initialize()``.
        cls._scene_data_backend = OvPhysxSceneDataBackend()
        cls._kinematics_dirty = False

    @classmethod
    def reset(cls, soft: bool = False) -> None:
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
            if not cls._warmup_done:
                if cls.backend is not None and cls.backend.stage is not None:
                    cls.dispatch_event(PhysicsEvent.STOP, payload={})
                cls._warmup_and_load()
            cls.dispatch_event(PhysicsEvent.PHYSICS_READY, payload={})
        cls._kinematics_dirty = True
        cls._scene_data_backend.transforms_version += 1
        cls._scene_data_backend.geometry_timestamp += 1

    @classmethod
    def forward(cls) -> None:
        """Evaluate and publish state changes made without stepping physics."""
        if cls.backend is not None and cls.backend.physx is not None:
            cls.backend.physx.update_articulations_kinematic()
            cls._kinematics_dirty = False
        cls._scene_data_backend.transforms_version += 1
        cls._scene_data_backend.geometry_timestamp += 1

    @classmethod
    def pre_render(cls) -> None:
        """Finish native kinematics before SDP publishes manually written joint poses."""
        if cls._kinematics_dirty and cls.backend is not None and cls.backend.physx is not None:
            cls.backend.physx.update_articulations_kinematic()
            cls._kinematics_dirty = False

    @classmethod
    def step(cls) -> None:
        """Step the simulation by one physics timestep."""
        if cls.backend is None or cls.backend.physx is None:
            return
        dt = cls.get_physics_dt()
        cls.backend.physx.step_sync(dt=dt)
        cls.backend.physx.update_articulations_kinematic()
        cls._kinematics_dirty = False
        cls._scene_data_backend.transforms_version += 1
        cls._scene_data_backend.geometry_timestamp += 1
        PhysicsManager._sim_time += dt

    @staticmethod
    def _warmup_physx(physx: Any) -> None:
        """Warm a runtime through its version-selected API."""
        entry_point = OVPHYSX_LIFECYCLE_ENTRY_POINTS["warmup"]
        warmup = getattr(physx, entry_point, None)
        if warmup is None:
            raise AttributeError(f"OVPhysX does not expose the selected {entry_point}() lifecycle entry point")
        warmup()

    @classmethod
    def close(cls) -> None:
        """Release ovphysx resources and clean up."""
        sim = SimulationContext.instance()
        # Dispatch STOP while the runtime is still live. Asset and sensor callbacks
        # invalidate raw native handles before the view registry drains the remaining
        # binding caches and the runtime is released.
        try:
            super().close()
        finally:
            try:
                if cls.backend is not None:
                    sim.close_backend(cls.backend)
                    cls.backend = None
            finally:
                cls._stage_usda = None
                cls._warmup_done = False
                cls._requires_full_stage = False
                cls._active_clone_recipes = []
                cls._pending_clones = []
                # Drop the SceneDataBackend singleton: its cached bindings and buffers
                # belong to the runtime instance just released. The next
                # SimulationContext re-creates it in initialize().
                cls._scene_data_backend = None
                cls._kinematics_dirty = False
                cls._next_control_ordinal = 2

    @classmethod
    def _attach_ovstage(cls, stage_usda: str) -> None:
        """Populate an OVStage from USDA text and attach it to the runtime."""
        import ovstage  # noqa: PLC0415

        stage = create_ovstage("isaaclab")
        try:
            ovstage.population.open_usd_from_string(
                stage,
                stage_usda,
                ordinal=1,
                # FIXME: Use PHYSICS once OVStage includes native-instance collider
                # dependencies in physics-only population.
                domains=ovstage.PopulationDomain.ALL,
            )
            # ovphysx reads sealed data only: population completes the writes but never
            # commits the ordinal, so attaching at an unsealed ordinal fails the parse
            # and silently yields an empty scene.
            stage.advance_write_floor(ordinal=1).wait()
            cls.backend.physx.attach_ovstage(stage, read_ordinal=1)
        except Exception:
            stage.destroy()
            raise
        cls.backend.stage = stage

        cls._next_control_ordinal = 2

    @classmethod
    def _prepare_physx_for_stage_reuse(cls) -> None:
        """Drain stage-bound handles before reusing the active runtime for another stage."""
        physx = cls.backend.physx
        if physx is None:
            return
        OvPhysxView._close_all_for(physx)
        physx.wait_op(physx.reset_stage())
        if cls.backend.stage is not None:
            cls.backend.stage.destroy()
            cls.backend.stage = None
        cls._next_control_ordinal = 2

    @classmethod
    def get_physx_instance(cls) -> Any:
        """Return the underlying ovphysx.PhysX instance (or None if not yet created)."""
        return None if cls.backend is None else cls.backend.physx

    @classmethod
    def get_gravity(cls) -> tuple[float, float, float]:
        """Return the world-frame gravity vector [m/s^2] currently applied to the scene.

        Mirrors PhysX's ``SimulationView.get_gravity()`` so backend-agnostic sensor code
        can read gravity through one classmethod. The value tracks :meth:`set_gravity`,
        falling back to the simulation cfg until the first live update.

        Raises:
            RuntimeError: If no simulation is active. Call :meth:`initialize` first.
        """
        if cls._sim is None or not hasattr(cls._sim, "cfg"):
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        if cls._gravity is None:
            return tuple(cls._sim.cfg.gravity)
        return cls._gravity

    @classmethod
    def set_gravity(cls, gravity: tuple[float, float, float]) -> None:
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
        if cls._sim is None or cls.get_physx_instance() is None or cls.backend.stage is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")

        gravity_array = np.asarray(gravity, dtype=np.float32)
        if gravity_array.shape != (3,) or not np.all(np.isfinite(gravity_array)):
            raise ValueError("Gravity must contain three finite values.")

        magnitude = float(np.linalg.norm(gravity_array))
        if math.isclose(magnitude, 0.0):
            direction = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        else:
            direction = (gravity_array / magnitude).reshape(1, 3)
        ordinal = cls._next_control_ordinal
        cls._next_control_ordinal += 1

        import ovstage  # noqa: PLC0415

        stage = cls.backend.stage
        with contextlib.ExitStack() as cleanup:
            paths = cleanup.enter_context(ovstage.PathDictionary(stage))
            path_list = paths.create_path_list_from_strings([cls._sim.cfg.physics_prim_path])
            cleanup.callback(paths.destroy_path_list, path_list)
            query = cleanup.enter_context(stage.query_from_path_list(path_list))
            stage.write_attribute(query, "physics:gravityDirection", ordinal, direction, is_array=False).wait()
            stage.write_attribute(
                query, "physics:gravityMagnitude", ordinal, np.array([magnitude], dtype=np.float32), is_array=False
            ).wait()
            stage.advance_write_floor(ordinal=ordinal).wait()
            cls.backend.physx.update_from_ovstage(ordinal, ordinal)

        # Only publish once the ordinal has been applied, so a failed write leaves
        # :meth:`get_gravity` reporting the gravity the scene is still running with.
        cls._gravity = (float(gravity_array[0]), float(gravity_array[1]), float(gravity_array[2]))

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

    @classmethod
    def _materialize_pending_clones_in_layer(cls, layer: Any) -> int:
        """Materialize queued clone targets into a flattened stage layer.

        OVPhysX runtime cloning is unsafe after a heterogeneous full-stage load:
        cloning one leaf can disturb tensor discovery for already loaded sibling
        assets. Missing targets are copied into the flattened layer. When another
        clone has already created a target ancestor, an internal reference overlays
        the source physics without replacing authored descendants. The live USD
        stage remains unchanged.

        Args:
            layer: Flattened stage layer to augment.

        Returns:
            Number of clone targets materialized in the layer.
        """
        from pxr import Sdf, Usd  # noqa: PLC0415

        pending_clones = list(cls._pending_clones)
        cls._pending_clones.clear()
        if not pending_clones:
            return 0

        exported_stage = Usd.Stage.Open(layer)
        if exported_stage is None:
            raise RuntimeError("OvPhysxManager: failed to open the flattened full-stage layer.")

        envs_path = Sdf.Path("/World/envs")
        operations: list[tuple[Sdf.Path, Sdf.Path, bool]] = []
        processed_targets: set[Sdf.Path] = set()
        for source, targets, _, _ in pending_clones:
            source_path = Sdf.Path(source)
            if layer.GetPrimAtPath(source_path) is None:
                raise RuntimeError(f"OvPhysxManager: clone source {source!r} is absent from the full stage.")
            for target in targets:
                target_path = Sdf.Path(target)
                if target_path in processed_targets:
                    continue
                boundary_path = target_path.GetParentPath()
                for prefix in target_path.GetPrefixes():
                    if prefix.GetParentPath() == envs_path:
                        boundary_path = prefix
                        break
                if layer.GetPrimAtPath(boundary_path) is None:
                    raise RuntimeError(f"OvPhysxManager: clone target parent is absent for {target!r}.")
                operations.append((source_path, target_path, layer.GetPrimAtPath(target_path) is not None))
                processed_targets.add(target_path)

        operations.sort(key=lambda operation: len(operation[1].GetPrefixes()))
        for source_path, target_path, target_exists in operations:
            parent_path = target_path.GetParentPath()
            parent_spec = layer.GetPrimAtPath(parent_path)
            if parent_spec is None:
                generated_paths: list[Sdf.Path] = []
                # ``CreatePrimInLayer`` authors missing ancestors as ``over`` specs. Track only
                # paths absent before creation so generated ancestors become defined without
                # changing existing authored specs.
                ancestor_path = parent_path
                while layer.GetPrimAtPath(ancestor_path) is None:
                    generated_paths.append(ancestor_path)
                    ancestor_path = ancestor_path.GetParentPath()
                if Sdf.CreatePrimInLayer(layer, parent_path) is None:
                    raise RuntimeError(
                        f"OvPhysxManager: failed to materialize clone target parent {str(parent_path)!r}."
                    )
                for generated_path in generated_paths:
                    generated_spec = layer.GetPrimAtPath(generated_path)
                    if generated_spec is not None and generated_spec.specifier == Sdf.SpecifierOver:
                        generated_spec.specifier = Sdf.SpecifierDef
            if target_exists:
                target_prim = exported_stage.GetPrimAtPath(target_path)
                if not target_prim.GetReferences().AddInternalReference(source_path):
                    raise RuntimeError(f"OvPhysxManager: failed to overlay clone target {str(target_path)!r}.")
            elif not Sdf.CopySpec(layer, source_path, layer, target_path):
                raise RuntimeError(f"OvPhysxManager: failed to materialize clone target {str(target_path)!r}.")

        if operations:
            logger.info("OvPhysxManager: materialized %d clone targets in the full-stage layer", len(operations))
        return len(operations)

    @staticmethod
    def _strip_non_source_environments(layer: Any, sources: list[str]) -> int:
        """Strip authored ``env_<i>`` prims that are not clone sources from a stage layer."""
        envs_spec = layer.GetPrimAtPath("/World/envs")
        if envs_spec is None or not envs_spec:
            return 0

        envs_path = Sdf.Path("/World/envs")
        source_environment_names = {"env_0"}
        for source in sources:
            for prefix in Sdf.Path(source).GetPrefixes():
                if prefix.GetParentPath() == envs_path:
                    source_environment_names.add(prefix.name)
                    break

        env_name_re = re.compile(r"^env_(\d+)$")
        names_to_remove = [
            child_name
            for child_name in list(envs_spec.nameChildren.keys())
            if env_name_re.match(child_name) and child_name not in source_environment_names
        ]
        for child_name in names_to_remove:
            del envs_spec.nameChildren[child_name]
        return len(names_to_remove)

    @classmethod
    def _serialize_selected_stage(cls, sim_stage: Any) -> str:
        """Serialize the selected stage representation for OVStage population."""
        layer = sim_stage.Flatten()
        if cls._requires_full_stage:
            cls._materialize_pending_clones_in_layer(layer)
            logger.info("OvPhysxManager: serialized the full USD stage in memory")
        else:
            sources = [source for source, _, _, _ in cls._pending_clones]
            # Remove runtime destinations even when their environment contains another source.
            source_paths = [Sdf.Path(source) for source in sources]
            for _, targets, _, _ in cls._pending_clones:
                for target in targets:
                    target_path = Sdf.Path(target)
                    if any(source.HasPrefix(target_path) for source in source_paths):
                        raise ValueError(f"OvPhysX clone target {target!r} overlaps a clone source.")
                    target_spec = layer.GetPrimAtPath(target_path)
                    if target_spec is not None:
                        del target_spec.nameParent.nameChildren[target_spec.name]
            removed_count = cls._strip_non_source_environments(layer, sources)
            if cls._has_nonzero_clone_source():
                cls._add_clone_collision_placeholders(layer, sim_stage)
            if removed_count:
                logger.info(
                    "OvPhysxManager: stripped %d non-source env_<i> subtrees from in-memory USD",
                    removed_count,
                )
            else:
                logger.debug("OvPhysxManager: no cloned environments to strip — serialized stage as-is.")
        return layer.ExportToString()

    @classmethod
    def _has_nonzero_clone_source(cls) -> bool:
        """Whether retained physics sources occupy more than the default environment."""
        return any(
            (match := re.search(r"/env_(\d+)(?:/|$)", source)) and int(match[1]) != 0
            for source, _, _, _ in cls._active_clone_recipes
        )

    @classmethod
    def _add_clone_collision_placeholders(cls, layer: Sdf.Layer, stage: Usd.Stage) -> None:
        """Keep collection membership resolvable without loading destination physics.

        OvPhysX looks up each cloned shape in the destination collision collection.
        Plain prims are enough for that lookup; no collision geometry is duplicated.
        """
        if not any(prim.IsA(UsdPhysics.CollisionGroup) for prim in stage.Traverse()):
            return
        exported_stage = Usd.Stage.Open(layer)
        xforms = UsdGeom.XformCache()
        for source, targets, transforms, _ in cls._pending_clones:
            source_path = Sdf.Path(source)
            physics_paths = [
                prim.GetPath()
                for prim in Usd.PrimRange(stage.GetPrimAtPath(source), Usd.TraverseInstanceProxies())
                if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            paths = sorted(
                {prefix for path in physics_paths for prefix in path.GetPrefixes() if prefix.HasPrefix(source_path)}
            )
            for i, target in enumerate(targets):
                for path in paths:
                    target_path = path.ReplacePrefix(source_path, Sdf.Path(target))
                    xform = UsdGeom.Xform.Define(exported_stage, target_path)
                    source_prim = stage.GetPrimAtPath(path)
                    if "PhysxContactReportAPI" in source_prim.GetPrimTypeInfo().GetAppliedAPISchemas():
                        # Contact bindings inspect authored targets before falling back to clone lineage.
                        xform.GetPrim().AddAppliedSchema("PhysxContactReportAPI")
                    world = xforms.GetLocalToWorldTransform(source_prim)
                    if path == source_path:
                        # The runtime prefers authored target poses over clone anchors.
                        if transforms:
                            pose = transforms[i]
                            world = Gf.Matrix4d().SetRotate(Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])))
                            world.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
                        xform.AddTransformOp().Set(world)
                        xform.SetResetXformStack(True)
                    else:
                        parent_world = xforms.GetLocalToWorldTransform(source_prim.GetParent())
                        xform.AddTransformOp().Set(world * parent_world.GetInverse())

    @classmethod
    def _replay_pending_clones(cls, physx: Any, requires_full_stage: bool) -> None:
        pending_clones = list(cls._pending_clones)
        cls._pending_clones.clear()

        if requires_full_stage:
            return

        for source, targets, target_transforms, target_env_ids in pending_clones:
            if not targets:
                continue
            logger.info(
                "OvPhysxManager: cloning %s -> %d targets (%s ... %s)",
                source,
                len(targets),
                targets[0],
                targets[-1],
            )
            transforms = target_transforms or None
            for start in range(0, len(targets), _MAX_CLONE_TARGETS_PER_CALL):
                end = start + _MAX_CLONE_TARGETS_PER_CALL
                op_idx = clone_physics(
                    physx,
                    source,
                    targets[start:end],
                    transforms[start:end] if transforms is not None else None,
                    target_env_ids[start:end] if target_env_ids is not None else None,
                )
                physx.wait_op(op_idx)

    @classmethod
    def _warmup_and_load(cls) -> None:
        """Serialize the USD stage and attach it to the ovphysx runtime.

        When no runtime is active, constructs a new :class:`ovphysx.PhysX`
        instance. The first construction also records IsaacLab's process device
        choice and registers process-exit cleanup. On a forced re-warm before
        :meth:`close`, it reuses the active instance, attaches the new USD through
        OVStage, rebuilds active clone recipes through full-stage materialization
        or runtime replay, and (on GPU) re-runs the supported warmup entry point
        so the new stage's bodies are resident.

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
        if cls._has_nonzero_clone_source() and not supports_clone_env_ids(OVPHYSX_VERSION):
            raise RuntimeError("Heterogeneous OvPhysX cloning requires ovphysx>=0.6.3; use uv run --extra ovphysx.")

        entries = None
        if (plan := sim.get_clone_plan()) is not None:
            entries = expand_deformable_entries(plan, deformable_prototypes(sim.stage, plan))

        ovphysx_device = "gpu" if "cuda" in PhysicsManager._device else "cpu"

        if cls._locked_device is not None and ovphysx_device != cls._locked_device:
            raise RuntimeError(
                f"OvPhysxManager is locked to device {cls._locked_device!r} for the lifetime of this process; "
                f"cannot switch to {ovphysx_device!r}. IsaacLab pins the first OVPhysX device choice because "
                "CPU-only mode cannot be reversed; restart the process to use a different device."
            )

        scene_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        if scene_prim.IsValid():
            if cls._active_clone_recipes:
                scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)
            cls._configure_physx_scene_prim(scene_prim, PhysicsManager._cfg, ovphysx_device)

        # Serialize sources without duplicating destination physics, even when a
        # USD cloning context also authored the targets. The live stage remains
        # available to USD-based sensors; only the physics export is stripped.
        # Heterogeneous scenes retain each source variant and lightweight target
        # placeholders for collision groups. Features requiring a full stage
        # instead materialize missing targets and bypass runtime cloning.
        cls._rearm_pending_clones()
        stage_usda = cls._serialize_selected_stage(sim.stage)
        cls._stage_usda = stage_usda

        previous_backend = cls.backend
        cls.backend = sim.get_or_create_backend(
            OvPhysxBackendCfg(
                device=PhysicsManager._device,
                cooked_collider_cache_dir=sim.cfg.physics.cooked_collider_cache_dir,
                use_env_ids=not cls._has_nonzero_clone_source(),
            )
        )
        cls._locked_device = ovphysx_device
        if not cls._atexit_registered:
            atexit.register(cls._close_at_exit)
            cls._atexit_registered = True
        if cls.backend is previous_backend or cls.backend.stage is not None:
            # Bindings are tied to the realized objects of one stage. Invalidate
            # asset/sensor handles and drain generic views before resetting the
            # cached runtime; PHYSICS_READY after this method rebuilds them.
            cls._prepare_physx_for_stage_reuse()

        cls._attach_ovstage(stage_usda)
        logger.info("OvPhysxManager: attached OVStage to ovphysx (device=%s)", ovphysx_device)

        cls._replay_pending_clones(cls.backend.physx, requires_full_stage=cls._requires_full_stage)

        # GPU bodies must be re-warmed after every OVStage attachment: the cached PhysX
        # instance carries its old buffer layout from the previous stage.
        if ovphysx_device == "gpu":
            cls._warmup_physx(cls.backend.physx)

        # The central SceneDataProvider can request these bindings later. Headless
        # training never consumes them, so avoid binding every rigid link here.
        if cls._scene_data_backend is None:
            cls._scene_data_backend = OvPhysxSceneDataBackend()
        cls._scene_data_backend._defer_setup(cls.backend.physx, sim.stage, PhysicsManager._device, entries)

        cls.dispatch_event(PhysicsEvent.MODEL_INIT, payload={})
        cls._warmup_done = True

    @classmethod
    def _close_at_exit(cls) -> None:
        """Release a live OVPhysX runtime without leaking an atexit exception."""
        if cls.backend is None or cls.backend.physx is None:
            return
        try:
            sim = PhysicsManager._sim
            is_active_manager = sim is not None and (
                sim.physics_manager is cls or sim.physics_manager == f"{cls.__module__}:{cls.__qualname__}"
            )
            if is_active_manager:
                cls.close()
            else:
                # Do not clear another backend's shared callbacks or simulation
                # state if this is only a stale OVPhysX runtime.
                cls.backend.close()
        except Exception:
            logger.exception("Failed to close OVPhysX during process exit.")

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
            # OvPhysX answers the backend-agnostic determinism request with enhanced determinism.
            # This is best-effort: reproducibility is not verified end to end.
            scene_prim.CreateAttribute("physxScene:enableEnhancedDeterminism", Sdf.ValueTypeNames.Bool).Set(
                cfg.enable_enhanced_determinism or cfg.deterministic
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
