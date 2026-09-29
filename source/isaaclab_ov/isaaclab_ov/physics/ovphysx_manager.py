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
import stat
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import warp as wp

from pxr import Sdf, UsdPhysics

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat
from isaaclab.scene_data.deformable_discovery import (
    deformable_geometry_batches,
    deformable_prototypes,
    expand_deformable_entries,
)
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils.buffers import TimestampedBuffer

from isaaclab_ov._clone import CloneRecipe, clone_transforms_from_positions
from isaaclab_ov._runtime import import_ovphysx
from isaaclab_ov.cloner import OvPhysxReplicateContext
from isaaclab_ov.cloner.replicate import _serialize_stage
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView
from isaaclab_ov.stage import create_ovstage

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


logger = logging.getLogger(__name__)


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

    The runtime's rigid-body view fills one pose buffer directly. Deformable views
    fill their native slices without an intermediate packing pass.
    """

    def __init__(self):
        self.backend: OvPhysxBackend | None = None
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
            from ovstage import PathDictionary

            body_paths = []
            with PathDictionary() as paths:
                for kind in (SimObjectType.RIGID_BODY, SimObjectType.ARTICULATION_LINK):
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
            OvPhysxManager.pre_render()
            if self._transforms.data.transforms is not None:
                self.backend.rigid_body_view.read(self._transforms.data.transforms)
            self._transforms.timestamp = self.transforms_timestamp
        return self._transforms.data


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
            "/ovphysx/clone/useEnvIds": is_gpu,
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
        self.rigid_body_view: Any = None

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
                if self.rigid_body_view is not None:
                    self.rigid_body_view.destroy()
                    self.rigid_body_view = None
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
    # Retain construction inputs for hard reset; serialization and replay never consume them.
    _clone_recipes: ClassVar[list[CloneRecipe]] = []
    _atexit_registered: ClassVar[bool] = False
    _scene_data_backend: ClassVar[OvPhysxSceneDataBackend | None] = None
    kinematics_dirty: ClassVar[bool] = False
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
        cls._clone_recipes.append((source, targets, target_transforms, None, 0))

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
        cls._ensure_physx_schemas_registered()
        cls._gravity = tuple(sim_context.cfg.gravity)
        cls._warmup_done = False
        cls._requires_full_stage = False
        cls._stage_usda = None
        cls._clone_recipes = []
        # Construct the SceneDataBackend eagerly so :class:`SimulationContext`
        # captures a real instance (not ``None``) when it builds the central
        # :class:`~isaaclab.scene.scene_data_provider.SceneDataProvider` in
        # its own ``__init__``. Bindings stay empty until :meth:`_warmup_and_load`
        # calls :meth:`OvPhysxSceneDataBackend.setup`, at which point the wheel
        # and the USD stage are live. Matches PhysX's pattern of constructing
        # the backend during ``initialize()``.
        cls._scene_data_backend = OvPhysxSceneDataBackend()
        cls.kinematics_dirty = False

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
        cls.kinematics_dirty = True
        cls._scene_data_backend.transforms_timestamp += 1
        cls._scene_data_backend.geometry_timestamp += 1

    @classmethod
    def forward(cls) -> None:
        """Evaluate and publish state changes made without stepping physics."""
        cls.update_kinematics()
        cls._scene_data_backend.transforms_timestamp += 1
        cls._scene_data_backend.geometry_timestamp += 1

    @classmethod
    def pre_render(cls) -> None:
        """Finish native kinematics before SDP publishes manually written joint poses."""
        cls.update_kinematics()

    @classmethod
    def update_kinematics(cls) -> None:
        """Update dirty articulation kinematics without publishing or rendering."""
        if not cls.kinematics_dirty:
            return
        if cls.backend is not None and cls.backend.physx is not None:
            cls.backend.physx.update_articulations_kinematic()
            cls.kinematics_dirty = False

    @classmethod
    def step(cls) -> None:
        """Step the simulation by one physics timestep."""
        if cls.backend is None or cls.backend.physx is None:
            return
        dt = cls.get_physics_dt()
        cls.backend.physx.step_sync(dt=dt)
        cls.kinematics_dirty = True
        cls.update_kinematics()
        cls._scene_data_backend.transforms_timestamp += 1
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
                cls._clone_recipes = []
                # Drop the SceneDataBackend singleton: its cached bindings and buffers
                # belong to the runtime instance just released. The next
                # SimulationContext re-creates it in initialize().
                cls._scene_data_backend = None
                cls.kinematics_dirty = False
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
        if cls.backend.rigid_body_view is not None:
            cls.backend.rigid_body_view.destroy()
            cls.backend.rigid_body_view = None
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
    def _warmup_and_load(cls) -> None:
        """Serialize the USD stage and attach it to the ovphysx runtime.

        When no runtime is active, constructs a new :class:`ovphysx.PhysX`
        instance. The first construction also records IsaacLab's process device
        choice and registers process-exit cleanup. On a forced re-warm before
        :meth:`close`, it reuses the active instance, attaches the new USD through
        OVStage, rebuilds active clone recipes through full-stage materialization
        or runtime replay, and re-runs the supported warmup entry point
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
        plan = sim.get_clone_plan()
        entries = None
        if plan is not None:
            env_ids = np.arange(len(plan.topology.world_prototype_layout))
            entries = expand_deformable_entries(deformable_prototypes(sim.stage, plan), plan, env_ids, plan.positions)

        ovphysx_device = "gpu" if "cuda" in PhysicsManager._device else "cpu"

        if cls._locked_device is not None and ovphysx_device != cls._locked_device:
            raise RuntimeError(
                f"OvPhysxManager is locked to device {cls._locked_device!r} for the lifetime of this process; "
                f"cannot switch to {ovphysx_device!r}. IsaacLab pins the first OVPhysX device choice because "
                "CPU-only mode cannot be reversed; restart the process to use a different device."
            )

        scene_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        if scene_prim.IsValid():
            if cls._clone_recipes:
                scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)
            cls._configure_physx_scene_prim(scene_prim, PhysicsManager._cfg, ovphysx_device)

        full_stage = cls._requires_full_stage or ovphysx_device == "cpu"
        stage_usda, native_clones = _serialize_stage(sim.stage, cls._clone_recipes, full_stage)
        cls._stage_usda = stage_usda

        previous_backend = cls.backend
        cls.backend = sim.get_or_create_backend(
            OvPhysxBackendCfg(
                device=PhysicsManager._device,
                cooked_collider_cache_dir=sim.cfg.physics.cooked_collider_cache_dir,
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

        for source, targets, transforms, env_ids, _ in native_clones:
            cls.backend.physx.wait_op(cls.backend.physx.clone(source, targets, transforms or None, env_ids=env_ids))

        # Native metadata and bindings must see the newly attached bodies, including on CPU.
        cls._warmup_physx(cls.backend.physx)

        # The central SceneDataProvider can request these bindings later. Headless
        # training never consumes them, so avoid binding every rigid link here.
        if cls._scene_data_backend is None:
            cls._scene_data_backend = OvPhysxSceneDataBackend()
        cls._scene_data_backend._defer_setup(cls.backend, PhysicsManager._device, entries)

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

        # PhysX uses the declared timestep to derive automatic collision contact offsets.
        sim_cfg = PhysicsManager._sim.cfg
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
