# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Native state and operations owned by one independently attached OvPhysX stage."""

from __future__ import annotations

import importlib.util
import math
import os
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import ovphysx
import ovstage
from ovphysx.types import CompiledStageCacheMode, GlobalConfigKey

from pxr import Plug

from isaaclab.utils import instantiate

from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView
from isaaclab_ov.stage import OvstageBackend, OvstageBackendCfg

from .ovphysx_backend_cfg import DEFAULT_COOKED_COLLIDER_CACHE_DIR, OvPhysxBackendCfg
from .ovphysx_scene_data import OvPhysxSceneDataBackend
from .schemas import physx_schema_paths

if TYPE_CHECKING:
    from isaaclab.scene_data.deformable_discovery import DeformableStageEntry

__all__ = ["OvPhysxBackend"]

_physx_schemas_registered = False


@contextmanager
def _compiled_stage_cache_for_attach(disable: bool) -> Iterator[None]:
    """Scope the runtime's process-wide cache mode to attach submission."""
    if not disable:
        yield
        return
    # OvPhysX captures the cache mode at submission during the synchronous simulation reset.
    key = GlobalConfigKey.COMPILED_STAGE_CACHE
    previous = CompiledStageCacheMode(ovphysx.get_global_config(key))
    ovphysx.set_global_config(key, CompiledStageCacheMode.DISABLED)
    try:
        yield
    finally:
        ovphysx.set_global_config(key, previous)


class OvPhysxBackend:
    """One native OvPhysX runtime and its attached stage, usable without a simulation context.

    Each backend needs its own stage and serialized native calls; process settings are shared.
    """

    def __init__(self, cfg: OvPhysxBackendCfg, *, scene: OvstageBackend | None = None):
        """Construct the native runtime without attaching a USD stage.

        Args:
            cfg: Native runtime configuration.
            scene: Shared OVStage resource to borrow, or ``None`` to own a private stage.
        """
        ovphysx.bootstrap()
        self.device = cfg.device
        is_gpu = cfg.device.startswith("cuda:")
        cache_dir = cfg.cooked_collider_cache_dir
        if cache_dir == DEFAULT_COOKED_COLLIDER_CACHE_DIR:
            cache_dir = _prepare_default_cache_dir(cache_dir)
        # Startup settings cannot go through PhysXConfig: reapplying them fails while any
        # stage is attached, even when their values have not changed.
        _apply_startup_config("NUM_THREADS", 8)
        if cache_dir is not None:
            _apply_startup_config("COOKED_COLLIDER_CACHE_DIRECTORY", cache_dir)
        # These Carbonite settings are process-wide, so CPU and CUDA backends share them;
        # each stage still selects its own CPU or GPU dynamics.
        carbonite_overrides = {
            "/physics/physxDispatcher": True,
            "/physics/updateToUsd": False,
            "/physics/updateVelocitiesToUsd": False,
            "/physics/updateParticlesToUsd": False,
            "/physics/suppressReadback": True,
            "/physics/suppressFabricUpdate": True,
            "/ovphysx/clone/useEnvIds": True,
        }
        # CPU scenes select CPU dynamics through their scene attributes (see _set_scene_dynamics_device).
        # OVPhysX's process-wide CPU-only mode is irreversible and would block later CUDA scenes.
        self.physx = ovphysx.PhysX(
            config=ovphysx.PhysXConfig(carbonite_overrides=carbonite_overrides),
            active_cuda_gpus=cfg.device.removeprefix("cuda:") if is_gpu else None,
        )
        self.scene = scene
        self.owns_scene = scene is None
        self.stage: Any = None
        self.rigid_body_view: Any = None
        self.scene_data = OvPhysxSceneDataBackend()
        """Publication of this stage's poses and geometry, refreshed by :meth:`step` and :meth:`forward`."""
        self.kinematics_dirty = False
        self.is_ready = False
        self.gravity: dict[str, tuple[float, float, float]] = {}
        """Last successfully applied gravity [m/s^2], keyed by physics scene path."""
        self._gravity_retry: set[str] = set()
        """Physics scenes whose last gravity update failed and may be partially applied."""

    def load(
        self,
        stage_usda: str,
        *,
        deformable_entries: Sequence[DeformableStageEntry] | None = None,
        disable_compiled_stage_cache: bool = False,
    ) -> None:
        """Replace the attached scene after draining its native bindings.

        Args:
            stage_usda: Exported USD scene to populate before physics attaches.
            deformable_entries: Declared deformables for scene-data publication, captured before
                native import. ``None`` leaves scene geometry uninitialized.
            disable_compiled_stage_cache: Bypass the compiled-stage cache for this attachment.

        Raises:
            RuntimeError: If the backend has closed.
        """
        if self.physx is None:
            raise RuntimeError("The OVPhysX backend has closed.")
        _ensure_physx_schemas_registered()
        self.is_ready = False
        self.kinematics_dirty = False
        # Cached gravity describes the outgoing stage, even if replacing it fails.
        self.gravity.clear()
        self.scene_data._clear()
        if self.stage is not None:
            if self.rigid_body_view is not None:
                self.rigid_body_view.destroy()
                self.rigid_body_view = None
            OvPhysxView._close_all_for(self.physx)
            self.physx.wait_task(self.physx.detach_ovstage())
            self.scene.reset()
            self.stage = None
        if self.scene is None:
            # Include rendering data for native-instance collider dependencies as well as physics.
            self.scene = instantiate(OvstageBackendCfg(populate_physics=True))
        try:
            self.scene.populate(stage_usda)
            with _compiled_stage_cache_for_attach(disable_compiled_stage_cache):
                operation = self.physx.attach_ovstage(self.scene.stage, read_ordinal=self.scene.ordinal - 1)
                # Accepted attach failures can retain native ownership. Keep the stage for cleanup.
                self.stage = self.scene.stage
            self.physx.wait_task(operation)
        except Exception:
            if self.owns_scene and self.stage is None:
                self.scene.close()
                self.scene = None
            raise
        self.kinematics_dirty = True
        self.scene_data._defer_setup(self, self.device, deformable_entries)

    def warmup(self) -> None:
        """Initialize the attached stage after native cloning, before reading scene data."""
        if self.physx is None or self.stage is None:
            raise RuntimeError("The OVPhysX backend has no attached stage.")
        # Pinned runtimes need explicit warmup before metadata and binding access, including on CPU.
        # Newer runtimes finish construction and DirectGPU upload in the attach and clone tasks.
        if hasattr(self.physx, "warmup"):
            self.physx.warmup()
        self.is_ready = True

    def step(self, dt: float) -> None:
        """Advance the attached stage synchronously by ``dt`` [s] and publish its state."""
        if self.physx is None:
            raise RuntimeError("The OVPhysX backend has closed.")
        self.physx.step_sync(dt=dt)
        self.is_ready = True
        self.kinematics_dirty = True
        self.forward()

    def forward(self) -> None:
        """Evaluate pending kinematics and mark the scene-data publication stale."""
        self.update_kinematics()
        self.scene_data.invalidate()

    def update_kinematics(self) -> None:
        """Evaluate dirty articulation kinematics without publishing or rendering."""
        if self.kinematics_dirty and self.is_ready:
            self.physx.update_articulations_kinematic()
            self.kinematics_dirty = False

    def set_gravity(
        self,
        gravity: tuple[float, float, float],
        *,
        physics_prim_path: str,
    ) -> None:
        """Apply a world-frame gravity vector [m/s^2] through a sealed OVStage control edit.

        A vector already applied to the scene on this stage creates no control update.

        Args:
            gravity: Three finite world-frame acceleration components [m/s^2].
            physics_prim_path: Path of a physics scene in the attached stage. OVStage does not
                report unknown paths, so a path without a scene leaves the simulation unchanged.

        Raises:
            RuntimeError: If the backend has no attached stage.
            ValueError: If gravity does not contain three finite values.
        """
        if self.physx is None or self.stage is None:
            raise RuntimeError("The OVPhysX backend has no attached stage.")
        gravity_array = np.asarray(gravity, dtype=np.float32)
        if gravity_array.shape != (3,) or not np.all(np.isfinite(gravity_array)):
            raise ValueError("Gravity must contain three finite values.")
        gravity_value = tuple(float(component) for component in gravity_array)
        # Reset events may resample the same vector. Skip it unless the last update of this scene failed.
        if physics_prim_path not in self._gravity_retry and self.gravity.get(physics_prim_path) == gravity_value:
            return
        magnitude = float(np.linalg.norm(gravity_array))
        if math.isclose(magnitude, 0.0):
            direction = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        else:
            direction = (gravity_array / magnitude).reshape(1, 3)
        scene = self.scene
        # A failed native update may apply partially, so the next request must run even if it repeats the cache.
        self._gravity_retry.add(physics_prim_path)
        # Consume only this edit, excluding pending rendering/output writes.
        scene.commit()
        ordinal = scene.ordinal
        with scene.query([physics_prim_path]) as query:
            scene.stage.write_attribute(query, "physics:gravityDirection", ordinal, direction, is_array=False).wait()
            scene.stage.write_attribute(
                query, "physics:gravityMagnitude", ordinal, np.array([magnitude], dtype=np.float32), is_array=False
            ).wait()
            scene.commit()
            self.physx.wait_task(self.physx.update_from_ovstage(ordinal, ordinal))
        # Publish only successful completion.
        self.gravity[physics_prim_path] = gravity_value
        self._gravity_retry.discard(physics_prim_path)

    def close(self) -> None:
        """Release bindings, the runtime, and its stage in native teardown order."""
        self.is_ready = False
        self.scene_data._clear()
        physx = self.physx
        if physx is None:
            if self.owns_scene and self.scene is not None:
                self.scene.close()
                self.scene = None
            self.stage = None
            self.gravity.clear()
            return

        # Destroy errors may retain native ownership and require another close.
        release_owners = False
        try:
            try:
                if self.rigid_body_view is not None:
                    self.rigid_body_view.destroy()
                    self.rigid_body_view = None
                OvPhysxView._close_all_for(physx)
            finally:
                try:
                    physx.wait_task(physx.detach_ovstage())
                finally:
                    try:
                        physx.destroy()
                    except Exception:
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
                self.kinematics_dirty = False
                if self.owns_scene and self.scene is not None:
                    self.scene.close()
                    self.scene = None
                self.stage = None
                self.gravity.clear()


def _apply_startup_config(name: str, value: int | str) -> None:
    """Apply a startup-only ``ovphysx.GlobalConfigKey`` setting unless an equivalent value is already set."""
    key = getattr(ovphysx.GlobalConfigKey, name)
    current = ovphysx.get_global_config(key)
    if isinstance(value, str) and current is not None:
        if os.path.realpath(str(current)) == os.path.realpath(value):
            return
    elif current == value:
        return
    try:
        ovphysx.set_global_config(key, value)
    except RuntimeError as exc:
        raise RuntimeError(
            f"OvPhysX process setting {name} is already {current!r}; an existing runtime prevents changing it"
            f" to {value!r}. Use the same value for every backend in this process."
        ) from exc


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


def _ensure_physx_schemas_registered() -> None:
    """Register process-wide USD schemas before either host or native population."""
    global _physx_schemas_registered
    if _physx_schemas_registered:
        return
    schema_paths = physx_schema_paths()
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
    _physx_schemas_registered = True
