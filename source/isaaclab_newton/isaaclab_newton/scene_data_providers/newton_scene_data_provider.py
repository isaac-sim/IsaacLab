# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider for Newton physics backend."""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any

from pxr import UsdGeom

from isaaclab.physics.base_scene_data_provider import BaseSceneDataProvider

logger = logging.getLogger(__name__)

_ENV_ID_RE = re.compile(r"/World/envs/env_(\d+)")


class NewtonSceneDataProvider(BaseSceneDataProvider):
    """Scene data provider for Newton physics backend.

    Provides access to Newton model, state, and USD stage for visualizers and renderers.
    Unlike PhysxSceneDataProvider which must build its own Newton model from USD and sync
    PhysX transforms into it, this provider delegates directly to NewtonManager since the
    Newton backend already owns the authoritative model and state.
    """

    def __init__(self, stage, simulation_context) -> None:
        self._simulation_context = simulation_context
        self._stage = stage
        self._metadata = {"physics_backend": "newton"}
        self._num_envs: int | None = None
        self._warned_once: set[str] = set()

        # Determine if usd stage sync is required for selected renderers and visualizers
        requirements = self._simulation_context.get_scene_data_requirements()
        self._needs_usd_sync = bool(requirements.requires_usd_stage)

    def _warn_once(self, key: str, message: str, *args) -> None:
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        logger.warning(message, *args)

    # ---- Environment discovery ---------------------------------------------------------------

    def get_num_envs(self) -> int:
        if self._num_envs is not None and self._num_envs > 0:
            return self._num_envs
        discovered = self._determine_num_envs_in_scene()
        if discovered > 0:
            self._num_envs = discovered
            return discovered
        return 0

    def _determine_num_envs_in_scene(self) -> int:
        if self._stage is None:
            return 0
        max_env_id = -1
        env_name_re = re.compile(r"^env_(\d+)$")
        envs_root = self._stage.GetPrimAtPath("/World/envs")
        if envs_root.IsValid():
            for child in envs_root.GetChildren():
                match = env_name_re.match(child.GetName())
                if match:
                    max_env_id = max(max_env_id, int(match.group(1)))
        return max_env_id + 1 if max_env_id >= 0 else 0

    # ---- Core provider API -------------------------------------------------------------------

    def update(self, env_ids: list[int] | None = None) -> None:
        """Sync Newton body transforms to USD Fabric when a Kit viewport is active.

        Called at render cadence by :meth:`~isaaclab.sim.SimulationContext.update_scene_data_provider`,
        after forward kinematics have been evaluated.  Only calls
        :meth:`~isaaclab_newton.physics.NewtonManager.sync_transforms_to_usd` when a Kit
        (or other USD-based) visualizer is in use. When both sim and rendering backend
        are Newton (or Rerun), the sync is skipped to avoid unnecessary slowdown.
        """
        if not self._needs_usd_sync:
            return
        try:
            from isaaclab_newton.physics import NewtonManager

            NewtonManager.sync_transforms_to_usd()
        except Exception:
            pass

    def get_newton_model(self) -> Any | None:
        """Return Newton model from NewtonManager."""
        from isaaclab_newton.physics import NewtonManager

        return NewtonManager.get_model()

    def get_newton_state(self, env_ids: list[int] | None = None) -> Any | None:
        """Return Newton state from NewtonManager.

        Args:
            env_ids: Optional list of environment IDs. Currently returns the full
                state for all environments (env_ids filtering is not yet implemented).

        Returns:
            The current Newton state (state_0) from NewtonManager.
        """
        from isaaclab_newton.physics import NewtonManager

        return NewtonManager.get_state_0()

    def get_model(self) -> Any | None:
        """Alias for get_newton_model (visualizer compatibility)."""
        return self.get_newton_model()

    def get_state(self, env_ids: list[int] | None = None) -> Any | None:
        """Alias for get_newton_state (visualizer compatibility)."""
        return self.get_newton_state(env_ids)

    def get_usd_stage(self) -> Any | None:
        """Return the USD stage handle."""
        if self._stage is not None:
            return self._stage
        return getattr(self._simulation_context, "stage", None)

    def get_metadata(self) -> dict[str, Any]:
        out = dict(self._metadata)
        out["num_envs"] = self.get_num_envs()
        out["needs_usd_sync"] = self._needs_usd_sync
        return out

    def get_transforms(self) -> dict[str, Any] | None:
        """Return body transforms from Newton state.

        Reads body_q from the authoritative Newton state and splits it into
        positions (vec3) and orientations (quaternion xyzw).
        """
        try:
            import warp as wp

            from isaaclab_newton.physics import NewtonManager

            state = NewtonManager.get_state_0()
            if state is None or state.body_q is None:
                return None

            body_q_t = wp.to_torch(state.body_q)
            positions = body_q_t[:, :3]
            orientations = body_q_t[:, 3:7]
            return {"positions": positions, "orientations": orientations}
        except Exception as exc:
            self._warn_once(
                "get-transforms-failed",
                "[NewtonSceneDataProvider] get_transforms() failed: %s",
                exc,
            )
            return None

    def get_velocities(self) -> dict[str, Any] | None:
        """Return body velocities from Newton state."""
        try:
            import warp as wp

            from isaaclab_newton.physics import NewtonManager

            state = NewtonManager.get_state_0()
            if state is None:
                return None

            body_qd = getattr(state, "body_qd", None)
            if body_qd is None:
                return None

            body_qd_t = wp.to_torch(body_qd)
            linear = body_qd_t[:, :3]
            angular = body_qd_t[:, 3:6]
            return {"linear": linear, "angular": angular, "source": "newton"}
        except Exception as exc:
            self._warn_once(
                "get-velocities-failed",
                "[NewtonSceneDataProvider] get_velocities() failed: %s",
                exc,
            )
            return None

    def get_contacts(self) -> dict[str, Any] | None:
        """Contacts not yet supported for Newton provider."""
        return None

    def get_camera_transforms(self) -> dict[str, Any] | None:
        """Return per-camera, per-env transforms (positions, orientations)."""
        if self._stage is None:
            return None

        import isaaclab.sim as isaaclab_sim

        env_pattern = re.compile(r"(?P<root>/World/envs/env_)(?P<id>\d+)(?P<path>/.*)")
        shared_paths: list[str] = []
        instances: dict[str, list[tuple[int, str]]] = {}
        num_envs = -1

        stage_prims = deque([self._stage.GetPseudoRoot()])
        while stage_prims:
            prim = stage_prims.popleft()
            prim_path = prim.GetPath().pathString

            world_id = 0
            template_path = prim_path
            if match := env_pattern.match(prim_path):
                world_id = int(match.group("id"))
                template_path = match.group("root") + "%d" + match.group("path")
                if world_id > num_envs:
                    num_envs = world_id

            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue

            if prim.IsA(UsdGeom.Camera):
                if template_path not in instances:
                    instances[template_path] = []
                instances[template_path].append((world_id, prim_path))
                if template_path not in shared_paths:
                    shared_paths.append(template_path)

            if hasattr(UsdGeom, "TraverseInstanceProxies"):
                child_prims = prim.GetFilteredChildren(UsdGeom.TraverseInstanceProxies())
            else:
                child_prims = prim.GetChildren()
            if child_prims:
                stage_prims.extend(child_prims)

        num_envs += 1
        positions: list[list[list[float] | None]] = []
        orientations: list[list[list[float] | None]] = []

        for template_path in shared_paths:
            per_world_pos: list[list[float] | None] = [None] * num_envs
            per_world_ori: list[list[float] | None] = [None] * num_envs
            for world_id, prim_path in instances.get(template_path, []):
                if world_id < 0 or world_id >= num_envs:
                    continue
                prim = self._stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                pos, ori = isaaclab_sim.resolve_prim_pose(prim)
                per_world_pos[world_id] = [float(pos[0]), float(pos[1]), float(pos[2])]
                per_world_ori[world_id] = [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])]
            positions.append(per_world_pos)
            orientations.append(per_world_ori)

        return {"order": shared_paths, "positions": positions, "orientations": orientations, "num_envs": num_envs}
