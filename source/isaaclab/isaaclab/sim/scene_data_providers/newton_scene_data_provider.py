# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed scene data provider."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NewtonSceneDataProvider:
    """Scene data provider for Newton Warp physics backend.

    Native (cheap): Newton Model/State from NewtonManager
    Adapted (future): USD stage (would need Newton→USD sync for OV visualizer)
    """

    def __init__(self, visualizer_cfgs: list[Any] | None, stage=None) -> None:
        self._has_ov_visualizer = False
        self._metadata: dict[str, Any] = {}
        self._stage = stage
        self._camera_data_cache: dict[str, Any] | None = None
        self._mesh_data_cache: dict[str, Any] | None = None

        if visualizer_cfgs:
            for cfg in visualizer_cfgs:
                if getattr(cfg, "visualizer_type", None) == "omniverse":
                    self._has_ov_visualizer = True

        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            self._metadata = {
                "physics_backend": "newton",
                "num_envs": NewtonManager._num_envs if NewtonManager._num_envs is not None else 0,
                "gravity_vector": NewtonManager._gravity_vector,
                "clone_physics_only": NewtonManager._clone_physics_only,
            }
        except Exception:
            self._metadata = {"physics_backend": "newton"}

    def update(self) -> None:
        """No-op for Newton backend (state updated by Newton solver)."""
        pass

    def get_newton_model(self) -> Any | None:
        """NATIVE: Newton Model from NewtonManager."""
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            return NewtonManager._model
        except Exception:
            return None

    def get_newton_state(self) -> Any | None:
        """NATIVE: Newton State from NewtonManager."""
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            return NewtonManager._state_0
        except Exception:
            return None

    def get_usd_stage(self) -> None:
        """Stage handle (if provided) for USD queries."""
        return self._stage

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_transforms(self) -> dict[str, Any] | None:
        """Extract transforms from Newton state (future work)."""
        return None

    def get_velocities(self) -> dict[str, Any] | None:
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            if NewtonManager._state_0 is None:
                return None
            return {"body_qd": NewtonManager._state_0.body_qd}
        except Exception:
            return None

    def get_contacts(self) -> dict[str, Any] | None:
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            if NewtonManager._contacts is None:
                return None
            return {"contacts": NewtonManager._contacts}
        except Exception:
            return None

    def get_mesh_data(self) -> dict[str, Any] | None:
        if self._stage is None:
            return None
        if self._mesh_data_cache is None:
            from .scene_data_provider import SceneDataProvider

            self._mesh_data_cache = SceneDataProvider._collect_mesh_data(self._stage)
        return dict(self._mesh_data_cache)

    def get_camera_data(self) -> dict[str, Any] | None:
        if self._stage is None:
            return None
        if self._camera_data_cache is None:
            self._camera_data_cache = self._collect_camera_data(self._stage)
        return dict(self._camera_data_cache)

    def get_camera_transforms(self) -> dict[str, Any] | None:
        if self._stage is None:
            return None
        camera_data = self.get_camera_data()
        if not camera_data:
            return None
        return self._collect_camera_transforms(self._stage, camera_data)


    @staticmethod
    def _collect_camera_data(stage, env_regex: str = r"(?P<root>/World/envs/env_)(?P<id>\d+)(?P<path>/.*)"):
        from pxr import Usd, UsdGeom
        import re

        env_pattern = re.compile(env_regex)
        shared_paths: list[str] = []
        instances: dict[str, list[tuple[int, str]]] = {}
        num_envs = -1

        stage_prims: list = [stage.GetPseudoRoot()]
        while stage_prims:
            prim = stage_prims.pop(0)
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

            if child_prims := prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
                stage_prims.extend(child_prims)

        return {"shared_paths": shared_paths, "instances": instances, "num_envs": num_envs + 1}

    @staticmethod
    def _collect_camera_transforms(stage, camera_data: dict[str, Any]):
        import isaaclab.sim as isaaclab_sim

        shared_paths = camera_data.get("shared_paths") or []
        instances = camera_data.get("instances") or {}
        num_envs = camera_data.get("num_envs", 0)

        positions: list[list[list[float] | None]] = []
        orientations: list[list[list[float] | None]] = []

        for template_path in shared_paths:
            per_world_pos: list[list[float] | None] = [None] * num_envs
            per_world_ori: list[list[float] | None] = [None] * num_envs
            for world_id, prim_path in instances.get(template_path, []):
                if world_id < 0 or world_id >= num_envs:
                    continue
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                pos, ori = isaaclab_sim.resolve_prim_pose(prim)
                per_world_pos[world_id] = [float(pos[0]), float(pos[1]), float(pos[2])]
                per_world_ori[world_id] = [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])]
            positions.append(per_world_pos)
            orientations.append(per_world_ori)

        return {"order": shared_paths, "positions": positions, "orientations": orientations, "num_envs": num_envs}
