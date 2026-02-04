# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider for visualizers and renderers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SceneDataProvider:
    """Creates appropriate data provider based on physics backend."""

    def __init__(
        self,
        backend: str,
        visualizer_cfgs: list[Any] | None,
        stage=None,
        simulation_context=None,
        force_newton_sync: bool = False,
    ) -> None:
        self._backend = backend
        self._provider = None

        if backend == "newton":
            from .newton_scene_data_provider import NewtonSceneDataProvider

            self._provider = NewtonSceneDataProvider(visualizer_cfgs, stage)
        elif backend == "omni":
            if stage is None or simulation_context is None:
                logger.warning("OV scene data provider requires stage and simulation context.")
                self._provider = None
            else:
                from .ov_scene_data_provider import OVSceneDataProvider

                self._provider = OVSceneDataProvider(visualizer_cfgs, stage, simulation_context, force_newton_sync)
        else:
            logger.warning(f"Unknown physics backend '{backend}'.")

    def update(self) -> None:
        if self._provider is not None:
            self._provider.update()

    def get_newton_model(self) -> Any | None:
        if self._provider is None:
            return None
        return self._provider.get_newton_model()

    def get_newton_state(self) -> Any | None:
        if self._provider is None:
            return None
        return self._provider.get_newton_state()

    def get_usd_stage(self) -> Any | None:
        if self._provider is None:
            return None
        return self._provider.get_usd_stage()

    def get_metadata(self) -> dict[str, Any]:
        if self._provider is None:
            return {}
        return self._provider.get_metadata()

    def get_transforms(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_transforms()

    def get_velocities(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_velocities()

    def get_contacts(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_contacts()

    def get_mesh_data(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_mesh_data()

    def get_camera_data(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_camera_data()

    def get_camera_transforms(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_camera_transforms()

    @staticmethod
    def _collect_mesh_data(stage, env_regex: str = r"(?P<root>/World/envs/env_)(?P<id>\d+)(?P<path>/.*)"):
        from pxr import UsdGeom
        from newton.sensors import TiledCameraSensor
        import isaaclab.sim as isaaclab_sim
        import re

        env_pattern = re.compile(env_regex)
        prim_data: dict[str, dict[str, Any]] = {}
        mesh_data: dict[str, dict[str, Any]] = {}
        shape_entries: list[dict[str, Any]] = []
        num_worlds = -1

        stage_prims: list = [stage.GetPseudoRoot()]
        while stage_prims:
            prim = stage_prims.pop(0)
            prim_path = prim.GetPath().pathString

            world_id = -1
            template_path = prim_path
            if match := env_pattern.match(prim_path):
                world_id = int(match.group("id"))
                template_path = match.group("root") + "%d" + match.group("path")
                if world_id > num_worlds:
                    num_worlds = world_id

            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue

            shape_type = TiledCameraSensor.RenderShapeType.NONE
            if prim.IsA(UsdGeom.Mesh):
                shape_type = TiledCameraSensor.RenderShapeType.MESH
            elif prim.IsA(UsdGeom.Sphere):
                shape_type = TiledCameraSensor.RenderShapeType.SPHERE
            elif prim.IsA(UsdGeom.Capsule):
                shape_type = TiledCameraSensor.RenderShapeType.CAPSULE
            elif prim.IsA(UsdGeom.Cube):
                shape_type = TiledCameraSensor.RenderShapeType.BOX
            elif prim.IsA(UsdGeom.Cylinder):
                shape_type = TiledCameraSensor.RenderShapeType.CYLINDER
            elif prim.IsA(UsdGeom.Cone):
                shape_type = TiledCameraSensor.RenderShapeType.CONE
            elif prim.IsA(UsdGeom.Plane):
                shape_type = TiledCameraSensor.RenderShapeType.PLANE

            if shape_type != TiledCameraSensor.RenderShapeType.NONE:
                if template_path not in prim_data:
                    prim_data[template_path] = {
                        "is_shared": world_id > -1,
                        "shape_type": shape_type,
                        "master_prim_path": prim_path,
                        "prims": [],
                    }
                elif world_id > -1 and not prim_data[template_path]["is_shared"]:
                    prim_data[template_path]["is_shared"] = True
                prim_data[template_path]["prims"].append((world_id, prim_path))

                if shape_type == TiledCameraSensor.RenderShapeType.MESH and template_path not in mesh_data:
                    mesh = UsdGeom.Mesh(prim)
                    mesh_data[template_path] = {
                        "points": mesh.GetPointsAttr().Get(),
                        "face_counts": mesh.GetFaceVertexCountsAttr().Get(),
                        "face_indices": mesh.GetFaceVertexIndicesAttr().Get(),
                    }

            if child_prims := prim.GetFilteredChildren(UsdGeom.TraverseInstanceProxies()):
                stage_prims.extend(child_prims)

        mesh_index_map: dict[str, int] = {}
        meshes: list[dict[str, Any]] = []
        for template_path, data in prim_data.items():
            if not data["is_shared"]:
                continue
            mesh_index = -1
            if data["shape_type"] == TiledCameraSensor.RenderShapeType.MESH:
                mesh_index = mesh_index_map.get(template_path, -1)
                if mesh_index == -1:
                    mesh_index = len(meshes)
                    mesh_index_map[template_path] = mesh_index
                    meshes.append(mesh_data[template_path])

            for world_id, prim_path in data["prims"]:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                pos, ori = isaaclab_sim.resolve_prim_pose(prim)
                size = SceneDataProvider._resolve_shape_size(data["shape_type"], prim)
                shape_entries.append(
                    {
                        "template_path": template_path,
                        "world_id": world_id,
                        "shape_type": data["shape_type"],
                        "mesh_index": mesh_index,
                        "prim_path": prim_path,
                        "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                        "orientation": [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])],
                        "size": [float(size[0]), float(size[1]), float(size[2])],
                    }
                )

        return {
            "num_worlds": num_worlds + 1,
            "prim_data": prim_data,
            "meshes": meshes,
            "shape_entries": shape_entries,
        }

    @staticmethod
    def _resolve_shape_size(shape_type, prim: Any):
        from pxr import UsdGeom
        from newton.sensors import TiledCameraSensor
        import isaaclab.sim as isaaclab_sim

        if shape_type == TiledCameraSensor.RenderShapeType.SPHERE:
            sphere = UsdGeom.Sphere(prim)
            radius = sphere.GetRadiusAttr().Get()
            return (radius, 0.0, 0.0)
        if shape_type == TiledCameraSensor.RenderShapeType.CAPSULE:
            capsule = UsdGeom.Capsule(prim)
            radius = capsule.GetRadiusAttr().Get()
            height = capsule.GetHeightAttr().Get()
            return (radius, height, 0.0)
        if shape_type == TiledCameraSensor.RenderShapeType.BOX:
            cube = UsdGeom.Cube(prim)
            size = cube.GetSizeAttr().Get()
            scale = isaaclab_sim.resolve_prim_scale(prim)
            return (size * scale[0], size * scale[1], size * scale[2])
        if shape_type == TiledCameraSensor.RenderShapeType.CYLINDER:
            cylinder = UsdGeom.Cylinder(prim)
            radius = cylinder.GetRadiusAttr().Get()
            height = cylinder.GetHeightAttr().Get()
            return (radius, height, 0.0)
        if shape_type == TiledCameraSensor.RenderShapeType.CONE:
            cone = UsdGeom.Cone(prim)
            radius = cone.GetRadiusAttr().Get()
            height = cone.GetHeightAttr().Get()
            return (radius, height, 0.0)
        if shape_type == TiledCameraSensor.RenderShapeType.PLANE:
            return (0.0, 0.0, 0.0)
        scale = isaaclab_sim.resolve_prim_scale(prim)
        return (scale[0], scale[1], scale[2])
