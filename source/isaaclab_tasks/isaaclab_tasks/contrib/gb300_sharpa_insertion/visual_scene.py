# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer-only scenery for the Sharpa RJ45 example."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from .workcell import (
    FLOOR_OFFSET_M,
    GB300_PRESENTATION_ROTATION_XYZW,
    GB300_PRESENTATION_TRANSLATION_E,
    GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M,
    GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M,
)

if TYPE_CHECKING:
    from pxr import Usd

    from .env import GB300SharpaInsertionEnv
    from .env_cfg import GB300SharpaInsertionEnvCfg

# The socket face is near composed CAD X=0.355 m; crop 55 mm behind it.
_RACK_FRONT_MIN_X_M = 0.30
# The two SN2201 rows and their gold bezel occupy this composed CAD Z range.
_SWITCH_FRONT_Z_RANGE_M = (1.97, 2.16)


def configure_visual_scene(cfg: GB300SharpaInsertionEnvCfg) -> str:
    """Configure renderer-only scenery and return the rack's USD path.

    Call after launcher resolution and before environment construction. USD-only
    cloning keeps every visual rack and facility mesh out of the Newton model.
    """
    usd_only = ("isaaclab.cloner:UsdReplicateContext",)
    usd_view = any(viz.backend.visualizer_type in ("kit", "newton_rtx") for viz in cfg.sim.visualizer_cfgs)
    root = os.environ.get(
        "ISAACLAB_SHARPA_RJ45_VISUAL_ROOT", f"{ISAACLAB_NUCLEUS_DIR}/Environments/RizonSharpaRJ45"
    ).rstrip("/")
    rack_path = f"{root}/Environments/GB300/external.usd"
    if usd_view:
        cfg.scene.visual_scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/VisualScene",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{root}/Environments/Datacenter/datacenter.usdc"),
            cloning_contexts=usd_only,
        )
        cfg.scene.visual_racks = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/VisualRacks",
            spawn=sim_utils.UsdFileCfg(usd_path=rack_path, func=_spawn_visual_racks),
            cloning_contexts=usd_only,
        )
    return rack_path


def add_gl_visuals(env: GB300SharpaInsertionEnv, rack_path: str, *, front_only: bool = False) -> None:
    """Register the detailed switch front or full rack without physics shapes.

    Call after the initial environment reset. Native USD mesh loading preserves
    the CAD normals and scalar materials; material batches avoid thousands
    of draw calls. The canonical rack has no albedo textures.
    """
    viewers = [viz for viz in env.sim.visualizers if viz.cfg.backend.visualizer_type == "newton_gl"]
    if not viewers:
        return
    import newton.usd

    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(retrieve_file_path(rack_path))
    transform = np.eye(4)
    transform[:3, :3] = ((0, 1, 0), (-1, 0, 0), (0, 0, 1))
    transform[:3, 3] = GB300_PRESENTATION_TRANSLATION_E
    transform[2, 3] += FLOOR_OFFSET_M
    cache = UsdGeom.XformCache()
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]) if front_only else None
    sources, groups = {}, {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        clip_front = False
        if front_only:
            if not str(prim.GetPath()).startswith(
                (
                    "/external/tn__0000_NV_MSN2201TOR_08132024",
                    "/external/tn__01_TORBEZELKIT2U_",
                    "/external/silver_meshes/",
                    "/external/bezel_back_plates/",
                )
            ):
                continue
            bound = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
            if bound.GetMin()[2] < _SWITCH_FRONT_Z_RANGE_M[0] or bound.GetMax()[2] > _SWITCH_FRONT_Z_RANGE_M[1]:
                continue
            # The merged switch shells include the rear chassis; retain their front faces.
            clip_front = bound.GetMin()[0] < _RACK_FRONT_MIN_X_M
        source = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
        key = str(source.GetPath())
        if key not in sources:
            sources[key] = newton.usd.get_mesh(prim, load_normals=True, compute_inertia=False)
        mesh = sources[key]
        matrix = transform @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        points = (mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
        normals = mesh.normals @ np.linalg.inv(matrix[:3, :3])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        indices = mesh.indices.reshape(-1, 3)
        if clip_front:
            front_y = GB300_PRESENTATION_TRANSLATION_E[1] - _RACK_FRONT_MIN_X_M
            indices = indices[np.all(points[indices, 1] <= front_y, axis=1)]
            used, indices = np.unique(indices, return_inverse=True)
            points, normals = points[used], normals[used]
            indices = indices.reshape(-1, 3)
        if not len(indices):
            continue
        material = str(UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0].GetPath())
        group = groups.setdefault(material, {"mesh": mesh, "points": [], "normals": [], "indices": [], "offset": 0})
        group["points"].append(points)
        group["normals"].append(normals.astype(np.float32))
        if np.linalg.det(matrix[:3, :3]) < 0:
            indices = indices[:, ::-1]
        group["indices"].append(indices.reshape(-1) + group["offset"])
        group["offset"] += len(points)
    for index, group in enumerate(groups.values()):
        mesh = group["mesh"]
        points = wp.array(np.concatenate(group["points"]), dtype=wp.vec3, device=env.device)
        normals = wp.array(np.concatenate(group["normals"]), dtype=wp.vec3, device=env.device)
        indices = wp.array(np.concatenate(group["indices"]), dtype=wp.int32, device=env.device)
        for viewer in viewers:
            viewer.log_mesh(
                f"/Presentation/GB300/Material{index}", points, indices, normals=normals,
                color=mesh.color, roughness=mesh.roughness, metallic=mesh.metallic,
                opacity=mesh.opacity, backface_culling=False,
            )  # fmt: skip


def _spawn_visual_racks(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Compose eight instanceable CAD references at the measured cabinet spacing [m]."""
    root = sim_utils.create_prim(prim_path, translation=translation, orientation=orientation)
    spacing = GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M[1] - GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M[1]
    x, y, z = GB300_PRESENTATION_TRANSLATION_E
    for index in range(8):
        rack_path = f"{prim_path}/Rack{index:02d}"
        # Compose placement with the CAD's authored root rotation and lift.
        sim_utils.create_prim(
            rack_path,
            translation=(x + index * spacing, y, z + FLOOR_OFFSET_M),
            orientation=GB300_PRESENTATION_ROTATION_XYZW,
        )  # fmt: skip
        rack = sim_utils.spawn_from_usd(f"{rack_path}/Asset", cfg)
        rack.SetInstanceable(True)
    return root
