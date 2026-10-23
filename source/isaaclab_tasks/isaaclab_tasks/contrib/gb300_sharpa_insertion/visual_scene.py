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

from .assets import GB300_SWITCH_FRONT_USD_PATH
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


def configure_visual_scene(cfg: GB300SharpaInsertionEnvCfg) -> str:
    """Configure renderer-only scenery and return the rack's USD path.

    Call after launcher resolution and before environment construction. USD-only
    cloning keeps every visual rack and facility mesh out of the Newton model.
    """
    root = os.environ.get(
        "ISAACLAB_SHARPA_RJ45_VISUAL_ROOT", f"{ISAACLAB_NUCLEUS_DIR}/Environments/RizonSharpaRJ45"
    ).rstrip("/")
    rack_path = f"{root}/Environments/GB300/external.usd"
    if any(viz.backend.visualizer_type in ("kit", "newton_rtx") for viz in cfg.sim.visualizer_cfgs):
        usd_only = ("isaaclab.cloner:UsdReplicateContext",)
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

    Call after the initial environment reset. The front-only asset contains
    three prepared material meshes in the demo's world frame, retaining every
    port and authored normal without processing the full rack at startup.
    """
    viewers = [viz for viz in env.sim.visualizers if viz.cfg.backend.visualizer_type == "newton_gl"]
    if not viewers:
        return
    import newton.usd

    from pxr import Usd, UsdGeom, UsdShade

    path = rack_path
    if front_only:
        path = GB300_SWITCH_FRONT_USD_PATH
        if "ISAACLAB_SHARPA_RJ45_VISUAL_ROOT" in os.environ:
            path = f"{rack_path.rsplit('/', 1)[0]}/{path.rsplit('/', 1)[1]}"
    stage = Usd.Stage.Open(retrieve_file_path(path))
    transform = np.eye(4)
    transform[:3, :3] = ((0, 1, 0), (-1, 0, 0), (0, 0, 1))
    transform[:3, 3] = GB300_PRESENTATION_TRANSLATION_E
    transform[2, 3] += FLOOR_OFFSET_M
    cache = UsdGeom.XformCache()
    sources, groups = {}, {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        source = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
        key = str(source.GetPath())
        if key not in sources:
            sources[key] = newton.usd.get_mesh(prim, load_normals=True, compute_inertia=False)
        mesh = sources[key]
        points, normals = mesh.vertices, mesh.normals
        indices = mesh.indices.reshape(-1, 3)
        if not front_only:
            matrix = transform @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
            points = (points @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
            normals = normals @ np.linalg.inv(matrix[:3, :3])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            if np.linalg.det(matrix[:3, :3]) < 0:
                indices = indices[:, ::-1]
        if not len(indices):
            continue
        material = str(UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0].GetPath())
        group = groups.setdefault(material, {"mesh": mesh, "points": [], "normals": [], "indices": [], "offset": 0})
        group["points"].append(points)
        group["normals"].append(normals.astype(np.float32, copy=False))
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
