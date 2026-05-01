# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-family implementation for :class:`VisualizationMarkers`."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import warp as wp
from newton import Axis, Mesh

import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply

logger = logging.getLogger(__name__)

_OMNIPBR_DEFAULTS = {
    "diffuse_color_constant": (0.2, 0.2, 0.2),
    "diffuse_tint": (1.0, 1.0, 1.0),
}
_UNBOUND_DEFAULT_FALLBACK_GRAY = (0.18, 0.18, 0.18)
_DEX_CUBE_TEXTURE_URL = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/Materials/dex_cube_mod.png"


@dataclass(frozen=True)
class _NewtonMarkerSpec:
    renderer: Literal["mesh", "frame", "none"]
    mesh_type: Literal["arrow", "box", "textured_box", "sphere", "cylinder", "capsule", "cone"] | None = None
    mesh_params: dict[str, float | tuple[float, float, float]] | None = None
    scale: tuple[float, float, float] | None = None
    color: tuple[float, float, float] | None = None
    texture: Any | None = None


@dataclass(frozen=True)
class _MeshData:
    vertices: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray


def render_newton_visualization_markers(viewer, visible_env_ids: list[int] | None, num_envs: int) -> None:
    """Render all active Newton visualization marker groups into a Newton-family viewer."""
    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        return

    for marker in sim.vis_marker_registry.get_groups().values():
        if isinstance(marker, NewtonVisualizationMarkers):
            marker.render(viewer, visible_env_ids=visible_env_ids, num_envs=num_envs)


class NewtonVisualizationMarkers:
    """Newton-family backend for visualization markers."""

    def __init__(self, cfg: VisualizationMarkersCfg, visible: bool = True):
        self.cfg = cfg
        self.group_id = f"{cfg.prim_path}::{id(self)}"
        self.visible = visible
        self.translations: torch.Tensor | None = None
        self.orientations: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.marker_indices: torch.Tensor | None = None
        self.count = len(cfg.markers)
        self._registered_meshes: set[tuple[int, str]] = set()
        self._warned_unsupported: set[str] = set()

        sim = sim_utils.SimulationContext.instance()
        if sim is not None:
            sim.vis_marker_registry.set_group(self.group_id, self)

    def close(self) -> None:
        """Remove marker backend from the simulation marker registry."""
        sim = sim_utils.SimulationContext.instance()
        if sim is not None:
            sim.vis_marker_registry.remove_group(self.group_id)

    def infer_device(self) -> torch.device:
        """Infer the device from current marker state."""
        for value in (self.translations, self.orientations, self.scales, self.marker_indices):
            if value is not None:
                return value.device
        return torch.device("cpu")

    def set_visibility(self, visible: bool) -> None:
        """Set marker visibility."""
        self.visible = visible

    def is_visible(self) -> bool:
        """Return whether this marker group is visible."""
        return self.visible

    def visualize(
        self,
        translations: torch.Tensor | None,
        orientations: torch.Tensor | None,
        scales: torch.Tensor | None,
        marker_indices: torch.Tensor | None,
    ) -> None:
        """Update marker state consumed by Newton-family visualizers."""
        if translations is not None:
            self.translations = translations.detach()
            self.count = translations.shape[0]
        if orientations is not None:
            self.orientations = orientations.detach()
            self.count = orientations.shape[0]
        if scales is not None:
            self.scales = scales.detach()
            self.count = scales.shape[0]
        if marker_indices is not None:
            self.marker_indices = marker_indices.detach().to(dtype=torch.int32)
            self.count = marker_indices.shape[0]
        elif self.count != 0:
            self.marker_indices = torch.zeros(self.count, dtype=torch.int32, device=self.infer_device())

    def render(self, viewer, visible_env_ids: list[int] | None, num_envs: int) -> None:
        """Render marker state to a Newton viewer."""
        state = _filter_marker_state(self, visible_env_ids=visible_env_ids, num_envs=num_envs)
        if state["count"] == 0:
            for name, marker_cfg in self.cfg.markers.items():
                self._hide_batch(viewer, name, _resolve_newton_marker_cfg(name, marker_cfg, self.cfg))
            return

        translations = state["translations"]
        if translations is None:
            return
        orientations = state["orientations"]
        if orientations is None:
            orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=translations.device).repeat(state["count"], 1)
        scales = state["scales"]
        if scales is None:
            scales = torch.ones((state["count"], 3), dtype=torch.float32, device=translations.device)
        marker_indices = state["marker_indices"]
        if marker_indices is None:
            marker_indices = torch.zeros(state["count"], dtype=torch.int64, device=translations.device)

        for proto_index, (name, marker_cfg) in enumerate(self.cfg.markers.items()):
            newton_cfg = _resolve_newton_marker_cfg(name, marker_cfg, self.cfg)
            batch_name = f"{self.group_id}/{name}"
            selected = marker_indices == proto_index
            if not state["visible"] or int(selected.sum().item()) == 0:
                self._hide_batch(viewer, name, newton_cfg)
                continue

            if newton_cfg.renderer == "none":
                unsupported_key = f"{self.group_id}:{name}"
                if unsupported_key not in self._warned_unsupported:
                    logger.warning(
                        "[NewtonVisualizationMarkers] Unsupported marker prototype '%s' in group '%s'; skipping.",
                        name,
                        self.group_id,
                    )
                    self._warned_unsupported.add(unsupported_key)
                continue

            selected_translations = translations[selected]
            selected_orientations = orientations[selected]
            default_scale = newton_cfg.scale or _extract_scale_hint(marker_cfg)
            selected_scales = scales[selected] * torch.tensor(
                default_scale, dtype=torch.float32, device=scales.device
            ).unsqueeze(0)

            if newton_cfg.renderer == "mesh":
                mesh_name = f"{self.group_id}/meshes/{name}"
                self._ensure_mesh_registered(viewer, mesh_name, newton_cfg)
                color = newton_cfg.color or _extract_color(marker_cfg)
                colors = torch.tensor(color, dtype=torch.float32, device=scales.device).repeat(
                    selected_scales.shape[0], 1
                )
                materials = torch.zeros((selected_scales.shape[0], 4), dtype=torch.float32, device=scales.device)
                if newton_cfg.texture is not None:
                    # ViewerGL gates texture sampling with material.w. Rerun and
                    # Viser ignore this flag but consume the mesh texture.
                    materials[:, 3] = 1.0
                xforms = torch.cat((selected_translations, selected_orientations), dim=1).detach().cpu().numpy()
                viewer.log_instances(
                    batch_name,
                    mesh_name,
                    wp.array(xforms.astype(np.float32), dtype=wp.transform),
                    wp.array(selected_scales.detach().cpu().numpy().astype(np.float32), dtype=wp.vec3),
                    wp.array(colors.detach().cpu().numpy().astype(np.float32), dtype=wp.vec3),
                    wp.array(materials.detach().cpu().numpy().astype(np.float32), dtype=wp.vec4),
                    hidden=False,
                )
            elif newton_cfg.renderer == "frame":
                starts, ends, colors = _build_frame_lines(selected_translations, selected_orientations, selected_scales)
                width = max(float(selected_scales.mean().item()) * 0.05, 0.0025)
                viewer.log_lines(
                    batch_name,
                    wp.array(starts.detach().cpu().numpy().astype(np.float32), dtype=wp.vec3),
                    wp.array(ends.detach().cpu().numpy().astype(np.float32), dtype=wp.vec3),
                    wp.array(colors.detach().cpu().numpy().astype(np.float32), dtype=wp.vec3),
                    width=width,
                    hidden=False,
                )

    def _hide_batch(self, viewer, name: str, newton_cfg: _NewtonMarkerSpec) -> None:
        batch_name = f"{self.group_id}/{name}"
        if newton_cfg.renderer == "mesh" and newton_cfg.mesh_type is not None:
            mesh_name = f"{self.group_id}/meshes/{name}"
            self._ensure_mesh_registered(viewer, mesh_name, newton_cfg)
            viewer.log_instances(batch_name, mesh_name, None, None, None, None, hidden=True)
        elif newton_cfg.renderer == "frame":
            viewer.log_lines(batch_name, None, None, None, hidden=True)

    def _ensure_mesh_registered(self, viewer, mesh_name: str, newton_cfg: _NewtonMarkerSpec) -> None:
        # The marker backend is shared by all Newton-family visualizers. Mesh
        # registration is viewer-local, so the same marker mesh must be logged
        # once per viewer (for example, once for Rerun and once for Viser).
        registered_key = (id(viewer), mesh_name)
        if registered_key in self._registered_meshes or newton_cfg.mesh_type is None:
            return
        mesh = _create_mesh(newton_cfg)
        viewer.log_mesh(
            mesh_name,
            wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3),
            wp.array(mesh.indices.astype(np.int32), dtype=wp.int32),
            normals=wp.array(mesh.normals.astype(np.float32), dtype=wp.vec3) if mesh.normals.size else None,
            uvs=wp.array(mesh.uvs.astype(np.float32), dtype=wp.vec2) if mesh.uvs.size else None,
            texture=newton_cfg.texture,
            hidden=True,
        )
        self._registered_meshes.add(registered_key)


def _resolve_newton_marker_cfg(name: str, marker_cfg: object, cfg: VisualizationMarkersCfg) -> _NewtonMarkerSpec:
    del name, cfg
    return _infer_newton_marker_cfg(marker_cfg)


def _infer_newton_marker_cfg(marker_cfg: object) -> _NewtonMarkerSpec:
    cfg_type = type(marker_cfg).__name__

    if cfg_type == "SphereCfg":
        return _NewtonMarkerSpec(renderer="mesh", mesh_type="sphere", mesh_params={"radius": float(marker_cfg.radius)})
    if cfg_type == "CuboidCfg":
        return _NewtonMarkerSpec(
            renderer="mesh", mesh_type="box", mesh_params={"size": tuple(float(v) for v in marker_cfg.size)}
        )
    if cfg_type == "CylinderCfg":
        return _NewtonMarkerSpec(
            renderer="mesh",
            mesh_type="cylinder",
            mesh_params={"radius": float(marker_cfg.radius), "height": float(marker_cfg.height)},
        )
    if cfg_type == "CapsuleCfg":
        return _NewtonMarkerSpec(
            renderer="mesh",
            mesh_type="capsule",
            mesh_params={"radius": float(marker_cfg.radius), "height": float(marker_cfg.height)},
        )
    if cfg_type == "ConeCfg":
        return _NewtonMarkerSpec(
            renderer="mesh",
            mesh_type="cone",
            mesh_params={"radius": float(marker_cfg.radius), "height": float(marker_cfg.height)},
        )

    if cfg_type == "UsdFileCfg":
        usd_path = str(marker_cfg.usd_path).lower()
        default_scale = _extract_scale_hint(marker_cfg)
        if usd_path.endswith("arrow_x.usd"):
            return _NewtonMarkerSpec(
                renderer="mesh",
                mesh_type="arrow",
                mesh_params={"base_radius": 0.08, "base_height": 0.7, "cap_radius": 0.16, "cap_height": 0.3},
                scale=(default_scale[0], default_scale[1] * 2.5, default_scale[2] * 2.5),
            )
        if usd_path.endswith("frame_prim.usd"):
            return _NewtonMarkerSpec(renderer="frame", scale=default_scale)
        if "dexcube" in usd_path or "dex_cube" in usd_path:
            # TODO: Remove this specialized DexCube mesh code when general
            # UsdFileCfg-to-Newton mesh conversion is supported.
            # DexCube USDs are roughly 6 cm wide. Keep scale separate so task
            # configs such as scale=(1.2, 1.2, 1.2) still apply naturally.
            return _NewtonMarkerSpec(
                renderer="mesh",
                mesh_type="textured_box",
                mesh_params={"size": (0.06, 0.06, 0.06)},
                color=(1.0, 1.0, 1.0),
                texture=_DEX_CUBE_TEXTURE_URL,
            )

        # TODO: Add generic UsdFileCfg -> Newton mesh extraction for mesh-backed USD marker assets.
        # For now, only common marker USDs are mapped to lightweight Newton-native fallbacks.

    return _NewtonMarkerSpec(renderer="none")


def _create_mesh(newton_cfg: _NewtonMarkerSpec):
    mesh_params = newton_cfg.mesh_params or {}
    if newton_cfg.mesh_type == "arrow":
        return Mesh.create_arrow(
            float(mesh_params["base_radius"]),
            float(mesh_params["base_height"]),
            cap_radius=float(mesh_params["cap_radius"]),
            cap_height=float(mesh_params["cap_height"]),
            up_axis=Axis.X,
        )
    if newton_cfg.mesh_type == "box":
        size = mesh_params["size"]
        return Mesh.create_box(float(size[0]) * 0.5, float(size[1]) * 0.5, float(size[2]) * 0.5)
    if newton_cfg.mesh_type == "textured_box":
        return _create_textured_box_mesh(mesh_params["size"])
    if newton_cfg.mesh_type == "sphere":
        return Mesh.create_sphere(radius=float(mesh_params["radius"]))
    if newton_cfg.mesh_type == "cylinder":
        return Mesh.create_cylinder(
            float(mesh_params["radius"]),
            float(mesh_params["height"]) * 0.5,
            up_axis=Axis.Z,
        )
    if newton_cfg.mesh_type == "capsule":
        return Mesh.create_capsule(
            float(mesh_params["radius"]),
            float(mesh_params["height"]) * 0.5,
            up_axis=Axis.Z,
        )
    if newton_cfg.mesh_type == "cone":
        return Mesh.create_cone(
            float(mesh_params["radius"]),
            float(mesh_params["height"]) * 0.5,
            up_axis=Axis.Z,
        )
    raise ValueError(f"Unsupported Newton mesh type: {newton_cfg.mesh_type}")


def _create_textured_box_mesh(size: tuple[float, float, float]) -> _MeshData:
    # TODO: Remove this specialized DexCube mesh code when general
    # UsdFileCfg-to-Newton mesh conversion is supported.
    half = np.asarray(size, dtype=np.float32) * 0.5
    usd_vertices = np.asarray(
        [
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
        dtype=np.float32,
    )
    uvs = np.asarray(
        [
            (1.0, 0.333333),
            (1.0, 0.666667),
            (0.5, 0.666667),
            (0.5, 0.333333),
            (0.5, 0.666667),
            (0.5, 1.0),
            (0.0, 1.0),
            (0.0, 0.666667),
            (0.5, 0.333333),
            (0.5, 0.666667),
            (0.0, 0.666667),
            (0.0, 0.333333),
            (1.0, 0.0),
            (1.0, 0.333333),
            (0.5, 0.333333),
            (0.5, 0.0),
            (0.5, 0.0),
            (0.5, 0.333333),
            (0.0, 0.333333),
            (0.0, 0.0),
            (1.0, 0.666667),
            (1.0, 1.0),
            (0.5, 1.0),
            (0.5, 0.666667),
        ],
        dtype=np.float32,
    )
    indices: list[int] = []
    for base in range(0, 24, 4):
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return _MeshData(
        vertices=usd_vertices * half,
        indices=np.asarray(indices, dtype=np.int32),
        normals=np.zeros((0, 3), dtype=np.float32),
        uvs=uvs,
    )


def _filter_marker_state(
    marker: NewtonVisualizationMarkers,
    visible_env_ids: list[int] | None,
    num_envs: int,
) -> dict[str, Any]:
    if visible_env_ids is None or marker.count == 0 or num_envs <= 0 or marker.count % num_envs != 0:
        return {
            "visible": marker.visible,
            "translations": marker.translations,
            "orientations": marker.orientations,
            "scales": marker.scales,
            "marker_indices": marker.marker_indices,
            "count": marker.count,
        }

    keep: list[int] = []
    repeat_count = marker.count // num_envs
    for block_idx in range(repeat_count):
        base = block_idx * num_envs
        for env_id in visible_env_ids:
            idx = base + env_id
            if idx < marker.count:
                keep.append(idx)

    if len(keep) == marker.count:
        return {
            "visible": marker.visible,
            "translations": marker.translations,
            "orientations": marker.orientations,
            "scales": marker.scales,
            "marker_indices": marker.marker_indices,
            "count": marker.count,
        }

    index = torch.tensor(keep, dtype=torch.long, device=marker.infer_device())
    return {
        "visible": marker.visible,
        "translations": marker.translations.index_select(0, index) if marker.translations is not None else None,
        "orientations": marker.orientations.index_select(0, index) if marker.orientations is not None else None,
        "scales": marker.scales.index_select(0, index) if marker.scales is not None else None,
        "marker_indices": marker.marker_indices.index_select(0, index) if marker.marker_indices is not None else None,
        "count": len(keep),
    }


def _extract_scale_hint(marker_cfg: object) -> tuple[float, float, float]:
    scale = marker_cfg.scale if type(marker_cfg).__name__ == "UsdFileCfg" else None
    if scale is None:
        return (1.0, 1.0, 1.0)
    return tuple(float(v) for v in scale)


def _extract_color(marker_cfg: object) -> tuple[float, float, float]:
    material_cfg = marker_cfg.visual_material
    if material_cfg is None:
        return _UNBOUND_DEFAULT_FALLBACK_GRAY

    if color := _extract_omnipbr_like_color(material_cfg):
        return color

    material_type = type(material_cfg).__name__
    if material_type == "PreviewSurfaceCfg":
        return _extract_rgb(material_cfg.diffuse_color) or _UNBOUND_DEFAULT_FALLBACK_GRAY
    if material_type == "GlassMdlCfg":
        return _extract_rgb(material_cfg.glass_color) or _UNBOUND_DEFAULT_FALLBACK_GRAY

    return _UNBOUND_DEFAULT_FALLBACK_GRAY


def _extract_omnipbr_like_color(material_cfg: object) -> tuple[float, float, float] | None:
    material_type = type(material_cfg).__name__
    if material_type == "MdlFileCfg":
        if not str(material_cfg.mdl_path).lower().endswith("omnipbr.mdl"):
            return None
        brightness = material_cfg.albedo_brightness
        if brightness is not None:
            diffuse_constant = (float(brightness), float(brightness), float(brightness))
        else:
            diffuse_constant = _OMNIPBR_DEFAULTS["diffuse_color_constant"]
        diffuse_tint = _OMNIPBR_DEFAULTS["diffuse_tint"]
    else:
        return None

    return (
        diffuse_constant[0] * diffuse_tint[0],
        diffuse_constant[1] * diffuse_tint[1],
        diffuse_constant[2] * diffuse_tint[2],
    )


def _extract_rgb(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        rgb = tuple(float(v) for v in value)
    except TypeError:
        return None
    if len(rgb) < 3:
        return None
    return (rgb[0], rgb[1], rgb[2])


def _build_frame_lines(
    translations: torch.Tensor,
    orientations: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unit_axes = (
        torch.eye(3, dtype=torch.float32, device=translations.device).unsqueeze(0).repeat(translations.shape[0], 1, 1)
    )
    scaled_axes = unit_axes * scales.unsqueeze(1)
    repeated_quats = orientations.unsqueeze(1).repeat(1, 3, 1).reshape(-1, 4)
    rotated_axes = quat_apply(repeated_quats, scaled_axes.reshape(-1, 3)).reshape(-1, 3, 3)
    starts = translations.unsqueeze(1).repeat(1, 3, 1).reshape(-1, 3)
    ends = (translations.unsqueeze(1) + rotated_axes).reshape(-1, 3)
    colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.35, 1.0]],
        dtype=torch.float32,
        device=translations.device,
    ).repeat(translations.shape[0], 1)
    return starts, ends, colors
