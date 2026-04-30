# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared marker rendering helpers for Newton-family visualizers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import warp as wp
from newton import Axis, Mesh

from isaaclab.markers.visualization_markers_cfg import NewtonMarkerCfg, VisualizationMarkersCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply

logger = logging.getLogger(__name__)

_OMNIPBR_DEFAULTS = {
    "diffuse_color_constant": (0.2, 0.2, 0.2),
    "diffuse_tint": (1.0, 1.0, 1.0),
}
_UNBOUND_DEFAULT_FALLBACK_GRAY = (0.18, 0.18, 0.18)


class NewtonMarkerRenderer:
    """Render Isaac Lab visualization markers through Newton viewer APIs."""

    def __init__(self):
        self._registered_meshes: set[str] = set()
        self._warned_unsupported: set[str] = set()

    def render(self, viewer, visible_env_ids: list[int] | None, num_envs: int) -> None:
        sim = SimulationContext.instance()
        if sim is None:
            return

        for group_id, state in sim.get_visualization_marker_groups().items():
            if not isinstance(state, dict) or "cfg" not in state:
                continue
            self._render_group(viewer, group_id, state, visible_env_ids=visible_env_ids, num_envs=num_envs)

    def _render_group(
        self,
        viewer,
        group_id: str,
        state: dict[str, Any],
        visible_env_ids: list[int] | None,
        num_envs: int,
    ) -> None:
        filtered = _filter_group_state(state, visible_env_ids=visible_env_ids, num_envs=num_envs)
        cfg: VisualizationMarkersCfg = filtered["cfg"]
        if filtered["count"] == 0:
            for name, marker_cfg in cfg.markers.items():
                self._hide_batch(viewer, group_id, name, _resolve_newton_marker_cfg(name, marker_cfg, cfg))
            return

        translations = filtered["translations"]
        if translations is None:
            return
        orientations = filtered["orientations"]
        if orientations is None:
            orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=translations.device).repeat(
                filtered["count"], 1
            )
        scales = filtered["scales"]
        if scales is None:
            scales = torch.ones((filtered["count"], 3), dtype=torch.float32, device=translations.device)
        marker_indices = filtered["marker_indices"]
        if marker_indices is None:
            marker_indices = torch.zeros(filtered["count"], dtype=torch.int64, device=translations.device)

        for proto_index, (name, marker_cfg) in enumerate(cfg.markers.items()):
            newton_cfg = _resolve_newton_marker_cfg(name, marker_cfg, cfg)
            batch_name = f"{group_id}/{name}"
            selected = marker_indices == proto_index
            if not filtered["visible"] or int(selected.sum().item()) == 0:
                self._hide_batch(viewer, group_id, name, newton_cfg)
                continue

            if newton_cfg.renderer == "none":
                unsupported_key = f"{group_id}:{name}"
                if unsupported_key not in self._warned_unsupported:
                    logger.warning(
                        "[NewtonMarkerRenderer] Unsupported marker prototype '%s' in group '%s'; skipping.",
                        name,
                        group_id,
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
                mesh_name = f"{group_id}/meshes/{name}"
                self._ensure_mesh_registered(viewer, mesh_name, newton_cfg)
                color = newton_cfg.color or _extract_color(marker_cfg)
                colors = torch.tensor(color, dtype=torch.float32, device=scales.device).repeat(
                    selected_scales.shape[0], 1
                )
                materials = torch.zeros((selected_scales.shape[0], 4), dtype=torch.float32, device=scales.device)
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

    def _hide_batch(self, viewer, group_id: str, name: str, newton_cfg: NewtonMarkerCfg) -> None:
        batch_name = f"{group_id}/{name}"
        if newton_cfg.renderer == "mesh" and newton_cfg.mesh_type is not None:
            mesh_name = f"{group_id}/meshes/{name}"
            self._ensure_mesh_registered(viewer, mesh_name, newton_cfg)
            viewer.log_instances(batch_name, mesh_name, None, None, None, None, hidden=True)
        elif newton_cfg.renderer == "frame":
            viewer.log_lines(batch_name, None, None, None, hidden=True)

    def _ensure_mesh_registered(self, viewer, mesh_name: str, newton_cfg: NewtonMarkerCfg) -> None:
        if mesh_name in self._registered_meshes or newton_cfg.mesh_type is None:
            return
        mesh = _create_mesh(newton_cfg)
        viewer.log_mesh(
            mesh_name,
            wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3),
            wp.array(mesh.indices.astype(np.int32), dtype=wp.int32),
            normals=wp.array(mesh.normals.astype(np.float32), dtype=wp.vec3) if mesh.normals.size else None,
            uvs=wp.array(mesh.uvs.astype(np.float32), dtype=wp.vec2) if mesh.uvs.size else None,
            hidden=True,
        )
        self._registered_meshes.add(mesh_name)


def _resolve_newton_marker_cfg(name: str, marker_cfg: object, cfg: VisualizationMarkersCfg) -> NewtonMarkerCfg:
    if name in cfg.newton_markers:
        return cfg.newton_markers[name]
    return _infer_newton_marker_cfg(marker_cfg)


def _infer_newton_marker_cfg(marker_cfg: object) -> NewtonMarkerCfg:
    cfg_type = type(marker_cfg).__name__

    if cfg_type == "SphereCfg":
        return NewtonMarkerCfg(renderer="mesh", mesh_type="sphere", mesh_params={"radius": float(marker_cfg.radius)})
    if cfg_type == "CuboidCfg":
        return NewtonMarkerCfg(
            renderer="mesh", mesh_type="box", mesh_params={"size": tuple(float(v) for v in marker_cfg.size)}
        )
    if cfg_type == "CylinderCfg":
        return NewtonMarkerCfg(
            renderer="mesh",
            mesh_type="cylinder",
            mesh_params={"radius": float(marker_cfg.radius), "height": float(marker_cfg.height)},
        )
    if cfg_type == "CapsuleCfg":
        return NewtonMarkerCfg(
            renderer="mesh",
            mesh_type="capsule",
            mesh_params={"radius": float(marker_cfg.radius), "height": float(marker_cfg.height)},
        )
    if cfg_type == "ConeCfg":
        return NewtonMarkerCfg(
            renderer="mesh",
            mesh_type="cone",
            mesh_params={"radius": float(marker_cfg.radius), "height": float(marker_cfg.height)},
        )

    if cfg_type == "UsdFileCfg":
        usd_path = str(marker_cfg.usd_path).lower()
        default_scale = _extract_scale_hint(marker_cfg)
        if usd_path.endswith("arrow_x.usd"):
            return NewtonMarkerCfg(
                renderer="mesh",
                mesh_type="arrow",
                mesh_params={"base_radius": 0.08, "base_height": 0.7, "cap_radius": 0.16, "cap_height": 0.3},
                scale=(default_scale[0], default_scale[1] * 2.5, default_scale[2] * 2.5),
            )
        if usd_path.endswith("frame_prim.usd"):
            return NewtonMarkerCfg(renderer="frame", scale=default_scale)
        if "dex_cube" in usd_path or "cube" in usd_path:
            return NewtonMarkerCfg(renderer="mesh", mesh_type="box", mesh_params={"size": (1.0, 1.0, 1.0)})

        # TODO: Add generic UsdFileCfg -> Newton mesh extraction for mesh-backed USD marker assets.
        # For now, only common marker USDs are mapped to lightweight Newton-native fallbacks.

    return NewtonMarkerCfg(renderer="none")


def _create_mesh(newton_cfg: NewtonMarkerCfg):
    if newton_cfg.mesh_type == "arrow":
        return Mesh.create_arrow(
            float(newton_cfg.mesh_params["base_radius"]),
            float(newton_cfg.mesh_params["base_height"]),
            cap_radius=float(newton_cfg.mesh_params["cap_radius"]),
            cap_height=float(newton_cfg.mesh_params["cap_height"]),
            up_axis=Axis.X,
        )
    if newton_cfg.mesh_type == "box":
        size = newton_cfg.mesh_params["size"]
        return Mesh.create_box(float(size[0]) * 0.5, float(size[1]) * 0.5, float(size[2]) * 0.5)
    if newton_cfg.mesh_type == "sphere":
        return Mesh.create_sphere(radius=float(newton_cfg.mesh_params["radius"]))
    if newton_cfg.mesh_type == "cylinder":
        return Mesh.create_cylinder(
            float(newton_cfg.mesh_params["radius"]),
            float(newton_cfg.mesh_params["height"]) * 0.5,
            up_axis=Axis.Z,
        )
    if newton_cfg.mesh_type == "capsule":
        return Mesh.create_capsule(
            float(newton_cfg.mesh_params["radius"]),
            float(newton_cfg.mesh_params["height"]) * 0.5,
            up_axis=Axis.Z,
        )
    if newton_cfg.mesh_type == "cone":
        return Mesh.create_cone(
            float(newton_cfg.mesh_params["radius"]),
            float(newton_cfg.mesh_params["height"]) * 0.5,
            up_axis=Axis.Z,
        )
    raise ValueError(f"Unsupported Newton mesh type: {newton_cfg.mesh_type}")


def _filter_group_state(
    state: dict[str, Any],
    visible_env_ids: list[int] | None,
    num_envs: int,
) -> dict[str, Any]:
    count = int(state["count"])
    if visible_env_ids is None or count == 0 or num_envs <= 0 or count % num_envs != 0:
        return state

    keep: list[int] = []
    repeat_count = count // num_envs
    for block_idx in range(repeat_count):
        base = block_idx * num_envs
        for env_id in visible_env_ids:
            idx = base + env_id
            if idx < count:
                keep.append(idx)

    if len(keep) == count:
        return state

    index = torch.tensor(keep, dtype=torch.long, device=_resolve_group_device(state))
    out = dict(state)
    out["translations"] = state["translations"].index_select(0, index) if state["translations"] is not None else None
    out["orientations"] = state["orientations"].index_select(0, index) if state["orientations"] is not None else None
    out["scales"] = state["scales"].index_select(0, index) if state["scales"] is not None else None
    out["marker_indices"] = (
        state["marker_indices"].index_select(0, index) if state["marker_indices"] is not None else None
    )
    out["count"] = len(keep)
    return out


def _resolve_group_device(state: dict[str, Any]) -> torch.device:
    for key in ("translations", "orientations", "scales", "marker_indices"):
        value = state[key]
        if value is not None:
            return value.device
    return torch.device("cpu")


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
