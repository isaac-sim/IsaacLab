"""Shared marker rendering helpers for Newton-family visualizers."""

from __future__ import annotations

import logging

import numpy as np
import torch
import warp as wp
from newton import Axis, Mesh

from isaaclab.markers.newton_marker_utils import NewtonMarkerGroupState, NewtonMarkerPrototype
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply

logger = logging.getLogger(__name__)


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
            if not isinstance(state, NewtonMarkerGroupState):
                continue
            self._render_group(viewer, group_id, state, visible_env_ids=visible_env_ids, num_envs=num_envs)

    def _render_group(
        self,
        viewer,
        group_id: str,
        state: NewtonMarkerGroupState,
        visible_env_ids: list[int] | None,
        num_envs: int,
    ) -> None:
        filtered = _filter_group_state(state, visible_env_ids=visible_env_ids, num_envs=num_envs)
        if filtered.count == 0:
            for proto_index, proto in enumerate(state.prototypes):
                self._hide_batch(viewer, group_id, proto_index, proto)
            return

        translations = filtered.translations
        if translations is None:
            return
        orientations = filtered.orientations
        if orientations is None:
            orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=translations.device).repeat(filtered.count, 1)
        scales = filtered.scales
        if scales is None:
            scales = torch.ones((filtered.count, 3), dtype=torch.float32, device=translations.device)
        marker_indices = filtered.marker_indices
        if marker_indices is None:
            marker_indices = torch.zeros(filtered.count, dtype=torch.int64, device=translations.device)

        for proto_index, proto in enumerate(filtered.prototypes):
            batch_name = f"{group_id}/{proto.name}"
            selected = marker_indices == proto_index
            if not filtered.visible or not proto.visible or int(selected.sum().item()) == 0:
                self._hide_batch(viewer, group_id, proto_index, proto)
                continue

            selected_translations = translations[selected]
            selected_orientations = orientations[selected]
            selected_scales = scales[selected] * torch.tensor(
                proto.default_scale, dtype=torch.float32, device=scales.device
            ).unsqueeze(0)

            if proto.renderer == "mesh":
                mesh_name = f"{group_id}/meshes/{proto.name}"
                self._ensure_mesh_registered(viewer, mesh_name, proto)
                colors = torch.tensor(proto.color, dtype=torch.float32, device=scales.device).repeat(
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
            elif proto.renderer == "frame":
                starts, ends, colors = _build_frame_lines(
                    selected_translations, selected_orientations, selected_scales
                )
                width = max(float(selected_scales.mean().item()) * 0.05, 0.0025)
                viewer.log_lines(
                    batch_name,
                    wp.array(starts.detach().cpu().numpy().astype(np.float32), dtype=wp.vec3),
                    wp.array(ends.detach().cpu().numpy().astype(np.float32), dtype=wp.vec3),
                    wp.array(colors.detach().cpu().numpy().astype(np.float32), dtype=wp.vec3),
                    width=width,
                    hidden=False,
                )
            else:
                unsupported_key = f"{group_id}:{proto.name}"
                if unsupported_key not in self._warned_unsupported:
                    logger.warning(
                        "[NewtonMarkerRenderer] Unsupported marker prototype '%s' in group '%s'; skipping.",
                        proto.name,
                        group_id,
                    )
                    self._warned_unsupported.add(unsupported_key)

    def _hide_batch(self, viewer, group_id: str, proto_index: int, proto: NewtonMarkerPrototype) -> None:
        batch_name = f"{group_id}/{proto.name}"
        if proto.renderer == "mesh" and proto.mesh_type is not None:
            mesh_name = f"{group_id}/meshes/{proto.name}"
            self._ensure_mesh_registered(viewer, mesh_name, proto)
            viewer.log_instances(batch_name, mesh_name, None, None, None, None, hidden=True)
        elif proto.renderer == "frame":
            viewer.log_lines(batch_name, None, None, None, hidden=True)

    def _ensure_mesh_registered(self, viewer, mesh_name: str, proto: NewtonMarkerPrototype) -> None:
        if mesh_name in self._registered_meshes or proto.mesh_type is None:
            return
        mesh = _create_mesh(proto)
        viewer.log_mesh(
            mesh_name,
            wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3),
            wp.array(mesh.indices.astype(np.int32), dtype=wp.int32),
            normals=wp.array(mesh.normals.astype(np.float32), dtype=wp.vec3) if mesh.normals.size else None,
            uvs=wp.array(mesh.uvs.astype(np.float32), dtype=wp.vec2) if mesh.uvs.size else None,
            hidden=False,
        )
        self._registered_meshes.add(mesh_name)


def _create_mesh(proto: NewtonMarkerPrototype):
    if proto.mesh_type == "arrow":
        return Mesh.create_arrow(
            float(proto.mesh_params["base_radius"]),
            float(proto.mesh_params["base_height"]),
            cap_radius=float(proto.mesh_params["cap_radius"]),
            cap_height=float(proto.mesh_params["cap_height"]),
            up_axis=Axis.X,
        )
    if proto.mesh_type == "box":
        size = proto.mesh_params["size"]
        return Mesh.create_box(float(size[0]) * 0.5, float(size[1]) * 0.5, float(size[2]) * 0.5)
    if proto.mesh_type == "sphere":
        return Mesh.create_sphere(radius=float(proto.mesh_params["radius"]))
    if proto.mesh_type == "cylinder":
        return Mesh.create_cylinder(
            float(proto.mesh_params["radius"]),
            float(proto.mesh_params["height"]) * 0.5,
            up_axis=Axis.Z,
        )
    if proto.mesh_type == "capsule":
        return Mesh.create_capsule(
            float(proto.mesh_params["radius"]),
            float(proto.mesh_params["height"]) * 0.5,
            up_axis=Axis.Z,
        )
    if proto.mesh_type == "cone":
        return Mesh.create_cone(
            float(proto.mesh_params["radius"]),
            float(proto.mesh_params["height"]) * 0.5,
            up_axis=Axis.Z,
        )
    raise ValueError(f"Unsupported Newton mesh type: {proto.mesh_type}")


def _filter_group_state(
    state: NewtonMarkerGroupState,
    visible_env_ids: list[int] | None,
    num_envs: int,
) -> NewtonMarkerGroupState:
    if visible_env_ids is None or state.count == 0 or num_envs <= 0 or state.count % num_envs != 0:
        return state

    keep: list[int] = []
    repeat_count = state.count // num_envs
    for block_idx in range(repeat_count):
        base = block_idx * num_envs
        for env_id in visible_env_ids:
            idx = base + env_id
            if idx < state.count:
                keep.append(idx)

    if len(keep) == state.count:
        return state

    index = torch.tensor(keep, dtype=torch.long, device=_resolve_group_device(state))
    return NewtonMarkerGroupState(
        group_id=state.group_id,
        prototypes=state.prototypes,
        visible=state.visible,
        translations=state.translations.index_select(0, index) if state.translations is not None else None,
        orientations=state.orientations.index_select(0, index) if state.orientations is not None else None,
        scales=state.scales.index_select(0, index) if state.scales is not None else None,
        marker_indices=state.marker_indices.index_select(0, index) if state.marker_indices is not None else None,
        count=len(keep),
    )


def _resolve_group_device(state: NewtonMarkerGroupState) -> torch.device:
    for value in (state.translations, state.orientations, state.scales, state.marker_indices):
        if value is not None:
            return value.device
    return torch.device("cpu")


def _build_frame_lines(
    translations: torch.Tensor,
    orientations: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unit_axes = torch.eye(3, dtype=torch.float32, device=translations.device).unsqueeze(0).repeat(translations.shape[0], 1, 1)
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
