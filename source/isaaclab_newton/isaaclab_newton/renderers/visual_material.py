# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton visual-material import and device-side color writes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import Model, ModelBuilder
from newton.selection import ArticulationView

from pxr import Usd, UsdGeom, UsdShade

if TYPE_CHECKING:
    from isaaclab.renderers.base_renderer import VisualMaterialBatch


def import_builder_visual_material_paths(builder: ModelBuilder, stage: Usd.Stage) -> None:
    """Record each shape's effective material binding."""
    builder.add_custom_attribute(
        ModelBuilder.CustomAttribute(
            name="visual_material_path",
            namespace="isaaclab",
            dtype=str,
            frequency=Model.AttributeFrequency.SHAPE,
            default="",
        )
    )
    paths = builder.custom_attributes["isaaclab:visual_material_path"].values
    for shape_index, shape_path in enumerate(builder.shape_label):
        shape_prim = stage.GetPrimAtPath(shape_path)
        imageable = UsdGeom.Imageable(shape_prim)
        if not shape_prim.IsValid() or (imageable and imageable.ComputePurpose() == UsdGeom.Tokens.guide):
            continue
        _material, relationship = UsdShade.MaterialBindingAPI(shape_prim).ComputeBoundMaterial()
        if relationship and (targets := relationship.GetTargets()):
            paths[shape_index] = targets[0].pathString


@wp.func
def _linear_to_srgb(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value <= 0.0031308:
        return 12.92 * value
    if value >= 1.0:
        return 1.0
    return 1.055 * wp.pow(value, 1.0 / 2.4) - 0.055


@wp.kernel(enable_backward=False)
def _scatter_material_colors(
    colors: wp.array(dtype=wp.vec3f),
    material_offsets: wp.array(dtype=wp.int32),
    env_ids: wp.array(dtype=wp.int32),
    shape_offsets: wp.array(dtype=wp.int32),
    shape_rows: wp.array(dtype=wp.int32),
    shape_colors: wp.array(dtype=wp.vec3f),
):
    material, env, shape_slot = wp.tid()
    material_row = material_offsets[material] + env_ids[env]
    shape_row = shape_offsets[material_row] + shape_slot
    if shape_row < shape_offsets[material_row + 1]:
        color = colors[material_row]
        shape_colors[shape_rows[shape_row]] = wp.vec3f(
            _linear_to_srgb(color[0]), _linear_to_srgb(color[1]), _linear_to_srgb(color[2])
        )


class VisualMaterialWriter:
    """Scatter one flat scene color buffer into bound Newton shape rows."""

    def __init__(self, model: Model, batches: tuple[VisualMaterialBatch, ...]):
        self._model = model
        # Newton exposes only shape_color; other channel batches remain available to other render consumers.
        batch = next((batch for batch in batches if batch.channel == "color"), None)
        if batch is None:
            self._colors = None
            self._all_offsets = self._all_env_ids = self._shape_offsets = self._shape_rows = wp.empty(
                0, dtype=wp.int32, device=model.device
            )
            self._max_shapes = 0
            return

        rows_by_path = dict(zip(batch.material_paths, range(len(batch.material_paths)), strict=True))
        shapes_by_material = [[] for _ in batch.material_paths]
        for shape_row, material_path in enumerate(model.isaaclab.visual_material_path):
            material_row = rows_by_path.get(material_path)
            if material_row is not None:
                shapes_by_material[material_row].append(shape_row)
        shape_offsets = [0]
        for rows in shapes_by_material:
            shape_offsets.append(shape_offsets[-1] + len(rows))
        self._colors = wp.from_torch(batch.values, dtype=wp.vec3f)
        self._all_offsets = wp.array(range(len(batch.values)), dtype=wp.int32, device=model.device)
        self._all_env_ids = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._shape_offsets = wp.array(shape_offsets, dtype=wp.int32, device=model.device)
        self._shape_rows = wp.array(
            [shape_row for rows in shapes_by_material for shape_row in rows], dtype=wp.int32, device=model.device
        )
        self._max_shapes = max(map(len, shapes_by_material), default=0)

    def __call__(self, material_offsets: dict[str, wp.array] | None = None, env_ids: wp.array | None = None) -> None:
        """Apply selected material/environment rows with one launch and no host-side work."""
        offsets = self._all_offsets if material_offsets is None else material_offsets.get("color")
        env_ids = self._all_env_ids if env_ids is None else env_ids
        if self._max_shapes and offsets is not None and len(offsets) and len(env_ids):
            wp.launch(
                _scatter_material_colors,
                dim=(len(offsets), len(env_ids), self._max_shapes),
                inputs=[
                    self._colors,
                    offsets,
                    env_ids,
                    self._shape_offsets,
                    self._shape_rows,
                    self._model.shape_color,
                ],
                device=self._model.device,
            )

    def close(self) -> None:
        """Release model and scene-buffer references."""
        self._model = self._colors = self._all_offsets = self._all_env_ids = None
        self._shape_offsets = self._shape_rows = None
        self._max_shapes = 0


@wp.kernel(enable_backward=False)
def _write_shape_colors(
    colors: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    shape_ids: wp.array(dtype=wp.int32),
    body_rows: wp.array(dtype=wp.int32),
    shape_colors: wp.array2d(dtype=wp.vec3f),
):
    env_row, shape_row = wp.tid()
    color = colors[env_row, body_rows[shape_row]]
    shape_colors[env_ids[env_row], shape_ids[shape_row]] = wp.vec3f(
        _linear_to_srgb(color[0]), _linear_to_srgb(color[1]), _linear_to_srgb(color[2])
    )


class VisualShapeColorWriter:
    """Write one sampled color per selected articulation body and environment."""

    def __init__(self, model: Model, view: ArticulationView, body_names: tuple[str, ...]):
        self.body_count = len(body_names)
        body_rows = {name: row for row, name in enumerate(body_names)}
        shape_ids: list[int] = []
        shape_body_rows: list[int] = []
        for body_id, body_name in enumerate(view.body_names):
            if body_name in body_rows:
                shapes = view.body_shapes[body_id]
                shape_ids.extend(shapes)
                shape_body_rows.extend([body_rows[body_name]] * len(shapes))
        self.device = torch.device(wp.device_to_torch(model.device))
        self._shape_ids = wp.array(shape_ids, dtype=wp.int32, device=model.device)
        self._body_rows = wp.array(shape_body_rows, dtype=wp.int32, device=model.device)
        self._view = view
        self.rebind(model)

    @property
    def model(self) -> Model:
        return self._view.model

    def rebind(self, model: Model) -> None:
        """Rebind the shape-color buffer after Newton replaces its model."""
        self._view.model = model
        self._shape_colors = self._view.get_attribute("shape_color", model)[:, 0]

    def __call__(self, colors: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Write colors shaped ``[len(env_ids), body_count, 3]`` with one launch."""
        if len(self._shape_ids) and env_ids.numel():
            stream = (
                wp.stream_from_torch(torch.cuda.current_stream(self.device)) if self.device.type == "cuda" else None
            )
            with wp.ScopedStream(stream, sync_enter=False):
                wp.launch(
                    _write_shape_colors,
                    dim=(env_ids.numel(), len(self._shape_ids)),
                    inputs=[
                        wp.from_torch(colors, dtype=wp.vec3f),
                        wp.from_torch(env_ids, dtype=wp.int32),
                        self._shape_ids,
                        self._body_rows,
                        self._shape_colors,
                    ],
                    device=self.model.device,
                )
