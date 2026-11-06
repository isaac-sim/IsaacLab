# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-side visual-material writes for Fabric rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.renderers.base_renderer import VisualMaterialBatch


@wp.kernel(enable_backward=False)
def _index_rows(rows: wp.fabricarray(dtype=wp.uint32), inverse: wp.array(dtype=wp.int32)):
    inverse[rows[wp.tid()]] = wp.tid()


@wp.kernel(enable_backward=False)
def _write_float(
    values: wp.array(dtype=wp.float32),
    material_offsets: wp.array(dtype=wp.int32),
    env_ids: wp.array(dtype=wp.int32),
    inverse: wp.array(dtype=wp.int32),
    output: wp.fabricarray(dtype=wp.float32),
):
    material, env = wp.tid()
    row = material_offsets[material] + env_ids[env]
    output_row = inverse[row]
    if output_row >= 0:
        output[output_row] = values[row]


@wp.kernel(enable_backward=False)
def _write_float2(
    values: wp.array(dtype=wp.vec2f),
    material_offsets: wp.array(dtype=wp.int32),
    env_ids: wp.array(dtype=wp.int32),
    inverse: wp.array(dtype=wp.int32),
    output: wp.fabricarray(dtype=wp.vec2f),
):
    material, env = wp.tid()
    row = material_offsets[material] + env_ids[env]
    output_row = inverse[row]
    if output_row >= 0:
        output[output_row] = values[row]


@wp.kernel(enable_backward=False)
def _write_float3(
    values: wp.array(dtype=wp.vec3f),
    material_offsets: wp.array(dtype=wp.int32),
    env_ids: wp.array(dtype=wp.int32),
    inverse: wp.array(dtype=wp.int32),
    output: wp.fabricarray(dtype=wp.vec3f),
):
    material, env = wp.tid()
    row = material_offsets[material] + env_ids[env]
    output_row = inverse[row]
    if output_row >= 0:
        output[output_row] = values[row]


_WRITES = {
    (torch.float32, ()): (_write_float, wp.float32),
    (torch.float32, (2,)): (_write_float2, wp.vec2f),
    (torch.float32, (3,)): (_write_float3, wp.vec3f),
}


class FabricVisualMaterialWriter:
    """Compile Fabric shader addresses once and update them from stable device buffers."""

    def __init__(self, batches: tuple[VisualMaterialBatch, ...]):
        import usdrt

        from isaaclab.sim.utils.stage import get_current_stage

        compiled = []
        for batch in batches:
            layout = (batch.values.dtype, tuple(batch.values.shape[1:]))
            if layout not in _WRITES:
                raise TypeError(f"Unsupported Fabric visual-material tensor layout: {layout}.")
            compiled.append((batch, *_WRITES[layout]))

        stage = get_current_stage(fabric=True)
        writes = {batch.channel: [] for batch, _kernel, _dtype in compiled}
        for batch_index, (batch, kernel, dtype) in enumerate(compiled):
            values = wp.from_torch(batch.values.detach(), dtype=dtype)
            groups: dict[tuple[str, Any], list[tuple[int, Any]]] = {}
            for row, (shader_path, input_name) in enumerate(zip(batch.shader_paths, batch.input_names, strict=True)):
                shader = stage.GetPrimAtPath(shader_path)
                if not shader.IsValid():
                    raise RuntimeError(f"Fabric visual-material shader prim {shader_path!r} does not exist.")
                attribute_name = f"inputs:{input_name}"
                attribute = shader.GetAttribute(attribute_name)
                if not attribute.IsValid():
                    raise RuntimeError(
                        f"Fabric visual-material input {attribute_name!r} does not exist on {shader_path!r}."
                    )
                groups.setdefault((attribute_name, attribute.GetTypeName()), []).append((row, shader))

            for group_index, ((attribute_name, attribute_type), rows) in enumerate(groups.items()):
                row_attribute = f"isaaclab:visualMaterialRow:{batch_index}:{group_index}"
                for row, shader in rows:
                    shader.CreateAttribute(row_attribute, usdrt.Sdf.ValueTypeNames.UInt, True).Set(row)
                selection = stage.SelectPrims(
                    require_attrs=[
                        (usdrt.Sdf.ValueTypeNames.UInt, row_attribute, usdrt.Usd.Access.Read),
                        (attribute_type, attribute_name, usdrt.Usd.Access.ReadWrite),
                    ],
                    device=str(batch.values.device),
                )
                if selection.GetCount() != len(rows):
                    raise RuntimeError(
                        f"Fabric visual-material selection for {attribute_name!r} matched "
                        f"{selection.GetCount()} shader prims, expected {len(rows)}."
                    )
                inverse = wp.full(len(batch.values), -1, dtype=wp.int32, device=str(batch.values.device))
                wp.launch(
                    _index_rows,
                    dim=len(rows),
                    inputs=[wp.fabricarray(selection, row_attribute), inverse],
                    device=str(batch.values.device),
                )
                writes[batch.channel].append((kernel, values, selection, inverse, attribute_name))

        self._writes = {channel: tuple(channel_writes) for channel, channel_writes in writes.items()}
        self._all_offsets = {
            batch.channel: wp.array(range(len(batch.values)), dtype=wp.int32, device=str(batch.values.device))
            for batch, _kernel, _dtype in compiled
        }
        device = next(iter(compiled))[0].values.device
        self._zero_env_id = wp.zeros(1, dtype=wp.int32, device=str(device))

    def __call__(self, material_offsets: dict[str, wp.array] | None = None, env_ids: wp.array | None = None) -> None:
        """Write selected material/environment rows through precompiled Fabric selections."""
        selected = self._all_offsets if material_offsets is None else material_offsets
        env_ids = self._zero_env_id if env_ids is None else env_ids
        for channel, offsets in selected.items():
            for kernel, values, selection, inverse, attribute_name in self._writes[channel]:
                selection.PrepareForReuse()
                wp.launch(
                    kernel,
                    dim=(len(offsets), len(env_ids)),
                    inputs=[values, offsets, env_ids, inverse, wp.fabricarray(selection, attribute_name)],
                    device=values.device,
                )

    def close(self) -> None:
        """Release references to stage-bound Fabric selections."""
        self._all_offsets = {}
        self._zero_env_id = None
        self._writes = {}
