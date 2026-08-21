# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-only visual-material writes for OVRTX-owned scenes."""

from __future__ import annotations

import itertools
import weakref
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from ovrtx import BindingFlag, DataAccess, PrimMode

if TYPE_CHECKING:
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from .ovrtx_renderer import OVRTXRenderer


class OVRTXVisualMaterialWriter:
    """Compile OVRTX material addresses once and publish dirty device buffers."""

    def __init__(self, renderer: OVRTXRenderer, batches: tuple[VisualMaterialBatch, ...]):
        self._renderer_ref = weakref.ref(renderer)
        self._buffers: dict[str, torch.Tensor] = {}
        self._dirty_channels: set[str] = set()
        self._operations: tuple[Any, ...] = ()
        self._addresses: list[tuple[str, Any, Any | None, str, slice]] = []

        groups = []
        device = None
        for batch in batches:
            values = batch.values.detach()
            if values.dtype == torch.float32 and values.ndim == 1:
                dtype, shape = "float32", None
            elif values.dtype == torch.float32 and values.ndim == 2 and values.shape[1] in (2, 3):
                dtype, shape = "float32", (values.shape[1],)
            else:
                raise TypeError(
                    f"OVRTX visual-material channel {batch.channel!r} requires float, float2, or float3; "
                    f"got dtype={values.dtype}, shape={tuple(values.shape)}."
                )
            if not values.is_cuda or (device is not None and values.device != device):
                raise RuntimeError("OVRTX visual-material attributes must reside on one CUDA device.")
            device = values.device
            self._buffers[batch.channel] = values
            start = 0
            for input_name, input_group in itertools.groupby(batch.input_names):
                end = start + sum(1 for _ in input_group)
                rows = slice(start, end)
                groups.append(
                    (batch.channel, f"inputs:{input_name}", list(batch.shader_paths[rows]), rows, dtype, shape)
                )
                start = end
        self._event = wp.Event(device=str(device))
        try:
            for channel, attribute_name, shader_paths, rows, dtype, shape in groups:
                if renderer._use_ovstage:
                    path_list = renderer._stage_paths.create_path_list_from_strings(shader_paths)
                    try:
                        address = renderer._stage.query_from_path_list(path_list)
                    except Exception:
                        renderer._stage_paths.destroy_path_list(path_list)
                        raise
                else:
                    path_list = None
                    address = renderer._renderer.bind_attribute(
                        prim_paths=shader_paths,
                        attribute_name=attribute_name,
                        dtype=dtype,
                        shape=shape,
                        prim_mode=PrimMode.EXISTING_ONLY,
                        flags=BindingFlag.OPTIMIZE,
                    )
                self._addresses.append((channel, address, path_list, attribute_name, rows))
        except Exception:
            self._release_backend_addresses(renderer)
            raise

    def __call__(self, material_offsets: dict[str, Any] | None = None, env_ids: Any | None = None) -> None:
        """Mark channels dirty; OVRTX currently copies each dirty channel's full device buffer."""
        del env_ids
        selected = self._buffers if material_offsets is None else material_offsets
        self._dirty_channels.update(selected)
        wp.record_event(self._event)

    def publish(self) -> None:
        """Submit dirty buffers to OVRTX after ordering their producer streams."""
        channels = self._dirty_channels
        if not channels:
            return
        renderer = self._renderer_ref()
        operations = []
        try:
            for channel, address, _path_list, attribute_name, rows in self._addresses:
                if channel not in channels:
                    continue
                if renderer._use_ovstage:
                    operation = renderer._stage.write_attribute(
                        address,
                        attribute_name,
                        ordinal=renderer._current_ordinal,
                        tensors=self._buffers[channel][rows],
                        is_array=False,
                        cuda_event=self._event.cuda_event,
                    )
                else:
                    operation = address.write_async(
                        self._buffers[channel][rows],
                        data_access=DataAccess.ASYNC,
                        cuda_event=self._event.cuda_event,
                    )
                operations.append(operation)
        finally:
            self._operations = tuple(operations)
        channels.clear()

    def drain(self) -> None:
        """Complete submitted writes before their scene buffers may change."""
        operations, self._operations = self._operations, ()
        for operation in operations:
            operation.wait()

    def _release_backend_addresses(self, renderer: OVRTXRenderer) -> None:
        for _channel, address, path_list, _attribute_name, _rows in self._addresses:
            if renderer._use_ovstage:
                renderer._stage.release_query(address).wait()
                renderer._stage_paths.destroy_path_list(path_list)
            else:
                address.unbind()
        self._addresses.clear()

    def close(self) -> None:
        """Drain writes and release every compiled backend address."""
        try:
            self.drain()
        finally:
            renderer = self._renderer_ref()
            if renderer is not None:
                self._release_backend_addresses(renderer)
            self._dirty_channels.clear()
            self._buffers.clear()
