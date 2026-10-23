# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-only visual-material writes for OVRTX-owned scenes."""

from __future__ import annotations

import contextlib
import itertools
import logging
import weakref
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from ovrtx import BindingFlag, DataAccess, PrimMode

if TYPE_CHECKING:
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from .ovrtx_renderer import OVRTXRenderer

logger = logging.getLogger(__name__)


class OVRTXVisualMaterialWriter:
    """Compile OVRTX material addresses once and publish dirty device buffers."""

    def __init__(self, renderer: OVRTXRenderer, batches: tuple[VisualMaterialBatch, ...]):
        self._renderer_ref = weakref.ref(renderer)
        self._buffers: dict[str, torch.Tensor] = {}
        self._dirty_channels: set[str] = set()
        self._operations: tuple[Any, ...] = ()
        self._addresses: list[tuple[str, Any, str, slice]] = []

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
        self._device = str(device)
        self._event = wp.Event(device=self._device)
        with contextlib.ExitStack() as resources:
            for channel, attribute_name, shader_paths, rows, dtype, shape in groups:
                if renderer._use_ovstage:
                    address = resources.enter_context(renderer.scene.query(shader_paths))
                else:
                    address = renderer.backend.renderer.bind_attribute(
                        prim_paths=shader_paths,
                        attribute_name=attribute_name,
                        dtype=dtype,
                        shape=shape,
                        prim_mode=PrimMode.EXISTING_ONLY,
                        flags=BindingFlag.OPTIMIZE,
                    )
                    resources.callback(address.unbind)
                self._addresses.append((channel, address, attribute_name, rows))
            self._resources = resources.pop_all()

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
            for channel, address, attribute_name, rows in self._addresses:
                if channel not in channels:
                    continue
                if renderer._use_ovstage:
                    operation = renderer.scene.stage.write_attribute(
                        address,
                        attribute_name,
                        ordinal=renderer.scene.ordinal,
                        tensors=self._buffers[channel][rows],
                        is_array=False,
                        cuda_event=self._event.cuda_event,
                    )
                else:
                    # The event orders the read after the producers (access sync). The stream
                    # receives the done fence: work later enqueued on it waits for the read, so
                    # the next refill of these zero-copy buffers cannot race it. Fills and
                    # refills run on this device's current Warp stream.
                    operation = address.write_async(
                        self._buffers[channel][rows],
                        data_access=DataAccess.ASYNC,
                        cuda_event=self._event.cuda_event,
                        cuda_stream=wp.get_stream(self._device).cuda_stream or 1,
                    )
                operations.append(operation)
        finally:
            self._operations = tuple(operations)
        channels.clear()

    def drain(self) -> None:
        """Complete submitted writes before their scene buffers may change.

        Every op is waited even when an earlier one fails. Skipping the rest would leave their
        writes pending with no remaining reference to wait on.
        """
        operations, self._operations = self._operations, ()
        errors = []
        for operation in operations:
            try:
                operation.wait()
            except Exception as e:
                errors.append(e)
        if errors:
            raise RuntimeError(f"{len(errors)} OVRTX material write(s) failed to complete") from errors[0]

    def close(self) -> None:
        """Drain writes and release every compiled backend address."""
        try:
            with self._resources:
                try:
                    self.drain()
                finally:
                    renderer = self._renderer_ref()
                    if renderer is not None:
                        # A render may still read these bindings when RenderContext closes its writers.
                        for error in renderer.drain_pending_renders():
                            logger.warning("Error draining in-flight render before material release: %s", error)
        finally:
            self._addresses.clear()
            self._dirty_channels.clear()
            self._buffers.clear()
