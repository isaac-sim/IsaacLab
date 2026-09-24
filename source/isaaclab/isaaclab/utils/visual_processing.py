# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composable pixel operations and persistent buffers for image observations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, dataclass, replace
from typing import Any

import warp as wp

from ..renderers.output_contract import RenderBufferSpec
from . import configclass
from .warp import ProxyArray


@dataclass(frozen=True)
class VisualProcessorContext:
    """Initialization context shared by processor factories, without a renderer reference."""

    stage: Any
    camera_prim_paths: tuple[str, ...]
    num_views: int
    height: int
    width: int
    device: str


@dataclass
class VisualProcessor:
    """Resolved buffer declarations and callbacks for one pixel operation.

    Factories may return closures or bound methods; no processor inheritance is required.
    ``initialize(inputs, outputs)`` binds persistent buffers once. Inputs are read-only
    unless also bound as an output. ``process(env_mask)``
    runs after rendering on the current Warp stream. Callbacks using another stream must
    order their work with that stream before returning. Buffers must never be replaced.
    ``reset(env_mask)`` resets selected environments, and ``close()`` releases owned state.
    Masks are device arrays of shape (N,) and dtype ``wp.bool``; do not retain them.
    Renderers currently refresh the entire image batch; processors may likewise transform
    the full batch while using the mask to advance only selected temporal state.

    Args:
        inputs: Required buffers and their contracts.
        outputs: Produced or modified buffers and their contracts.
        process: Apply the operation to bound buffers for a new camera frame.
        initialize: Bind buffers and allocate scratch storage before the first frame.
        reset: Reset state for the selected environments.
        close: Release processor resources; must be safe after partial initialization.
        neutral_exposure: Request scene-linear input without renderer exposure.
        in_place: Allow outputs to reuse matching input storage. The operation must support aliasing.
    """

    inputs: dict[str, RenderBufferSpec]
    outputs: dict[str, RenderBufferSpec]
    process: Callable[[wp.array], None]
    initialize: Callable[[dict[str, ProxyArray], dict[str, ProxyArray]], None] | None = None
    reset: Callable[[wp.array], None] | None = None
    close: Callable[[], None] | None = None
    neutral_exposure: bool = False
    in_place: bool = False


@configclass
class VisualProcessorCfg:
    """Configure a factory that creates independent state for each observation term.

    ``func(cfg, context)`` resolves configuration before renderer setup and returns a
    :class:`VisualProcessor`, or ``None`` to disable this operation. Static ``inputs``
    declare potential renderer requirements needed before simulation startup (e.g. HDR).
    The returned processor declares the actual inputs and outputs after discovery.
    """

    func: Callable[[VisualProcessorCfg, VisualProcessorContext], VisualProcessor | None] = MISSING
    inputs: dict[str, RenderBufferSpec] = {}
    outputs: dict[str, RenderBufferSpec] = {}
    params: dict[str, Any] = {}


class VisualProcessingPipeline:
    """Resolve an ordered processor chain and bind storage once per observation term.

    Resolution validates every input against the preceding output or renderer contract.
    Allocation keeps intermediate buffers private and preserves the camera's RGB/RGBA
    alias. There are no implicit dtype, layout, device, or color-space conversions.

    Args:
        configs: Ordered processor factory configurations.
        context: Camera dimensions, device, and stage for configuration discovery.
        renderer_specs: Buffers supported by the selected renderer.
        requested_outputs: Output names, including processor-produced names.
    """

    def __init__(
        self,
        configs: list[VisualProcessorCfg],
        context: VisualProcessorContext,
        renderer_specs: dict[str, RenderBufferSpec],
        requested_outputs: list[str],
    ):
        self.context = context
        self._processors: list[VisualProcessor] = []
        self._renderer_specs: dict[str, RenderBufferSpec] = {}
        self.render_outputs: dict[str, ProxyArray] = {}
        self._buffers: list[dict[str, ProxyArray]] = []
        self._outputs: dict[str, ProxyArray] | None = None
        self._requested_outputs = list(requested_outputs)
        available: dict[str, RenderBufferSpec] = {}
        try:
            for index, cfg in enumerate(configs):
                processor = cfg.func(cfg, context)
                if processor is None:
                    continue
                self._processors.append(processor)
                for name, required in processor.inputs.items():
                    self._validate_spec(name, required)
                    actual = available.get(name)
                    if actual is None:
                        actual = renderer_specs.get(name)
                        if actual is None:
                            raise ValueError(
                                f"Processor {index} requires unavailable input {name!r}. "
                                "Select a supporting renderer or place its producer earlier in the chain."
                            )
                        self._renderer_specs[name] = actual
                        available[name] = actual
                    self._validate_input(index, name, required, actual)
                for name, output in processor.outputs.items():
                    self._validate_spec(name, output)
                available.update(self._with_color_aliases(processor.outputs))
            for name in self._requested_outputs:
                if name not in available:
                    if name not in renderer_specs:
                        raise ValueError(f"Output {name!r} is not produced by the renderer or processor chain.")
                    self._renderer_specs[name] = renderer_specs[name]
                    available[name] = renderer_specs[name]
            color_names = {"rgb", "rgba"}
            if color_names <= renderer_specs.keys() and not color_names.isdisjoint(self._renderer_specs):
                self._renderer_specs.update({name: renderer_specs[name] for name in color_names})
                self._renderer_specs = self._with_color_aliases(self._renderer_specs)
            for name, spec in self._renderer_specs.items():
                self._validate_spec(name, spec)
        except Exception as exc:
            try:
                self.close()
            finally:
                raise exc

    @property
    def render_data_types(self) -> tuple[str, ...]:
        """Private and public inputs that must be supplied by the renderer."""
        return tuple(self._renderer_specs)

    @property
    def neutral_exposure(self) -> bool:
        """Whether any active processor requires neutral renderer exposure."""
        return any(processor.neutral_exposure for processor in self._processors)

    def allocate(self, render_outputs: dict[str, ProxyArray] | None = None) -> dict[str, ProxyArray]:
        """Bind inputs and allocate persistent intermediates and requested outputs.

        Args:
            render_outputs: Existing camera buffers. These are borrowed read-only, so
                even in-place processors cannot alter inputs shared by other terms.
                If omitted, allocate owned renderer inputs for the legacy ISP adapter.
        """
        if self._outputs is not None:
            return self._outputs
        try:
            if render_outputs is None:
                self.render_outputs = self._allocate_buffers(self._renderer_specs)
                borrowed_pointers = set()
            else:
                self.render_outputs = {}
                for name, spec in self._renderer_specs.items():
                    buffer = render_outputs.get(name)
                    shape = (self.context.num_views, self.context.height, self.context.width, spec.channels)
                    if (
                        buffer is None
                        or buffer.warp.shape != shape
                        or buffer.warp.dtype != spec.dtype
                        or buffer.warp.device != wp.get_device(self.context.device)
                    ):
                        raise ValueError(f"Renderer input {name!r} does not match the resolved buffer contract {spec}.")
                    self.render_outputs[name] = buffer
                borrowed_pointers = {buffer.warp.ptr for buffer in self.render_outputs.values()}
            current = dict(self.render_outputs)
            for processor in self._processors:
                inputs = {name: current[name] for name in processor.inputs}
                specs = self._with_color_aliases(processor.outputs)
                outputs: dict[str, ProxyArray] = {}
                for name, spec in specs.items():
                    buffer = current.get(name)
                    if processor.in_place and buffer is not None and buffer.warp.ptr not in borrowed_pointers:
                        if buffer.warp.dtype == spec.dtype and buffer.warp.shape[-1] == spec.channels:
                            outputs[name] = buffer
                # Allocate RGB and RGBA together to retain their alias even when only one is declared.
                if not {"rgb", "rgba"}.isdisjoint(specs) and not {"rgb", "rgba"} <= outputs.keys():
                    outputs.pop("rgb", None)
                    outputs.pop("rgba", None)
                outputs.update(self._allocate_buffers({k: v for k, v in specs.items() if k not in outputs}))
                if processor.initialize is not None:
                    processor.initialize(inputs, outputs)
                current.update(outputs)
            names = set(self._requested_outputs)
            if not names.isdisjoint({"rgb", "rgba"}):
                names.update({"rgb", "rgba"})
            self._outputs = {name: current[name] for name in current if name in names}
            return self._outputs
        except Exception as exc:
            try:
                self.close()
            finally:
                raise exc

    def process(self, env_mask: wp.array) -> None:
        """Apply each operation once after a fresh render, on the current Warp stream."""
        for processor in self._processors:
            processor.process(env_mask)

    def reset(self, env_mask: wp.array) -> None:
        """Reset state only for environments selected by the boolean device mask."""
        for processor in self._processors:
            if processor.reset is not None:
                processor.reset(env_mask)

    def close(self) -> None:
        """Release processor state in reverse order. Repeated calls are safe."""
        processors, self._processors = self._processors, []
        error = None
        for processor in reversed(processors):
            if processor.close is not None:
                try:
                    processor.close()
                except Exception as exc:
                    if error is None:
                        error = exc
        self.render_outputs = {}
        self._outputs = None
        self._buffers.clear()
        if error is not None:
            raise RuntimeError("Failed to close a visual processor.") from error

    def _validate_spec(self, name: str, spec: RenderBufferSpec) -> None:
        if spec.layout != "NHWC":
            raise ValueError(f"Buffer {name!r} requires unsupported layout {spec.layout!r}; expected 'NHWC'.")
        if spec.device is not None and wp.get_device(spec.device) != wp.get_device(self.context.device):
            raise ValueError(
                f"Buffer {name!r} requires device {spec.device!r}, but the camera uses {self.context.device!r}."
            )
        if spec.channels < 1:
            raise ValueError(f"Buffer {name!r} must declare at least one channel.")
        if name in {"rgb", "rgba"} and (spec.dtype != wp.uint8 or spec.channels != (3 if name == "rgb" else 4)):
            raise ValueError(f"Buffer {name!r} must preserve the camera's uint8 RGB/RGBA output layout.")

    def _validate_input(self, index: int, name: str, required: RenderBufferSpec, actual: RenderBufferSpec) -> None:
        if (
            required.channels != actual.channels
            or required.dtype != actual.dtype
            or required.layout != actual.layout
            or (required.color_space is not None and required.color_space != actual.color_space)
        ):
            raise ValueError(
                f"Processor {index} input {name!r} has an incompatible buffer contract: "
                f"requires {required}, received {actual}. Check processor ordering and renderer support."
            )

    @staticmethod
    def _with_color_aliases(specs: dict[str, RenderBufferSpec]) -> dict[str, RenderBufferSpec]:
        result = dict(specs)
        if "rgba" in result:
            result.setdefault("rgb", replace(result["rgba"], channels=3))
        elif "rgb" in result:
            result["rgba"] = replace(result["rgb"], channels=4)
        if "rgb" in result and replace(result["rgb"], channels=4) != result["rgba"]:
            raise ValueError("RGB and RGBA must declare matching dtype, layout, device, and color space for aliasing.")
        return result

    def _allocate_buffers(self, specs: dict[str, RenderBufferSpec]) -> dict[str, ProxyArray]:
        context = self.context
        buffers = {}
        for name, spec in specs.items():
            if name == "rgb" and "rgba" in specs:
                continue
            buffers[name] = ProxyArray(
                wp.zeros(
                    (context.num_views, context.height, context.width, spec.channels),
                    dtype=spec.dtype,
                    device=context.device,
                )
            )
        if "rgb" in specs and "rgba" in buffers:
            rgba = buffers["rgba"].warp
            buffers["rgb"] = ProxyArray(
                wp.array(
                    ptr=rgba.ptr,
                    shape=(*rgba.shape[:3], 3),
                    strides=rgba.strides,
                    dtype=rgba.dtype,
                    device=rgba.device,
                    copy=False,
                )
            )
        # Raw-pointer RGB aliases do not own their RGBA allocation. Retain every
        # stage's storage even when a later stage replaces those output names.
        self._buffers.append(buffers)
        return buffers
