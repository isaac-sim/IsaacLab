# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from builtins import ExceptionGroup
from collections.abc import Mapping, Sequence

import torch
import warp as wp

from ..configclass import instantiate
from ..warp import ProxyArray
from .dict_tensor_processor import DictTensorProcessor
from .dict_tensor_processor_cfg import DictTensorProcessorCfg


class _DictTensorProcessorPipeline:
    """Compose dictionary processors in order, preserving untouched input keys.

    Each processor receives current named tensors and returns its named outputs. Preparation
    constructs processors once; calls reuse their output storage. Each consumer owns its own
    pipeline. Input buffers are borrowed, and returned mappings are independent snapshots of
    references, not tensor copies. Use ``as_tensor`` when an observation must retain values.
    """

    def __init__(self, source_buffers: Mapping[str, ProxyArray], processors: Sequence[DictTensorProcessorCfg]):
        self._source_buffers = dict(source_buffers)
        self.required_inputs = self.required_source_names(processors)
        self._processors: list[tuple[DictTensorProcessor, dict[str, ProxyArray]]] = []
        self._closed = False
        self.outputs = dict(source_buffers)
        self.produced_outputs = frozenset(name for cfg in processors for name in cfg.outputs)
        try:
            for cfg in processors:
                if not cfg.outputs or len(set(cfg.outputs)) != len(cfg.outputs):
                    raise ValueError("A processor must declare distinct output names.")
                missing = set(cfg.inputs) - self.outputs.keys()
                if missing:
                    raise ValueError(f"Missing input buffers for preparation: {sorted(missing)}.")
                processor = instantiate(cfg, dict(self.outputs))
                if not isinstance(processor, DictTensorProcessor):
                    raise TypeError("Dictionary processors must inherit DictTensorProcessor.")
                self._processors.append((processor, dict(processor.outputs)))
                if set(processor.outputs) != set(cfg.outputs):
                    raise ValueError("Processor outputs must match its declared output names.")
                if any(not isinstance(buffer, ProxyArray) for buffer in processor.outputs.values()):
                    raise TypeError("Processor outputs must be ProxyArray objects.")
                devices = {str(self.outputs[name].device) for name in cfg.inputs}
                devices.update(str(buffer.device) for buffer in processor.outputs.values())
                if len(devices) > 1:
                    raise ValueError("A processor must use one device for its inputs and outputs.")
                self.outputs.update(processor.outputs)
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    @property
    def has_reset_callbacks(self) -> bool:
        """Whether any processor owns resettable state."""
        return any(type(processor).reset is not DictTensorProcessor.reset for processor, _ in self._processors)

    @staticmethod
    def _validate_buffer(name: str, value: ProxyArray, reference: ProxyArray) -> None:
        if not isinstance(value, ProxyArray):
            raise TypeError(f"Buffer {name!r} must be a ProxyArray.")
        if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(
                f"Buffer {name!r} must match shape, dtype and device"
                f" {reference.shape}, {reference.dtype}, {reference.device}."
            )

    @staticmethod
    def required_source_names(
        processors: Sequence[DictTensorProcessorCfg], published_names: Sequence[str] = ()
    ) -> frozenset[str]:
        """Resolve external keys before allocating buffers or initializing a renderer."""
        required: set[str] = set()
        produced: set[str] = set()
        for cfg in processors:
            required.update(set(cfg.inputs) - produced)
            produced.update(cfg.outputs)
        required.update(set(published_names) - produced)
        return frozenset(required)

    def __call__(self, data: dict[str, ProxyArray]) -> dict[str, ProxyArray]:
        """Apply processors and return their results with untouched inputs passed through."""
        if self._closed:
            raise RuntimeError("Cannot execute a closed processor pipeline.")
        missing = self.required_inputs - data.keys()
        if missing:
            raise ValueError(f"Missing required buffers: {sorted(missing)}.")
        for name in self._source_buffers.keys() & data.keys():
            self._validate_buffer(name, data[name], self._source_buffers[name])
        current = dict(data)
        for processor, outputs in self._processors:
            result = processor(dict(current))
            if result.keys() != outputs.keys() or any(result[name] is not output for name, output in outputs.items()):
                raise ValueError("Processors must return their persistent named output buffers.")
            current.update(result)
        self._source_buffers.update({name: data[name] for name in self._source_buffers.keys() & data.keys()})
        self.outputs = dict(current)
        return current

    def apply_to_tensor(
        self,
        data: torch.Tensor,
        *,
        input_name: str,
        output_name: str,
        auxiliary_buffers: Mapping[str, ProxyArray] | None = None,
    ) -> torch.Tensor:
        """Process an incoming observation tensor and return an independently owned snapshot.

        The supplied tensor is authoritative: earlier observation transformations are preserved.
        Auxiliary inputs must already be ready on the current Warp/Torch stream. CUDA work is
        ordered with the caller's Torch stream; no host/device transfer is performed.
        """
        if auxiliary_buffers is not None and input_name in auxiliary_buffers:
            raise ValueError("Auxiliary buffers must not replace the incoming observation tensor.")
        buffers = dict(auxiliary_buffers or {})
        buffers[input_name] = ProxyArray(wp.from_torch(data))
        return self.as_tensor(buffers, output_name=output_name)

    def as_tensor(
        self, buffers: Mapping[str, ProxyArray], *, output_name: str, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Execute on the current Torch stream and snapshot one output for an observation.

        Source buffers must be ready on the current Warp or Torch stream. Stream ordering
        is established without a device-wide synchronization; no host/device copy is made.
        A supplied ``out`` must match the prepared output's shape, dtype and device and is overwritten in place.
        """
        if self._closed:
            raise RuntimeError("Cannot execute a closed buffer pipe.")
        reference = self.outputs[output_name] if output_name in self.produced_outputs else buffers[output_name]
        if out is not None:
            self._validate_buffer("out", ProxyArray(wp.from_torch(out)), reference)
        device = wp.get_device(reference.device)
        stream = wp.stream_from_torch(torch.cuda.current_stream(str(device))) if device.is_cuda else None
        with wp.ScopedStream(stream, sync_enter=True, sync_exit=True):
            result = self(buffers)[output_name].torch
            return result.clone() if out is None else out.copy_(result)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset processor-owned state for selected batch entries."""
        for processor, _ in self._processors:
            processor.reset(env_ids)

    def close(self) -> None:
        """Close processors in reverse order, including after failed construction."""
        if self._closed:
            return
        self._closed = True
        processors, self._processors = self._processors, []
        self._source_buffers.clear()
        self.outputs = {}
        errors = []
        for processor, _ in reversed(processors):
            try:
                processor.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Failed to close dictionary processors.", errors)
