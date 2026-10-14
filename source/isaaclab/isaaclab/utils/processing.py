# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named buffer processing shared by sensors and observations."""

from __future__ import annotations

from builtins import ExceptionGroup
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass
from types import MappingProxyType
from typing import Any

import torch
import warp as wp

from .configclass import configclass
from .string import string_to_callable
from .warp import ProxyArray


@dataclass(frozen=True)
class PreparedBufferOperation:
    """One operation with persistent output buffers and resolved resources.

    ``apply`` receives input and destination dictionaries keyed by operation roles, not sensor
    output names. Preparation allocates the outputs once. Inputs are borrowed read-only by
    convention; outputs must use independent storage. The operation must fill each supplied
    destination and enqueue GPU work on the current Warp stream. The pipe retains the outputs
    and controls execution order; private workspace belongs to the operation.
    """

    outputs: Mapping[str, ProxyArray]
    apply: Callable[[Mapping[str, ProxyArray], Mapping[str, ProxyArray]], None]
    reset: Callable[[Sequence[int] | None], None] | None = None
    close: Callable[[], None] | None = None


@configclass
class ProcessingBufferStepCfg:
    """Bind named buffers to the input/output roles of one processing operation.

    The factory ``prepare(inputs, **params)`` runs once during pipe construction and returns
    a :class:`PreparedBufferOperation` with allocated output buffers. ``inputs`` contains actual
    :class:`ProxyArray` objects keyed by operation roles. Factories inspect array shape, dtype
    and device directly; each operation documents its expected layout and value semantics.
    Factories should not read input values during preparation or retain input pointers for execution:
    ``apply`` receives the current inputs on every call. Scene resources must be supplied explicitly.
    """

    prepare: Callable[..., PreparedBufferOperation] | str = MISSING
    """Operation factory or an import string in ``module:attribute`` form."""

    inputs: dict[str, str] = MISSING
    """Map operation input roles to buffer names read from the preceding stage."""

    outputs: dict[str, str] = MISSING
    """Map operation output roles to names created or replaced in the bundle."""

    params: dict[str, Any] = {}
    """Resolved operation parameters passed to the factory.

    Configurations are deep-copied by ``configclass`` and managers. Borrowed device allocations
    or models need a resource wrapper that preserves identity through ``__deepcopy__``, or a
    factory closure retaining the resource. Do not put a raw Warp allocation in this dictionary.
    """


class ProcessingBufferPipe:
    """Apply an ordered sequence of operations to named :class:`ProxyArray` buffers.

    Buffers may hold images, LiDAR points, or other sensor and observation data. The pipe imposes
    no image layout; each operation defines its input requirements and allocates its outputs.

    A pipe retains persistent output storage allocated by its prepared operations. Create one instance per
    consumer; instances are not reentrant. Borrowed inputs are never modified by the pipe, and
    untouched channels pass through by reference. Returned buffers remain valid until the next
    call, reset or close; consumers retaining an observation must take a snapshot.

    Buffer dependencies are resolved in order, so an operation reads preceding replacements.
    GPU execution uses the caller's current Warp stream without a device-wide synchronization.
    The caller owns producer readiness and keeps borrowed storage alive until consumption ends.

    Args:
        source_buffers: Available external buffers, used to prepare operations and validate later inputs.
        steps: Ordered operation configurations.
    """

    def __init__(self, source_buffers: Mapping[str, ProxyArray], steps: Sequence[ProcessingBufferStepCfg]):
        self._source_buffers = dict(source_buffers)
        self.required_inputs = self.required_source_names(steps)
        self._operations: list[
            tuple[PreparedBufferOperation, dict[str, str], dict[str, str], Mapping[str, ProxyArray]]
        ] = []
        self._closed = False
        self.outputs: Mapping[str, ProxyArray] = MappingProxyType({})
        available = dict(source_buffers)
        produced: set[str] = set()
        try:
            for step in steps:
                inputs, outputs = dict(step.inputs), dict(step.outputs)
                if not outputs or len(set(outputs.values())) != len(outputs):
                    raise ValueError("A processing step must bind distinct output names to its output roles.")
                missing = set(inputs.values()) - available.keys()
                if missing:
                    raise ValueError(f"Missing input buffers for preparation: {sorted(missing)}.")
                prepare = string_to_callable(step.prepare) if isinstance(step.prepare, str) else step.prepare
                input_buffers = {role: available[name] for role, name in inputs.items()}
                operation = prepare(MappingProxyType(input_buffers), **step.params)
                if not isinstance(operation, PreparedBufferOperation):
                    raise TypeError("A buffer operation factory must return PreparedBufferOperation.")
                # Own the operation before checking its outputs so failed validation releases its resources.
                output_buffers = {}
                destinations = MappingProxyType(output_buffers)
                self._operations.append((operation, inputs, outputs, destinations))
                output_buffers.update(operation.outputs)
                if destinations.keys() != outputs.keys():
                    raise ValueError("Operation output roles must match the step's output bindings.")
                if any(not isinstance(buffer, ProxyArray) for buffer in destinations.values()):
                    raise TypeError("Prepared operation outputs must be ProxyArray objects.")
                devices = {str(buffer.device) for buffer in (*input_buffers.values(), *destinations.values())}
                if len(devices) > 1:
                    raise ValueError("A buffer operation must use one device for its inputs and outputs.")
                for role, name in outputs.items():
                    available[name] = destinations[role]
                produced.update(outputs.values())
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise
        self.outputs = MappingProxyType(available)
        self.produced_outputs = frozenset(produced)

    @property
    def has_reset_callbacks(self) -> bool:
        """Whether any operation needs selected batch entries reset."""
        return any(operation.reset is not None for operation, *_ in self._operations)

    @staticmethod
    def _validate_buffer(name: str, value: ProxyArray, reference: ProxyArray) -> None:
        """Check replacement storage against the actual buffer used to prepare the pipe."""
        if not isinstance(value, ProxyArray):
            raise TypeError(f"Buffer {name!r} must be a ProxyArray.")
        if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(
                f"Buffer {name!r} must match shape, dtype and device"
                f" {reference.shape}, {reference.dtype}, {reference.device}."
            )

    @staticmethod
    def required_source_names(
        steps: Sequence[ProcessingBufferStepCfg], published_names: Sequence[str] = ()
    ) -> frozenset[str]:
        """Resolve external names without constructing operations or allocating buffer storage.

        Call this before producer initialization to request private processing inputs. Public
        pass-through outputs also require a source, even when no operation reads them.
        """
        required: set[str] = set()
        produced: set[str] = set()
        for step in steps:
            required.update(set(step.inputs.values()) - produced)
            produced.update(step.outputs.values())
        required.update(set(published_names) - produced)
        return frozenset(required)

    def __call__(self, buffers: Mapping[str, ProxyArray]) -> dict[str, ProxyArray]:
        """Execute the pipe on ready source buffers and return the resulting bundle."""
        if self._closed:
            raise RuntimeError("Cannot execute a closed buffer pipe.")
        missing = self.required_inputs - buffers.keys()
        if missing:
            raise ValueError(f"Missing required buffers: {sorted(missing)}.")
        for name in self._source_buffers.keys() & buffers.keys():
            self._validate_buffer(name, buffers[name], self._source_buffers[name])
        current = dict(buffers)
        for operation, inputs, outputs, destinations in self._operations:
            operation.apply({role: current[name] for role, name in inputs.items()}, destinations)
            current.update({name: destinations[role] for role, name in outputs.items()})
        # Release replaced source references, such as the preceding observation's tensor.
        self._source_buffers.update({name: buffers[name] for name in self._source_buffers.keys() & buffers.keys()})
        self.outputs = MappingProxyType(dict(current))
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
        """Reset operation-owned state for selected batch entries."""
        for operation, *_ in self._operations:
            if operation.reset is not None:
                operation.reset(env_ids)

    def close(self) -> None:
        """Release operation resources; repeated calls have no effect."""
        if self._closed:
            return
        self._closed = True
        operations, self._operations = self._operations, []
        self._source_buffers.clear()
        self.outputs = MappingProxyType({})
        errors = []
        for operation, *_ in reversed(operations):
            if operation.close is not None:
                try:
                    operation.close()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise ExceptionGroup("Failed to close buffer operations.", errors)
