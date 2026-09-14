# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synchronous, single-environment prototype; lifecycle belongs to the application."""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class DRObservation:
    """Application identity; sequence increases across steps AND resets.

    Set consumed for policy, history, and value-bootstrap reads, not physics ticks.
    A caller executing K actions can mark only its actual decision boundaries.
    """

    sequence: int
    episode: int
    seed: int
    consumed: bool = True


@dataclass(frozen=True)
class DRFrame:
    """Aligned NHWC CUDA tensors: uint8 RGB, float32 depth [m], bool preserve mask."""

    rgb: torch.Tensor
    depth: torch.Tensor
    preserve: torch.Tensor

    def validate(self) -> None:
        """Reject unsupported layouts and devices before calling a backend."""
        if self.rgb.ndim != 4 or self.rgb.shape[0] != 1 or self.rgb.shape[-1] != 3:
            raise ValueError("The first DR sketch supports one environment, NHWC RGB only")
        shape = (*self.rgb.shape[:-1], 1)
        for value, dtype, expected in (
            (self.rgb, torch.uint8, self.rgb.shape),
            (self.depth, torch.float32, shape),
            (self.preserve, torch.bool, shape),
        ):
            if not value.is_cuda or value.device != self.rgb.device:
                raise ValueError("DR payloads must share a CUDA device; no CPU transport fallback")
            if value.dtype != dtype or value.shape != expected:
                raise ValueError("DR requires aligned uint8 RGB, float32 depth and bool mask")


class DRBackend(Protocol):
    """Worker-owned model interface. Methods finish their GPU work before returning."""

    def activate(self) -> None:
        """Acquire model residency before collection."""
        ...

    def generate(self, frame: DRFrame, observation: DRObservation, camera: str) -> torch.Tensor:
        """Return NHWC uint8 RGB on the input device; never stage images on the CPU."""
        ...

    def offload(self) -> None:
        """Drain inference AND transfers, then release model residency."""
        ...

    def close(self) -> None:
        """Drain and release resources owned by this backend."""
        ...


class VisualDRRuntime:
    """Cache camera results within an explicitly declared observation scope.

    No automatic wakeup, scheduling thread, reset hook, or trainer dependency.
    The first version is single-threaded and requires explicit reset scopes.
    """

    def __init__(self, backend: DRBackend):
        self.backend = backend
        self.observation: DRObservation | None = None
        self._cache: dict[str, torch.Tensor] = {}
        self._active = False
        self._closed = False
        self._needs_scope = True

    def activate(self) -> None:
        """Let the application hand GPU memory to collection."""
        if self._closed:
            raise RuntimeError("DR runtime is closed")
        if not self._active:
            self.backend.activate()
            self._active = True

    def begin(self, observation: DRObservation) -> None:
        """Declare the next scope before computing observations, including reset."""
        if self._closed:
            raise RuntimeError("DR runtime is closed")
        if observation == self.observation:
            return
        if self.observation is not None and observation.sequence <= self.observation.sequence:
            raise ValueError("Use a new increasing sequence for every step/reset observation")
        self.observation = observation
        self._needs_scope = False
        self._cache.clear()

    @torch.no_grad()
    def process(self, camera: str, frame: DRFrame) -> torch.Tensor:
        """Generate once per camera/scope and preserve foreground exactly."""
        if self._closed or self.observation is None or self._needs_scope:
            raise RuntimeError("Call begin() on an open runtime before reading DR observations")
        if not self.observation.consumed:
            return frame.rgb.clone()
        if not self._active:
            raise RuntimeError("Activate DR before collection; reads cannot wake an offloaded model")
        if camera not in self._cache:
            frame.validate()
            # Isolate renderer buffers and the final preservation mask from backend mutation.
            owned = DRFrame(frame.rgb.clone(), frame.depth.clone(), frame.preserve.clone())
            generated = self.backend.generate(owned, self.observation, camera)
            if (generated.shape, generated.dtype, generated.device) != (
                frame.rgb.shape,
                frame.rgb.dtype,
                frame.rgb.device,
            ):
                raise ValueError("Backend changed RGB shape, dtype or device")
            self._cache[camera] = torch.where(frame.preserve, frame.rgb, generated)
        return self._cache[camera].clone()

    def offload(self) -> None:
        """Close admission before the backend drains; failures leave admission closed."""
        if self._active:
            self._active = False
            self._needs_scope = True
            self._cache.clear()
            self.backend.offload()

    def close(self) -> None:
        """Release this runtime once; callers own shared-worker shutdown separately."""
        if not self._closed:
            self._closed = True
            self._active = False
            self._cache.clear()
            self.backend.close()
