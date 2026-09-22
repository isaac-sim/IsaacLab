# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The contract between the runtime and whatever generates pixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from .cfg import DRBackendCfg


@dataclass(frozen=True)
class DRFrame:
    """Aligned per-camera signals for a batch of environments.

    NHWC throughout, all on one CUDA device: uint8 ``rgb``, float32 ``depth`` in
    metres, bool ``preserve`` marking pixels that must survive untouched.
    """

    rgb: torch.Tensor
    depth: torch.Tensor
    preserve: torch.Tensor
    segmentation: torch.Tensor | None = None
    """Raw semantic IDs, needed only by backends using a segmentation control hint."""

    @property
    def num_envs(self) -> int:
        return self.rgb.shape[0]

    def validate(self) -> None:
        """Reject layouts and devices a backend cannot consume."""
        if self.rgb.ndim != 4 or self.rgb.shape[-1] != 3:
            raise ValueError(f"Visual DR needs NHWC uint8 RGB, got {tuple(self.rgb.shape)}")
        single = (*self.rgb.shape[:-1], 1)
        checks = [
            ("rgb", self.rgb, torch.uint8, self.rgb.shape),
            ("depth", self.depth, torch.float32, single),
            ("preserve", self.preserve, torch.bool, single),
        ]
        if self.segmentation is not None:
            checks.append(("segmentation", self.segmentation, self.segmentation.dtype, single))
        for name, value, dtype, expected in checks:
            if not value.is_cuda or value.device != self.rgb.device:
                raise ValueError(f"Visual DR payload '{name}' must share the RGB CUDA device; no CPU staging")
            if value.dtype != dtype or value.shape != expected:
                raise ValueError(
                    f"Visual DR payload '{name}' must be {dtype} of shape {tuple(expected)}, "
                    f"got {value.dtype} of shape {tuple(value.shape)}"
                )

    def index(self, indices: torch.Tensor) -> DRFrame:
        """Narrow to a subset of environments without copying pixel data."""
        segmentation = None if self.segmentation is None else self.segmentation[indices]
        return DRFrame(self.rgb[indices], self.depth[indices], self.preserve[indices], segmentation)


@dataclass(frozen=True)
class DRRequest:
    """Per-sample generation context, aligned with a :class:`DRFrame`'s batch.

    ``seeds`` and ``prompts`` are drawn by the runtime from the episode identity,
    so the same environment on the same episode gets the same background no
    matter which replica serves it or how often the frame is read.
    """

    seeds: torch.Tensor
    prompts: tuple[str, ...]
    camera: str

    def __post_init__(self):
        if self.seeds.numel() != len(self.prompts):
            raise ValueError("DRRequest needs one prompt per seed")


@runtime_checkable
class DRBackend(Protocol):
    """Owns model weights and their GPU residency. Methods finish their GPU work."""

    def activate(self) -> None:
        """Acquire model residency before collection."""
        ...

    def generate(self, frame: DRFrame, request: DRRequest) -> torch.Tensor:
        """Return NHWC uint8 RGB on the input device, matching the input grid."""
        ...

    def offload(self) -> None:
        """Drain inference, then release residency so a learner can use the memory."""
        ...

    def close(self) -> None:
        """Drain and release everything this backend owns."""
        ...


class PassthroughBackend:
    """Returns frames unchanged. For wiring up a task before a model is available.

    Deliberately not a silent fallback: a task has to ask for this by name, so a
    misconfigured Cosmos backend can never masquerade as successful generation.
    """

    def __init__(self, cfg: DRBackendCfg):
        self.cfg = cfg

    def activate(self) -> None:
        pass

    def generate(self, frame: DRFrame, request: DRRequest) -> torch.Tensor:
        return frame.rgb.clone()

    def offload(self) -> None:
        pass

    def close(self) -> None:
        pass
