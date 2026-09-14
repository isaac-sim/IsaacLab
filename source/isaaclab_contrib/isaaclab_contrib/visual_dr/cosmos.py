# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-process tensor inference and residency; checkpoint-specific sampling is injected.

Install isaaclab_contrib[cosmos-runtime], then the patched Cosmos checkout with --no-deps.
The shared torch 2.10/FlashAttention build is not validated with IsaacLab's 2.11.
Mask-guidance sampler hooks and checkpoint aliases must accompany that checkout.
No file-oriented Cosmos inference API is used for live camera observations.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .runtime import DRBackend, DRFrame, DRObservation


@dataclass(frozen=True)
class CosmosDRCfg:
    """Model-worker settings; precision and compilation are capabilities to validate."""

    checkpoint: str
    prompt: str
    fp8: bool = False
    compile: bool = False
    device: str = "cuda:0"
    num_inference_steps: int = 6
    depth_range_m: tuple[float, float] = (0.1, 2.0)

    def __post_init__(self):
        near, far = self.depth_range_m
        device = torch.device(self.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("Cosmos runtime DR requires an explicit CUDA device, for example cuda:0")
        if self.num_inference_steps < 1 or not 0 < near < far or not math.isfinite(far):
            raise ValueError("Inference steps and finite increasing depth bounds must be positive")


@dataclass(frozen=True)
class CosmosInput:
    """Single-frame NCTHW signals: RGB [-1, 1], depth [m], boolean preserve mask.

    The injected model maps depth to its control representation, resizes all
    signals together, applies masked sampling, and restores this image grid.
    """

    rgb: torch.Tensor
    depth: torch.Tensor
    preserve: torch.Tensor
    prompt: str
    seed: int
    num_inference_steps: int

    @classmethod
    def prepare(cls, frame: DRFrame, observation: DRObservation, camera: str, cfg: CosmosDRCfg) -> CosmosInput:
        """Convert a checked frame without image downloads or global RNG changes."""
        near, far = cfg.depth_range_m
        rgb = frame.rgb.permute(0, 3, 1, 2).unsqueeze(2).float().div(127.5).sub(1)
        depth = frame.depth.permute(0, 3, 1, 2).unsqueeze(2)
        depth = torch.nan_to_num(depth, nan=far, posinf=far, neginf=near).clamp(near, far)
        preserve = frame.preserve.permute(0, 3, 1, 2).unsqueeze(2).contiguous()
        identity = f"{observation.seed}:{observation.episode}:{observation.sequence}:{camera}".encode()
        seed = int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "little") % (2**63)
        return cls(rgb.contiguous(), depth.contiguous(), preserve, cfg.prompt, seed, cfg.num_inference_steps)

    def decode(self, generated: torch.Tensor) -> torch.Tensor:
        """Check normalized model output and restore uint8 NHWC on the same device."""
        if (
            generated.shape != self.rgb.shape
            or generated.device != self.rgb.device
            or not generated.is_floating_point()
        ):
            raise ValueError("Cosmos must return floating NCTHW RGB on the input device and grid")
        if not torch.isfinite(generated).all():
            raise ValueError("Cosmos returned nonfinite RGB")
        rgb = generated.float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
        return rgb.squeeze(2).permute(0, 2, 3, 1).contiguous()


class CosmosBackend:
    """Own an nn.Module whose forward(CosmosInput) returns normalized NCTHW RGB.

    model_factory wraps the patched Cosmos sampler and owns all its model weights.
    It must not capture GPU weights outside the module or retain request tensors.
    The backend handles evaluation, optional forward compilation, paging and CUDA
    completion. FP8 requires a qualified custom backend; it is never silently ignored.
    """

    def __init__(self, cfg: CosmosDRCfg, model_factory: Callable[[CosmosDRCfg], torch.nn.Module]):
        if cfg.fp8:
            raise NotImplementedError("FP8 needs a qualified custom backend for the selected torch/torchao versions")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._factory = model_factory
        self._model: torch.nn.Module | None = None
        self._active = self._compiled = self._closed = self._faulted = False

    def activate(self) -> None:
        """Load once and restore residency; compilation is applied only once."""
        if self._closed or self._faulted:
            raise RuntimeError("Cosmos backend is closed or faulted")
        if self._active:
            return
        try:
            if self._model is None:
                self._model = self._factory(self.cfg).eval().requires_grad_(False)
            self._model.to(self.device)
            if self.cfg.compile and not self._compiled:
                self._model = torch.compile(self._model, dynamic=False)
                self._compiled = True
            self._active = True
        except Exception:
            self._faulted = True
            raise

    @torch.no_grad()
    def generate(self, frame: DRFrame, observation: DRObservation, camera: str) -> torch.Tensor:
        """Run synchronous tensor inference; the runtime performs final foreground compositing."""
        if not self._active or self._closed or self._faulted:
            raise RuntimeError("Activate Cosmos before generating observations")
        frame.validate()
        if frame.rgb.device != self.device:
            raise ValueError("In-process Cosmos must reside on the camera device")
        request = CosmosInput.prepare(frame, observation, camera, self.cfg)
        try:
            result = request.decode(self._model(request))
            torch.cuda.synchronize(self.device)
            return result
        except Exception:
            self._active = False
            self._faulted = True
            raise

    def offload(self) -> None:
        """Drain GPU work before moving weights to host memory for the learner phase."""
        if not self._active:
            return
        self._active = False
        try:
            torch.cuda.synchronize(self.device)
            self._model.to("cpu")
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        except Exception:
            self._faulted = True
            raise

    def close(self) -> None:
        """Release owned weights after completion; a failed drain remains retryable."""
        if self._closed:
            return
        self._active = False
        try:
            if self._model is not None:
                torch.cuda.synchronize(self.device)
                self._model = None
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
        except Exception:
            self._faulted = True
            raise
        self._closed = True


def create_cosmos_backend(
    cfg: CosmosDRCfg,
    factory: Callable[[CosmosDRCfg], DRBackend] | None = None,
    *,
    model_factory: Callable[[CosmosDRCfg], torch.nn.Module] | None = None,
) -> DRBackend:
    """Select an owned tensor model or a custom backend (for example a remote worker)."""
    if factory is not None and model_factory is not None:
        raise ValueError("Supply either a model_factory or a custom backend factory")
    if factory is not None:
        return factory(cfg)
    if model_factory is None:
        raise NotImplementedError(
            "Provide a tensor-native model_factory wrapping the patched Cosmos sampler, or a custom backend factory"
        )
    return CosmosBackend(cfg, model_factory)
