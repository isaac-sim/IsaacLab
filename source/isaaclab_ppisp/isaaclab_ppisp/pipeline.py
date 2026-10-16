# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP numerical execution with owned controller workspace and optional shared weights."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .cfg import PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN, PpispCfg, normalize_ppisp_cfg
from .kernels import (
    PPISP_CONTROLLER_FEATURE_LEN,
    PPISP_CONTROLLER_PARAM_COUNT,
    apply_ppisp_to_rgba,
    apply_ppisp_to_rgba_with_controller_params,
    compute_ppisp_controller_params,
)


@dataclass(frozen=True, eq=False, slots=True)
class PpispControllerWeights:
    """Shared, read-only device weights with identity preserved when configurations are copied.

    The caller supplies the allocation and orders its upload before processing. The wrapper retains
    that allocation for every consumer; neither processors nor configuration copies mutate it.
    """

    values: wp.array

    def __post_init__(self) -> None:
        if self.values.shape != (PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN,) or self.values.dtype != wp.float32:
            raise ValueError("PPISP controller weights require the expected one-dimensional float32 buffer.")

    def __deepcopy__(self, memo: dict) -> PpispControllerWeights:
        # Config cloning must share model resources, while each processor creates its own workspace.
        return self


class PpispPipeline:
    """Apply concrete PPISP settings to explicitly supplied image buffers.

    This numerical object performs no camera, scene, or renderer discovery. Resolve USD settings
    with :func:`~isaaclab_ppisp.ppisp_cfg_from_usd_camera` before construction. Each instance owns
    its controller workspace and is not reentrant; coefficients apply to its whole image batch.
    :class:`~isaaclab_ppisp.PpispProcessor` owns an independent instance for each consumer.

    Args:
        cfg: Concrete PPISP settings, normalized in place for direct numerical use.
        controller_weights: Optional device-resident controller weights to borrow read-only.
            Callers may share these across instances, but must keep them unchanged and order their
            upload before execution. Workspace and outputs are never shared automatically.
    """

    def __init__(self, cfg: PpispCfg, *, controller_weights: PpispControllerWeights | None = None):
        normalized_cfg = normalize_ppisp_cfg(cfg)
        if normalized_cfg is None:
            raise ValueError("PpispPipeline requires a concrete PpispCfg.")
        self.cfg = normalized_cfg
        self._shared_controller_weights = controller_weights
        self._controller_weights_by_device: dict[str, wp.array] = {}
        self._controller_buffers_by_shape: dict[tuple[str, int, int, int], tuple[wp.array, ...]] = {}
        self._closed = False

    def initialize(self, shape: tuple[int, ...], device: str | wp.Device) -> None:
        """Prepare controller weights and scratch before the first image is processed."""
        if self._closed:
            raise RuntimeError("Cannot initialize a closed PPISP pipeline.")
        if self.cfg.controller_weights is None and self._shared_controller_weights is None:
            return
        if not wp.get_device(device).is_cuda:
            raise ValueError("PPISP controller requires a CUDA device.")
        device_name = str(wp.get_device(device))
        if device_name not in self._controller_weights_by_device:
            if self._shared_controller_weights is None:
                if len(self.cfg.controller_weights) != PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN:
                    raise ValueError(f"PPISP controller requires {PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN} weights.")
                weights = wp.array(self.cfg.controller_weights, dtype=wp.float32, device=device_name)
            else:
                weights = self._shared_controller_weights.values
            if weights.device != wp.get_device(device_name):
                raise ValueError("Shared PPISP controller weights must match the image device.")
            self._controller_weights_by_device[device_name] = weights
        self._controller_buffers(shape, device_name)

    def apply(self, hdr: wp.array, rgba: wp.array) -> None:
        """Write RGB or RGBA to the supplied destination on the current Warp stream."""
        if self._closed:
            raise RuntimeError("Cannot apply a closed PPISP pipeline.")
        if self.cfg.controller_weights is None and self._shared_controller_weights is None:
            apply_ppisp_to_rgba(hdr, rgba, self.cfg)
            return
        self.initialize(hdr.shape, hdr.device)
        device = str(hdr.device)
        features, controller_params = self._controller_buffers(hdr.shape, device)
        compute_ppisp_controller_params(
            hdr,
            self._controller_weights_by_device[device],
            features,
            controller_params,
            self.cfg.controller_prior_exposure,
            float(self.cfg.controller_responsivity),
        )
        apply_ppisp_to_rgba_with_controller_params(hdr, rgba, self.cfg, controller_params)

    def close(self) -> None:
        """Release references to owned workspace and borrowed weights. Repeated calls are safe."""
        self._controller_weights_by_device.clear()
        self._controller_buffers_by_shape.clear()
        self._shared_controller_weights = None
        self._closed = True

    def _controller_buffers(self, shape: tuple[int, ...], device: str) -> tuple[wp.array, ...]:
        key = (device, *shape[:3])
        buffers = self._controller_buffers_by_shape.get(key)
        if buffers is None:
            buffers = (
                wp.empty((shape[0], PPISP_CONTROLLER_FEATURE_LEN), dtype=wp.float32, device=device),
                wp.empty((shape[0], PPISP_CONTROLLER_PARAM_COUNT), dtype=wp.float32, device=device),
            )
            self._controller_buffers_by_shape[key] = buffers
        return buffers
