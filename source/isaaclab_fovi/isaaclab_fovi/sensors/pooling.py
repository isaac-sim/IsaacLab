# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared retinal-pooling helpers for FoveatedCamera backends.

Builds a pair of (lowres, highres) :class:`SamplingCoords` and a retinal
pooler analogous to :class:`fovi.sensing.samplers.PoolingGridSampler`, but driven by a
:class:`FoveatedCameraCfg` and operating on rays rather than images.

Key deviation from ``PoolingGridSampler``: the highres grid shares ``cmf_a`` with
the lowres grid (no ``cmf_a_mult``).  Only ``res`` is scaled by
``pool_res_mult`` so the photoreceptor and RGC layers represent the same
visual-field geometry at different densities.

Construction happens in two stages to accommodate Isaac's sensor init order:
1. ``build_lowres_coords`` + ``build_highres_coords`` run *before*
   ``super()._initialize_impl()`` (CPU) so the backend can size its render
   buffers from the highres sample count.
2. ``build_pooler`` runs *after* coords have been moved to the sensor's
   device, so the KNN indices/distances/Gaussian weights materialise on the
   correct device (they are plain tensor attributes and are not moved by
   ``nn.Module.to``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .calibrated_geometry import sampling_geometry

if TYPE_CHECKING:
    from fovi.arch.knn import KNNPoolingLayer
    from fovi.arch.pooling import GridPoolingLayer
    from fovi.sensing.coords import SamplingCoords

    from .foveated_camera_cfg import FoveatedCameraCfg


def build_lowres_coords(cfg: FoveatedCameraCfg, device: str) -> tuple[SamplingCoords, int]:
    """Build the lowres (RGC / public output) SamplingCoords.

    Returns ``(lowres_coords, matched_res)`` where ``matched_res`` is the
    auto-matched radial resolution also needed for the highres grid.
    """
    from fovi.sensing.coords import SamplingCoords, auto_match_num_coords

    fov, cmf_a = sampling_geometry(cfg)

    matched_res, _ = auto_match_num_coords(
        fov,
        cmf_a,
        cfg.foveated_res,
        cfg.foveated_style,
        1,
        device,
        quiet=True,
        fov_type=cfg.fov_type,
        radius_norm=cfg.radius_norm,
        field_geometry=cfg.field_geometry,
    )
    lowres = SamplingCoords(
        fov=fov,
        cmf_a=cmf_a,
        res=matched_res,
        device=device,
        style=cfg.foveated_style,
        isotropic_plotting_type=cfg.isotropic_plotting_type,
        fov_type=cfg.fov_type,
        radius_norm=cfg.radius_norm,
        field_geometry=cfg.field_geometry,
    )
    return lowres, matched_res


def build_highres_coords(cfg: FoveatedCameraCfg, matched_res: int, device: str) -> SamplingCoords | None:
    """Build the highres (photoreceptor) SamplingCoords when pooling is on."""
    if not cfg.pool_enabled:
        return None
    from fovi.sensing.coords import SamplingCoords

    fov, cmf_a = sampling_geometry(cfg)

    # pool_res_mult is a float to support fractional ratios (e.g. sqrt(2) for
    # a 2× N_hi/N_lo ratio).  SamplingCoords requires an integer radial res.
    highres_res = int(round(cfg.pool_res_mult * matched_res))
    return SamplingCoords(
        fov=fov,
        cmf_a=cmf_a,
        res=highres_res,
        device=device,
        style=cfg.foveated_style,
        isotropic_plotting_type=cfg.isotropic_plotting_type,
        fov_type=cfg.fov_type,
        radius_norm=cfg.radius_norm,
        field_geometry=cfg.field_geometry,
    )


def build_pooler(
    cfg: FoveatedCameraCfg,
    lowres: SamplingCoords,
    highres: SamplingCoords | None,
    device: str,
    mode: str | None = None,
) -> tuple[KNNPoolingLayer | GridPoolingLayer | None, int | None]:
    """Build a retinal pooling layer. Both coord sets must already be on ``device``.

    If ``mode`` is None, uses ``cfg.pool_mode`` — the standard pooler for
    floating-point outputs.  Pass ``mode="max"`` (or similar) to build a
    separate pooler for integer segmentation outputs.
    """
    if highres is None:
        return None, None

    from fovi.arch.pooling import build_retinal_pooler

    return build_retinal_pooler(
        highres,
        lowres,
        k=cfg.pool_k,
        mode=mode if mode is not None else cfg.pool_mode,
        device=device,
        sample_cortex=cfg.pool_sample_cortex,
        gauss_sigma=cfg.pool_gauss_sigma,
    )


def is_segmentation_output(name: str) -> bool:
    """Whether a sensor output key should be pooled with ``pool_int_mode``.

    Uses a key-name heuristic (``"segmentation" in name``) rather than dtype,
    since all IsaacLab segmentation variants — ``semantic_segmentation``,
    ``instance_segmentation_fast``, ``instance_id_segmentation_fast`` —
    produce class-like values whose per-pixel semantics are broken by
    averaging, regardless of whether they're stored as int32 class IDs or
    uint8 colorized RGBA.
    """
    return "segmentation" in name


def pool_hwc_buffer(pooler: KNNPoolingLayer | GridPoolingLayer, buf: torch.Tensor) -> torch.Tensor:
    """Pool a ``(B, N_hi, 1, C)`` sensor buffer into ``(B, N_lo, 1, C)``.

    Both poolers expect ``(B, C, N)`` inputs, so we permute in and out.
    """
    B, N_hi, one, C = buf.shape
    assert one == 1, f"expected (B, N, 1, C) buffer, got {buf.shape}"
    x = buf[:, :, 0, :].transpose(1, 2).contiguous()  # (B, C, N_hi)
    y = pooler(x)  # (B, C, N_lo)
    return y.transpose(1, 2).unsqueeze(2).contiguous()  # (B, N_lo, 1, C)
