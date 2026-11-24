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
1. ``build_retinal_coords`` runs *before*
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
from fovi.arch.knn import KNNPoolingLayer
from fovi.arch.pooling import GridPoolingLayer, build_retinal_pooler
from fovi.sensing.coords import SamplingCoords, auto_match_num_coords
from fovi.sensing.projection import CameraModel

if TYPE_CHECKING:
    from .foveated_camera_cfg import FoveatedCameraCfg


# Upper bound on ``batch * outputs * k`` elements per modal-pooling chunk.  Each chunk holds about ten
# int64 tensors of this size, so peak scratch memory stays near 300 MB regardless of the batch size.
_MODAL_CHUNK_ELEMENTS = 2**22


def build_retinal_coords(
    cfg: FoveatedCameraCfg, camera: CameraModel, device: str
) -> tuple[SamplingCoords, SamplingCoords | None]:
    """Build the public retina and, when pooling, the denser retina with the same angular geometry."""
    fov = camera.field_of_view(cfg.retinal_window_reference_side, cfg.retinal_window_fraction)

    matched_res, _ = auto_match_num_coords(
        fov,
        cfg.cmf_a_deg,
        cfg.foveated_res,
        cfg.foveated_style,
        1,
        device,
        quiet=True,
        fov_type=cfg.fov_type,
        radius_norm=cfg.radius_norm,
        field_geometry="spherical",
    )
    lowres = SamplingCoords(
        fov=fov,
        cmf_a=cfg.cmf_a_deg,
        res=matched_res,
        device=device,
        style=cfg.foveated_style,
        isotropic_plotting_type=cfg.isotropic_plotting_type,
        fov_type=cfg.fov_type,
        radius_norm=cfg.radius_norm,
        field_geometry="spherical",
    )
    highres = None
    if cfg.pool_enabled:
        # Fractional ratios (e.g. sqrt(2) for twice as many samples) still need an integer radial resolution.
        highres = lowres.clone(resolution=int(round(cfg.pool_res_mult * lowres.resolution)))
    return lowres, highres


def build_pooler(
    cfg: FoveatedCameraCfg,
    lowres: SamplingCoords,
    highres: SamplingCoords | None,
    device: str,
) -> tuple[KNNPoolingLayer | GridPoolingLayer | None, int | None]:
    """Build the retinal pooling layer of ``cfg.pool_mode``, which pools every output except segmentation. Both coord
    sets must already be on ``device``.

    Segmentation pools by modal label over this pooler's neighbourhoods: pass :func:`segmentation_neighbor_indices` of
    it to :func:`pool_segmentation_buffer`.
    """
    if highres is None:
        return None, None
    return build_retinal_pooler(
        highres,
        lowres,
        k=cfg.pool_k,
        mode=cfg.pool_mode,
        device=device,
        sample_cortex=cfg.pool_sample_cortex,
        gauss_sigma=cfg.pool_gauss_sigma,
    )


def segmentation_neighbor_indices(pooler: KNNPoolingLayer | GridPoolingLayer) -> torch.Tensor:
    """Return a pooler's ``(k, N_lo)`` neighbour table in tie-break order.

    Entries are input-sample indices; ``N_hi`` (the number of input samples)
    marks a neighbour excluded by the pooler (spatial padding or a masked input
    sample), and every slot of a masked output sample is excluded.

    * :class:`KNNPoolingLayer`: its padded KNN indices, nearest neighbour first.
    * :class:`GridPoolingLayer`: the samples in each pooling window (same
      padding, stride and masks as the pooler), ordered by distance from the
      window centre and then in row-major window order.
    """
    if isinstance(pooler, KNNPoolingLayer):
        return pooler.knn_indices_pad_token
    if not isinstance(pooler, GridPoolingLayer):
        raise TypeError(f"Modal pooling needs a KNN or grid pooler, got {type(pooler).__name__}")

    num_inputs = len(pooler.in_coords)
    device = pooler.input_mask.device
    # Pool sample indices (float64 is exact for any realistic count) with the pooler's own padding;
    # -1 marks padding and masked samples.
    grid = pooler.in_coords.as_grid(torch.arange(num_inputs, device=device, dtype=torch.float64)[None, None])
    grid = pooler._pad(torch.where(pooler.input_mask, grid, -1.0), -1.0)
    side, stride = pooler.kernel_size, pooler.stride
    windows = grid[0, 0].unfold(0, side, stride).unfold(1, side, stride)  # (H_lo, W_lo, side, side)
    windows = windows.flatten(-2).permute(2, 0, 1)  # (k, H_lo, W_lo)
    axis = torch.arange(side, device=device, dtype=torch.float64) - (side - 1) / 2
    radius = (axis[:, None].square() + axis[None, :].square()).flatten()
    windows = windows[torch.sort(radius, stable=True).indices]
    windows = torch.where(pooler.output_mask[0], windows, -1.0)
    if not pooler.polar:
        # Invert the upright Cartesian layout produced by ``as_grid``, as GridPoolingLayer.forward does.
        windows = windows.flip(-2).transpose(-2, -1)
    indices = windows.flatten(-2).long()
    return torch.where(indices < 0, num_inputs, indices)


def is_segmentation_output(name: str) -> bool:
    """Whether a sensor output holds segmentation labels, which pool with ``pool_int_mode``, the modal label
    (:func:`pool_segmentation_buffer`).

    Uses the key name (``"segmentation" in name``) rather than the dtype: every Isaac Lab segmentation variant
    (``semantic_segmentation``, ``instance_segmentation_fast``, ``instance_id_segmentation_fast``) holds labels that
    averaging would corrupt, whether stored as integer IDs or colorized RGBA.
    """
    return "segmentation" in name


def pool_hwc_buffer(
    pooler: KNNPoolingLayer | GridPoolingLayer, buf: torch.Tensor, visible: torch.Tensor | None = None
) -> torch.Tensor:
    """Pool a ``(B, N_hi, 1, C)`` sensor buffer into ``(B, N_lo, 1, C)``.

    Both poolers expect ``(B, C, N)`` inputs, so we permute in and out.

    Args:
        pooler: The retinal pooler.
        buf: ``(B, N_hi, 1, C)`` floating-point samples.
        visible: ``(B, N_hi)`` photoreceptors the physical camera sees, or None for all. The others are left out of
            each pooled sample, as the pooler leaves out photoreceptors outside the retina, and a pooled sample that
            sees none is zero.
    """
    B, N_hi, one, C = buf.shape
    assert one == 1, f"expected (B, N, 1, C) buffer, got {buf.shape}"
    x = buf[:, :, 0, :].transpose(1, 2).contiguous()  # (B, C, N_hi)
    if visible is None:
        y = pooler(x)  # (B, C, N_lo)
    else:
        seen = visible[:, None, :]
        weight = seen.to(x.dtype)
        if pooler.mode == "sum":
            y = pooler(x * weight)
        else:
            # Pool the samples with their visibility in one call: a weighted mean over the visible photoreceptors is
            # the pooled masked samples over the pooled mask, and a max over them ignores -inf.
            masked = torch.where(seen, x, -torch.inf) if pooler.mode == "max" else x * weight
            pooled = pooler(torch.cat((masked, weight), dim=1))
            values, coverage = pooled[:, :C], pooled[:, C:]
            if pooler.mode != "max":
                values = values / coverage.clamp_min(torch.finfo(x.dtype).tiny)
            y = torch.where(coverage > 0, values, 0)
    return y.transpose(1, 2).unsqueeze(2).contiguous()  # (B, N_lo, 1, C)


def pool_segmentation_buffer(
    neighbors: torch.Tensor, buf: torch.Tensor, visible: torch.Tensor | None = None
) -> torch.Tensor:
    """Pool a ``(B, N_hi, 1, C)`` segmentation buffer into ``(B, N_lo, 1, C)`` by majority vote.

    Each output sample takes the most common full channel vector among its
    pooling neighbours (so neighbours ``[0, 0, 0]``, ``[0, 0, 0]``,
    ``[0, 0, 1]`` give ``[0, 0, 0]``), keeping the buffer's dtype: values are
    compared bitwise and copied from an input sample, never converted or mixed.
    This holds for integer IDs of any width and for colorized RGBA vectors.

    Ties go to the value whose first occurrence comes earliest in the pooler's
    neighbour order (see :func:`segmentation_neighbor_indices`; the nearest
    neighbour wins for KNN pooling).  Neighbours the pooler excludes, or the
    physical camera does not see, do not vote, and outputs without any valid
    neighbour are zero, as in fovi.

    Args:
        neighbors: ``(k, N_lo)`` neighbour table from :func:`segmentation_neighbor_indices`, on the buffer's device.
        buf: ``(B, N_hi, 1, C)`` buffer of any dtype.
        visible: ``(B, N_hi)`` photoreceptors the physical camera sees, or None for all.

    Returns:
        ``(B, N_lo, 1, C)`` pooled buffer with the dtype of ``buf``.
    """
    B, N_hi, one, C = buf.shape
    assert one == 1, f"expected (B, N, 1, C) buffer, got {buf.shape}"
    indices = neighbors  # (k, N_lo), N_hi = excluded
    k, N_lo = indices.shape
    values = buf[:, :, 0, :].contiguous()  # (B, N_hi, C)
    # Equality keys, padded with one never-voting sample at index N_hi.
    keys = torch.nn.functional.pad(_equality_keys(values), (0, 1))  # (B, N_hi + 1)
    valid = (indices != N_hi).T  # (N_lo, k)
    table = indices.T  # (N_lo, k)
    chunk_size = max(1, _MODAL_CHUNK_ELEMENTS // max(1, N_lo * k))
    # Visibility, padded like the keys; the never-voting sample is unseen.
    seen = None if visible is None else torch.nn.functional.pad(visible, (0, 1)).split(chunk_size)
    slots = []
    for index, chunk in enumerate(keys.split(chunk_size)):
        chunk_valid = valid if seen is None else valid & seen[index][:, table]  # (N_lo, k) or (b, N_lo, k)
        slots.append(_modal_slot(chunk[:, table], chunk_valid))
    slot = torch.cat(slots)  # (B, N_lo); k where no neighbour is valid
    has_value = slot < k
    # (B, N_lo) input sample index; outputs without a value read sample 0 and are zeroed below.
    source = table.gather(1, slot.clamp(max=k - 1).T).T.clamp(max=N_hi - 1)
    pooled = values.gather(1, source.unsqueeze(-1).expand(-1, -1, C))
    pooled = torch.where(has_value.unsqueeze(-1), pooled, torch.zeros((), dtype=buf.dtype, device=buf.device))
    return pooled.unsqueeze(2)


def _equality_keys(values: torch.Tensor) -> torch.Tensor:
    """Map ``(B, N, C)`` channel vectors to ``(B, N)`` int64 keys that are equal iff the vectors are bitwise equal.

    Vectors of at most 8 bytes are packed into the key directly; wider vectors
    fall back to dense ranks from :func:`torch.unique`.
    """
    B, N, _ = values.shape
    raw = values.view(torch.uint8) if values.dtype != torch.bool else values.to(torch.uint8)
    num_bytes = raw.shape[-1]
    if num_bytes <= 8:
        return torch.nn.functional.pad(raw, (0, 8 - num_bytes)).view(torch.int64).reshape(B, N)
    return torch.unique(raw.reshape(B * N, num_bytes), dim=0, return_inverse=True)[1].reshape(B, N)


def _modal_slot(keys: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Pick the modal neighbour slot of ``(B, N_lo, k)`` keys; ``valid`` is the ``(N_lo, k)`` or ``(B, N_lo, k)``
    voting mask.

    Returns the ``(B, N_lo)`` earliest slot holding the most common valid key,
    or ``k`` where no slot is valid.
    """
    k = keys.shape[-1]
    sorted_keys, order = torch.sort(keys, dim=-1)
    # Label equal-key runs in sorted order, then count valid votes and find the earliest valid slot per run.
    new_run = torch.ones_like(sorted_keys, dtype=torch.bool)
    new_run[..., 1:] = sorted_keys[..., 1:] != sorted_keys[..., :-1]
    run = new_run.cumsum(-1) - 1
    valid_sorted = valid.expand_as(order).gather(-1, order)
    votes = torch.zeros_like(run).scatter_add_(-1, run, valid_sorted.long())
    earliest = torch.full_like(run, k).scatter_reduce_(-1, run, torch.where(valid_sorted, order, k), "amin")
    # Most votes first, then earliest slot; runs without valid votes score zero.
    best = (votes * (k + 1) + (k - earliest)).argmax(-1, keepdim=True)
    return earliest.gather(-1, best).squeeze(-1)
