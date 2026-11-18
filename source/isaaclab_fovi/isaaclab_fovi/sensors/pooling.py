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
        field_geometry="spherical",
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
        field_geometry="spherical",
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
        field_geometry="spherical",
    )


MODAL_POOL_MODE = "mode"
"""Pooling mode that takes the most common neighbour value (see :func:`pool_segmentation_buffer`)."""

# Upper bound on ``batch * outputs * k`` elements per modal-pooling chunk.  Each chunk holds about ten
# int64 tensors of this size, so peak scratch memory stays near 300 MB regardless of the batch size.
_MODAL_CHUNK_ELEMENTS = 2**22


class ModalPooler(torch.nn.Module):
    """Modal pooler for segmentation outputs, built by :func:`build_pooler` with ``mode="mode"``.

    Holds the fixed neighbour table of a fovi pooler and pools by majority vote
    (see :func:`pool_segmentation_buffer`).  Like the fovi poolers, calling it
    maps ``(B, C, N_hi)`` to ``(B, C, N_lo)``, so :func:`pool_hwc_buffer` also
    accepts it, but only in the native dtype: casting IDs to float first loses
    IDs above ``2**24``.
    """

    def __init__(self, neighbor_indices: torch.Tensor, num_inputs: int):
        """Initialize the pooler.

        Args:
            neighbor_indices: ``(k, N_lo)`` int64 input-sample indices of each output's neighbours, ordered by
                tie-break preference (nearest first).  The value ``num_inputs`` marks an excluded neighbour.
            num_inputs: Number of input samples ``N_hi``.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.register_buffer("neighbor_indices", neighbor_indices, persistent=False)

    @property
    def k(self) -> int:
        """Number of neighbour slots per output sample."""
        return self.neighbor_indices.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool ``(B, C, N_hi)`` values to ``(B, C, N_lo)`` by majority vote over each channel vector."""
        pooled = pool_segmentation_buffer(self, x.transpose(1, 2).unsqueeze(2))
        return pooled[:, :, 0, :].transpose(1, 2)


def build_pooler(
    cfg: FoveatedCameraCfg,
    lowres: SamplingCoords,
    highres: SamplingCoords | None,
    device: str,
    mode: str | None = None,
) -> tuple[KNNPoolingLayer | GridPoolingLayer | ModalPooler | None, int | None]:
    """Build a retinal pooling layer. Both coord sets must already be on ``device``.

    If ``mode`` is None, uses ``cfg.pool_mode`` — the standard pooler for
    floating-point outputs.  ``mode="mode"`` (the default ``cfg.pool_int_mode``)
    builds a :class:`ModalPooler` over the same neighbourhoods for segmentation
    outputs; pool those with :func:`pool_segmentation_buffer`.  Other modes are
    passed to fovi (``"max"``, ``"avg"``, ``"sum"``, ``"gaussian"``).
    """
    if highres is None:
        return None, None

    from fovi.arch.pooling import build_retinal_pooler

    mode = mode if mode is not None else cfg.pool_mode
    pooler, k = build_retinal_pooler(
        highres,
        lowres,
        k=cfg.pool_k,
        # The modal pooler only reuses the fovi neighbourhoods; "max" needs no Gaussian weights.
        mode="max" if mode == MODAL_POOL_MODE else mode,
        device=device,
        sample_cortex=cfg.pool_sample_cortex,
        gauss_sigma=cfg.pool_gauss_sigma,
    )
    if mode == MODAL_POOL_MODE:
        pooler = ModalPooler(segmentation_neighbor_indices(pooler), len(highres))
    return pooler, k


def segmentation_neighbor_indices(pooler: KNNPoolingLayer | GridPoolingLayer | ModalPooler) -> torch.Tensor:
    """Return a pooler's ``(k, N_lo)`` neighbour table in tie-break order.

    Entries are input-sample indices; ``N_hi`` (the number of input samples)
    marks a neighbour excluded by the pooler (spatial padding or a masked input
    sample), and every slot of a masked output sample is excluded.

    * :class:`KNNPoolingLayer`: its padded KNN indices, nearest neighbour first.
    * :class:`GridPoolingLayer`: the samples in each pooling window (same
      padding, stride and masks as the pooler), ordered by distance from the
      window centre and then in row-major window order.
    """
    from fovi.arch.knn import KNNPoolingLayer
    from fovi.arch.pooling import GridPoolingLayer

    if isinstance(pooler, ModalPooler):
        return pooler.neighbor_indices
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
    """Whether a sensor output key should be pooled with ``pool_int_mode`` (modal pooling by default).

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


def pool_segmentation_buffer(
    pooler: KNNPoolingLayer | GridPoolingLayer | ModalPooler, buf: torch.Tensor
) -> torch.Tensor:
    """Pool a ``(B, N_hi, 1, C)`` segmentation buffer into ``(B, N_lo, 1, C)`` by majority vote.

    Each output sample takes the most common full channel vector among its
    pooling neighbours (so neighbours ``[0, 0, 0]``, ``[0, 0, 0]``,
    ``[0, 0, 1]`` give ``[0, 0, 0]``), keeping the buffer's dtype: values are
    compared bitwise and copied from an input sample, never converted or mixed.
    This holds for integer IDs of any width and for colorized RGBA vectors.

    Ties go to the value whose first occurrence comes earliest in the pooler's
    neighbour order (see :func:`segmentation_neighbor_indices`; the nearest
    neighbour wins for KNN pooling).  Neighbours the pooler excludes do not
    vote, and outputs without any valid neighbour are zero, as in fovi.

    Args:
        pooler: A :class:`ModalPooler` from ``build_pooler(..., mode="mode")``, or a fovi KNN or grid pooler
            whose neighbourhoods to use.
        buf: ``(B, N_hi, 1, C)`` buffer of any dtype, on the pooler's device.

    Returns:
        ``(B, N_lo, 1, C)`` pooled buffer with the dtype of ``buf``.
    """
    B, N_hi, one, C = buf.shape
    assert one == 1, f"expected (B, N, 1, C) buffer, got {buf.shape}"
    indices = segmentation_neighbor_indices(pooler)  # (k, N_lo), N_hi = excluded
    k, N_lo = indices.shape
    values = buf[:, :, 0, :].contiguous()  # (B, N_hi, C)
    # Equality keys, padded with one never-voting sample at index N_hi.
    keys = torch.nn.functional.pad(_equality_keys(values), (0, 1))  # (B, N_hi + 1)
    valid = (indices != N_hi).T  # (N_lo, k)
    table = indices.T  # (N_lo, k)
    slots = []
    for chunk in keys.split(max(1, _MODAL_CHUNK_ELEMENTS // max(1, N_lo * k))):
        slots.append(_modal_slot(chunk[:, table], valid))
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
    """Pick the modal neighbour slot of ``(B, N_lo, k)`` keys; ``valid`` is the ``(N_lo, k)`` voting mask.

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
