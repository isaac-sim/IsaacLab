# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

"""Merged upstream teacher-input routing and yaw interpolation.

Adapted from Pollen Robotics ``microduck_rl`` distill.py at bae39f5900b318f7e87fc485c30a31213cfa14eb,
licensed Apache-2.0 (see LICENSE-microduck_rl.md). Changes: Isaac Lab group names and formatting.
The proposed angular-velocity gate in upstream PR54 is deliberately absent.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


def fallen_mask_from_obs(obs: torch.Tensor, gravity_slice: tuple[int, int], gate_tilt_deg: float) -> torch.Tensor:
    """tilt > gate from the projected-gravity block: upright → g = (0, 0, -1), cos(tilt) = -g_z."""
    g = obs[:, gravity_slice[0] : gravity_slice[1]]
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)  # obs noise / IMU DR: renormalize
    cos_tilt = -g[:, 2]
    return cos_tilt < torch.cos(torch.deg2rad(torch.tensor(gate_tilt_deg, device=obs.device)))


def turn_in_place_mask(obs: torch.Tensor, twist_slice: tuple[int, int], eps: float = 1e-6) -> torch.Tensor:
    """Linear command exactly 0 and a non-zero yaw command (command obs carry no noise)."""
    tw = obs[:, twist_slice[0] : twist_slice[1]]
    return (tw[:, :2].abs().amax(dim=1) <= eps) & (tw[:, 2].abs() > eps)


def body_frame_masks(
    obs: torch.Tensor,
    twist_slice: tuple[int, int],
    body_slice: tuple[int, int],
    active_axes: tuple[int, ...],
    no_teacher_axes: tuple[int, ...],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(body frames → off the walk anchor, teach frames → stand expert). Command obs carry
    no noise, so "twist == 0" and "axis == 0" are exact tests."""
    standing = obs[:, twist_slice[0] : twist_slice[1]].abs().amax(dim=1) <= eps
    body = obs[:, body_slice[0] : body_slice[1]]
    active = body[:, list(active_axes)].abs().amax(dim=1) > eps
    frames = standing & active
    if no_teacher_axes:
        frames_teach = frames & (body[:, list(no_teacher_axes)].abs().amax(dim=1) <= eps)
    else:
        frames_teach = frames
    return frames, frames_teach


def expert_input(
    obs: torch.Tensor,
    twist_slice: tuple[int, int],
    mask_cols: tuple[int, ...] = (),
    mask_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    out = obs.clone()
    out[:, twist_slice[0] : twist_slice[1]] = 0.0
    if mask_cols and mask_rows is not None and mask_rows.any():
        cols = torch.as_tensor(mask_cols, device=obs.device, dtype=torch.long)
        rows = mask_rows.nonzero().flatten()
        out[rows.unsqueeze(1), cols.unsqueeze(0)] = 0.0
    return out


def _interp_rows(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Interpolate x on xp for per-row fp values, clamping x to the knot interval."""
    x = x.clamp(xp[0], xp[-1])
    hi = torch.searchsorted(xp, x, right=True).clamp(1, xp.numel() - 1)
    lo = hi - 1
    w = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp.gather(1, lo.unsqueeze(1)).squeeze(1) * (1 - w) + fp.gather(1, hi.unsqueeze(1)).squeeze(1) * w


def remap_anchor_yaw(obs: torch.Tensor, twist_slice: tuple[int, int], table: dict) -> torch.Tensor:
    """Copy of the flat obs with the yaw command replaced by the one the walk anchor needs to
    actually turn at that rate (bilinear in |v_xy| and |wz|, sign kept, 0 → 0)."""
    out = obs.clone()
    tw = obs[:, twist_slice[0] : twist_slice[1]]
    wz = tw[:, 2]
    lin = tw[:, :2].norm(dim=1)
    dev = obs.device
    lk = torch.as_tensor(table["lin"], dtype=obs.dtype, device=dev)
    tk = torch.as_tensor([0.0, *table["targets"]], dtype=obs.dtype, device=dev)
    fed = torch.zeros_like(wz)
    for key, sign in (("pos", 1.0), ("neg", -1.0)):
        rows = (wz * sign) > 0
        if not rows.any():
            continue
        grid = torch.as_tensor([[0.0, *r] for r in table[key]], dtype=obs.dtype, device=dev)  # (L, K)
        c = wz[rows].abs()
        n = c.numel()
        per_lin = torch.stack([_interp_rows(c, tk, grid[i].expand(n, -1)) for i in range(lk.numel())], dim=1)  # (n, L)
        fed[rows] = sign * _interp_rows(lin[rows], lk, per_lin)
    out[:, twist_slice[0] + 2] = fed
    return out


def _with_flat(td: TensorDict, groups: list[str], flat: torch.Tensor) -> TensorDict:
    """Rebuild the obs groups of ``td`` from a modified flat obs."""
    off = 0
    for g in groups:
        d = td[g].shape[-1]
        td[g] = flat[:, off : off + d]
        off += d
    return td
