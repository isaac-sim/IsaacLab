# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime implementation and start-sampler classes for assembly profiles.

These classes are geometry providers — they derive cached tensors from their
configs once, and the :class:`AssemblyProfile` hoists all segment geometry into
stacked tensors for fully vectorized interpolation at sample time.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from isaaclab_tasks.contrib.nist.assembly_keypoints import Offset

if TYPE_CHECKING:
    from isaaclab_tasks.contrib.nist.assembly_profile_cfg import (
        AssemblyProfileCfg,
        DiscreteYawCfg,
        EndPointsSegmentCfg,
        IncrementalSegmentCfg,
        UniformPoseNoiseCfg,
        UniformYawCfg,
    )

# ---------------------------------------------------------------------------
# Start-sampler type and callable classes
# ---------------------------------------------------------------------------

StartSampler = Callable[[int, str | torch.device], tuple[torch.Tensor, torch.Tensor]]
"""Callable that generates noise for the start pose of a segment.

Args:
    num_envs: Number of environments.
    device: Torch device.

Returns:
    ``(pos_offset, quat)`` noise tensors, shapes ``(num_envs, 3)`` and ``(num_envs, 4)``.
"""


class UniformYaw:
    """Uniformly random yaw in ``[-pi, pi]``, no position noise."""

    def __init__(self, cfg: UniformYawCfg):
        self.cfg = cfg

    def __call__(self, num_envs: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.zeros(num_envs, 3, device=device)
        yaw = math_utils.sample_uniform(-math.pi, math.pi, (num_envs,), device=device)
        quat = math_utils.quat_from_euler_xyz(
            torch.zeros(num_envs, device=device),
            torch.zeros(num_envs, device=device),
            yaw,
        )
        return pos, quat


class DiscreteYaw:
    """Randomly chosen from a discrete set of yaw angles [rad], no position noise."""

    def __init__(self, cfg: DiscreteYawCfg):
        self.cfg = cfg
        self.yaws: list[float] = cfg.yaws if cfg.yaws is not None else []

    def __call__(self, num_envs: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.zeros(num_envs, 3, device=device)
        yaw_options = torch.tensor(self.yaws, device=device)
        indices = torch.randint(0, len(self.yaws), (num_envs,), device=device)
        yaw = yaw_options[indices]
        quat = math_utils.quat_from_euler_xyz(
            torch.zeros(num_envs, device=device),
            torch.zeros(num_envs, device=device),
            yaw,
        )
        return pos, quat


class UniformPoseNoise:
    """Uniform noise over user-defined position and euler-angle ranges."""

    def __init__(self, cfg: UniformPoseNoiseCfg):
        self.cfg = cfg

    def __call__(self, num_envs: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ranges = torch.tensor(
            [self.cfg.x, self.cfg.y, self.cfg.z, self.cfg.roll, self.cfg.pitch, self.cfg.yaw],
            device=device,
        )
        samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=device)
        pos = samples[:, :3]
        quat = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
        return pos, quat


# ---------------------------------------------------------------------------
# Segment and profile implementation classes
# ---------------------------------------------------------------------------


class EndPointsSegment:
    """Geometry provider: derives cached tensors from explicit start/end poses."""

    def __init__(self, cfg: EndPointsSegmentCfg):
        self.cfg = cfg
        self._pos_start: torch.Tensor | None = None
        self._pos_end: torch.Tensor | None = None
        self._quat_start: torch.Tensor | None = None
        self._aa_delta: torch.Tensor | None = None
        self._euler_total: torch.Tensor | None = None

    def _ensure_tensors(self, device: str | torch.device):
        if self._pos_start is not None and self._pos_start.device == torch.device(device):
            return
        self._pos_start = torch.tensor([self.cfg.start_pose.pos], device=device)
        self._pos_end = torch.tensor([self.cfg.end_pose.pos], device=device)
        self._quat_start = torch.tensor([self.cfg.start_pose.quat], device=device)
        quat_end = torch.tensor([self.cfg.end_pose.quat], device=device)
        quat_rel = math_utils.quat_mul(math_utils.quat_conjugate(self._quat_start), quat_end)
        self._aa_delta = math_utils.axis_angle_from_quat(quat_rel)
        revs = torch.tensor([self.cfg.revolutions], device=device)
        self._euler_total = revs * 2.0 * math.pi


class IncrementalSegment:
    """Geometry provider: derives cached tensors from start + distance + ratio."""

    def __init__(self, cfg: IncrementalSegmentCfg):
        self.cfg = cfg
        self._pos_start: torch.Tensor | None = None
        self._pos_end: torch.Tensor | None = None
        self._quat_start: torch.Tensor | None = None
        self._aa_delta: torch.Tensor | None = None
        self._euler_total: torch.Tensor | None = None

    def _ensure_tensors(self, device: str | torch.device):
        if self._pos_start is not None and self._pos_start.device == torch.device(device):
            return
        self._pos_start = torch.tensor([self.cfg.start_pose.pos], device=device)
        dist = torch.tensor([self.cfg.distance], device=device)
        self._pos_end = self._pos_start + dist
        self._quat_start = torch.tensor([self.cfg.start_pose.quat], device=device)
        self._aa_delta = torch.zeros(1, 3, device=device)
        ratio = torch.tensor([self.cfg.ratio], device=device)
        self._euler_total = torch.where(ratio != 0, dist / ratio, torch.zeros_like(dist))


class AssemblyProfile:
    """Vectorized assembly sampler.

    All segment geometry is hoisted into stacked profile-level tensors at cache
    time.  The :meth:`sample` hot path uses gather + fused-multiply-add for
    position, axis-angle interpolation for the start-to-end orientation delta,
    euler-based revolutions, and a noise dispatch loop (only when segments have
    non-trivial noise).
    """

    def __init__(self, cfg: AssemblyProfileCfg):
        self.cfg = cfg
        assert cfg.segments is not None, "AssemblyProfileCfg.segments must be set"
        self.segments: list[EndPointsSegment | IncrementalSegment] = [
            seg_cfg.class_type(seg_cfg)
            for seg_cfg in cfg.segments  # type: ignore[misc]
        ]
        self._samplers: list[StartSampler | None] = [
            seg_cfg.start_sampler.class_type(seg_cfg.start_sampler) if seg_cfg.start_sampler is not None else None
            for seg_cfg in cfg.segments
        ]
        self._boundaries: torch.Tensor | None = None
        self._seg_widths: torch.Tensor | None = None
        self._seg_pos_starts: torch.Tensor | None = None
        self._seg_pos_deltas: torch.Tensor | None = None
        self._seg_quat_starts: torch.Tensor | None = None
        self._seg_aa_deltas: torch.Tensor | None = None
        self._seg_euler_totals: torch.Tensor | None = None
        self._all_no_noise: bool = all(s is None for s in self._samplers)

    @property
    def assembled_offset(self) -> Offset:
        """The assembled pose (fraction=0), i.e. the first segment's start_pose."""
        return self.segments[0].cfg.start_pose

    def _ensure_profile_tensors(self, device: str | torch.device):
        if self._boundaries is not None and self._boundaries.device == torch.device(device):
            return
        for seg in self.segments:
            seg._ensure_tensors(device)
        bounds = [self.segments[0].cfg.fraction[0]]
        for seg in self.segments:
            bounds.append(seg.cfg.fraction[1])
        self._boundaries = torch.tensor(bounds, device=device)
        self._seg_widths = torch.tensor(
            [seg.cfg.fraction[1] - seg.cfg.fraction[0] for seg in self.segments],
            device=device,
        )
        self._seg_pos_starts = torch.cat([seg._pos_start for seg in self.segments], dim=0)  # type:ignore
        self._seg_pos_deltas = torch.cat([seg._pos_end - seg._pos_start for seg in self.segments], dim=0)  # type:ignore
        self._seg_quat_starts = torch.cat([seg._quat_start for seg in self.segments], dim=0)  # type:ignore
        self._seg_aa_deltas = torch.cat([seg._aa_delta for seg in self.segments], dim=0)  # type:ignore
        self._seg_euler_totals = torch.cat([seg._euler_total for seg in self.segments], dim=0)  # type:ignore

    def sample(
        self,
        fraction_range: tuple[float, float],
        num_envs: int,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample assembly poses for all envs in one vectorized pass.

        Args:
            fraction_range: ``(lo, hi)`` fraction range to sample from.
            num_envs: Number of environments.
            device: Torch device.

        Returns:
            ``(pos, quat)`` each of shape ``(num_envs, 3)`` and ``(num_envs, 4)``.
        """
        self._ensure_profile_tensors(device)
        assert self._boundaries is not None
        assert self._seg_widths is not None
        assert self._seg_pos_starts is not None
        assert self._seg_pos_deltas is not None
        assert self._seg_quat_starts is not None
        assert self._seg_aa_deltas is not None
        assert self._seg_euler_totals is not None

        fractions = math_utils.sample_uniform(fraction_range[0], fraction_range[1], (num_envs, 1), device=device)

        seg_idx = torch.searchsorted(self._boundaries[1:], fractions.squeeze(-1)).clamp(0, len(self.segments) - 1)

        seg_lo = self._boundaries[:-1][seg_idx]
        t = ((fractions.squeeze(-1) - seg_lo) / self._seg_widths[seg_idx]).unsqueeze(-1)

        pos = self._seg_pos_starts[seg_idx] + t * self._seg_pos_deltas[seg_idx]

        aa_interp = t * self._seg_aa_deltas[seg_idx]
        angle = aa_interp.norm(dim=-1)
        axis = torch.where(
            angle.unsqueeze(-1) > 1e-8,
            aa_interp / angle.unsqueeze(-1).clamp(min=1e-8),
            torch.tensor([[0.0, 0.0, 1.0]], device=device).expand_as(aa_interp),
        )
        quat_delta = math_utils.quat_from_angle_axis(angle, axis)

        euler = t * self._seg_euler_totals[seg_idx]
        quat_rev = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

        quat = math_utils.quat_mul(
            self._seg_quat_starts[seg_idx],
            math_utils.quat_mul(quat_delta, quat_rev),
        )

        if not self._all_no_noise:
            if len(self._samplers) == 1 and self._samplers[0] is not None:
                noise_pos, noise_quat = self._samplers[0](num_envs, device)
                pos = pos + noise_pos
                quat = math_utils.quat_mul(noise_quat, quat)
            else:
                for i, sampler in enumerate(self._samplers):
                    if sampler is None:
                        continue
                    mask = seg_idx == i
                    if not mask.any():
                        continue
                    n = int(mask.sum().item())
                    noise_pos, noise_quat = sampler(n, device)
                    pos[mask] = pos[mask] + noise_pos
                    quat[mask] = math_utils.quat_mul(noise_quat, quat[mask])

        return pos, quat
