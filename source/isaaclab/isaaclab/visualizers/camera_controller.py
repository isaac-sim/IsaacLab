# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared environment-relative and asset-tracking visualizer cameras."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_slerp, yaw_quat

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from .visualizer_cfg import VisualizerCfg


class _CameraController:
    """Resolve one camera's origin and retain its selected environment across resets."""

    def __init__(self, cfg: VisualizerCfg, scene: InteractiveScene, visible_env_ids: list[int] | None) -> None:
        self.cfg = cfg
        self.scene = scene
        self.env_index = self._resolve_env_index(visible_env_ids)
        self.origin = scene.env_origins[self.env_index].detach().clone()
        self.has_pose = False
        self._heading: torch.Tensor | None = None
        self._body_index: int | None = None
        self._asset = None
        self._body_name = ""
        if cfg.origin_type == "asset":
            if not cfg.origin_track_path:
                raise ValueError("origin_type='asset' requires origin_track_path to be set.")
            asset_name, _, self._body_name = cfg.origin_track_path.partition("/")
            try:
                self._asset = scene[asset_name]
            except KeyError as exc:
                raise ValueError(f"Camera origin_track_path refers to an unknown scene asset: {asset_name!r}.") from exc
        elif cfg.origin_type != "env":
            raise ValueError(f"Unknown camera origin_type: {cfg.origin_type!r}.")

    def update(self, dt: float = 0.0) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Return a world-space eye/target pose after elapsed time ``dt`` [s], or defer until ready."""
        orientation = None
        if self._asset is None:
            if self.has_pose:
                return None
        else:
            if not self._asset.is_initialized:
                return None
            if self._body_name:
                if self._body_index is None:
                    body_ids, _ = self._asset.find_bodies(self._body_name)
                    if len(body_ids) != 1:
                        raise ValueError(
                            f"Camera origin_track_path must match exactly one body: {self.cfg.origin_track_path!r}."
                        )
                    self._body_index = body_ids[0]
                self.origin = self._asset.data.body_pos_w.torch[self.env_index, self._body_index]
                if self.cfg.origin_follow_heading:
                    orientation = self._asset.data.body_quat_w.torch[self.env_index, self._body_index]
            else:
                self.origin = self._asset.data.root_pos_w.torch[self.env_index]
                if self.cfg.origin_follow_heading:
                    orientation = self._asset.data.root_quat_w.torch[self.env_index]
        offsets = self.origin.new_tensor((self.cfg.eye, self.cfg.lookat))
        heading = yaw_quat(orientation) if orientation is not None else None
        time_constant = self.cfg.origin_heading_smoothing_time_constant
        if heading is not None and self._heading is not None and time_constant > 0.0:
            # Interpolate yaw only; slerp handles the shortest path across the +/-pi boundary.
            weight = -math.expm1(-max(dt, 0.0) / time_constant)
            heading = quat_slerp(self._heading, heading, weight)
        self._heading = heading
        if self._heading is not None:
            offsets = quat_apply(self._heading.expand(2, -1), offsets)
        eye, target = (offsets + self.origin).detach().cpu().tolist()
        self.has_pose = True
        return tuple(eye), tuple(target)

    def world_to_offsets(
        self, eye: tuple[float, float, float], target: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Convert a world-space camera edit back into the last rendered tracking frame."""
        offsets = self.origin.new_tensor((eye, target)) - self.origin
        if self._heading is not None:
            offsets = quat_apply_inverse(self._heading.expand(2, -1), offsets)
        eye_offset, target_offset = offsets.detach().cpu().tolist()
        return tuple(eye_offset), tuple(target_offset)

    def _resolve_env_index(self, visible_env_ids: list[int] | None) -> int:
        """Choose a visible environment using physical layout rather than array ordering."""
        num_envs = self.scene.num_envs
        candidates = sorted(set(visible_env_ids)) if visible_env_ids is not None else list(range(num_envs))
        if not candidates:
            raise ValueError("Camera origin requires at least one visible environment.")
        if self.cfg.origin_env_index == "center":
            xy = self.scene.env_origins[:, :2]
            center = (xy.amin(dim=0) + xy.amax(dim=0)) / 2
            distances = (xy[candidates] - center).square().sum(dim=-1)
            return candidates[int(distances.argmin().item())]
        index = self.cfg.origin_env_index
        if not isinstance(index, int) or not 0 <= index < num_envs:
            raise ValueError(f"Camera origin_env_index {index!r} is outside the environment range [0, {num_envs - 1}].")
        if index not in candidates:
            raise ValueError(f"Camera origin_env_index {index} is not a visible environment.")
        return index
