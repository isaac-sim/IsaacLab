# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared environment-relative and asset-tracking visualizer cameras."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ..scene import InteractiveScene
    from .visualizer_cfg import VisualizerCfg

logger = logging.getLogger(__name__)


class CameraController:
    """Resolve one camera's origin and retain its selected environment across resets."""

    def __init__(self, cfg: VisualizerCfg, scene: InteractiveScene, visible_env_ids: list[int] | None) -> None:
        self.cfg = cfg
        env_origins = scene.env_origins.cpu().numpy()
        self.env_index = self._resolve_env_index(env_origins, visible_env_ids)
        self.origin = env_origins[self.env_index].astype(float)
        self.has_pose = False
        self._yaw: float | None = None
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
        if self._asset is None:
            if self.has_pose:
                return None
        else:
            if not self._asset.is_initialized:
                return None
            data = self._asset.data
            follow_heading = self.cfg.origin_follow_heading
            if self._body_name:
                if self._body_index is None:
                    body_ids, _ = self._asset.find_bodies(self._body_name)
                    if len(body_ids) != 1:
                        raise ValueError(
                            f"Camera origin_track_path must match exactly one body: {self.cfg.origin_track_path!r}."
                        )
                    self._body_index = body_ids[0]
                position = data.body_pos_w.torch[self.env_index, self._body_index]
                quat = data.body_quat_w.torch[self.env_index, self._body_index] if follow_heading else None
            else:
                position = data.root_pos_w.torch[self.env_index]
                quat = data.root_quat_w.torch[self.env_index] if follow_heading else None
            if quat is None:
                self.origin = position.cpu().numpy().astype(float)
            else:
                # One device-to-host copy per frame.
                state = torch.cat((position, quat)).cpu().numpy().astype(float)
                self.origin = state[:3]
                self._update_yaw(*state[3:], dt)
        eye, target = self._rotate(np.array((self.cfg.eye, self.cfg.lookat), dtype=float)) + self.origin
        self.has_pose = True
        return tuple(eye.tolist()), tuple(target.tolist())

    def world_to_offsets(
        self, eye: tuple[float, float, float], target: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Convert a world-space camera edit back into the last rendered tracking frame."""
        offsets = np.array((eye, target), dtype=float) - self.origin
        if self._yaw is not None:
            offsets = self._rotate(offsets, inverse=True)
        eye_offset, target_offset = offsets.tolist()
        return tuple(eye_offset), tuple(target_offset)

    def _update_yaw(self, x: float, y: float, z: float, w: float, dt: float) -> None:
        """Filter the tracked yaw toward the quaternion's yaw along the shortest rotation."""
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        time_constant = self.cfg.origin_heading_smoothing_time_constant
        if self._yaw is not None and time_constant > 0.0:
            weight = -math.expm1(-dt / time_constant)
            yaw = self._yaw + weight * math.remainder(yaw - self._yaw, math.tau)
        self._yaw = yaw

    def _rotate(self, offsets: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Rotate offsets about +Z by the tracked yaw; offsets stay world-aligned without heading."""
        if self._yaw is None:
            return offsets
        cos, sin = math.cos(self._yaw), math.sin(self._yaw)
        rotation = np.array(((cos, -sin, 0.0), (sin, cos, 0.0), (0.0, 0.0, 1.0)))
        return offsets @ rotation if inverse else offsets @ rotation.T

    def _resolve_env_index(self, env_origins: np.ndarray, visible_env_ids: list[int] | None) -> int:
        """Choose a visible environment using physical layout rather than array ordering."""
        num_envs = len(env_origins)
        candidates = sorted(set(visible_env_ids)) if visible_env_ids is not None else list(range(num_envs))
        if not candidates:
            raise ValueError("Camera origin requires at least one visible environment.")
        xy = env_origins[:, :2]
        index = self.cfg.origin_env_index
        if index == "center":
            target = (xy.min(axis=0) + xy.max(axis=0)) / 2
        elif not 0 <= index < num_envs:
            raise ValueError(f"Camera origin_env_index {index} is outside the environment range [0, {num_envs - 1}].")
        elif index in candidates:
            return index
        else:
            target = xy[index]
        # Sorted candidates make argmin break ties by lowest index.
        nearest = candidates[int(np.argmin(((xy[candidates] - target) ** 2).sum(axis=-1)))]
        if index != "center":
            logger.warning("Camera environment %d is not visible; following environment %d instead.", index, nearest)
        return nearest
