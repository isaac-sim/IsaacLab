# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World poses for perspective views tied to an environment or an asset."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..scene import InteractiveScene
    from .visualizer_cfg import TrackingCameraCfg

_RESET_JUMP = 1.0
"""Distance [m] an asset moves between two updates only when its environment resets."""

_RESET_YAW_JUMP = 1.5
"""Heading change [rad] between two updates that a robot only makes when its environment resets.

A fixed angle, unlike a rate, does not misfire when two updates are a few milliseconds apart."""


class CameraFollower:
    """Turns a :class:`~isaaclab.visualizers.TrackingCameraCfg` tied to an environment into world poses.

    A view that only has :attr:`~isaaclab.visualizers.TrackingCameraCfg.env_index` stays fixed relative to that
    environment's origin. A view with :attr:`~isaaclab.visualizers.TrackingCameraCfg.track_path` follows the
    asset's position, and its yaw when ``follow_heading`` is set, with the heading filtered over time.
    """

    def __init__(self, cfg: TrackingCameraCfg, scene: InteractiveScene) -> None:
        self.cfg = cfg
        self._scene = scene
        self._env = 0
        self._yaw: float | None = None
        self._last_position: list[float] | None = None
        self._time: float | None = None
        self.set_env(cfg.env_index)
        self._asset = None
        self._body_index: int | None = None
        self._body_name = ""
        if cfg.track_path:
            self.set_track_path(cfg.track_path)

    def set_track_path(self, track_path: str | None) -> None:
        """Follow another asset or body, or only the environment origin when *track_path* is empty.

        Args:
            track_path: ``"asset"`` for an asset root or ``"asset/body"`` for one of its bodies; None or an empty
                string stops following an asset.

        Raises:
            ValueError: If the asset is unknown, or the body does not match exactly one body of an initialized asset.
        """
        asset, body_name = None, ""
        if track_path:
            asset_name, _, body_name = track_path.partition("/")
            try:
                asset = self._scene[asset_name]
            except KeyError as exc:
                raise ValueError(f"track_path refers to an unknown scene asset: {asset_name!r}.") from exc
            if body_name and asset.is_initialized and len(asset.find_bodies(body_name)[0]) != 1:
                raise ValueError(f"track_path must match exactly one body: {track_path!r}.")
        self.cfg.track_path = track_path or None
        self._asset, self._body_name, self._body_index = asset, body_name, None
        self._yaw = self._last_position = self._time = None

    def asset_names(self) -> list[str]:
        """Return the names of the scene assets the view can follow."""
        scene = self._scene
        return [*getattr(scene, "articulations", {}), *getattr(scene, "rigid_objects", {})]

    def body_names(self, asset_name: str) -> list[str]:
        """Return the body names of the scene asset *asset_name*, or an empty list when it has none."""
        try:
            return list(getattr(self._scene[asset_name], "body_names", []))
        except KeyError:
            return []

    @property
    def env(self) -> int:
        """Environment whose origin or asset the view follows."""
        return self._env

    def set_env(self, index: int) -> None:
        """Follow environment *index* from now on, with the heading filter starting afresh.

        Raises:
            ValueError: If *index* is outside the scene's environments.
        """
        if not 0 <= index < self._scene.num_envs:
            raise ValueError(f"env_index {index} is outside the {self._scene.num_envs} environments of the scene.")
        self._env = int(index)
        self._yaw = self._last_position = self._time = None

    def pose(self, time: float) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Return the world eye and look-at target at simulation *time*, or None until the asset is ready.

        Args:
            time: Simulation time [s]. Only the heading filter uses it, through the time since the previous call.
        """
        cfg = self.cfg
        origin = self._scene.env_origins[self._env]
        offsets = torch.tensor((cfg.eye, cfg.lookat), dtype=torch.float32, device=origin.device)
        if self._asset is None:
            eye, target = (origin + offsets).tolist()
            return tuple(eye), tuple(target)
        asset = self._asset
        if not asset.is_initialized:
            return None
        data = asset.data
        if self._body_name:
            if self._body_index is None:
                body_ids, _ = asset.find_bodies(self._body_name)
                if len(body_ids) != 1:
                    raise ValueError(f"track_path must match exactly one body: {cfg.track_path!r}.")
                self._body_index = body_ids[0]
            position = data.body_pos_w.torch[self._env, self._body_index]
            quat = data.body_quat_w.torch[self._env, self._body_index]
        else:
            position, quat = data.root_pos_w.torch[self._env], data.root_quat_w.torch[self._env]
        if cfg.follow_heading:
            # one device-to-host copy per frame
            state = torch.cat((position, quat)).tolist()
            dt, self._time = (0.0 if self._time is None else max(time - self._time, 0.0)), time
            yaw = self._filtered_yaw(state[3:], state[:3], dt)
            cos, sin = math.cos(yaw), math.sin(yaw)
            rotation = torch.tensor(((cos, -sin, 0.0), (sin, cos, 0.0), (0.0, 0.0, 1.0)), device=offsets.device)
            offsets = offsets @ rotation.T
        eye, target = (position + offsets).tolist()
        return tuple(eye), tuple(target)

    def _filtered_yaw(self, quat: list[float], position: list[float], dt: float) -> float:
        """Return the yaw of the xyzw *quat*, filtered along the shortest rotation.

        The first sample, and the sample after an environment resets, is not filtered.
        """
        x, y, z, w = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        last, self._last_position = self._last_position, position
        reset = self._yaw is None or last is None or math.dist(position, last) > _RESET_JUMP
        if not reset:
            # an environment can reset in place with a new heading
            reset = abs(math.remainder(yaw - self._yaw, math.tau)) > _RESET_YAW_JUMP
        tau = self.cfg.heading_smoothing_time_constant
        if tau > 0.0 and not reset:
            yaw = self._yaw + -math.expm1(-dt / tau) * math.remainder(yaw - self._yaw, math.tau)
        self._yaw = yaw
        return yaw
