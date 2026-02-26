# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for visualizers."""

from __future__ import annotations

import logging
import random
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.sim.scene_data_providers import SceneDataProvider

    from .visualizer_cfg import VisualizerCfg


logger = logging.getLogger(__name__)


class Visualizer(ABC):
    """Base class for all visualizer backends.

    Lifecycle: __init__() -> initialize() -> step() (repeated) -> close()
    """

    def __init__(self, cfg: VisualizerCfg):
        """Initialize visualizer with config."""
        self.cfg = cfg
        self._scene_data_provider = None
        self._is_initialized = False
        self._is_closed = False

    @abstractmethod
    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Initialize visualizer resources."""
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float, state: Any | None = None) -> None:
        """Update visualization for one step.

        Args:
            dt: Time step in seconds.
            state: Updated physics state (e.g., newton.State).
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        raise NotImplementedError

    @abstractmethod
    def is_running(self) -> bool:
        """Check if visualizer is still running (e.g., window not closed)."""
        raise NotImplementedError

    def is_training_paused(self) -> bool:
        """Check if training is paused by visualizer controls."""
        return False

    def is_rendering_paused(self) -> bool:
        """Check if rendering is paused by visualizer controls."""
        return False

    @property
    def is_initialized(self) -> bool:
        """Check if initialize() has been called."""
        return self._is_initialized

    @property
    def is_closed(self) -> bool:
        """Check if close() has been called."""
        return self._is_closed

    def supports_markers(self) -> bool:
        """Check if visualizer supports VisualizationMarkers."""
        return False

    def supports_live_plots(self) -> bool:
        """Check if visualizer supports LivePlots."""
        return False

    def requires_forward_before_step(self) -> bool:
        """Whether simulation should run forward() before step()."""
        return False

    def get_visualized_env_ids(self) -> list[int] | None:
        """Return env IDs this visualizer is displaying, if any."""
        return getattr(self, "_env_ids", None)

    def _compute_visualized_env_ids(self) -> list[int] | None:
        """Compute which env indices to show from config."""
        if self._scene_data_provider is None:
            return None
        num_envs = self._scene_data_provider.get_metadata().get("num_envs", 0)
        if num_envs <= 0:
            logger.warning(
                "[Visualizer] num_envs is 0 or missing from provider metadata; partial visualization disabled."
            )
            return None
        filter_mode = getattr(self.cfg, "env_filter_mode", "none")
        if filter_mode == "none":
            return None
        if filter_mode == "env_ids":
            env_ids_cfg = getattr(self.cfg, "env_filter_ids", None)
            if env_ids_cfg is not None and len(env_ids_cfg) > 0:
                return [i for i in env_ids_cfg if 0 <= i < num_envs]
            return None
        if filter_mode == "random_n":
            count = int(getattr(self.cfg, "env_filter_random_n", 0))
            if count <= 0:
                return None
            count = min(count, num_envs)
            seed = int(getattr(self.cfg, "env_filter_seed", 0))
            rng = random.Random(seed)
            return sorted(rng.sample(range(num_envs), count))
        logger.warning("[Visualizer] Unknown env_filter_mode='%s'; defaulting to all envs.", filter_mode)
        return None

    def get_rendering_dt(self) -> float | None:
        """Get rendering time step. Returns None to use interface default."""

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Set camera view position. No-op by default."""
        pass

    def _resolve_camera_pose_from_usd_path(
        self, usd_path: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Resolve camera pose/target from provider camera transforms."""
        if self._scene_data_provider is None:
            return None
        transforms = self._scene_data_provider.get_camera_transforms()
        if not transforms:
            return None

        env_id, template_path = self._resolve_template_camera_path(usd_path)
        camera_transform = self._lookup_camera_transform(transforms, template_path, env_id)
        if camera_transform is None:
            return None
        pos, ori = camera_transform

        pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
        ori_t = (float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]))
        forward = self._quat_rotate_vec(ori_t, (0.0, 0.0, -1.0))
        target = (pos_t[0] + forward[0], pos_t[1] + forward[1], pos_t[2] + forward[2])
        return pos_t, target

    def _resolve_template_camera_path(self, usd_path: str) -> tuple[int, str]:
        """Normalize concrete env camera path to templated camera path."""
        env_pattern = re.compile(r"(?P<root>/World/envs/env_)(?P<id>\d+)(?P<path>/.*)")
        if match := env_pattern.match(usd_path):
            return int(match.group("id")), match.group("root") + "%d" + match.group("path")
        return 0, usd_path

    def _lookup_camera_transform(
        self, transforms: dict[str, Any], template_path: str, env_id: int
    ) -> tuple[list[float], list[float]] | None:
        """Fetch (position, orientation) for a templated camera path and env index."""
        order = transforms.get("order", [])
        positions = transforms.get("positions", [])
        orientations = transforms.get("orientations", [])

        if template_path not in order:
            return None
        idx = order.index(template_path)
        if idx >= len(positions) or idx >= len(orientations):
            return None
        if env_id < 0 or env_id >= len(positions[idx]):
            return None
        pos = positions[idx][env_id]
        ori = orientations[idx][env_id]
        if pos is None or ori is None:
            return None
        return pos, ori

    @staticmethod
    def _quat_rotate_vec(
        quat_xyzw: tuple[float, float, float, float], vec: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        import torch

        from isaaclab.utils.math import quat_apply

        quat = torch.tensor(quat_xyzw, dtype=torch.float32).unsqueeze(0)
        vector = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        rotated = quat_apply(quat, vector)[0]
        return (float(rotated[0]), float(rotated[1]), float(rotated[2]))

    def reset(self, soft: bool = False) -> None:
        """Reset visualizer state. No-op by default."""
        pass

    def play(self) -> None:
        """Handle simulation play/start. No-op by default."""
        pass

    def pause(self) -> None:
        """Handle simulation pause. No-op by default."""
        pass

    def stop(self) -> None:
        """Handle simulation stop. No-op by default."""
        pass
