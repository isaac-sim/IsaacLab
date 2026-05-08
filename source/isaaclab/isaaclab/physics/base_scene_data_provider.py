# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider interface for visualizers and renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSceneDataProvider(ABC):
    """Backend-agnostic scene data provider interface."""

    def set_interactive_scene(self, scene: Any) -> None:
        """Attach the interactive scene so visualizers can discover scene-owned sensors."""
        self._interactive_scene = scene

    def get_interactive_scene(self) -> Any | None:
        """Return the registered interactive scene, if available."""
        return getattr(self, "_interactive_scene", None)

    def get_camera_sensors(self) -> dict[str, Any]:
        """Return Isaac Lab camera sensors keyed by scene sensor name."""
        scene = getattr(self, "_interactive_scene", None)
        if scene is None:
            return {}
        try:
            from isaaclab.sensors.camera import Camera
        except ImportError:
            return {}
        return {name: sensor for name, sensor in scene.sensors.items() if isinstance(sensor, Camera)}

    @abstractmethod
    def update(self) -> None:
        """Refresh any cached scene data (full model/state)."""
        raise NotImplementedError

    @abstractmethod
    def get_newton_model(self) -> Any | None:
        """Return Newton model handle when available."""
        raise NotImplementedError

    @abstractmethod
    def get_newton_state(self) -> Any | None:
        """Return Newton state handle when available (full state)."""
        raise NotImplementedError

    @abstractmethod
    def get_usd_stage(self) -> Any | None:
        """Return USD stage handle when available."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return backend metadata (num_envs, gravity, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def get_transforms(self) -> dict[str, Any] | None:
        """Return body transforms, if supported."""
        raise NotImplementedError

    @abstractmethod
    def get_velocities(self) -> dict[str, Any] | None:
        """Return body velocities, if supported."""
        raise NotImplementedError

    @abstractmethod
    def get_contacts(self) -> dict[str, Any] | None:
        """Return contacts, if supported."""
        raise NotImplementedError

    @abstractmethod
    def get_camera_transforms(self) -> dict[str, Any] | None:
        """Return per-camera, per-env transforms, if supported."""
        raise NotImplementedError
