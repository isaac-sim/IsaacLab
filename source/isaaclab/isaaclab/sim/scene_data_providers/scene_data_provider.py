# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider interface for visualizers and renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SceneDataProvider(ABC):
    """Backend-agnostic scene data provider interface."""

    @abstractmethod
    def update(self, env_ids: list[int] | None = None) -> None:
        """Refresh any cached scene data."""
        raise NotImplementedError

    @abstractmethod
    def get_newton_model(self) -> Any | None:
        """Return Newton model handle when available."""
        raise NotImplementedError

    @abstractmethod
    def get_newton_state(self, env_ids: list[int] | None = None) -> Any | None:
        """Return Newton state handle when available."""
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
