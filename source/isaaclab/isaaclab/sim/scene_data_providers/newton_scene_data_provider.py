# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed scene data provider (stub)."""

from __future__ import annotations

from typing import Any


class NewtonSceneDataProvider:
    """Scene data provider for Newton Warp physics backend.

    This stub exists to keep the interface stable while the Newton provider is
    implemented. All optional data accessors return None.
    """

    def __init__(self, visualizer_cfgs: list[Any] | None) -> None:
        self._metadata = {"physics_backend": "newton"}

    def update(self, env_ids: list[int] | None = None) -> None:
        """No-op for Newton backend (stub)."""
        pass

    def get_newton_model(self) -> Any | None:
        """Return Newton model handle when available."""
        return None

    def get_newton_state(self, env_ids: list[int] | None = None) -> Any | None:
        """Return Newton state handle when available. env_ids ignored in stub."""
        return None

    def get_usd_stage(self) -> Any | None:
        """UNAVAILABLE: Newton backend doesn't provide USD stage."""
        return None

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_transforms(self) -> dict[str, Any] | None:
        """Return body transforms when available."""
        return None

    def get_velocities(self) -> dict[str, Any] | None:
        """Return body velocities when available."""
        return None

    def get_contacts(self) -> dict[str, Any] | None:
        """Return contacts when available."""
        return None

    def get_camera_transforms(self) -> dict[str, Any] | None:
        """Return camera transforms when available."""
        return None
