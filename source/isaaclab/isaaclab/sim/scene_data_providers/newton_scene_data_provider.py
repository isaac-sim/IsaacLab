# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TODO: implement when the newton physics backend is added."""

from __future__ import annotations

from typing import Any


class NewtonSceneDataProvider:
    """TODO: implement when the newton physics backend is added."""

    def __init__(self, visualizer_cfgs: list[Any] | None) -> None:
        self._metadata = {"physics_backend": "newton"}

    def update(self, env_ids: list[int] | None = None) -> None:
        """TODO: implement when the newton physics backend is added."""
        pass

    def get_newton_model(self) -> Any | None:
        """TODO: implement when the newton physics backend is added."""
        return None

    def get_newton_state(self, env_ids: list[int] | None = None) -> Any | None:
        """TODO: implement when the newton physics backend is added."""
        return None

    def get_usd_stage(self) -> Any | None:
        """TODO: implement when the newton physics backend is added."""
        return None

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_transforms(self) -> dict[str, Any] | None:
        """TODO: implement when the newton physics backend is added."""
        return None

    def get_velocities(self) -> dict[str, Any] | None:
        """TODO: implement when the newton physics backend is added."""
        return None

    def get_contacts(self) -> dict[str, Any] | None:
        """TODO: implement when the newton physics backend is added."""
        return None

    def get_camera_transforms(self) -> dict[str, Any] | None:
        """TODO: implement when the newton physics backend is added."""
        return None
