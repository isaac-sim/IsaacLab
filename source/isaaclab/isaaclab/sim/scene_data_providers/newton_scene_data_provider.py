#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed scene data provider."""

from __future__ import annotations

import logging
from typing import Any

from .scene_data_provider import SceneDataProviderBase

logger = logging.getLogger(__name__)


class NewtonSceneDataProvider(SceneDataProviderBase):
    """Newton-backed scene data provider (when Newton physics is the active backend)."""

    def __init__(self, visualizer_cfgs: list[Any] | None) -> None:
        self._has_newton_visualizer = False
        self._has_rerun_visualizer = False
        self._has_ov_visualizer = False
        self._metadata: dict[str, Any] = {}

        if visualizer_cfgs:
            for cfg in visualizer_cfgs:
                viz_type = getattr(cfg, "visualizer_type", None)
                if viz_type == "newton":
                    self._has_newton_visualizer = True
                elif viz_type == "rerun":
                    self._has_rerun_visualizer = True
                elif viz_type == "omniverse":
                    self._has_ov_visualizer = True

        # Lazy import to keep develop usable without Newton installed.
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            self._metadata = {
                "physics_backend": "newton",
                "num_envs": NewtonManager._num_envs if NewtonManager._num_envs is not None else 0,
                "gravity_vector": NewtonManager._gravity_vector,
                "clone_physics_only": NewtonManager._clone_physics_only,
            }
        except Exception:
            self._metadata = {"physics_backend": "newton"}

    def update(self) -> None:
        return None

    def get_model(self):
        if not (self._has_newton_visualizer or self._has_rerun_visualizer):
            return None
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            return NewtonManager._model
        except Exception:
            return None

    def get_state(self):
        if not (self._has_newton_visualizer or self._has_rerun_visualizer):
            return None
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            return NewtonManager._state_0
        except Exception:
            return None

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_transforms(self) -> dict[str, Any] | None:
        return None

    def get_velocities(self) -> dict[str, Any] | None:
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            state = NewtonManager._state_0
            if state is None:
                return None
            return {"body_qd": state.body_qd}
        except Exception:
            return None

    def get_contacts(self) -> dict[str, Any] | None:
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager

            if NewtonManager._contacts is None:
                return None
            return {"contacts": NewtonManager._contacts}
        except Exception:
            return None
