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
    """Scene data provider for Newton Warp physics backend.
    
    Native (cheap): Newton Model/State from NewtonManager
    Adapted (future): USD stage (would need Newton→USD sync for OV visualizer)
    """

    def __init__(self, visualizer_cfgs: list[Any] | None) -> None:
        self._has_ov_visualizer = False
        self._metadata: dict[str, Any] = {}

        if visualizer_cfgs:
            for cfg in visualizer_cfgs:
                if getattr(cfg, "visualizer_type", None) == "omniverse":
                    self._has_ov_visualizer = True

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
        """No-op for Newton backend (state updated by Newton solver)."""
        pass

    def get_newton_model(self) -> Any | None:
        """NATIVE: Newton Model from NewtonManager."""
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager
            return NewtonManager._model
        except Exception:
            return None

    def get_newton_state(self) -> Any | None:
        """NATIVE: Newton State from NewtonManager."""
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager
            return NewtonManager._state_0
        except Exception:
            return None

    def get_usd_stage(self) -> None:
        """UNAVAILABLE: Newton backend doesn't provide USD (future: Newton→USD sync)."""
        return None

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_transforms(self) -> dict[str, Any] | None:
        """Extract transforms from Newton state (future work)."""
        return None

    def get_velocities(self) -> dict[str, Any] | None:
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager
            if NewtonManager._state_0 is None:
                return None
            return {"body_qd": NewtonManager._state_0.body_qd}
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

    def get_mesh_data(self) -> dict[str, Any] | None:
        """ADAPTED: Extract mesh data from Newton shapes (future work)."""
        return None