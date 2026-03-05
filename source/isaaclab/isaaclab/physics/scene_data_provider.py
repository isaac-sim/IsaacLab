# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory for creating scene data provider instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_scene_data_provider import BaseSceneDataProvider

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


class SceneDataProvider(FactoryBase, BaseSceneDataProvider):
    """Factory for creating scene data provider instances."""

    _backend_class_names = {"physx": "PhysxSceneDataProvider", "newton": "NewtonSceneDataProvider"}

    @classmethod
    def _get_backend(cls, visualizer_cfgs, stage, simulation_context: SimulationContext, *args, **kwargs) -> str:
        manager_name = simulation_context.physics_manager.__name__.lower()
        if "newton" in manager_name:
            return "newton"
        if "physx" in manager_name:
            return "physx"
        raise ValueError(f"Unknown physics manager: {manager_name}")

    @classmethod
    def _get_module_name(cls, backend: str) -> str:
        return f"isaaclab_{backend}.scene_data_providers"

    def __new__(
        cls, visualizer_cfgs, stage, simulation_context: SimulationContext, *args, **kwargs
    ) -> BaseSceneDataProvider:
        """Create a new scene data provider based on the active physics backend."""
        result = super().__new__(cls, visualizer_cfgs, stage, simulation_context, *args, **kwargs)
        if not isinstance(result, BaseSceneDataProvider):
            name = type(result).__name__
            raise TypeError(f"Backend scene data provider {name!r} must inherit from BaseSceneDataProvider.")
        return result
