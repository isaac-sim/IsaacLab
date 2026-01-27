#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider abstraction for visualizers and renderers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class SceneDataProviderBase(ABC):
    """Base interface for scene data providers.
    
    Provides simulation data in multiple formats for visualizers, renderers,
    and other downstream consumers. Each backend provider implements this interface,
    exposing native data cheaply and adapted data when needed.
    """

    @abstractmethod
    def update(self) -> None:
        """Update adapted data for current simulation step."""

    @abstractmethod
    def get_newton_model(self) -> Any | None:
        """Get Newton Model."""

    @abstractmethod
    def get_newton_state(self) -> Any | None:
        """Get Newton State."""

    @abstractmethod
    def get_usd_stage(self) -> Any | None:
        """Get USD stage."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Get provider metadata and performance hints."""

    @abstractmethod
    def get_transforms(self) -> dict[str, Any] | None:
        """Get world-space transforms in backend-agnostic format."""

    def get_velocities(self) -> dict[str, Any] | None:
        """Get velocities in backend-agnostic format."""
        return None

    def get_contacts(self) -> dict[str, Any] | None:
        """Get contact data in backend-agnostic format."""
        return None

    def get_mesh_data(self) -> dict[str, Any] | None:
        """Get mesh geometry and materials."""
        return None


class SceneDataProvider:
    """Facade that creates appropriate provider based on physics backend."""

    def __init__(
        self,
        backend: str,
        visualizer_cfgs: list[Any] | None,
        stage=None,
        simulation_context=None,
    ) -> None:
        self._backend = backend
        self._provider: SceneDataProviderBase | None = None

        if backend == "newton":
            from .newton_scene_data_provider import NewtonSceneDataProvider

            self._provider = NewtonSceneDataProvider(visualizer_cfgs)
        elif backend == "omni":
            if stage is None or simulation_context is None:
                logger.warning("OV scene data provider requires stage and simulation context.")
                self._provider = None
            else:
                from .ov_scene_data_provider import OVSceneDataProvider

                self._provider = OVSceneDataProvider(visualizer_cfgs, stage, simulation_context)
        else:
            logger.warning(f"Unknown physics backend '{backend}'.")

    def update(self) -> None:
        if self._provider is not None:
            self._provider.update()

    def get_newton_model(self) -> Any | None:
        if self._provider is None:
            return None
        return self._provider.get_newton_model()

    def get_newton_state(self) -> Any | None:
        if self._provider is None:
            return None
        return self._provider.get_newton_state()

    def get_usd_stage(self) -> Any | None:
        if self._provider is None:
            return None
        return self._provider.get_usd_stage()

    def get_metadata(self) -> dict[str, Any]:
        if self._provider is None:
            return {}
        return self._provider.get_metadata()

    def get_transforms(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_transforms()

    def get_velocities(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_velocities()

    def get_contacts(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_contacts()

    def get_mesh_data(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_mesh_data()