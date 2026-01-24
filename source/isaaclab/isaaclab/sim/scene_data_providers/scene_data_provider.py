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
    """Base interface for scene data providers."""

    @abstractmethod
    def update(self) -> None:
        """Sync any per-step data needed by visualizers."""

    @abstractmethod
    def get_model(self):
        """Return a physics-backend model object, if applicable."""

    @abstractmethod
    def get_state(self):
        """Return a physics-backend state object, if applicable."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return basic metadata about the scene/backend."""

    @abstractmethod
    def get_transforms(self) -> dict[str, Any] | None:
        """Return transform data keyed by semantic names."""

    def get_velocities(self) -> dict[str, Any] | None:
        """Return velocity data keyed by semantic names."""
        # TODO: Populate linear/angular velocities once available per backend.
        return None

    def get_contacts(self) -> dict[str, Any] | None:
        """Return contact data keyed by semantic names."""
        # TODO: Populate contact data once available per backend.
        return None

    def get_meshes(self) -> dict[str, Any] | None:
        """Return mesh/material data keyed by semantic names."""
        # TODO: Populate mesh/material data once available per backend.
        return None


class SceneDataProvider:
    """Facade that selects the correct scene data provider for the active backend."""

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
                logger.warning(
                    "Omni scene data provider requires stage and simulation context; skipping initialization."
                )
                self._provider = None
            else:
                from .ov_scene_data_provider import OmniSceneDataProvider

                self._provider = OmniSceneDataProvider(visualizer_cfgs, stage, simulation_context)
        else:
            logger.warning(f"Unknown physics backend '{backend}'. No scene data provider created.")

    def update(self) -> None:
        if self._provider is not None:
            self._provider.update()

    def get_model(self):
        if self._provider is None:
            return None
        return self._provider.get_model()

    def get_state(self):
        if self._provider is None:
            return None
        return self._provider.get_state()

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

    def get_meshes(self) -> dict[str, Any] | None:
        if self._provider is None:
            return None
        return self._provider.get_meshes()
