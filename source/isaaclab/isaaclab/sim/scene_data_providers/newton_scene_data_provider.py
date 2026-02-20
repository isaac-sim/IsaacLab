# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider for Newton physics backend."""

from __future__ import annotations

from typing import Any


class NewtonSceneDataProvider:
    """Scene data provider for Newton physics backend.

    Provides access to Newton model, state, and USD stage for visualizers and renderers.
    """

    def __init__(self, visualizer_cfgs: list[Any] | None, simulation_context) -> None:
        self._simulation_context = simulation_context
        self._metadata = {"physics_backend": "newton"}

    def update(self, env_ids: list[int] | None = None) -> None:
        """Refresh any cached scene data.

        For Newton backend, this is a no-op as the model and state are managed by NewtonManager.
        """
        pass

    def get_newton_model(self) -> Any | None:
        """Return Newton model from NewtonManager."""
        from isaaclab_newton.physics import NewtonManager

        return NewtonManager.get_model()

    def get_newton_state(self, env_ids: list[int] | None = None) -> Any | None:
        """Return Newton state from NewtonManager.

        Args:
            env_ids: Optional list of environment IDs. Currently not supported for filtering.
                    Returns the full state for all environments.

        Returns:
            The current Newton state (state_0) from NewtonManager.
        """
        from isaaclab_newton.physics import NewtonManager

        # For now, return state_0 (current state) for all environments
        # TODO: Implement env_ids filtering if needed
        return NewtonManager.get_state_0()

    def get_model(self) -> Any | None:
        """Return Newton model (alias for get_newton_model for visualizer compatibility)."""
        return self.get_newton_model()

    def get_state(self, env_ids: list[int] | None = None) -> Any | None:
        """Return Newton state (alias for get_newton_state for visualizer compatibility)."""
        return self.get_newton_state(env_ids)

    def get_usd_stage(self) -> Any | None:
        """Return the USD stage handle from simulation context."""
        return getattr(self._simulation_context, "stage", None)

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_transforms(self) -> dict[str, Any] | None:
        """TODO: implement when the newton physics backend is added."""
        pass

    def get_velocities(self) -> dict[str, Any] | None:
        """TODO: implement when the newton physics backend is added."""
        pass

    def get_contacts(self) -> dict[str, Any] | None:
        """TODO: implement when the newton physics backend is added."""
        pass

    def get_camera_transforms(self) -> dict[str, Any] | None:
        """TODO: implement when the newton physics backend is added."""
        pass
