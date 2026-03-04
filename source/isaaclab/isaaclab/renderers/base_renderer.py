# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for renderer implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from isaaclab.sensors import SensorBase


class BaseRenderer(ABC):
    """Abstract base class for renderer implementations."""

    @abstractmethod
    def create_render_data(self, sensor: SensorBase) -> Any:
        """Create render data for the given sensor.

        The returned object is opaque to the interface: callers pass it to other
        renderer methods without inspecting its contents. Its structure is
        implementation-specific (each renderer defines its own type).

        Args:
            sensor: The camera sensor to create render data for.

        Returns:
            Renderer-specific data object holding resources needed for rendering.
            Passed to subsequent render calls.
        """
        pass

    @abstractmethod
    def set_outputs(self, render_data: Any, output_data: dict[str, torch.Tensor]) -> None:
        """Store reference to output buffers for writing during render.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            output_data: Dictionary mapping output names (e.g. ``"rgb"``, ``"depth"``)
                to pre-allocated tensors where rendered data will be written.
        """
        pass

    @abstractmethod
    def update_transforms(self) -> None:
        """Update scene transforms before rendering.

        Called to sync physics/asset state into the renderer's scene representation.
        """
        pass

    @abstractmethod
    def update_camera(
        self, render_data: Any, positions: torch.Tensor, orientations: torch.Tensor, intrinsics: torch.Tensor
    ) -> None:
        """Update camera poses and intrinsics for the next render.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            positions: Camera positions in world frame, shape ``(N, 3)``.
            orientations: Camera orientations as quaternions (x, y, z, w), shape ``(N, 4)``.
            intrinsics: Camera intrinsic matrices, shape ``(N, 3, 3)``.
        """
        pass

    @abstractmethod
    def render(self, render_data: Any) -> None:
        """Perform rendering and write to output buffers.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
        """
        pass

    @abstractmethod
    def write_output(self, render_data: Any, output_name: str, output_data: torch.Tensor) -> None:
        """Write a specific output type to the given buffer.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            output_name: Name of the output (e.g. ``"rgba"``, ``"depth"``).
            output_data: Pre-allocated tensor to write the output into.
        """
        pass

    @abstractmethod
    def cleanup(self, render_data: Any) -> None:
        """Release renderer resources associated with the given render data.

        Args:
            render_data: The render data object to clean up, or ``None``.
        """
        pass
