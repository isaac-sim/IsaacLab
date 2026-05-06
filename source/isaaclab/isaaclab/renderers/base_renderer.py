# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for renderer implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .camera_render_spec import CameraRenderSpec
from .output_contract import RenderBufferKind, RenderBufferSpec

if TYPE_CHECKING:
    import torch

    from isaaclab.sensors.camera.camera_data import CameraData


class BaseRenderer(ABC):
    """Abstract base class for renderer implementations."""

    @abstractmethod
    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Per-output layout (channels + dtype) this renderer can produce.

        Outputs absent from the mapping are not produced by this backend.

        Returns:
            Mapping from supported :class:`RenderBufferKind` to its :class:`RenderBufferSpec`.
        """
        pass

    @abstractmethod
    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Prepare the stage for rendering before :meth:`create_render_data` is called.

        Some renderers need to export or preprocess the USD stage before
        creating render data. This method is called after the renderer is
        instantiated and before :meth:`create_render_data`.

        Args:
            stage: USD stage to prepare, or None if not applicable.
            num_envs: Number of environments.
        """
        pass

    @abstractmethod
    def create_render_data(self, spec: CameraRenderSpec) -> Any:
        """Create render data for the given camera :class:`CameraRenderSpec`.

        Args:
            spec: Immutable description of the tiled camera (paths, config, device).

        Returns:
            Renderer-specific data for subsequent :meth:`render` / :meth:`read_output` calls.
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
    def read_output(self, render_data: Any, camera_data: CameraData) -> None:
        """Read rendered outputs from the renderer into the camera data container.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            camera_data: The :class:`~isaaclab.sensors.camera.camera_data.CameraData`
                instance to populate.
        """
        pass

    @abstractmethod
    def cleanup(self, render_data: Any) -> None:
        """Release renderer resources associated with the given render data.

        Args:
            render_data: The render data object to clean up, or ``None``.
        """
        pass
