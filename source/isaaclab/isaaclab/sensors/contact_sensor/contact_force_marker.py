# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared contact-force visualization."""

from __future__ import annotations

import torch

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class ContactForceVisualizer:
    """Visualize contact-force vectors with arrow markers."""

    def __init__(
        self,
        cfg: VisualizationMarkersCfg,
        force_scale: float,
        tail_offset_ratio: float = 0.25,
    ):
        """Initialize the force visualizer.

        Args:
            cfg: Arrow marker configuration.
            force_scale: Arrow length per force magnitude [m/N].
            tail_offset_ratio: Offset from the marker pivot to its tail, as a fraction of its displayed length.

        Raises:
            ValueError: If the arrow marker has no three-dimensional scale.
        """
        marker_scale = getattr(cfg.markers.get("arrow"), "scale", None)
        if marker_scale is None or len(marker_scale) != 3:
            raise ValueError("Contact-force visualization requires an 'arrow' marker with a three-dimensional scale.")

        self._visualizer = VisualizationMarkers(cfg)
        self._force_scale = force_scale
        self._prototype_length = marker_scale[2]
        self._tail_offset_ratio = tail_offset_ratio

    def set_visibility(self, visible: bool) -> None:
        """Set arrow marker visibility."""
        self._visualizer.set_visibility(visible)

    def visualize(self, positions: torch.Tensor, forces: torch.Tensor, force_threshold: float) -> None:
        """Visualize force vectors from the requested tail positions.

        Args:
            positions: Desired arrow tail positions [m], shape ``(..., 3)``.
            forces: Force vectors [N] in world frame, shape ``(..., 3)``.
            force_threshold: Minimum force magnitude to visualize [N].
        """
        marker_positions = positions.reshape(-1, 3).clone()
        forces = forces.reshape(-1, 3)
        magnitudes = torch.linalg.norm(forces, dim=-1)
        valid = magnitudes > force_threshold

        force_directions = torch.zeros_like(forces)
        force_directions[valid] = forces[valid] / magnitudes[valid].unsqueeze(-1)

        orientations = torch.zeros((forces.shape[0], 4), device=forces.device, dtype=forces.dtype)
        orientations[:, 3] = 1.0
        if valid.any():
            local_x = torch.zeros_like(force_directions[valid])
            local_x[:, 0] = 1.0
            quaternion_xyz = torch.cross(local_x, force_directions[valid], dim=-1)
            quaternion_w = 1.0 + force_directions[valid, 0:1]
            quaternions = torch.cat((quaternion_xyz, quaternion_w), dim=-1)

            opposite = force_directions[valid, 0] < -1.0 + 1.0e-6
            quaternions[opposite] = quaternions.new_tensor((0.0, 0.0, 1.0, 0.0))
            orientations[valid] = torch.nn.functional.normalize(quaternions, dim=-1)

        scales = torch.ones_like(forces)
        scales[:, 0] = magnitudes * self._force_scale
        scales[~valid] = 0.0

        displayed_lengths = self._prototype_length * scales[:, 0]
        marker_positions[valid] += (
            self._tail_offset_ratio * displayed_lengths[valid].unsqueeze(-1) * force_directions[valid]
        )
        self._visualizer.visualize(marker_positions, orientations, scales)
