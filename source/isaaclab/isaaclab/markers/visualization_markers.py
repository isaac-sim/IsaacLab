# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic visualization marker facade."""

from __future__ import annotations

import logging

import numpy as np
import torch

import isaaclab.sim as sim_utils

from .visualization_markers_cfg import VisualizationMarkersCfg

logger = logging.getLogger(__name__)


class VisualizationMarkers:
    """Coordinate groups of visual markers across active visualizer backends.

    The public API intentionally remains the historical marker API:
    :meth:`set_visibility`, :meth:`is_visible`, and :meth:`visualize`. Backend
    details are delegated to Kit and Newton marker implementations.
    """

    def __init__(self, cfg: VisualizationMarkersCfg):
        """Initialize visualization marker backends from the active simulation context."""
        if len(cfg.markers) == 0:
            raise ValueError(f"The `cfg.markers` cannot be empty. Received: {cfg.markers}")

        self.cfg = cfg
        self.prim_path = cfg.prim_path
        self._count = len(cfg.markers)
        self._is_visible = True
        self._backends: list[object] = []
        self._ensure_backends_initialized()

    def __str__(self) -> str:
        """Return a string representation of the marker group."""
        msg = f"VisualizationMarkers(prim_path={self.prim_path})"
        msg += f"\n\tCount: {self.count}"
        msg += f"\n\tNumber of prototypes: {self.num_prototypes}"
        msg += "\n\tMarkers Prototypes:"
        for index, (name, marker) in enumerate(self.cfg.markers.items()):
            msg += f"\n\t\t[Index: {index}]: {name}: {marker.to_dict()}"
        return msg

    @property
    def num_prototypes(self) -> int:
        """The number of marker prototypes available."""
        return len(self.cfg.markers)

    @property
    def count(self) -> int:
        """The total number of marker instances."""
        return self._count

    def set_visibility(self, visible: bool):
        """Set marker visibility for all initialized backends."""
        self._is_visible = visible
        self._ensure_backends_initialized()
        for backend in self._backends:
            backend.set_visibility(visible)

    def is_visible(self) -> bool:
        """Return whether the marker group is visible."""
        if self._backends:
            return any(backend.is_visible() for backend in self._backends)
        return self._is_visible

    def visualize(
        self,
        translations: np.ndarray | torch.Tensor | None = None,
        orientations: np.ndarray | torch.Tensor | None = None,
        scales: np.ndarray | torch.Tensor | None = None,
        marker_indices: list[int] | np.ndarray | torch.Tensor | None = None,
    ):
        """Update markers in all initialized visualizer backends."""
        self._ensure_backends_initialized()
        if not self.is_visible():
            return

        norm_translations = self._to_tensor(translations, expected_width=3, name="translations")
        norm_orientations = self._to_tensor(orientations, expected_width=4, name="orientations")
        norm_scales = self._to_tensor(scales, expected_width=3, name="scales")
        norm_marker_indices = self._to_index_tensor(marker_indices)
        target_device = self._resolve_target_device(
            norm_translations, norm_orientations, norm_scales, norm_marker_indices
        )
        if norm_translations is not None:
            norm_translations = norm_translations.to(device=target_device)
        if norm_orientations is not None:
            norm_orientations = norm_orientations.to(device=target_device)
        if norm_scales is not None:
            norm_scales = norm_scales.to(device=target_device)
        if norm_marker_indices is not None:
            norm_marker_indices = norm_marker_indices.to(device=target_device)

        num_markers = 0
        for value in (norm_translations, norm_orientations, norm_scales, norm_marker_indices):
            if value is not None:
                num_markers = value.shape[0]

        if norm_marker_indices is None and num_markers != 0 and num_markers != self._count:
            norm_marker_indices = torch.zeros(num_markers, dtype=torch.int32, device=target_device)
        elif norm_marker_indices is None and num_markers == 0:
            if all(value is None for value in (norm_translations, norm_orientations, norm_scales)):
                raise ValueError("Number of markers cannot be zero! Hint: The function was called with no inputs?")
            num_markers = self._count

        for backend in self._backends:
            backend.visualize(norm_translations, norm_orientations, norm_scales, norm_marker_indices)

        if num_markers != 0:
            self._count = num_markers

    def __del__(self):
        for backend in getattr(self, "_backends", []):
            if hasattr(backend, "close"):
                backend.close()

    def _ensure_backends_initialized(self) -> None:
        sim = sim_utils.SimulationContext.instance()
        if sim is None:
            from .kit_visualization_markers import KitVisualizationMarkers

            if not any(isinstance(backend, KitVisualizationMarkers) for backend in self._backends):
                self._backends.append(KitVisualizationMarkers(self.cfg, visible=self._is_visible))
            return

        has_kit_marker_backend = any(
            viz.supports_markers() and viz.pumps_app_update() and viz.cfg.enable_markers for viz in sim.visualizers
        )
        has_newton_marker_backend = any(
            viz.supports_markers() and not viz.pumps_app_update() and viz.cfg.enable_markers for viz in sim.visualizers
        )

        if has_kit_marker_backend:
            from .kit_visualization_markers import KitVisualizationMarkers

            if not any(isinstance(backend, KitVisualizationMarkers) for backend in self._backends):
                self._backends.append(KitVisualizationMarkers(self.cfg, visible=self._is_visible))
        if has_newton_marker_backend:
            from .newton_visualization_markers import NewtonVisualizationMarkers

            if not any(isinstance(backend, NewtonVisualizationMarkers) for backend in self._backends):
                self._backends.append(NewtonVisualizationMarkers(self.cfg, visible=self._is_visible))

    def _resolve_target_device(self, *values: torch.Tensor | None) -> torch.device:
        for value in values:
            if value is not None:
                return value.device
        for backend in self._backends:
            if hasattr(backend, "infer_device"):
                return backend.infer_device()
        return torch.device("cpu")

    @staticmethod
    def _to_tensor(
        value: np.ndarray | torch.Tensor | None,
        expected_width: int,
        name: str,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        else:
            tensor = value.detach()
        if tensor.ndim != 2 or tensor.shape[1] != expected_width:
            raise ValueError(f"Expected `{name}` to have shape (M, {expected_width}). Received: {tuple(tensor.shape)}.")
        return tensor.to(dtype=torch.float32)

    @staticmethod
    def _to_index_tensor(value: list[int] | np.ndarray | torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, list):
            tensor = torch.tensor(value)
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        else:
            tensor = value.detach()
        if tensor.ndim != 1:
            raise ValueError(f"Expected `marker_indices` to have shape (M,). Received: {tuple(tensor.shape)}.")
        return tensor.to(dtype=torch.int32)
