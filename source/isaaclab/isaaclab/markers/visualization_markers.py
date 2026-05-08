# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic facade for coordinating groups of visual markers.

The :class:`VisualizationMarkers` class is used to create a group of visual
markers and visualize them through the active visualizer backends. The marker
prototypes are configured with :class:`VisualizationMarkersCfg`, and individual
marker instances can be updated by passing prototype indices and their
translations, orientations, and scales.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

import isaaclab.sim as sim_utils

from .visualization_markers_cfg import VisualizationMarkersCfg

logger = logging.getLogger(__name__)


class VisualizationMarkers:
    """Coordinate groups of visual markers across active visualizer backends.

    This class allows visualization of different UI markers in the scene, such
    as points, frames, arrows, and shapes. Marker prototypes are reusable
    templates that define variations of objects to visualize. For example, a
    sphere marker prototype can be used to create many sphere marker instances
    at different locations.

    The class parses the configuration to create the marker prototypes in each
    active backend. The marker prototype name comes from the key in the
    :attr:`VisualizationMarkersCfg.markers` dictionary, and prototype indices
    are based on the dictionary order. For example, if the dictionary has two
    markers, ``"marker1"`` and ``"marker2"``, their prototype indices are 0
    and 1 respectively. These indices can be passed to :meth:`visualize` as a
    list or array of integers.

    Switching between marker prototypes is possible by calling
    :meth:`visualize` with the corresponding prototype indices. The marker
    transforms are updated only for the arguments that are provided; omitted
    translations, orientations, scales, or marker indices are left unchanged
    when supported by the active backend.

    Usage:
        The following snippet creates 24 sphere markers at random translations.
        The first 12 markers use the first prototype and the rest use the
        second prototype.

        .. code-block:: python

            import numpy as np

            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/testMarkers",
                markers={
                    "marker1": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "marker2": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )

            marker = VisualizationMarkers(cfg)
            marker_translations = np.random.uniform(-1.0, 1.0, (24, 3))

            # This creates 24 markers using the first prototype because marker
            # indices are not given.
            marker.visualize(translations=marker_translations)

            # 0 -> marker1, 1 -> marker2. Since translations are omitted here,
            # only the marker prototypes are changed.
            marker_indices = [0] * 12 + [1] * 12
            marker.visualize(marker_indices=marker_indices)

            # Update both marker prototypes and translations.
            marker.visualize(marker_indices=marker_indices, translations=marker_translations)

    The public API intentionally remains the historical marker API:
    :meth:`set_visibility`, :meth:`is_visible`, and :meth:`visualize`. Backend
    details are delegated to Kit and Newton marker implementations.
    """

    def __init__(self, cfg: VisualizationMarkersCfg):
        """Initialize visualization marker backends from the active simulation context.

        Args:
            cfg: The configuration for the markers.

        Raises:
            ValueError: When no markers are provided in the :obj:`cfg`.
        """
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
        """Update markers in all initialized visualizer backends.

        .. note::
            If the markers are hidden, the function returns without updating
            backend marker state. This avoids unnecessary work while debug
            visualization is disabled.

        Whenever updating the markers, the input arrays must have the same
        number of elements in the first dimension. Backends generally require
        all per-marker arrays to describe the same number of marker instances.

        The function supports dynamic updates of the marker count. For example,
        if you have 24 points to visualize, you can pass 24 translations,
        orientations, and scales. If you later want to visualize only 12
        points, you can pass arrays with 12 rows and the backends will update
        the number of marker instances.

        The function also updates marker prototypes based on prototype indices.
        For instance, if there are two marker prototypes and you pass marker
        indices ``[0, 1, 0, 1]``, the first and third markers use the first
        prototype and the second and fourth markers use the second prototype.

        .. caution::
            This function updates all markers instanced from the prototypes. If
            you want to update only a subset of markers, handle the indexing
            externally and pass complete arrays to this function.

        Args:
            translations: Translations w.r.t. parent prim frame. Shape is
                (M, 3). Defaults to None, which means left unchanged.
            orientations: Quaternion orientations (x, y, z, w) w.r.t. parent
                prim frame. Shape is (M, 4). Defaults to None, which means left
                unchanged.
            scales: Scale applied before any rotation is applied. Shape is
                (M, 3). Defaults to None, which means left unchanged.
            marker_indices: Decides which marker prototype to visualize. Shape
                is (M). Defaults to None, which means left unchanged provided
                that the total number of markers is the same as the previous
                call. If the number of markers is different, the function will
                update the number of markers.

        Raises:
            ValueError: When input arrays do not follow the expected shapes.
            ValueError: When the function is called with all None arguments.
        """
        self._ensure_backends_initialized()
        # If markers are hidden, do not spend time normalizing or dispatching
        # marker state to the active backends.
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
            self._ensure_kit_backend()
            return

        if any(viz.supports_markers() and viz.pumps_app_update() and viz.cfg.enable_markers for viz in sim.visualizers):
            self._ensure_kit_backend()
        if any(
            viz.supports_markers() and not viz.pumps_app_update() and viz.cfg.enable_markers for viz in sim.visualizers
        ):
            self._ensure_newton_backend()

    def _ensure_kit_backend(self) -> None:
        """Create the Kit marker backend if it is not already active."""
        from isaaclab_visualizers.kit.kit_visualization_markers import KitVisualizationMarkers

        if not any(isinstance(backend, KitVisualizationMarkers) for backend in self._backends):
            self._backends.append(KitVisualizationMarkers(self.cfg, visible=self._is_visible))

    def _ensure_newton_backend(self) -> None:
        """Create the Newton-family marker backend if it is not already active."""
        from isaaclab_visualizers.newton.newton_visualization_markers import NewtonVisualizationMarkers

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
