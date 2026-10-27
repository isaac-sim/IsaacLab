# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated: ViewportCameraController compatibility shim.

Camera tracking is now built into :class:`~isaaclab_visualizers.kit.KitVisualizer`.
Declare a :class:`~isaaclab.visualizers.PerspectiveCameraCfg` with ``env_index`` or ``track_path`` in the
``cameras`` of :class:`~isaaclab.visualizers.VisualizerCfg`, and add it to
:attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .. import ViewerCfg


def _warn_method(name: str) -> None:
    warnings.warn(
        f"ViewportCameraController.{name}() is deprecated. "
        "Declare a PerspectiveCameraCfg with env_index or track_path in VisualizerCfg.cameras, and call "
        "visualizer.follow() or visualizer.unfollow() to change it at runtime.",
        DeprecationWarning,
        stacklevel=3,
    )


class ViewportCameraController:
    """Deprecated compatibility shim for :class:`ViewportCameraController`.

    .. deprecated::
        :class:`ViewportCameraController` has been removed. Camera tracking is now built into
        :class:`~isaaclab_visualizers.kit.KitVisualizer`. Declare a
        :class:`~isaaclab.visualizers.PerspectiveCameraCfg` with ``env_index`` or ``track_path`` in
        :class:`~isaaclab.visualizers.VisualizerCfg` ``cameras`` and pass it to
        :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs`.
    """

    def __init__(self, env: object, cfg: ViewerCfg):
        warnings.warn(
            "ViewportCameraController is deprecated and has been removed. "
            "Camera tracking is now built into KitVisualizer: declare a PerspectiveCameraCfg with "
            "env_index or track_path in VisualizerCfg.cameras and add it to SimulationCfg.visualizer_cfgs.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._env = env
        self._cfg = cfg
        self.default_cam_eye = np.array(getattr(cfg, "eye", (7.5, 7.5, 7.5)), dtype=float)
        self.default_cam_lookat = np.array(getattr(cfg, "lookat", (0.0, 0.0, 0.0)), dtype=float)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cfg(self) -> ViewerCfg:
        return self._cfg

    # ------------------------------------------------------------------
    # Public methods (deprecated forwarding stubs)
    # ------------------------------------------------------------------

    def set_view_env_index(self, env_index: int) -> None:
        """Deprecated. Call :meth:`~isaaclab_visualizers.kit.KitVisualizer.set_follow_env` instead."""
        _warn_method("set_view_env_index")
        num_envs = getattr(self._env, "num_envs", None)
        if num_envs is not None and not (0 <= env_index < num_envs):
            raise ValueError(
                f"Out of range value for attribute 'env_index': {env_index}."
                f" Expected a value between 0 and {num_envs - 1}."
            )
        self._cfg.env_index = env_index
        for viz in self._get_kit_visualizers():
            viz.set_follow_env(env_index)

    def update_view_to_world(self) -> None:
        """Deprecated. Call :meth:`~isaaclab_visualizers.kit.KitVisualizer.unfollow` instead."""
        _warn_method("update_view_to_world")
        self._cfg.origin_type = "world"
        for viz in self._get_kit_visualizers():
            viz.unfollow()
            viz.set_view_offsets(tuple(self.default_cam_eye.tolist()), tuple(self.default_cam_lookat.tolist()))

    def update_view_to_env(self) -> None:
        """Deprecated. Call :meth:`~isaaclab_visualizers.kit.KitVisualizer.follow` instead."""
        _warn_method("update_view_to_env")
        self._cfg.origin_type = "env"
        for viz in self._get_kit_visualizers():
            viz.follow(None)

    def update_view_to_asset_root(self, asset_name: str) -> None:
        """Deprecated. Call :meth:`~isaaclab_visualizers.kit.KitVisualizer.follow` with the asset instead."""
        _warn_method("update_view_to_asset_root")
        self._cfg.asset_name = asset_name
        self._cfg.origin_type = "asset_root"
        for viz in self._get_kit_visualizers():
            viz.follow(asset_name)

    def update_view_to_asset_body(self, asset_name: str, body_name: str) -> None:
        """Deprecated. Call :meth:`~isaaclab_visualizers.kit.KitVisualizer.follow` with the body instead."""
        _warn_method("update_view_to_asset_body")
        self._cfg.asset_name = asset_name
        self._cfg.body_name = body_name
        self._cfg.origin_type = "asset_body"
        for viz in self._get_kit_visualizers():
            viz.follow(f"{asset_name}/{body_name}")

    def update_view_location(
        self,
        eye: tuple[float, float, float] | None = None,
        lookat: tuple[float, float, float] | None = None,
    ) -> None:
        """Deprecated. Call :meth:`~isaaclab_visualizers.kit.KitVisualizer.set_view_offsets` directly instead."""
        _warn_method("update_view_location")
        if eye is not None:
            self.default_cam_eye = np.asarray(eye, dtype=float)
        if lookat is not None:
            self.default_cam_lookat = np.asarray(lookat, dtype=float)
        for viz in self._get_kit_visualizers():
            viz.set_view_offsets(tuple(self.default_cam_eye.tolist()), tuple(self.default_cam_lookat.tolist()))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_kit_visualizers(self) -> list:
        """Return active Kit visualizers from the env's sim, if any."""
        return [v for v in self._env.sim.visualizers if v.cfg.backend.visualizer_type == "kit"]
