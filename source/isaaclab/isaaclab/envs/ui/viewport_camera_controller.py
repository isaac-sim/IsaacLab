# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated: ViewportCameraController compatibility shim.

Camera tracking is now built into :class:`~isaaclab_visualizers.kit.KitVisualizer`.
Configure ``eye``, ``lookat``, ``origin_type``, and ``origin_track_path`` directly on
:class:`~isaaclab_visualizers.kit.KitVisualizerCfg` and add it to
:attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ViewerCfg


def _warn_method(name: str) -> None:
    warnings.warn(
        f"ViewportCameraController.{name}() is deprecated. "
        "Set cfg fields (origin_type, origin_track_path, eye, lookat) on KitVisualizerCfg and "
        "call visualizer.reapply_origin() to update at runtime.",
        DeprecationWarning,
        stacklevel=3,
    )


class ViewportCameraController:
    """Deprecated compatibility shim for :class:`ViewportCameraController`.

    .. deprecated::
        :class:`ViewportCameraController` has been removed. Camera tracking is now built into
        :class:`~isaaclab_visualizers.kit.KitVisualizer`. Set ``origin_type``,
        ``origin_track_path``, ``eye``, and ``lookat`` directly on
        :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` and pass it to
        :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs`.
    """

    def __init__(self, env: object, cfg: ViewerCfg):
        warnings.warn(
            "ViewportCameraController is deprecated and has been removed. "
            "Camera tracking is now built into KitVisualizer — configure "
            "origin_type, origin_track_path, eye, and lookat directly on KitVisualizerCfg "
            "and add it to SimulationCfg.visualizer_cfgs.",
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
        """Deprecated. Update :attr:`~KitVisualizerCfg.origin_env_index` on the visualizer instead."""
        _warn_method("set_view_env_index")
        num_envs = getattr(self._env, "num_envs", None)
        if num_envs is not None and not (0 <= env_index < num_envs):
            raise ValueError(
                f"Out of range value for attribute 'env_index': {env_index}."
                f" Expected a value between 0 and {num_envs - 1}."
            )
        self._cfg.env_index = env_index
        for viz in self._get_kit_visualizers():
            viz.cfg.origin_env_index = env_index
            if getattr(viz.cfg, "origin_type", None) == "env":
                viz.reapply_origin()

    def update_view_to_world(self) -> None:
        """Deprecated. Set ``origin_type='world'`` on :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` instead."""
        _warn_method("update_view_to_world")
        self._cfg.origin_type = "world"
        origin = torch.zeros(3)
        for viz in self._get_kit_visualizers():
            viz.cfg.origin_type = "world"
            eye = (origin + torch.tensor(self.default_cam_eye, dtype=torch.float32)).tolist()
            target = (origin + torch.tensor(self.default_cam_lookat, dtype=torch.float32)).tolist()
            viz.set_camera_view(tuple(eye), tuple(target))

    def update_view_to_env(self) -> None:
        """Deprecated. Set ``origin_type='env'`` on :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` instead."""
        _warn_method("update_view_to_env")
        self._cfg.origin_type = "env"
        for viz in self._get_kit_visualizers():
            viz.cfg.origin_type = "env"
            viz.reapply_origin()

    def update_view_to_asset_root(self, asset_name: str) -> None:
        """Deprecated. Set ``origin_type='asset'`` and ``origin_track_path`` on
        :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` instead."""
        _warn_method("update_view_to_asset_root")
        self._cfg.asset_name = asset_name
        self._cfg.origin_type = "asset_root"
        for viz in self._get_kit_visualizers():
            viz.cfg.origin_type = "asset"
            viz.cfg.origin_track_path = asset_name
            viz.reapply_origin()

    def update_view_to_asset_body(self, asset_name: str, body_name: str) -> None:
        """Deprecated. Set ``origin_type='asset'`` and ``origin_track_path`` on
        :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` instead."""
        _warn_method("update_view_to_asset_body")
        self._cfg.asset_name = asset_name
        self._cfg.body_name = body_name
        self._cfg.origin_type = "asset_body"
        for viz in self._get_kit_visualizers():
            viz.cfg.origin_type = "asset"
            viz.cfg.origin_track_path = f"{asset_name}/{body_name}"
            viz.reapply_origin()

    def update_view_location(
        self,
        eye: tuple[float, float, float] | None = None,
        lookat: tuple[float, float, float] | None = None,
    ) -> None:
        """Deprecated. Call :meth:`~isaaclab_visualizers.kit.KitVisualizer.set_camera_view` directly instead."""
        _warn_method("update_view_location")
        if eye is not None:
            self.default_cam_eye = np.asarray(eye, dtype=float)
        if lookat is not None:
            self.default_cam_lookat = np.asarray(lookat, dtype=float)
        for viz in self._get_kit_visualizers():
            # Persist relative offsets into the visualizer config so that
            # subsequent reapply_origin() calls (e.g. from asset tracking) use
            # the updated offsets rather than the stale original values.
            viz.cfg.eye = tuple(self.default_cam_eye.tolist())
            viz.cfg.lookat = tuple(self.default_cam_lookat.tolist())
            # reapply_origin() adds the current world/env/asset origin and pushes
            # the result to the viewport, preserving origin-relative behaviour.
            viz.reapply_origin()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_kit_visualizers(self) -> list:
        """Return active Kit visualizers from the env's sim, if any."""
        sim = getattr(self._env, "sim", None)
        if sim is None:
            return []
        return [
            v
            for v in getattr(sim, "visualizers", [])
            if getattr(getattr(v, "cfg", None), "visualizer_type", None) == "kit"
        ]
