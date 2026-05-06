# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim Kit perspective RGB capture via ``/OmniverseKit_Persp`` and omni.replicator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .isaacsim_kit_perspective_video_cfg import IsaacsimKitPerspectiveVideoCfg


class IsaacsimKitPerspectiveVideo:
    """Stateful capture of one RGB frame per call from the Kit perspective camera."""

    def __init__(self, cfg: IsaacsimKitPerspectiveVideoCfg):
        self.cfg = cfg
        self._rgb_annotator = None
        self._render_product = None

    def render_rgb_array(self) -> np.ndarray:
        """Return one RGB frame. Blank frame during warmup; raises on other failures."""
        import omni.kit.app
        import omni.replicator.core as rep

        omni.kit.app.get_app().update()

        h, w = self.cfg.window_height, self.cfg.window_width
        if self._rgb_annotator is None:
            from isaacsim.core.rendering_manager import ViewportManager

            ViewportManager.set_camera_view(
                self.cfg.camera_prim_path,
                eye=list(self.cfg.eye),
                target=list(self.cfg.lookat),
            )
            self._render_product = rep.create.render_product(self.cfg.camera_prim_path, (w, h))
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])

        rgb_data = self._rgb_annotator.get_data()
        if isinstance(rgb_data, dict):
            rgb_data = rgb_data.get("data", np.array([], dtype=np.uint8))
        rgb_data = np.asarray(rgb_data, dtype=np.uint8)
        if rgb_data.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
        if rgb_data.ndim == 1:
            rgb_data = rgb_data.reshape(h, w, -1)
        return rgb_data[:, :, :3]


def create_isaacsim_kit_perspective_video(cfg: IsaacsimKitPerspectiveVideoCfg) -> IsaacsimKitPerspectiveVideo:
    """Instantiate the perspective video capture implementation from ``cfg.class_type``."""
    ct = cfg.class_type
    if isinstance(ct, type):
        return ct(cfg)
    from isaaclab.utils.string import string_to_callable

    cls = string_to_callable(str(ct))
    return cls(cfg)
