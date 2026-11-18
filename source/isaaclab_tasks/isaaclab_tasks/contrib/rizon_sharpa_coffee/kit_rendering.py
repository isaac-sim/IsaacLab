# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit USD presentation of native coffee grains and the coloured water surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings
from isaaclab_visualizers.kit import KitVisualizer

from pxr import Usd, UsdGeom, Vt

from isaaclab.visualizers import VisualizerCfg

from .rendering import create_coffee_points, create_water_mesh

if TYPE_CHECKING:
    from .rendering import CoffeeRendering


class CoffeeKitVisualizer(KitVisualizer):
    """Render the coffee task through Kit while Newton remains the physics owner."""

    cfg: VisualizerCfg
    rendering: CoffeeRendering | None = None
    _update_subscription = None

    def initialize(self, sim, *, cameras) -> None:
        if self._is_initialized:
            return
        apply_isaac_rtx_global_settings(self.cfg.backend.render_settings)
        super().initialize(sim, cameras=cameras)
        self._stage = sim.stage
        self._root_path = "/World/CoffeePresentation/Kit"
        UsdGeom.Xform.Define(self._stage, self._root_path)
        self._coffee = create_coffee_points(self._stage, f"{self._root_path}/Coffee")
        self._water, self._water_shader = create_water_mesh(self._stage, f"{self._root_path}/Water")
        self._coffee_attributes_ready = False
        import omni.kit.app

        self._update_subscription = (
            omni.kit.app.get_app()
            .get_pre_update_event_stream()
            .create_subscription_to_pop(self._update_rendering, name="Coffee presentation")
        )

    def update_coffee(
        self, points: wp.array, radii: wp.array, colors: wp.array, *, dynamic_radii: bool = False
    ) -> None:
        """Publish dry grains, omitting coffee that has converted to liquid."""
        positions = points.numpy()
        if dynamic_radii or not self._coffee_attributes_ready:
            widths = 2.0 * radii.numpy()
            display_colors = colors.numpy()
            if dynamic_radii:
                visible = widths > 0.0
                positions, widths, display_colors = positions[visible], widths[visible], display_colors[visible]
            self._coffee.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(widths))
            self._coffee.GetDisplayColorPrimvar().Set(Vt.Vec3fArray.FromNumpy(display_colors))
            self._coffee_attributes_ready = True
        self._coffee.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))

    def update_water(self, points, indices, normals, colors: np.ndarray | None, opacity: float) -> None:
        """Publish shared liquid vertices with smoothly interpolated local colours."""
        if points is None or colors is None:
            self._water.MakeInvisible()
            return
        self._water.MakeVisible()
        self._water.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.numpy()))
        triangles = indices.numpy().reshape(-1)
        self._water.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles) // 3, 3, dtype=np.int32)))
        self._water.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(triangles))
        self._water.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.numpy()))
        self._water.GetDisplayColorPrimvar().Set(Vt.Vec3fArray.FromNumpy(colors))
        self._water_shader.GetInput("opacity").Set(float(opacity))

    def _update_rendering(self, event) -> None:
        """Publish scene geometry before both viewport and on-demand Kit frames."""
        if self.rendering is not None:
            self.rendering.draw_kit(self)

    def close(self) -> None:
        if self._is_closed:
            return
        if self._update_subscription is not None:
            self._update_subscription.unsubscribe()
            self._update_subscription = None
        self.rendering = None
        initialized = self._is_initialized
        try:
            super().close()
        finally:
            if initialized:
                self._stage.RemovePrim(self._root_path)

    def _set_usd_camera_pose(self, camera_path: str, position, target) -> bool:
        """Publish this demo's camera above Kit's session-layer defaults."""
        with Usd.EditContext(self._sim.stage, self._sim.stage.GetSessionLayer()):
            return super()._set_usd_camera_pose(camera_path, position, target)
