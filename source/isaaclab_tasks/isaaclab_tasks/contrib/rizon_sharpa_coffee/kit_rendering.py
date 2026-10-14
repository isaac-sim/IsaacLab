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

from pxr import Usd, UsdGeom, UsdShade, Vt

from .rendering import COFFEE_COLOR, _CoffeePresentationCfg, _vertex_color_material, _water_material

if TYPE_CHECKING:
    from .demo_env_cfg import CoffeeKitVisualizerCfg
    from .rendering import CoffeeRendering


class CoffeeKitVisualizer(KitVisualizer):
    """Render the coffee task through Kit while Newton remains the physics owner."""

    cfg: CoffeeKitVisualizerCfg
    rendering: CoffeeRendering | None = None
    _update_subscription = None

    def initialize(self, sim, *, cameras) -> None:
        if self._is_initialized:
            return
        apply_isaac_rtx_global_settings(self.cfg.render_settings)
        sim.get_or_create_backend(_CoffeePresentationCfg(backdrop=self.cfg.backdrop), stage=sim.stage)
        super().initialize(sim, cameras=cameras)
        self._stage = sim.stage
        self._root_path = "/World/CoffeePresentation/Kit"
        UsdGeom.Xform.Define(self._stage, self._root_path)
        self._coffee = UsdGeom.Points.Define(self._stage, f"{self._root_path}/Coffee")
        self._coffee.CreateWidthsAttr()
        self._coffee.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        self._water = UsdGeom.Mesh.Define(self._stage, f"{self._root_path}/Water")
        self._water.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self._water.CreateDoubleSidedAttr(True)
        self._water.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        self._coffee_colors = self._coffee.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        self._water_colors = self._water.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        coffee_material, _ = _vertex_color_material(
            self._stage, f"{self._root_path}/Looks/Coffee", COFFEE_COLOR, roughness=0.85
        )
        water_material, self._water_shader = _water_material(self._stage, f"{self._root_path}/Looks/Water")
        UsdShade.MaterialBindingAPI.Apply(self._coffee.GetPrim()).Bind(coffee_material)
        UsdShade.MaterialBindingAPI.Apply(self._water.GetPrim()).Bind(water_material)
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
            self._coffee_colors.Set(Vt.Vec3fArray.FromNumpy(display_colors))
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
        self._water_colors.Set(Vt.Vec3fArray.FromNumpy(colors))
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
