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

from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from isaaclab.utils.assets import retrieve_file_path

from .assets import COFFEE_ASSET_DIR
from .presentation import COFFEE_COLOR, WATER_COLOR, _add_coffee_studio_lights

if TYPE_CHECKING:
    from .kit_presentation_cfg import CoffeeKitVisualizerCfg
    from .presentation import CoffeePresentation


class CoffeeKitVisualizer(KitVisualizer):
    """Render the coffee task through Kit while Newton remains the physics owner."""

    cfg: CoffeeKitVisualizerCfg
    presentation: CoffeePresentation | None = None
    _update_subscription = None

    def initialize(self, sim, *, cameras) -> None:
        if self._is_initialized:
            return
        apply_isaac_rtx_global_settings(self.cfg.render_settings)
        super().initialize(sim, cameras=cameras)
        self._stage = self._sim.stage
        self._root_path = "/World/CoffeePresentation"
        UsdGeom.Xform.Define(self._stage, self._root_path)
        self._visibility_backup = {}
        self._light_backup = {}
        for path in ("/World/DomeLight", "/World/KeyLight"):
            intensity = self._stage.GetPrimAtPath(path).GetAttribute("inputs:intensity")
            if intensity:
                self._light_backup[path] = intensity.Get()
                intensity.Set(0.0)
        _add_coffee_studio_lights(self._stage, f"{self._root_path}/Lights")
        if self.cfg.backdrop == "kitchen":
            from .kitchen import add_kitchen

            add_kitchen(self._stage, f"{self._root_path}/Kitchen")
            for prim in self._stage.Traverse():
                if (
                    prim.GetPath().HasPrefix("/World/envs")
                    and prim.GetName() == "StudioFloor"
                    and prim.IsA(UsdGeom.Imageable)
                ):
                    imageable = UsdGeom.Imageable(prim)
                    self._visibility_backup[str(prim.GetPath())] = imageable.GetVisibilityAttr().Get()
                    imageable.MakeInvisible()
        elif self.cfg.backdrop != "studio":
            raise ValueError("Coffee backdrop must be 'kitchen' or 'studio'.")
        self._coffee = UsdGeom.Points.Define(self._stage, f"{self._root_path}/Coffee")
        self._coffee.CreateWidthsAttr()
        self._coffee.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        self._water = UsdGeom.Mesh.Define(self._stage, f"{self._root_path}/Water")
        self._water.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self._water.CreateDoubleSidedAttr(True)
        self._water.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        self._coffee_colors = self._coffee.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        self._water_colors = self._water.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        self._coffee_material, _ = self._color_material("Coffee", COFFEE_COLOR, 0.85)
        water_material, self._water_shader = self._color_material("Water", WATER_COLOR, 0.08)
        # The MDL surface uses the local tracer colour as optical absorption.
        # Opacity remains available on the PreviewSurface fallback.
        water_mdl = UsdShade.Shader.Define(self._stage, f"{water_material.GetPath()}/Liquid")
        water_mdl.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        water_mdl.SetSourceAsset(Sdf.AssetPath(retrieve_file_path(f"{COFFEE_ASSET_DIR}/CoffeeWater.mdl")), "mdl")
        water_mdl.SetSourceAssetSubIdentifier("CoffeeWater", "mdl")
        water_material.CreateSurfaceOutput("mdl").ConnectToSource(water_mdl.ConnectableAPI(), "out")
        UsdShade.MaterialBindingAPI.Apply(self._coffee.GetPrim()).Bind(self._coffee_material)
        UsdShade.MaterialBindingAPI.Apply(self._water.GetPrim()).Bind(water_material)
        self._coffee_attributes_ready = False
        import omni.kit.app

        self._update_subscription = (
            omni.kit.app.get_app()
            .get_pre_update_event_stream()
            .create_subscription_to_pop(self._update_presentation, name="Coffee presentation")
        )

    def prepare_presentation(self) -> None:
        """Hide native point displays while retaining all physical particles."""
        for prim in self._stage.Traverse():
            if prim.IsA(UsdGeom.Points) and prim.GetPath().HasPrefix("/World/envs"):
                imageable = UsdGeom.Imageable(prim)
                self._visibility_backup[str(prim.GetPath())] = imageable.GetVisibilityAttr().Get()
                imageable.MakeInvisible()

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

    def _update_presentation(self, event) -> None:
        """Publish scene geometry before both viewport and on-demand Kit frames."""
        if self.presentation is not None:
            self.presentation.draw_kit(self)

    def close(self) -> None:
        if self._update_subscription is not None:
            self._update_subscription.unsubscribe()
            self._update_subscription = None
        self.presentation = None
        if self._is_initialized:
            for path, visibility in self._visibility_backup.items():
                UsdGeom.Imageable(self._stage.GetPrimAtPath(path)).GetVisibilityAttr().Set(visibility)
            for path, intensity in self._light_backup.items():
                self._stage.GetPrimAtPath(path).GetAttribute("inputs:intensity").Set(intensity)
            self._stage.RemovePrim(self._root_path)
        super().close()

    def _color_material(self, name: str, color: tuple[float, float, float], roughness: float):
        material = UsdShade.Material.Define(self._stage, f"{self._root_path}/Looks/{name}")
        shader = UsdShade.Shader.Define(self._stage, f"{material.GetPath()}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.33)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        reader = UsdShade.Shader.Define(self._stage, f"{material.GetPath()}/Colour")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        shader.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "result")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material, shader
