# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime appearance of the OVRTX view in the Newton GL window: its background and the scene's lights.

The OpenGL view's Appearance options (sky, shadows, wireframe, colors) configure Newton's rasterizer and do not touch
the OVRTX image, whose lighting and materials come from the scene's USD. What an OVRTX view can change while running is
the attributes its render product and the scene's lights expose, which this module writes through the renderer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BACKGROUNDS = ("Scene Sky", "Solid Color")
"""Background choices: the scene's dome light, or one color."""

LIGHTINGS = ("Scene Lights", "Dome Light", "Distant Light", "Dome + Distant")
"""Lighting choices: the lights the scene authored, or a dome (sky) light, a distant (sun) light, or both."""

_RIG_PATH = "/NewtonViewerLights"
_RIG_USD = """#usda 1.0
(defaultPrim = "Lights")

def Scope "Lights"
{
    def DomeLight "Dome"
    {
        float inputs:intensity = 1000
        color3f inputs:color = (0.53, 0.68, 0.9)
        token visibility = "invisible"
    }

    def DistantLight "Sun"
    {
        float inputs:intensity = 3000
        color3f inputs:color = (1, 1, 1)
        float inputs:angle = 1.0
        float3 xformOp:rotateXYZ = (25, 0, 35)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
        token visibility = "invisible"
    }
}
"""
"""Lights the viewer adds to the OVRTX stage: a sky-colored dome light, and a sun that shines down and sideways."""


@dataclass
class SceneLight:
    """A light of the scene stage and the intensity and color it was authored with."""

    path: str
    intensity: float
    color: tuple[float, float, float]


def find_scene_lights(stage: Any) -> list[SceneLight]:
    """Return the lights of ``stage`` that carry an intensity and a color."""
    lights = []
    for prim in stage.Traverse():
        if "Light" not in prim.GetTypeName():
            continue
        intensity, color = prim.GetAttribute("inputs:intensity"), prim.GetAttribute("inputs:color")
        if intensity and color and intensity.HasValue() and color.HasValue():
            lights.append(SceneLight(str(prim.GetPath()), float(intensity.Get()), tuple(float(c) for c in color.Get())))
    return lights


class RtxAppearance:
    """Changes the background and lighting of one OVRTX render product while it runs."""

    def __init__(
        self,
        stage: Any,
        renderer: Any,
        render_product: str,
        background_color: tuple[float, float, float] | None = None,
    ):
        """Create the controller.

        Args:
            stage: USD stage the lights are read from.
            renderer: The OVRTX renderer that owns the render product.
            render_product: Path of the render product whose background is changed.
            background_color: Solid background the camera was configured with, or None for the scene sky.
        """
        self._renderer = renderer
        self._product = render_product
        self._initial = (0, (0.2, 0.4, 0.6)) if background_color is None else (1, tuple(background_color))
        self.lights = find_scene_lights(stage)
        self.rig = [
            SceneLight(f"{_RIG_PATH}/Dome", 1000.0, (0.53, 0.68, 0.9)),
            SceneLight(f"{_RIG_PATH}/Sun", 3000.0, (1.0, 1.0, 1.0)),
        ]
        self._rig_reference = None
        self.lighting = 0
        self.light_scale = 1.0
        self.light_tint = (1.0, 1.0, 1.0)
        self.background, self.background_color = self._initial
        self.error = ""

    @property
    def available(self) -> bool:
        """Whether the renderer accepts attribute writes; the shared-stage path does not yet."""
        return not self._renderer._use_ovstage and self._renderer.backend.renderer is not None

    def _write(self, paths: list[str], attribute: str, tensor: Any, **kwargs: Any) -> bool:
        from ovrtx import PrimMode

        try:
            self._renderer.backend.renderer.write_attribute(
                prim_paths=paths, attribute_name=attribute, tensor=tensor, prim_mode=PrimMode.MUST_EXIST, **kwargs
            )
        except Exception as exc:
            logger.exception("[NewtonRtxAppearance] Could not write %s.", attribute)
            self.error = f"Could not change {attribute}: {exc}"
            return False
        self.error = ""
        return True

    def _ensure_rig(self) -> bool:
        """Add the viewer's dome and distant lights to the OVRTX stage the first time they are needed."""
        if self._rig_reference is not None:
            return True
        try:
            self._rig_reference = self._renderer.backend.renderer.add_usd_reference_from_string(_RIG_USD, _RIG_PATH)
        except Exception as exc:
            logger.exception("[NewtonRtxAppearance] Could not add the viewer lights.")
            self.error = f"Could not add lights: {exc}"
            return False
        return True

    def _active(self) -> list[SceneLight]:
        """Return the lights the current lighting turns on."""
        if self.lighting == 0:
            return self.lights
        dome, sun = self.rig
        return {1: [dome], 2: [sun], 3: [dome, sun]}[self.lighting]

    def apply_lights(self) -> bool:
        """Show the lights of the chosen lighting, with their intensity scaled and color tinted, and hide the rest."""
        if self.lighting and not self._ensure_rig():
            return False
        every = [*self.lights, *(self.rig if self._rig_reference is not None else [])]
        if not every:
            return False
        from ovrtx import Semantic

        active = {light.path for light in self._active()}
        paths = [light.path for light in every]
        intensities = np.array([light.intensity * self.light_scale for light in every], np.float32)
        colors = np.array([np.multiply(light.color, self.light_tint) for light in every], np.float32)
        visibility = ["inherited" if path in active else "invisible" for path in paths]
        return (
            self._write(paths, "visibility", visibility, semantic=Semantic.TOKEN_STRING)
            and self._write(paths, "inputs:intensity", intensities)
            and self._write(paths, "inputs:color", colors)
        )

    def close(self) -> None:
        """Remove the lights the viewer added to the stage."""
        reference, self._rig_reference = self._rig_reference, None
        if reference is not None:
            try:
                self._renderer.backend.renderer.remove_usd(reference)
            except Exception:
                logger.debug("[NewtonRtxAppearance] The viewer lights were already removed.", exc_info=True)

    def apply_background(self) -> bool:
        """Write the background type and, for a solid color, the color."""
        from ovrtx import Semantic

        solid = self.background == 1
        wrote = self._write(
            [self._product],
            "omni:rtx:background:source:type",
            ["color" if solid else "domeLight"],
            semantic=Semantic.TOKEN_STRING,
        )
        if solid and wrote:
            wrote = self._write(
                [self._product],
                "omni:rtx:background:source:color",
                np.array([self.background_color], np.float32),
            )
        return wrote

    def reset(self) -> bool:
        """Return to the scene's own lights and background."""
        self.lighting, self.light_scale, self.light_tint = 0, 1.0, (1.0, 1.0, 1.0)
        self.background, self.background_color = self._initial
        return self.apply_lights() & self.apply_background()
