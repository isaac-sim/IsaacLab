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

    def apply_lights(self) -> bool:
        """Write every light's intensity, scaled, and color, tinted."""
        if not self.lights:
            return False
        paths = [light.path for light in self.lights]
        intensities = np.array([light.intensity * self.light_scale for light in self.lights], np.float32)
        colors = np.array([np.multiply(light.color, self.light_tint) for light in self.lights], np.float32)
        return self._write(paths, "inputs:intensity", intensities) and self._write(paths, "inputs:color", colors)

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
        self.light_scale, self.light_tint = 1.0, (1.0, 1.0, 1.0)
        self.background, self.background_color = self._initial
        return self.apply_lights() & self.apply_background()
