# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime appearance of the OVRTX view in the Newton GL window: its background and the lights that illuminate it.

The OpenGL view's Appearance options (sky, shadows, wireframe, colors) configure Newton's rasterizer and do not touch
the OVRTX image, whose lighting and materials come from the scene's USD. What an OVRTX view can change while running is
the attributes its render product and the scene's lights expose, which this module writes through the renderer.

Three lights are controlled independently: the scene's own lights, and a dome light and a distant light that the viewer
adds to the OVRTX stage the first time either is enabled.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, retrieve_file_path

logger = logging.getLogger(__name__)

BACKGROUNDS = ("Scene Sky", "Solid Color")
"""Background choices: the scene's dome light, or one color."""

SKY_TEXTURES = {
    "Cloudy: Abandoned Parking": f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
    "Cloudy: Evening Road": f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
    "Cloudy: Lakeside": f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
    "Clear: Kloofendal": f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
}
"""Outdoor sky textures that Isaac Lab tasks already use for dome lights, by display name."""

_DOME_PATH, _SUN_PATH = "/NewtonViewerDome", "/NewtonViewerSun"
_DOME_DEFAULTS = (1000.0, (0.53, 0.68, 0.9))
_SUN_DEFAULTS = (3000.0, (1.0, 1.0, 1.0), 1.0, 65.0, 35.0)  # Intensity, color, angle [deg], elevation, azimuth.

_DOME_USD = """#usda 1.0
(defaultPrim = "Dome")

def DomeLight "Dome"
{{
    float inputs:intensity = 1000
    color3f inputs:color = (0.53, 0.68, 0.9)
    asset inputs:texture:file = @{texture}@
    token inputs:texture:format = "latlong"
    token visibility = "invisible"
}}
"""
"""The dome light the viewer adds to the OVRTX stage. A texture is a USD asset path, which cannot be written to a
running stage, so changing it adds the light again."""

_SUN_USD = """#usda 1.0
(defaultPrim = "Sun")

def DistantLight "Sun"
{
    float inputs:intensity = 3000
    color3f inputs:color = (1, 1, 1)
    float inputs:angle = 1.0
    token visibility = "invisible"
}
"""
"""The distant light the viewer adds to the OVRTX stage; its direction is written as a transform."""


def sun_matrix(elevation: float, azimuth: float) -> np.ndarray:
    """Return the renderer's row-vector transform that makes a distant light shine from the given direction.

    Args:
        elevation: Height of the light above the horizon [deg]; 90 is straight down.
        azimuth: Direction the light comes from, around the vertical axis [deg].
    """
    el, az = np.radians(elevation), np.radians(azimuth)
    toward_light = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # Local +Z.
    x_axis = np.cross((0.0, 0.0, 1.0), toward_light)
    x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 1e-6 else np.array([1.0, 0.0, 0.0])
    y_axis = np.cross(toward_light, x_axis)
    matrix = np.eye(4)
    matrix[0, :3], matrix[1, :3], matrix[2, :3] = x_axis, y_axis, toward_light  # Rows, for row vectors.
    return matrix


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
            stage: USD stage the scene's lights are read from.
            renderer: The OVRTX renderer that owns the render product.
            render_product: Path of the render product whose background is changed.
            background_color: Solid background the camera was configured with, or None for the scene sky.
        """
        self._renderer = renderer
        self._product = render_product
        self._initial = (0, (0.2, 0.4, 0.6)) if background_color is None else (1, tuple(background_color))
        self.lights = find_scene_lights(stage)
        self._dome_reference = self._sun_reference = None
        self._texture_thread: threading.Thread | None = None
        self._texture_result: str | Exception | None = None
        self.error = ""
        self.reset_settings()

    def reset_settings(self) -> None:
        """Restore every setting to its default without writing it."""
        self.background, self.background_color = self._initial
        self.scene_enabled, self.scene_scale, self.scene_tint = True, 1.0, (1.0, 1.0, 1.0)
        self.dome_enabled = False
        self.dome_intensity, self.dome_color = _DOME_DEFAULTS
        self.dome_texture = ""
        """Sky texture of the dome light: a path or URL, or an empty string for none."""
        self._dome_texture_file = ""
        self.texture_status = ""
        self.sun_enabled = False
        self.sun_intensity, self.sun_color, self.sun_angle, self.sun_elevation, self.sun_azimuth = _SUN_DEFAULTS

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

    def _add(self, usd: str, path: str, what: str) -> Any | None:
        """Add a light layer to the OVRTX stage and return its reference, or None after reporting why not."""
        try:
            return self._renderer.backend.renderer.add_usd_reference_from_string(usd, path)
        except Exception as exc:
            logger.exception("[NewtonRtxAppearance] Could not add the %s.", what)
            self.error = f"Could not add the {what}: {exc}"
            return None

    def _ensure_lights(self) -> bool:
        """Add the viewer's dome and distant lights to the OVRTX stage the first time each is enabled."""
        if self.dome_enabled and self._dome_reference is None:
            self._dome_reference = self._add(
                _DOME_USD.format(texture=self._dome_texture_file), _DOME_PATH, "dome light"
            )
        if self.sun_enabled and self._sun_reference is None:
            self._sun_reference = self._add(_SUN_USD, _SUN_PATH, "distant light")
        return (not self.dome_enabled or self._dome_reference is not None) and (
            not self.sun_enabled or self._sun_reference is not None
        )

    def apply_lights(self) -> bool:
        """Show the enabled lights with their own intensity and color, hide the rest, and aim the distant light."""
        if not self._ensure_lights():
            return False
        from ovrtx import Semantic

        paths = [light.path for light in self.lights]
        intensities = [light.intensity * self.scene_scale for light in self.lights]
        colors = [np.multiply(light.color, self.scene_tint) for light in self.lights]
        visible = [self.scene_enabled] * len(paths)
        for reference, path, enabled, intensity, color in (
            (self._dome_reference, _DOME_PATH, self.dome_enabled, self.dome_intensity, self.dome_color),
            (self._sun_reference, _SUN_PATH, self.sun_enabled, self.sun_intensity, self.sun_color),
        ):
            if reference is not None:
                paths.append(path)
                intensities.append(intensity)
                colors.append(color)
                visible.append(enabled)
        if not paths:
            return False
        states = ["inherited" if show else "invisible" for show in visible]
        wrote = (
            self._write(paths, "visibility", states, semantic=Semantic.TOKEN_STRING)
            and self._write(paths, "inputs:intensity", np.array(intensities, np.float32))
            and self._write(paths, "inputs:color", np.array(colors, np.float32))
        )
        if wrote and self._sun_reference is not None:
            wrote = self._write([_SUN_PATH], "inputs:angle", np.array([self.sun_angle], np.float32)) and self._write(
                [_SUN_PATH],
                "omni:xform",
                sun_matrix(self.sun_elevation, self.sun_azimuth)[None],
                semantic=Semantic.XFORM_MAT4x4,
            )
        return wrote

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

    # ------------------------------------------------------------------
    # Dome light sky texture
    # ------------------------------------------------------------------

    def set_dome_texture(self, source: str) -> None:
        """Give the dome light a sky texture, downloading it first when it is a remote file.

        Args:
            source: A path or URL of an HDR or EXR sky image, or an empty string to remove the texture. The change
                shows once :meth:`poll` reports that the file is ready.
        """
        self.dome_texture = source
        if not source:
            self._dome_texture_file, self.texture_status = "", ""
            self._apply_texture()
            return
        if self._texture_thread is not None and self._texture_thread.is_alive():
            return
        self.texture_status = "Loading sky..."

        def work() -> None:
            try:
                self._texture_result = retrieve_file_path(source)
            except Exception as exc:
                logger.exception("[NewtonRtxAppearance] Could not load the sky texture %s.", source)
                self._texture_result = exc

        self._texture_thread = threading.Thread(target=work, name="rtx-sky-texture", daemon=True)
        self._texture_thread.start()

    def poll(self) -> bool:
        """Apply a sky texture that finished loading; returns whether the image changed."""
        result, self._texture_result = self._texture_result, None
        if result is None:
            return False
        if isinstance(result, Exception):
            self.texture_status, self.dome_texture = f"Could not load the sky: {result}", ""
            return False
        self._dome_texture_file, self.texture_status = result, ""
        return self._apply_texture()

    def _apply_texture(self) -> bool:
        """Add the dome light again with the current sky texture and write its settings."""
        reference, self._dome_reference = self._dome_reference, None
        if reference is not None:
            try:
                self._renderer.backend.renderer.remove_usd(reference)
            except Exception:
                logger.debug("[NewtonRtxAppearance] The dome light was already removed.", exc_info=True)
        self.dome_enabled = True if self._dome_texture_file else self.dome_enabled
        return self.apply_lights()

    def use_sky(self, source: str) -> None:
        """Choose the dome light's sky texture; the light is tinted white for a texture and sky blue without one.

        Args:
            source: A path or URL of an HDR or EXR sky image, or an empty string for no texture.
        """
        self.dome_color = (1.0, 1.0, 1.0) if source else _DOME_DEFAULTS[1]
        self.set_dome_texture(source)

    def reset(self) -> bool:
        """Return to the scene's own lights and background, and remove the sky texture."""
        had_texture = bool(self._dome_texture_file)
        self.reset_settings()
        if had_texture:
            self._apply_texture()  # Adds the dome light again, without the texture.
        return self.apply_lights() & self.apply_background()

    def close(self) -> None:
        """Remove the lights the viewer added to the stage."""
        references, self._dome_reference, self._sun_reference = (self._dome_reference, self._sun_reference), None, None
        for reference in references:
            if reference is not None:
                try:
                    self._renderer.backend.renderer.remove_usd(reference)
                except Exception:
                    logger.debug("[NewtonRtxAppearance] A viewer light was already removed.", exc_info=True)
