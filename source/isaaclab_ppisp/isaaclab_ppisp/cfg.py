# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP configuration and USD/shader parsing helpers.

The implementation follows the physically plausible ISP model described in
https://arxiv.org/abs/2601.18336.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any

from isaaclab.utils.configclass import configclass

PPISP_SHADER_NAME = "PPISP"
"""Conventional prim name for a PPISP shader child under a ``RenderProduct``."""

PPISP_FLOAT2_INPUTS = {
    "vignettingCenterR",
    "vignettingCenterG",
    "vignettingCenterB",
    "colorLatentBlue",
    "colorLatentRed",
    "colorLatentGreen",
    "colorLatentNeutral",
}

PPISP_DEFAULT_INPUTS: dict[str, float | tuple[float, float]] = {
    "responsivity": 1.0,
    "exposureOffset": 0.0,
    "vignettingCenterR": (0.0, 0.0),
    "vignettingAlpha1R": 0.0,
    "vignettingAlpha2R": 0.0,
    "vignettingAlpha3R": 0.0,
    "vignettingCenterG": (0.0, 0.0),
    "vignettingAlpha1G": 0.0,
    "vignettingAlpha2G": 0.0,
    "vignettingAlpha3G": 0.0,
    "vignettingCenterB": (0.0, 0.0),
    "vignettingAlpha1B": 0.0,
    "vignettingAlpha2B": 0.0,
    "vignettingAlpha3B": 0.0,
    "colorLatentBlue": (0.0, 0.0),
    "colorLatentRed": (0.0, 0.0),
    "colorLatentGreen": (0.0, 0.0),
    "colorLatentNeutral": (0.0, 0.0),
    "crfToeR": 0.013659,
    "crfShoulderR": 0.013659,
    "crfGammaR": 0.378165,
    "crfCenterR": 0.0,
    "crfToeG": 0.013659,
    "crfShoulderG": 0.013659,
    "crfGammaG": 0.378165,
    "crfCenterG": 0.0,
    "crfToeB": 0.013659,
    "crfShoulderB": 0.013659,
    "crfGammaB": 0.378165,
    "crfCenterB": 0.0,
}


def default_ppisp_inputs() -> dict[str, float | tuple[float, float]]:
    """Return a copy of the PPISP identity/default input dictionary."""
    return dict(PPISP_DEFAULT_INPUTS)


@configclass
class PpispCfg:
    """Configuration for PPISP post-processing.

    PPISP inputs are static in IsaacLab. If imported from animated USD shader inputs,
    the first authored time sample is used and later samples are ignored.
    """

    shader_prim_path: str | None = None
    """Optional source USD shader prim path used to populate :attr:`inputs`."""

    inputs: dict[str, float | tuple[float, float]] = field(default_factory=default_ppisp_inputs)
    """Flat PPISP shader input values keyed by USD input name.

    Coordinate conventions for spatial inputs:

    * ``vignettingCenter{R,G,B}`` is a 2D offset in UV space normalised by
      ``max(width, height)`` with the image center at ``(0.0, 0.0)``. The
      :data:`PPISP_DEFAULT_INPUTS` defaults place every channel's
      optical center at the image center.
    * Radial vignetting coefficients ``vignettingAlpha{1,2,3}{R,G,B}`` are
      polynomial coefficients in the same normalised radius, applied as
      ``factor = clamp(1 + a1*r^2 + a2*r^4 + a3*r^6, 0, 1)``; with a square
      frame the image corners sit at ``r^2 = 0.5``.
    """


def normalize_ppisp_cfg(
    ppisp_cfg: PpispCfg | None,
    stage: Any | None = None,
) -> PpispCfg | None:
    """Normalise a :class:`PpispCfg` for downstream consumption.

    * If ``ppisp_cfg`` is ``None``, returns ``None``.
    * If ``ppisp_cfg.shader_prim_path`` is set and ``stage`` is supplied,
      merges the shader-authored values with the cfg's explicit overrides
      (see :func:`_merge_shader_inputs_with_cfg`).
    * Otherwise validates ``ppisp_cfg.inputs`` and fills in defaults.
    """
    if ppisp_cfg is None:
        return None
    if not isinstance(ppisp_cfg, PpispCfg):
        raise TypeError(f"Unsupported PPISP configuration type: {type(ppisp_cfg)!r}")
    input_overrides = dict(ppisp_cfg.inputs)
    if ppisp_cfg.shader_prim_path and stage is not None:
        return _merge_shader_inputs_with_cfg(ppisp_cfg, stage, input_overrides)
    ppisp_cfg.inputs = _normalized_inputs(input_overrides)
    return ppisp_cfg


def ppisp_cfg_from_usd_shader(shader: Any) -> PpispCfg:
    """Create :class:`PpispCfg` from a ``UsdShade.Shader`` prim.

    Animated inputs are collapsed to their first authored time sample.
    """
    cfg = PpispCfg(shader_prim_path=str(shader.GetPath()))
    values = default_ppisp_inputs()
    for input_name in values:
        shader_input = shader.GetInput(input_name)
        if not shader_input:
            continue
        attr = shader_input.GetAttr()
        value = _read_first_authored_value(attr)
        if value is not None:
            values[input_name] = _normalize_input_value(input_name, value)
    cfg.inputs = values
    return cfg


def ppisp_cfg_from_usd_stage(stage: Any, shader_prim_path: str) -> PpispCfg:
    """Create :class:`PpispCfg` from a shader prim path in a USD stage."""
    from pxr import UsdShade

    shader = UsdShade.Shader(stage.GetPrimAtPath(shader_prim_path))
    if not shader:
        raise ValueError(f"PPISP shader prim not found at path: {shader_prim_path}")
    return ppisp_cfg_from_usd_shader(shader)


def _normalized_inputs(inputs: dict[str, Any]) -> dict[str, float | tuple[float, float]]:
    values = default_ppisp_inputs()
    for input_name, value in inputs.items():
        if input_name not in values:
            raise ValueError(f"Unknown PPISP input: {input_name}")
        values[input_name] = _normalize_input_value(input_name, value)
    return values


def _merge_shader_inputs_with_cfg(
    ppisp_cfg: PpispCfg,
    stage: Any,
    input_overrides: dict[str, Any],
) -> PpispCfg:
    parsed_cfg = ppisp_cfg_from_usd_stage(stage, ppisp_cfg.shader_prim_path)
    if input_overrides != PPISP_DEFAULT_INPUTS:
        parsed_cfg.inputs.update(_normalized_input_overrides(input_overrides))
    return parsed_cfg


def _normalized_input_overrides(inputs: dict[str, Any]) -> dict[str, float | tuple[float, float]]:
    values = {}
    for input_name, value in inputs.items():
        if input_name not in PPISP_DEFAULT_INPUTS:
            raise ValueError(f"Unknown PPISP input: {input_name}")
        values[input_name] = _normalize_input_value(input_name, value)
    return values


def _normalize_input_value(input_name: str, value: Any) -> float | tuple[float, float]:
    if input_name in PPISP_FLOAT2_INPUTS:
        if len(value) != 2:
            raise ValueError(f"PPISP input '{input_name}' expects two values.")
        return (float(value[0]), float(value[1]))
    return float(value)


def _read_first_authored_value(attr: Any) -> Any:
    time_samples = attr.GetTimeSamples()
    if time_samples:
        return attr.Get(time_samples[0])
    return attr.Get()


def resolve_and_normalize(isp_cfg: Any, stage: Any, camera_prim_path: str) -> PpispCfg | None:
    """Resolve a Camera sensor batch's ``isp_cfg`` to a normalised cfg or ``None``.

    Handles all three legal forms of :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg`:

    * ``None`` → returns ``None``.
    * :class:`~isaaclab.sensors.camera.CameraISPMode` sentinel — walks the stage
      via :func:`auto_camera_ppisp_cfg` (and :func:`auto_any_ppisp_cfg` for
      ``AUTO_ANY``) to discover a PPISP shader. Returns the parsed +
      normalised :class:`PpispCfg`, or ``None`` if no shader matched.
    * Concrete :class:`PpispCfg` — normalises in place (validates input keys,
      fills defaults, and merges shader-authored values when ``shader_prim_path``
      is set).

    This is the single entry point renderer backends call inside their
    ``prepare_cameras`` hook so :mod:`isaaclab.sensors.camera` does not need
    to know about PPISP types at all. The returned cfg applies to the whole
    Camera sensor batch; callers pass the first matched camera prim path for
    the camera-bound discovery phase.

    Args:
        isp_cfg: The Camera sensor's :attr:`isp_cfg` value (``None``, ``CameraISPMode``, or :class:`PpispCfg`).
        stage: USD stage used for sentinel discovery and shader-path resolution.
        camera_prim_path: Absolute path of the first matched camera prim in the
            Camera sensor batch (target of the ``camera`` relationship for the
            camera-bound discovery phase).

    Returns:
        A fully-normalised :class:`PpispCfg`, or ``None`` if the batch has no ISP.
    """
    # Local import avoids a top-of-module dep on isaaclab.sensors.
    from isaaclab.sensors.camera.camera_isp import CameraISPMode

    if isp_cfg is None:
        return None
    if isinstance(isp_cfg, CameraISPMode):
        resolved = auto_camera_ppisp_cfg(stage, camera_prim_path)
        if resolved is None and isp_cfg == CameraISPMode.AUTO_ANY:
            resolved = auto_any_ppisp_cfg(stage)
        if resolved is None:
            return None
        return normalize_ppisp_cfg(resolved, stage=stage)
    return normalize_ppisp_cfg(isp_cfg, stage=stage)


def auto_camera_ppisp_cfg(stage: Any, camera_prim_path: str) -> PpispCfg | None:
    """Find the first PPISP shader on ``stage`` bound to ``camera_prim_path``.

    Walks ``stage`` looking for the first ``RenderProduct`` whose ``camera``
    relationship targets ``camera_prim_path``. If that ``RenderProduct`` has a
    child shader at ``<render_product>/<PPISP_SHADER_NAME>``, parses
    and returns the corresponding :class:`PpispCfg`. Returns ``None`` if
    no matching ``RenderProduct`` is found, or if it has no PPISP shader child.

    Args:
        stage: USD stage to search.
        camera_prim_path: Absolute camera prim path the ``RenderProduct``'s
            ``camera`` relationship must target.

    Returns:
        Parsed :class:`PpispCfg` if a matching shader was found, else ``None``.
    """
    from pxr import Sdf, UsdShade

    target_path = Sdf.Path(camera_prim_path)
    for prim in stage.Traverse():
        if prim.GetTypeName() != "RenderProduct":
            continue
        camera_rel = prim.GetRelationship("camera")
        if not camera_rel:
            continue
        if target_path not in camera_rel.GetTargets():
            continue
        shader_path = prim.GetPath().AppendChild(PPISP_SHADER_NAME)
        shader_prim = stage.GetPrimAtPath(shader_path)
        if shader_prim and shader_prim.IsValid():
            return ppisp_cfg_from_usd_shader(UsdShade.Shader(shader_prim))
    return None


def auto_any_ppisp_cfg(stage: Any) -> PpispCfg | None:
    """Find the first PPISP shader anywhere on ``stage``, regardless of binding.

    Walks ``stage`` looking for the first prim named
    :data:`PPISP_SHADER_NAME` that resolves to a valid ``UsdShade.Shader``.
    Returns ``None`` if no such shader exists.

    Used as a fallback when no ``RenderProduct`` binds a PPISP shader to a
    given camera but the scene still contains a PPISP configuration that
    should apply.

    Args:
        stage: USD stage to search.

    Returns:
        Parsed :class:`PpispCfg` for the first matching shader, else ``None``.
    """
    from pxr import UsdShade

    for prim in stage.Traverse():
        if prim.GetName() != PPISP_SHADER_NAME:
            continue
        shader = UsdShade.Shader(prim)
        if shader:
            return ppisp_cfg_from_usd_shader(shader)
    return None
