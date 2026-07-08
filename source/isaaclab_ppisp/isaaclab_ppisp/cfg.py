# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP configuration and USD parsing helpers.

The implementation follows the physically plausible ISP model described in
https://arxiv.org/abs/2601.18336.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any

from isaaclab.utils.configclass import configclass

PPISP_ATTR_NAMESPACE = "ppisp:"
"""Namespace prefix for authoritative PPISP attributes authored on a USD camera."""

PPISP_CONTROLLER_WEIGHTS_CAMERA_ATTR = "controllerWeights"
"""Camera ``ppisp:*`` attribute name containing flattened controller weights."""

PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN = 241_961
"""Flattened element count of the camera-authored controller weight array exported by NRE.

This is a frozen architectural constant tied to the exported controller network
shape (see :mod:`isaaclab_ppisp.kernels` for the offset layout). USD parsing
and Warp execution validate against it and fail loudly on a mismatch.
"""

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

    PPISP inputs are static in IsaacLab. NRE exports store the authoritative
    values on a USD camera as ``ppisp:*`` attributes. If animated USD
    attributes are imported, the first authored time sample is used and later
    samples are ignored.
    """

    camera_prim_path: str | None = None
    """Optional USD camera prim path used to import PPISP camera attributes."""

    inputs: dict[str, float | tuple[float, float]] = field(default_factory=default_ppisp_inputs)
    """Flat PPISP values keyed by PPISP parameter name.

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

    controller_prior_exposure: float = 0.0
    """Controller prior exposure [EV] used by the native controller path."""

    controller_responsivity: float | None = None
    """Controller feature-extraction responsivity [dimensionless].

    When ``None``, the controller uses the static PPISP ``responsivity`` camera
    attribute so feature extraction sees the same responsivity-scaled HDR
    radiance as the image PPISP transform.
    """

    controller_weights: tuple[float, ...] | None = None
    """Flattened controller weights.

    USD imports read these from the camera's ``ppisp:controllerWeights``
    attribute. When present, the native controller predicts ``exposureOffset``
    and the four color latents from the HDR image each frame. Static PPISP
    inputs still provide responsivity, vignetting, and CRF.
    """


def normalize_ppisp_cfg(
    ppisp_cfg: PpispCfg | None,
    stage: Any | None = None,
) -> PpispCfg | None:
    """Normalise a :class:`PpispCfg` for downstream consumption.

    * If ``ppisp_cfg`` is ``None``, returns ``None``.
    * If ``ppisp_cfg.camera_prim_path`` is set, requires ``stage`` and merges
      camera-authored USD values with the cfg's explicit overrides (see
      :func:`_merge_camera_attrs_with_cfg`).
    * Otherwise validates ``ppisp_cfg.inputs`` and fills in defaults.
    """
    if ppisp_cfg is None:
        return None
    if not isinstance(ppisp_cfg, PpispCfg):
        raise TypeError(f"Unsupported PPISP configuration type: {type(ppisp_cfg)!r}")
    input_overrides = dict(ppisp_cfg.inputs)
    if ppisp_cfg.camera_prim_path:
        if stage is None:
            raise ValueError("PpispCfg.camera_prim_path requires a USD stage for normalization.")
        return _merge_camera_attrs_with_cfg(ppisp_cfg, stage, input_overrides)
    ppisp_cfg.inputs = _normalized_inputs(input_overrides)
    _finalize_ppisp_cfg(ppisp_cfg)
    return ppisp_cfg


def ppisp_cfg_from_usd_camera(camera_prim: Any) -> PpispCfg:
    """Create :class:`PpispCfg` from a USD camera prim.

    PPISP values are read from camera ``ppisp:*`` attributes. Animated
    attributes are collapsed to their first authored time sample.
    """
    cfg = _ppisp_cfg_from_usd_camera(camera_prim)
    _finalize_ppisp_cfg(cfg)
    cfg.camera_prim_path = None
    return cfg


def _ppisp_cfg_from_usd_camera(camera_prim: Any) -> PpispCfg:
    values = _read_ppisp_inputs_from_camera(camera_prim)
    controller_weights = _read_controller_weights_from_camera(camera_prim)
    if values is None and controller_weights is None:
        camera_path = str(camera_prim.GetPath()) if camera_prim and camera_prim.IsValid() else "<none>"
        raise ValueError(
            f"PPISP camera attributes were not found on camera {camera_path}; expected ppisp:* attributes."
        )

    cfg = PpispCfg(camera_prim_path=str(camera_prim.GetPath()))
    if values is None:
        values = default_ppisp_inputs()
    cfg.inputs = values
    if controller_weights is not None:
        cfg.controller_weights = controller_weights
    return cfg


def ppisp_cfg_from_usd_stage(stage: Any, camera_prim_path: str) -> PpispCfg:
    """Create :class:`PpispCfg` from a camera prim path in a USD stage."""

    return ppisp_cfg_from_usd_camera(_get_camera_prim_at_path(stage, camera_prim_path))


def _get_camera_prim_at_path(stage: Any, camera_prim_path: str) -> Any:
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim or not camera_prim.IsValid():
        raise ValueError(f"PPISP camera prim not found at path: {camera_prim_path}")
    if camera_prim.GetTypeName() != "Camera":
        raise ValueError(f"PPISP prim is not a Camera: {camera_prim_path} ({camera_prim.GetTypeName()})")
    return camera_prim


def _normalized_inputs(inputs: dict[str, Any]) -> dict[str, float | tuple[float, float]]:
    values = default_ppisp_inputs()
    for input_name, value in inputs.items():
        if input_name not in values:
            raise ValueError(f"Unknown PPISP input: {input_name}")
        values[input_name] = _normalize_input_value(input_name, value)
    return values


def _merge_camera_attrs_with_cfg(
    ppisp_cfg: PpispCfg,
    stage: Any,
    input_overrides: dict[str, Any],
) -> PpispCfg:
    assert ppisp_cfg.camera_prim_path is not None
    parsed_cfg = _ppisp_cfg_from_usd_camera(_get_camera_prim_at_path(stage, ppisp_cfg.camera_prim_path))
    normalized_overrides = _normalized_input_overrides(input_overrides)
    if normalized_overrides != PPISP_DEFAULT_INPUTS:
        parsed_cfg.inputs.update(normalized_overrides)
    if ppisp_cfg.controller_weights is not None:
        parsed_cfg.controller_prior_exposure = ppisp_cfg.controller_prior_exposure
        parsed_cfg.controller_weights = ppisp_cfg.controller_weights
    elif ppisp_cfg.controller_prior_exposure != 0.0:
        parsed_cfg.controller_prior_exposure = ppisp_cfg.controller_prior_exposure
    if ppisp_cfg.controller_responsivity is not None:
        parsed_cfg.controller_responsivity = ppisp_cfg.controller_responsivity
    _finalize_ppisp_cfg(parsed_cfg)
    parsed_cfg.camera_prim_path = None
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


def _finalize_ppisp_cfg(ppisp_cfg: PpispCfg) -> None:
    if ppisp_cfg.controller_responsivity is None:
        ppisp_cfg.controller_responsivity = float(ppisp_cfg.inputs["responsivity"])
    else:
        ppisp_cfg.controller_responsivity = float(ppisp_cfg.controller_responsivity)


def _read_first_authored_value(attr: Any) -> Any:
    time_samples = attr.GetTimeSamples()
    if time_samples:
        return attr.Get(time_samples[0])
    return attr.Get()


def _read_ppisp_inputs_from_camera(camera_prim: Any | None) -> dict[str, float | tuple[float, float]] | None:
    if camera_prim is None or not camera_prim.IsValid():
        return None

    values = default_ppisp_inputs()
    found = False
    for input_name in values:
        attr = camera_prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{input_name}")
        if not attr or not attr.IsValid():
            continue
        value = _read_first_authored_value(attr)
        if value is not None:
            values[input_name] = _normalize_input_value(input_name, value)
            found = True
    return values if found else None


def _read_controller_weights_from_camera(camera_prim: Any | None) -> tuple[float, ...] | None:
    if camera_prim is None or not camera_prim.IsValid():
        return None
    attr = camera_prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{PPISP_CONTROLLER_WEIGHTS_CAMERA_ATTR}")
    if not attr or not attr.IsValid():
        return None
    value = _read_first_authored_value(attr)
    if value is None:
        return None
    weights = tuple(float(v) for v in value)
    if len(weights) != PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN:
        raise ValueError(
            "Expected "
            f"{PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN} PPISP controller weights on camera "
            f"{camera_prim.GetPath()}, got {len(weights)}."
        )
    return weights


def _has_ppisp_camera_attrs(camera_prim: Any | None) -> bool:
    if camera_prim is None or not camera_prim.IsValid() or camera_prim.GetTypeName() != "Camera":
        return False
    for input_name in PPISP_DEFAULT_INPUTS:
        attr = camera_prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{input_name}")
        if attr and attr.IsValid() and _read_first_authored_value(attr) is not None:
            return True
    weights_attr = camera_prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{PPISP_CONTROLLER_WEIGHTS_CAMERA_ATTR}")
    return bool(weights_attr and weights_attr.IsValid() and _read_first_authored_value(weights_attr) is not None)


def has_ppisp_camera_attrs(camera_prim: Any | None) -> bool:
    """Return whether a USD camera prim contains recognized PPISP camera attributes.

    Args:
        camera_prim: USD prim to inspect.

    Returns:
        True when ``camera_prim`` is a camera with at least one recognized
        ``ppisp:*`` attribute, otherwise false.
    """
    return _has_ppisp_camera_attrs(camera_prim)


def resolve_and_normalize(isp_cfg: Any, stage: Any, camera_prim_path: str | None = None) -> PpispCfg | None:
    """Resolve a Camera sensor batch's ``isp_cfg`` to a normalised cfg or ``None``.

    Handles all three legal forms of :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg`:

    * ``None`` → returns ``None``.
    * :class:`~isaaclab.sensors.camera.CameraISPMode` sentinel — checks the
      target camera via :func:`auto_camera_ppisp_cfg` (and uses
      :func:`auto_any_ppisp_cfg` for ``AUTO_ANY`` or when no camera path is
      supplied) to discover a PPISP camera. Returns the parsed + normalised
      :class:`PpispCfg`, or ``None`` if no PPISP camera matched.
    * Concrete :class:`PpispCfg` — normalises in place (validates input keys,
      fills defaults, and merges camera-authored USD values when
      ``camera_prim_path`` is set).

    This is the single entry point renderer backends call inside their
    ``prepare_cameras`` hook so :mod:`isaaclab.sensors.camera` does not need
    to know about PPISP types at all. The returned cfg applies to the whole
    Camera sensor batch; callers pass the first matched camera prim path for
    the camera-local discovery phase.

    Args:
        isp_cfg: The Camera sensor's :attr:`isp_cfg` value (``None``, ``CameraISPMode``, or :class:`PpispCfg`).
        stage: USD stage used for sentinel discovery and camera-path resolution.
        camera_prim_path: Optional absolute path of the first matched camera
            prim in the Camera sensor batch. When omitted, discovery uses the
            first camera on the stage with PPISP camera attributes.

    Returns:
        A fully-normalised :class:`PpispCfg`, or ``None`` if the batch has no ISP.
    """
    # Local import avoids a top-of-module dep on isaaclab.sensors.
    from isaaclab.sensors.camera.camera_isp import CameraISPMode

    if isp_cfg is None:
        return None
    if isinstance(isp_cfg, CameraISPMode):
        resolved = auto_camera_ppisp_cfg(stage, camera_prim_path) if camera_prim_path else None
        if resolved is None and (isp_cfg == CameraISPMode.AUTO_ANY or not camera_prim_path):
            resolved = auto_any_ppisp_cfg(stage)
        if resolved is None:
            return None
        return normalize_ppisp_cfg(resolved)
    return normalize_ppisp_cfg(isp_cfg, stage=stage)


def auto_camera_ppisp_cfg(stage: Any, camera_prim_path: str) -> PpispCfg | None:
    """Find PPISP camera attributes for ``camera_prim_path`` on ``stage``.

    Checks only the target camera itself. Use :func:`auto_any_ppisp_cfg` when
    the caller intentionally wants a stage-wide fallback.

    Args:
        stage: USD stage to search.
        camera_prim_path: Absolute camera prim path to match.

    Returns:
        Parsed :class:`PpispCfg` if a matching camera was found, else ``None``.
    """
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not _has_ppisp_camera_attrs(camera_prim):
        return None
    cfg = ppisp_cfg_from_usd_camera(camera_prim)
    cfg.camera_prim_path = None
    return cfg


def auto_any_ppisp_cfg(stage: Any) -> PpispCfg | None:
    """Find the first camera with PPISP attributes anywhere on ``stage``.

    Used as a fallback when no camera is provided, or when the caller requests
    the first available PPISP camera attributes regardless of camera binding.

    Args:
        stage: USD stage to search.

    Returns:
        Parsed :class:`PpispCfg` for the first matching camera, else ``None``.
    """
    for prim in stage.Traverse():
        if _has_ppisp_camera_attrs(prim):
            cfg = ppisp_cfg_from_usd_camera(prim)
            cfg.camera_prim_path = None
            return cfg
    return None
