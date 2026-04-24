"""Utilities for compiling Isaac Lab marker configs to Newton-friendly draw specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

import torch

# Match the color-resolution semantics used in ``sim.utils.newton_model_utils``:
# - OmniPBR defaults: diffuse_color_constant * diffuse_tint
# - otherwise use authored/tint-like color when we can infer it from config
# - fall back to a neutral 18% gray rather than an arbitrary debug color
_OMNIPBR_DEFAULTS = {
    "diffuse_color_constant": (0.2, 0.2, 0.2),
    "diffuse_tint": (1.0, 1.0, 1.0),
}
_UNBOUND_DEFAULT_FALLBACK_GRAY = (0.18, 0.18, 0.18)


@dataclass(frozen=True)
class NewtonMarkerPrototype:
    """Newton-friendly description of a marker prototype."""

    name: str
    renderer: Literal["mesh", "frame", "unsupported"]
    mesh_type: Literal["arrow", "box", "sphere", "cylinder", "capsule", "cone"] | None
    mesh_params: dict[str, float | tuple[float, float, float]]
    color: tuple[float, float, float]
    default_scale: tuple[float, float, float]
    visible: bool = True


@dataclass
class NewtonMarkerGroupState:
    """Mutable per-group marker state consumed by Newton-family visualizers."""

    group_id: str
    prototypes: tuple[NewtonMarkerPrototype, ...]
    visible: bool = True
    translations: torch.Tensor | None = None
    orientations: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    marker_indices: torch.Tensor | None = None
    count: int = 0


def compile_markers_cfg_for_newton(markers_cfg: dict[str, object]) -> tuple[NewtonMarkerPrototype, ...]:
    """Compile marker spawn configs into Newton-friendly prototype specs."""

    return tuple(_compile_single_marker(name, cfg) for name, cfg in markers_cfg.items())


def _compile_single_marker(name: str, cfg: object) -> NewtonMarkerPrototype:
    cfg_type = type(cfg).__name__
    color = _extract_diffuse_color(cfg)
    visible = bool(getattr(cfg, "visible", True))
    default_scale = _extract_scale_hint(cfg)

    if cfg_type == "SphereCfg":
        return NewtonMarkerPrototype(
            name=name,
            renderer="mesh",
            mesh_type="sphere",
            mesh_params={"radius": float(cfg.radius)},
            color=color,
            default_scale=(1.0, 1.0, 1.0),
            visible=visible,
        )

    if cfg_type == "CuboidCfg":
        size = tuple(float(v) for v in cfg.size)
        return NewtonMarkerPrototype(
            name=name,
            renderer="mesh",
            mesh_type="box",
            mesh_params={"size": size},
            color=color,
            default_scale=(1.0, 1.0, 1.0),
            visible=visible,
        )

    if cfg_type == "CylinderCfg":
        return NewtonMarkerPrototype(
            name=name,
            renderer="mesh",
            mesh_type="cylinder",
            mesh_params={"radius": float(cfg.radius), "height": float(cfg.height)},
            color=color,
            default_scale=(1.0, 1.0, 1.0),
            visible=visible,
        )

    if cfg_type == "CapsuleCfg":
        return NewtonMarkerPrototype(
            name=name,
            renderer="mesh",
            mesh_type="capsule",
            mesh_params={"radius": float(cfg.radius), "height": float(cfg.height)},
            color=color,
            default_scale=(1.0, 1.0, 1.0),
            visible=visible,
        )

    if cfg_type == "ConeCfg":
        return NewtonMarkerPrototype(
            name=name,
            renderer="mesh",
            mesh_type="cone",
            mesh_params={"radius": float(cfg.radius), "height": float(cfg.height)},
            color=color,
            default_scale=(1.0, 1.0, 1.0),
            visible=visible,
        )

    if cfg_type == "UsdFileCfg":
        usd_path = str(getattr(cfg, "usd_path", "")).lower()
        if usd_path.endswith("arrow_x.usd"):
            widened_arrow_scale = (
                default_scale[0],
                default_scale[1] * 2.5,
                default_scale[2] * 2.5,
            )
            return NewtonMarkerPrototype(
                name=name,
                renderer="mesh",
                mesh_type="arrow",
                mesh_params={
                    "base_radius": 0.08,
                    "base_height": 0.7,
                    "cap_radius": 0.16,
                    "cap_height": 0.3,
                },
                color=color,
                default_scale=widened_arrow_scale,
                visible=visible,
            )
        if usd_path.endswith("frame_prim.usd"):
            return NewtonMarkerPrototype(
                name=name,
                renderer="frame",
                mesh_type=None,
                mesh_params={},
                color=color,
                default_scale=default_scale,
                visible=visible,
            )
        if "dex_cube" in usd_path or "cube" in usd_path:
            return NewtonMarkerPrototype(
                name=name,
                renderer="mesh",
                mesh_type="box",
                mesh_params={"size": (1.0, 1.0, 1.0)},
                color=color,
                default_scale=default_scale,
                visible=visible,
            )

    return NewtonMarkerPrototype(
        name=name,
        renderer="unsupported",
        mesh_type=None,
        mesh_params={},
        color=color,
        default_scale=default_scale,
        visible=visible,
    )


def _extract_scale_hint(cfg: object) -> tuple[float, float, float]:
    scale = getattr(cfg, "scale", None)
    if scale is None:
        return (1.0, 1.0, 1.0)
    return tuple(float(v) for v in scale)


def _extract_diffuse_color(cfg: object) -> tuple[float, float, float]:
    """Resolve a marker color using the same policy as Newton model color replacement."""

    material_cfg = getattr(cfg, "visual_material", None)
    if material_cfg is None:
        return _UNBOUND_DEFAULT_FALLBACK_GRAY

    if color := _extract_omnipbr_like_color(material_cfg):
        return color

    for attr_name in ("diffuse_color", "glass_color", "color"):
        if color := _extract_rgb(getattr(material_cfg, attr_name, None)):
            return color

    return _UNBOUND_DEFAULT_FALLBACK_GRAY


def _extract_omnipbr_like_color(material_cfg: object) -> tuple[float, float, float] | None:
    """Resolve OmniPBR-style albedo as diffuse_color_constant * diffuse_tint."""

    diffuse_constant = _extract_rgb(getattr(material_cfg, "diffuse_color_constant", None))
    diffuse_tint = _extract_rgb(getattr(material_cfg, "diffuse_tint", None))

    # Some config classes only expose a tint/brightness-like view of OmniPBR; honor those too.
    if diffuse_constant is None and hasattr(material_cfg, "albedo_brightness"):
        brightness = getattr(material_cfg, "albedo_brightness", None)
        if brightness is not None:
            diffuse_constant = (float(brightness), float(brightness), float(brightness))

    if diffuse_constant is None and diffuse_tint is None:
        mdl_path = str(getattr(material_cfg, "mdl_path", "")).lower()
        if not mdl_path.endswith("omnipbr.mdl"):
            return None

    diffuse_constant = diffuse_constant or _OMNIPBR_DEFAULTS["diffuse_color_constant"]
    diffuse_tint = diffuse_tint or _OMNIPBR_DEFAULTS["diffuse_tint"]
    return (
        diffuse_constant[0] * diffuse_tint[0],
        diffuse_constant[1] * diffuse_tint[1],
        diffuse_constant[2] * diffuse_tint[2],
    )


def _extract_rgb(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        rgb = tuple(float(v) for v in value)
    except TypeError:
        return None
    if len(rgb) < 3:
        return None
    return (rgb[0], rgb[1], rgb[2])
