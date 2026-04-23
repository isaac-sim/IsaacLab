"""Utilities for compiling Isaac Lab marker configs to Newton-friendly draw specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


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
            return NewtonMarkerPrototype(
                name=name,
                renderer="mesh",
                mesh_type="arrow",
                mesh_params={
                    # Keep the native Newton arrow roughly unit-sized so Isaac Lab's
                    # existing cfg.scale and per-instance scales control the final width.
                    "base_radius": 0.2,
                    "base_height": 0.65,
                    "cap_radius": 0.5,
                    "cap_height": 0.35,
                },
                color=color,
                default_scale=default_scale,
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
    """Resolve the simple diffuse color Newton markers support in v1."""

    # Newton markers intentionally support only a simple diffuse color mapping for now.
    material_cfg = getattr(cfg, "visual_material", None)
    if material_cfg is None:
        return (1.0, 0.2, 0.2)

    diffuse_color = getattr(material_cfg, "diffuse_color", None)
    if diffuse_color is None:
        return (1.0, 0.2, 0.2)

    return tuple(float(v) for v in diffuse_color)
