# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure keyboard geometry, labels, validation, and math helpers."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .keyboard_layouts import sample_layout
from .keyboard_schema import BaseVisual, KeyboardStyle, ResolvedKey, ResolvedKeyboard, UnitKey
from .keyboard_styles import sample_style

if TYPE_CHECKING:
    from .keyboard_gen_cfg import KeyboardSpawnerCfg


def generate_keyboard(cfg: KeyboardSpawnerCfg) -> ResolvedKeyboard:
    """Generate deterministic in-memory keyboard geometry from a spawner configuration."""

    _validate_generation_cfg(cfg)
    rng = random.Random(cfg.seed)
    layout = sample_layout(cfg.family, cfg.families, rng)
    family = layout.family
    unit_keys = tuple(layout.keys)
    style = sample_style(cfg.style, family, rng)
    deck_quat = _deck_quat(style)
    active_keys = _generate_keys(unit_keys, style, cfg)
    slot_count = _slot_count(len(active_keys), cfg)
    slot_count = _partition_slot_count(slot_count, cfg)
    if len(active_keys) > slot_count:
        raise ValueError(
            f"Layout '{family}' produced {len(active_keys)} keys, but topology '{cfg.topology_mode}' "
            f"only exposes {slot_count} slots."
        )
    keys = list(active_keys)
    for slot in range(len(keys), slot_count):
        keys.append(_inactive_key(slot, style))
    case_center, case_size, plate_center, plate_size = _case_geometry(unit_keys, style)
    case_center = quat_rotate(deck_quat, case_center)
    plate_center = quat_rotate(deck_quat, plate_center)
    warnings = tuple(_validate_keyboard(keys, case_size))
    if cfg.strict_validation and warnings:
        raise ValueError("Invalid generated keyboard:\n  - " + "\n  - ".join(warnings))
    return ResolvedKeyboard(
        family=family,
        style_name=style.name,
        seed=cfg.seed,
        topology_mode=cfg.topology_mode,
        active_key_count=len(active_keys),
        slot_count=slot_count,
        partition_mode=cfg.partition_mode,
        partition_dof=cfg.partition_dof,
        partition_tail_policy=cfg.partition_tail_policy,
        partition_count=_partition_count(slot_count, cfg),
        keys=tuple(keys),
        case_center=case_center,
        case_size=case_size,
        plate_center=plate_center,
        plate_size=plate_size,
        deck_quat_xyzw=deck_quat,
        style=style,
        warnings=warnings,
    )


def _validate_generation_cfg(cfg: KeyboardSpawnerCfg) -> None:
    if cfg.topology_mode not in ("exact", "global_padded", "bucketed"):
        raise ValueError(f"Unsupported topology_mode '{cfg.topology_mode}'.")
    if cfg.max_slots < 1:
        raise ValueError("max_slots must be >= 1.")
    if cfg.topology_mode == "bucketed" and not cfg.bucket_sizes:
        raise ValueError("bucketed topology requires at least one bucket size.")
    if cfg.partition_mode not in ("single", "fixed_dof"):
        raise ValueError(f"Unsupported partition_mode '{cfg.partition_mode}'.")
    if cfg.partition_tail_policy not in ("pad_tail", "ragged"):
        raise ValueError(f"Unsupported partition_tail_policy '{cfg.partition_tail_policy}'.")
    if cfg.partition_dof < 1:
        raise ValueError("partition_dof must be >= 1.")


def _slot_count(active_count: int, cfg: KeyboardSpawnerCfg) -> int:
    if cfg.topology_mode == "exact":
        return active_count
    if cfg.topology_mode == "global_padded":
        return cfg.max_slots
    for bucket in sorted(cfg.bucket_sizes):
        if active_count <= bucket:
            return bucket
    if active_count <= cfg.max_slots:
        return cfg.max_slots
    return active_count


def _partition_slot_count(slot_count: int, cfg: KeyboardSpawnerCfg) -> int:
    if cfg.partition_mode != "fixed_dof" or cfg.partition_tail_policy != "pad_tail":
        return slot_count
    return _partition_count(slot_count, cfg) * cfg.partition_dof


def _partition_count(slot_count: int, cfg: KeyboardSpawnerCfg) -> int:
    if cfg.partition_mode != "fixed_dof":
        return 1
    return max(1, math.ceil(slot_count / cfg.partition_dof))


def _generate_keys(unit_keys: Sequence[UnitKey], style: KeyboardStyle, cfg: KeyboardSpawnerCfg) -> list[ResolvedKey]:
    min_x = min(key.x for key in unit_keys)
    min_y = min(key.y for key in unit_keys)
    max_x = max(key.x + key.w for key in unit_keys)
    max_y = max(key.y + key.h for key in unit_keys)
    center_u = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
    plate_top = style.base_thickness * 0.5
    cap_center_z = plate_top + style.cap_clearance + style.cap_height * 0.5
    deck_quat = _deck_quat(style)

    resolved: list[ResolvedKey] = []
    for slot, key in enumerate(unit_keys):
        size_x = max(key.w * style.pitch - style.gap, style.pitch * 0.25)
        size_y = max(key.h * style.pitch - style.gap, style.pitch * 0.25)
        size = (size_x, size_y, style.cap_height)
        local_center = (
            (key.x + key.w * 0.5 - center_u[0]) * style.pitch,
            (key.y + key.h * 0.5 - center_u[1]) * style.pitch,
            cap_center_z,
        )
        local_quat = _quat_from_euler(key.roll, key.pitch, key.yaw)
        center = quat_rotate(deck_quat, local_center)
        quat_xyzw = _quat_mul(deck_quat, local_quat)
        resolved.append(
            ResolvedKey(
                slot=slot,
                active=True,
                name=_safe_name(key.name),
                label=key.label,
                group=key.group,
                center=center,
                size=size,
                collision_size=inset_box(size, cfg.key_collision_margin),
                quat_xyzw=quat_xyzw,
                travel=style.travel,
                lower_limit=-style.travel,
                upper_limit=style.upper_limit,
                mass=style.key_mass,
                stiffness=style.stiffness,
                damping=style.damping,
                max_force=style.max_force,
                color=_key_color(key.group, style),
            )
        )
    return resolved


def _case_geometry(
    unit_keys: Sequence[UnitKey], style: KeyboardStyle
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    min_x = min(key.x for key in unit_keys) - style.case_margin_u
    min_y = min(key.y for key in unit_keys) - style.case_margin_u
    max_x = max(key.x + key.w for key in unit_keys) + style.case_margin_u
    max_y = max(key.y + key.h for key in unit_keys) + style.case_margin_u
    width = (max_x - min_x) * style.pitch
    depth = (max_y - min_y) * style.pitch
    case_size = (width, depth, style.base_thickness)
    case_center = (0.0, 0.0, 0.0)
    plate_size = (
        max(width - style.pitch * 0.35, style.pitch),
        max(depth - style.pitch * 0.35, style.pitch),
        style.plate_thickness,
    )
    plate_center = (0.0, 0.0, style.base_thickness * 0.5 + style.plate_thickness * 0.5)
    return case_center, case_size, plate_center, plate_size


def _inactive_key(slot: int, style: KeyboardStyle) -> ResolvedKey:
    return ResolvedKey(
        slot=slot,
        active=False,
        name=f"inactive_{slot:03d}",
        label="",
        group="inactive",
        center=(0.0, 0.0, -0.05 - 0.0001 * slot),
        size=(0.002, 0.002, 0.002),
        collision_size=(0.001, 0.001, 0.001),
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        travel=0.0,
        lower_limit=0.0,
        upper_limit=0.0,
        mass=0.001,
        stiffness=style.stiffness,
        damping=style.damping,
        max_force=style.max_force,
        color=style.inactive_color,
    )


def _validate_keyboard(keys: Sequence[ResolvedKey], case_size: tuple[float, float, float]) -> list[str]:
    warnings: list[str] = []
    slots = [key.slot for key in keys]
    if len(slots) != len(set(slots)):
        warnings.append("duplicate key slots")
    active = [key for key in keys if key.active]
    names = [key.name for key in active]
    if len(names) != len(set(names)):
        warnings.append("duplicate active key names")
    for key in active:
        if key.lower_limit > key.upper_limit:
            warnings.append(f"{key.name}: lower limit greater than upper limit")
        if key.travel < 0.0:
            warnings.append(f"{key.name}: negative travel")
        if key.mass <= 0.0:
            warnings.append(f"{key.name}: non-positive mass")

    aabbs = [(key, *_key_aabb_xy(key)) for key in active]
    aabbs.sort(key=lambda item: item[1][0])
    for i, (key_a, a_min, a_max) in enumerate(aabbs):
        for key_b, b_min, b_max in aabbs[i + 1 :]:
            if b_min[0] - a_max[0] > 1.0e-5:
                break
            overlap_y = min(a_max[1], b_max[1]) - max(a_min[1], b_min[1])
            if overlap_y > 1.0e-5:
                warnings.append(f"{key_a.name} overlaps {key_b.name}")
                if len(warnings) > 16:
                    warnings.append("additional overlap warnings truncated")
                    return warnings
    if case_size[0] <= 0.0 or case_size[1] <= 0.0 or case_size[2] <= 0.0:
        warnings.append("case has non-positive dimensions")
    return warnings


def _key_aabb_xy(key: ResolvedKey) -> tuple[tuple[float, float], tuple[float, float]]:
    half_x = key.size[0] * 0.5
    half_y = key.size[1] * 0.5
    corners = [(-half_x, -half_y, 0.0), (half_x, -half_y, 0.0), (half_x, half_y, 0.0), (-half_x, half_y, 0.0)]
    rotated = [quat_rotate(key.quat_xyzw, corner) for corner in corners]
    xs = [key.center[0] + point[0] for point in rotated]
    ys = [key.center[1] + point[1] for point in rotated]
    return (min(xs), min(ys)), (max(xs), max(ys))


def _key_color(group: str, style: KeyboardStyle) -> tuple[float, float, float]:
    if group == "accent":
        return style.accent_color
    if group == "modifier":
        return style.modifier_color
    if group == "inactive":
        return style.inactive_color
    return style.alpha_color


def _deck_quat(style: KeyboardStyle) -> tuple[float, float, float, float]:
    if abs(style.deck_tilt) < 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return _quat_from_euler(style.deck_tilt, 0.0, 0.0)


def _apply_deck_quat(
    keyboard: ResolvedKeyboard,
    visual: BaseVisual,
) -> BaseVisual:
    return BaseVisual(
        visual.name,
        quat_rotate(keyboard.deck_quat_xyzw, visual.center),
        visual.size,
        visual.color,
        keyboard.deck_quat_xyzw,
    )


# Fixed number of case-trim visual boxes authored per keyboard, regardless of base profile. Every
# variant pads to this count so the per-env shape topology is identical (required by Newton's shared
# ArticulationView). Must be >= the most any single profile emits.
_BASE_VISUAL_SLOTS = 6


def base_profile_visuals(keyboard: ResolvedKeyboard) -> tuple[BaseVisual, ...]:
    style = keyboard.style
    profile = style.base_profile
    cx, cy, cz = keyboard.case_center
    sx, sy, sz = keyboard.case_size
    top_z = cz + sz * 0.5
    bottom_z = cz - sz * 0.5
    pitch = style.pitch
    edge = max(min(pitch * 0.18, sx * 0.06, sy * 0.06), 0.0011)
    rail_h = max(style.plate_thickness * 1.15, 0.0008)
    visuals: list[BaseVisual] = []

    if profile == "tray":
        visuals.extend(
            _perimeter_rails("tray", cx, cy, sx, sy, edge * 1.25, rail_h * 1.35, top_z, style.case_edge_color)
        )
    elif profile == "low_bezel":
        visuals.extend(
            _perimeter_rails("bezel", cx, cy, sx, sy, edge * 0.70, rail_h * 0.75, top_z, style.case_edge_color)
        )
    elif profile == "sandwich":
        strip_h = max(sz * 0.22, 0.0012)
        strip_z = cz - sz * 0.12
        visuals.extend(
            _perimeter_rails(
                "sandwich",
                cx,
                cy,
                sx,
                sy,
                edge * 0.95,
                strip_h,
                strip_z - strip_h * 0.5,
                style.case_edge_color,
            )
        )
        front_depth = max(edge * 1.5, pitch * 0.12)
        visuals.append(
            BaseVisual(
                "front_accent_lip",
                (cx, cy - sy * 0.5 + front_depth * 0.5, top_z + rail_h * 0.45),
                (sx, front_depth, rail_h * 0.90),
                style.case_edge_color,
            )
        )
    elif profile == "floating":
        foot_h = max(sz * 0.34, 0.0011)
        foot_x = min(max(pitch * 0.85, 0.004), sx * 0.18)
        foot_y = min(max(pitch * 0.58, 0.003), sy * 0.18)
        x_offset = max(sx * 0.34, foot_x * 0.75)
        y_offset = max(sy * 0.31, foot_y * 0.75)
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                visuals.append(
                    BaseVisual(
                        f"floating_foot_{'r' if x_sign > 0 else 'l'}_{'b' if y_sign > 0 else 'f'}",
                        (cx + x_sign * x_offset, cy + y_sign * y_offset, bottom_z - foot_h * 0.5),
                        (foot_x, foot_y, foot_h),
                        style.foot_color,
                    )
                )
    elif profile == "wedge":
        rear_depth = min(max(pitch * 1.25, edge * 2.2), sy * 0.28)
        riser_h = max(sz * 0.62, 0.0024)
        front_h = max(sz * 0.16, 0.0008)
        visuals.extend(
            [
                BaseVisual(
                    "rear_riser",
                    (cx, cy + sy * 0.5 - rear_depth * 0.5, bottom_z - riser_h * 0.5),
                    (sx * 0.96, rear_depth, riser_h),
                    style.foot_color,
                ),
                BaseVisual(
                    "front_low_foot",
                    (cx, cy - sy * 0.5 + rear_depth * 0.22, bottom_z - front_h * 0.5),
                    (sx * 0.74, rear_depth * 0.34, front_h),
                    style.foot_color,
                ),
                BaseVisual(
                    "rear_top_lip",
                    (cx, cy + sy * 0.5 - edge * 0.65, top_z + rail_h * 0.55),
                    (sx, edge * 1.30, rail_h * 1.10),
                    style.case_edge_color,
                ),
            ]
        )
    elif profile == "rimless":
        bevel_h = max(style.plate_thickness * 0.45, 0.00045)
        bevel_depth = max(edge * 0.65, pitch * 0.06)
        visuals.extend(
            [
                BaseVisual(
                    "front_micro_bevel",
                    (cx, cy - sy * 0.5 + bevel_depth * 0.5, top_z + bevel_h * 0.35),
                    (sx * 0.96, bevel_depth, bevel_h),
                    style.case_edge_color,
                ),
                BaseVisual(
                    "rear_micro_bevel",
                    (cx, cy + sy * 0.5 - bevel_depth * 0.5, top_z + bevel_h * 0.35),
                    (sx * 0.96, bevel_depth, bevel_h),
                    style.case_edge_color,
                ),
            ]
        )
    elif profile != "slab":
        raise ValueError(f"Unsupported keyboard base_profile '{profile}'.")
    # Pad to a fixed trim-box count so every variant authors identical shape topology. Newton's shared
    # ArticulationView requires uniform per-world strides (visual shapes included), so a variant-
    # dependent trim count breaks it. Padding boxes are tiny (effectively invisible) and visual-only.
    if len(visuals) > _BASE_VISUAL_SLOTS:
        raise ValueError(
            f"base_profile '{profile}' emits {len(visuals)} trim pieces (> _BASE_VISUAL_SLOTS="
            f"{_BASE_VISUAL_SLOTS}); raise _BASE_VISUAL_SLOTS to keep variant topology uniform."
        )
    while len(visuals) < _BASE_VISUAL_SLOTS:
        visuals.append(
            BaseVisual(f"trim_pad_{len(visuals):02d}", (cx, cy, bottom_z), (1.0e-5, 1.0e-5, 1.0e-5), style.foot_color)
        )
    return tuple(_apply_deck_quat(keyboard, visual) for visual in visuals)


def _perimeter_rails(
    prefix: str,
    cx: float,
    cy: float,
    width: float,
    depth: float,
    edge: float,
    height: float,
    base_z: float,
    color: tuple[float, float, float],
) -> list[BaseVisual]:
    edge = min(edge, width * 0.24, depth * 0.24)
    center_z = base_z + height * 0.5
    inner_width = max(width - 2.0 * edge, edge)
    return [
        BaseVisual(
            f"{prefix}_left_rail",
            (cx - (width - edge) * 0.5, cy, center_z),
            (edge, depth, height),
            color,
        ),
        BaseVisual(
            f"{prefix}_right_rail",
            (cx + (width - edge) * 0.5, cy, center_z),
            (edge, depth, height),
            color,
        ),
        BaseVisual(
            f"{prefix}_front_rail",
            (cx, cy - (depth - edge) * 0.5, center_z),
            (inner_width, edge, height),
            color,
        ),
        BaseVisual(
            f"{prefix}_rear_rail",
            (cx, cy + (depth - edge) * 0.5, center_z),
            (inner_width, edge, height),
            color,
        ),
    ]


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def inset_box(size: tuple[float, float, float], margin: float) -> tuple[float, float, float]:
    return (
        max(size[0] - 2.0 * margin, size[0] * 0.25),
        max(size[1] - 2.0 * margin, size[1] * 0.25),
        max(size[2] - 2.0 * min(margin, size[2] * 0.2), size[2] * 0.5),
    )


def is_identity_quat(quat_xyzw: tuple[float, float, float, float]) -> bool:
    return (
        abs(quat_xyzw[0]) < 1.0e-9
        and abs(quat_xyzw[1]) < 1.0e-9
        and abs(quat_xyzw[2]) < 1.0e-9
        and abs(quat_xyzw[3] - 1.0) < 1.0e-9
    )


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def _quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_conj(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def quat_rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    p = (v[0], v[1], v[2], 0.0)
    r = _quat_mul(_quat_mul(q, p), _quat_conj(q))
    return (r[0], r[1], r[2])


def quat_to_matrix(
    q: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def mat3_rotate(
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2],
        matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2],
        matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2],
    )
