# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure data structures for procedural SO101 keyboard assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

KeyboardFamily = Literal[
    "ansi_60",
    "ansi_tkl",
    "ansi_full",
    "numpad",
    "macro_pad",
    "ortholinear",
    "split_columnar",
    "compact",
    "random",
]
TopologyMode = Literal["exact", "global_padded", "bucketed"]
PartitionMode = Literal["single", "fixed_dof"]
PartitionTailPolicy = Literal["pad_tail", "ragged"]

DEFAULT_FAMILIES: tuple[str, ...] = (
    "ansi_60",
    "ansi_tkl",
    "ansi_full",
    "numpad",
    "macro_pad",
    "ortholinear",
    "split_columnar",
    "compact",
)
DEFAULT_BUCKET_SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 88, 108)
DEFAULT_MAX_SLOTS = 108
DEFAULT_PARTITION_DOF = 6

# --- Key press convention -------------------------------------------------------------------------
# Every key is a prismatic joint along the deck normal (-Z) authored with drive target 0, so:
#   * released (rest):       joint position == 0
#   * bottomed (full press): joint position == lower_limit == -travel
# A keystroke registers when the key is depressed past ``KEY_ACTUATION_FRACTION`` of its travel,
# i.e. ``joint_position <= upper_limit - KEY_ACTUATION_FRACTION * travel``. The detection is
# edge-triggered: holding the key past the actuation point is a single keystroke, and the key must
# return above the actuation point to re-arm.
KEY_ACTUATION_FRACTION = 0.5


@dataclass(frozen=True)
class UnitKey:
    """Logical key geometry in keyboard units before metric resolution."""

    name: str
    label: str
    x: float
    y: float
    w: float = 1.0
    h: float = 1.0
    row: int = 0
    group: str = "alpha"
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class LogicalKeyboardLayout:
    """Selected logical keyboard layout before metric geometry resolution."""

    family: str
    keys: tuple[UnitKey, ...]


@dataclass(frozen=True)
class KeyboardStyle:
    """Coherent visual and physical archetype."""

    name: str
    pitch: float
    gap: float
    cap_height: float
    cap_top_scale: float
    travel: float
    upper_limit: float
    cap_clearance: float
    case_margin_u: float
    base_thickness: float
    plate_thickness: float
    key_mass: float
    base_mass: float
    stiffness: float
    damping: float
    max_force: float
    case_color: tuple[float, float, float]
    plate_color: tuple[float, float, float]
    alpha_color: tuple[float, float, float]
    modifier_color: tuple[float, float, float]
    accent_color: tuple[float, float, float]
    label_color: tuple[float, float, float]
    label_scale: float
    label_pixel_fill: float
    label_mode: str
    label_spacebar: bool
    base_profile: str = "slab"
    deck_tilt: float = 0.0
    case_edge_color: tuple[float, float, float] = (0.12, 0.12, 0.12)
    foot_color: tuple[float, float, float] = (0.035, 0.035, 0.035)
    inactive_color: tuple[float, float, float] = (0.08, 0.08, 0.08)


@dataclass(frozen=True)
class BaseVisual:
    """Additional visual-only case pieces shared by USD export and viser inspection."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ResolvedKey:
    """Metric key slot ready for inspection or USD authoring."""

    slot: int
    active: bool
    name: str
    label: str
    group: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    collision_size: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]
    travel: float
    lower_limit: float
    upper_limit: float
    mass: float
    stiffness: float
    damping: float
    max_force: float
    color: tuple[float, float, float]


@dataclass(frozen=True)
class ResolvedKeyboard:
    """Fully resolved keyboard geometry and metadata."""

    family: str
    style_name: str
    seed: int
    topology_mode: str
    active_key_count: int
    slot_count: int
    partition_mode: str
    partition_dof: int
    partition_tail_policy: str
    partition_count: int
    keys: tuple[ResolvedKey, ...]
    case_center: tuple[float, float, float]
    case_size: tuple[float, float, float]
    plate_center: tuple[float, float, float]
    plate_size: tuple[float, float, float]
    deck_quat_xyzw: tuple[float, float, float, float]
    style: KeyboardStyle
    warnings: tuple[str, ...] = ()

    @property
    def active_keys(self) -> tuple[ResolvedKey, ...]:
        """Return active key slots only."""

        return tuple(key for key in self.keys if key.active)

    def key_partition(self, slot: int) -> tuple[int, int]:
        """Return ``(partition_index, local_slot)`` for a global key slot."""

        if self.partition_mode == "fixed_dof":
            return slot // self.partition_dof, slot % self.partition_dof
        return 0, slot

    def manifest(self) -> dict[str, Any]:
        """Return JSON-serializable metadata for debugging or downstream lookup."""

        from .keyboard_labels import legend_text

        return {
            "family": self.family,
            "style_name": self.style_name,
            "base_profile": self.style.base_profile,
            "deck_tilt": self.style.deck_tilt,
            "deck_quat_xyzw": self.deck_quat_xyzw,
            "seed": self.seed,
            "topology_mode": self.topology_mode,
            "active_key_count": self.active_key_count,
            "slot_count": self.slot_count,
            "partition_mode": self.partition_mode,
            "partition_dof": self.partition_dof,
            "partition_tail_policy": self.partition_tail_policy,
            "partition_count": self.partition_count,
            "case_center": self.case_center,
            "case_size": self.case_size,
            "plate_center": self.plate_center,
            "plate_size": self.plate_size,
            "active_key_mask": [key.active for key in self.keys],
            "keys": [
                {
                    "slot": key.slot,
                    "active": key.active,
                    "name": key.name,
                    "label": key.label,
                    "group": key.group,
                    "joint_name": f"key_{key.slot:03d}_joint",
                    "link_name": f"key_{key.slot:03d}",
                    "partition_index": self.key_partition(key.slot)[0],
                    "partition_local_slot": self.key_partition(key.slot)[1],
                    "partition_name": f"part_{self.key_partition(key.slot)[0]:03d}",
                    "center": key.center,
                    "size": key.size,
                    "collision_size": key.collision_size,
                    "quat_xyzw": key.quat_xyzw,
                    "travel": key.travel,
                    "lower_limit": key.lower_limit,
                    "upper_limit": key.upper_limit,
                    "label_text": legend_text(key.label, self.style),
                }
                for key in self.keys
            ],
            "warnings": list(self.warnings),
        }
