# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named grasp candidates with separate reset positions and preload targets.

This adapts the reset-authoring contract from Isaac Lab PR 7735, rather than
transferring its KUKA/Allegro-specific joint values. A saved candidate is not a
contact-valid grasp unless its separate free-spoon validation says so.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .robot import (
    ARM_JOINT_NAMES,
    ASSET_BUNDLE_SHA256,
    COFFEE_READY_POSITIONS,
    DEFAULT_SPOON_OFFSET_POS,
    DEFAULT_SPOON_OFFSET_ROT,
    HAND_HOLD_POSITIONS,
    HAND_JOINT_LIMITS,
    HAND_JOINT_NAMES,
)


@dataclass
class CoffeeGraspReset:
    """One robot-specific zero-velocity reset and its intended motor targets.

    Joint mappings are in radians. The spoon transform is relative to the
    canonical palm, in metres and XYZW quaternion order. The optional contact
    report describes an isolated rigid-body retention test, not MPM tool use.
    """

    joint_positions: dict[str, float]
    joint_targets: dict[str, float]
    spoon_offset_pos: tuple[float, float, float] = DEFAULT_SPOON_OFFSET_POS
    spoon_offset_rot: tuple[float, float, float, float] = DEFAULT_SPOON_OFFSET_ROT
    schema_version: int = 1
    asset_bundle_sha256: str = ASSET_BUNDLE_SHA256
    source: str = "measured coffee-ready mounted candidate"
    contact_validation: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Reject mismatched assets, incomplete orders and invalid joint values."""
        if self.schema_version != 1 or self.asset_bundle_sha256 != ASSET_BUNDLE_SHA256:
            raise ValueError("The grasp reset schema or Rizon/Sharpa asset fingerprint does not match.")
        names = set(ARM_JOINT_NAMES + HAND_JOINT_NAMES)
        for label, values in (("positions", self.joint_positions), ("targets", self.joint_targets)):
            if set(values) != names or not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Grasp joint {label} must contain all 29 named, finite values.")
            for name, (lower, upper) in zip(HAND_JOINT_NAMES, HAND_JOINT_LIMITS, strict=True):
                if not lower <= values[name] <= upper:
                    raise ValueError(f"Grasp joint {label} for {name} exceeds the Sharpa limits.")
        if len(self.spoon_offset_pos) != 3 or len(self.spoon_offset_rot) != 4:
            raise ValueError("The spoon transform requires XYZ position and XYZW rotation.")
        if not all(math.isfinite(value) for value in (*self.spoon_offset_pos, *self.spoon_offset_rot)):
            raise ValueError("The spoon transform must be finite.")
        norm = math.sqrt(sum(value * value for value in self.spoon_offset_rot))
        if abs(norm - 1.0) > 1.0e-4:
            raise ValueError("The spoon rotation must be a unit XYZW quaternion.")

    def save(self, path: str | Path) -> None:
        """Write an editable JSON candidate after validating its contract."""
        self.validate()
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")


def default_grasp_reset() -> CoffeeGraspReset:
    """Return the measured display pinch, explicitly without contact certification."""
    positions = dict(zip(ARM_JOINT_NAMES + HAND_JOINT_NAMES, COFFEE_READY_POSITIONS + HAND_HOLD_POSITIONS, strict=True))
    return CoffeeGraspReset(joint_positions=positions, joint_targets=positions.copy())


def load_grasp_reset(path: str | Path) -> CoffeeGraspReset:
    """Load an authored candidate with an exact asset and named-joint contract."""
    state = CoffeeGraspReset(**json.loads(Path(path).expanduser().read_text()))
    state.validate()
    state.spoon_offset_pos = tuple(state.spoon_offset_pos)
    state.spoon_offset_rot = tuple(state.spoon_offset_rot)
    return state


def capture_grasp_reset(env, env_id: int = 0) -> CoffeeGraspReset:
    """Capture measured robot joints, motor targets and palm-local tool pose."""
    from isaaclab.utils import math as math_utils

    robot = env.scene["robot"]
    names = ARM_JOINT_NAMES + HAND_JOINT_NAMES
    joint_ids = robot.find_joints(names, preserve_order=True)[0]
    palm = robot.data.body_link_pose_w.torch[env_id, env._palm_id]
    spoon = env.scene["spoon"].data.root_link_pose_w.torch[env_id]
    position, rotation = math_utils.subtract_frame_transforms(palm[:3], palm[3:], spoon[:3], spoon[3:])
    state = CoffeeGraspReset(
        joint_positions=dict(zip(names, robot.data.joint_pos.torch[env_id, joint_ids].tolist(), strict=True)),
        joint_targets=dict(
            zip(names, robot.actuators.target_command.position.torch[env_id, joint_ids].tolist(), strict=True)
        ),
        spoon_offset_pos=tuple(position.tolist()),
        spoon_offset_rot=tuple(rotation.tolist()),
        source=f"measured simulator snapshot; grasp mode {env.cfg.grasp_mode}",
    )
    state.validate()
    return state


def configure_grasp_reset(cfg) -> None:
    """Apply an optional saved candidate before the environment is constructed."""
    if cfg.grasp_reset_path is None and cfg.grasp_mode == "contact":
        cfg.grasp_reset_path = str(Path(__file__).with_name("assets") / "contact_grasp.json")
    if cfg.grasp_reset_path is None:
        return
    state = load_grasp_reset(cfg.grasp_reset_path)
    cfg.scene.robot.init_state.joint_pos = state.joint_positions.copy()
    cfg.grasp_reset_joint_targets = state.joint_targets.copy()
    cfg.spoon_offset_pos = state.spoon_offset_pos
    cfg.spoon_offset_rot = state.spoon_offset_rot


def configured_hand_targets(cfg) -> tuple[float, ...]:
    """Return authored preload targets, or the default spoon-holding posture."""
    if cfg.grasp_reset_joint_targets is None:
        return HAND_HOLD_POSITIONS
    return tuple(cfg.grasp_reset_joint_targets[name] for name in HAND_JOINT_NAMES)
