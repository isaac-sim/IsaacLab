# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured lid assembly and cup docking targets; no object-state writes.

This candidate uses physical grasp/contact gates and ordinary robot actions. CPU
trajectory checks establish neither collision clearance nor physical success.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

OPTIONS = {
    "algorithm": "anchored_lid_feedback_preinsert_docking_v5",
    "position_tolerance_m": 0.008,
    "rotation_tolerance_rad": 0.060,
    "endpoint_hold_s": 0.25,
    "lid_release_support_hold_s": 0.25,
    "lid_release_target_separation_m": 0.110,
    "cup_release_support_hold_s": 0.25,
    "maximum_stage_wall_sim_s": 90.0,
    "transit_tcp_z_m": 0.50,
    "lid_empty_transit_tcp_z_m": 0.60,
    "lid_empty_joint_waypoint_rad": [
        0.4207511672711693,
        -0.7518490166840933,
        -0.34360682206715804,
        -1.9573984580515424,
        -0.24441385486798692,
        1.2257383113611597,
        -0.7005853407533812,
    ],
    "lid_grasp_local_m": [0.0, 0.0, 0.018],
    "lid_hand_yaw_rad": math.pi / 2,
    "cup_grasp_local_m": [0.0, -0.057, 0.155],
    "cup_hand_local_xyzw": [0.5, 0.5, 0.5, -0.5],
    "cup_placement_tolerance_m": [0.0003, 0.0003, 0.0003],
    "cup_placement_rotation_tolerance_rad": 0.0025,
    "cup_approach_integral_gain_s_inv": 2.0,
    "cup_approach_bias_lower_b_m": [-0.002, -0.002, 0.0],
    "cup_approach_bias_upper_b_m": [0.002, 0.002, 0.008],
    "cup_approach_bias_rate_m_s": 0.003,
    "cup_approach_rotation_bias_limit_rad": 0.0075,
    "cup_approach_rotation_bias_rate_rad_s": 0.020,
    "cup_pregrasp_backoff_m": 0.05,
    "cup_empty_transit_tcp_z_m": 0.60,
    "cup_empty_joint_waypoint_rad": [
        -1.4434593462045706,
        1.1258680751677366,
        1.4417102527766354,
        -1.2891767046596574,
        0.4635819171476493,
        1.389154505537075,
        -2.348137811750025,
    ],
    "joint_tracking_tolerance_rad": 0.020,
    "joint_endpoint_tolerance_rad": 0.008,
    "joint_action": {
        "arm_alpha": 0.2,
        "arm_scale": 0.03,
        "maximum_joint_increment_rad": 0.012,
        "joint_limit_interior_margin_rad": 0.020,
    },
    "lid_seated_axial_m": 0.220,
    "thread_pitch_m": 0.008,
    "clockwise_turns_per_grasp": 0.505,
    "motor_cap_target_m": [0.64, 0.25, 0.145],
    "motor_preinsert_lid_z_m": 0.210,
    "motor_alignment_free_min_z_m": 0.205,
    "motor_alignment_free_rotation_limit_rad": 0.075,
    "motor_alignment_xy_tolerance_m": 0.0002,
    "motor_alignment_rotation_tolerance_rad": 0.0025,
    "motor_preinsert_z_tolerance_m": 0.001,
    "motor_bias_lower_w_m": [-0.003, -0.003, 0.0],
    "motor_bias_upper_w_m": [0.003, 0.003, 0.012],
    "motor_rotation_bias_limit_rad": 0.020,
    "motor_bias_rate_m_s": 0.003,
    "motor_rotation_bias_rate_rad_s": 0.020,
    "motor_integral_gain_s_inv": 2.0,
    "physical_validation_passed": False,
}

# Duration is a minimum tracking-gated motion time, never proof of completion.
STAGES = (
    ("raise_empty_hand", 8.0),
    ("lid_joint_posture", 25.0),
    ("lid_approach", 8.0),
    ("lid_close", 2.5),
    ("lid_lift", 4.0),
    ("lid_carry", 6.0),
    ("lid_lower_thread", 5.0),
    ("lid_turn_first", 12.0),
    ("lid_open_regrasp", 2.5),
    ("lid_clear_regrasp", 3.0),
    ("lid_unwind_hand", 7.0),
    ("lid_lower_regrasp", 3.0),
    ("lid_close_second", 2.5),
    ("lid_turn_second", 12.0),
    ("lid_seat_hold", 1.0),
    ("lid_release", 2.5),
    ("lid_retreat", 3.0),
    ("cup_raise_empty_high", 8.0),
    ("cup_joint_posture", 25.0),
    ("cup_pregrasp_descent", 8.0),
    ("cup_approach", 4.0),
    ("cup_close", 2.5),
    ("cup_lift", 7.0),
    ("cup_invert", 12.0),
    ("cup_carry_motor", 7.0),
    ("cup_preinsert_motor", 4.0),
    ("cup_lower_motor", 6.0),
    ("cup_support_hold", 1.0),
    ("cup_release", 2.5),
    ("cup_retreat", 4.0),
    ("complete", 1.0),
)


def assembly_controller_identity() -> dict:
    """Return candidate controls and source identity without validation claims."""
    return {
        "schema": "measured_assembly_controller_v1",
        "options": copy.deepcopy(OPTIONS),
        "stages": [list(row) for row in STAGES],
        "source_sha256": {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "physics_state_modified": False,
    }


def _pose(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.isfinite(result).all() or np.linalg.norm(result[3:]) < 1e-12:
        raise ValueError("Require a finite position [m] and nonzero XYZW quaternion.")
    result = result.copy()
    result[3:] /= np.linalg.norm(result[3:])
    return result


def _make_pose(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    return np.r_[position, rotation.as_quat()]


def _joint_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"Require finite {name} [rad] with shape {shape}.")
    return result.astype(float, copy=False)


def bounded_joint_posture_step(
    target: np.ndarray, joints: np.ndarray, limits: np.ndarray, previous_delta: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Return raw joint actions with unchanged live EMA and increment constraints.

    Args:
        target: Desired named arm joint positions [rad], shape (7,).
        joints: Measured named arm joint positions [rad], shape (7,).
        limits: Authored joint limits [rad], shape (7, 2).
        previous_delta: Live processed arm action [rad], shape (7,).

    Returns:
        Raw seven-joint actions and diagnostics. Unreachable inset bounds cause
        the strongest reachable restoration or braking, without a history reset.
    """
    target = _joint_array(target, (7,), "joint target")
    joints = _joint_array(joints, (7,), "joint positions")
    limits = _joint_array(limits, (7, 2), "joint limits")
    previous_delta = _joint_array(previous_delta, (7,), "processed action")
    options = OPTIONS["joint_action"]
    margin = options["joint_limit_interior_margin_rad"]
    if np.any(limits[:, 1] - limits[:, 0] <= 2 * margin):
        raise ValueError("Joint limits must have a nonempty declared interior.")
    if np.any(target < limits[:, 0] + margin) or np.any(target > limits[:, 1] - margin):
        raise ValueError("The requested joint posture must lie inside the authored limit inset.")
    alpha, scale = options["arm_alpha"], options["arm_scale"]
    center, radius = (1 - alpha) * previous_delta, alpha * scale
    reachable_lower, reachable_upper = center - radius, center + radius
    cap = options["maximum_joint_increment_rad"]
    lower, upper = np.maximum(reachable_lower, -cap), np.minimum(reachable_upper, cap)
    increment_unreachable = lower > upper
    braking = np.clip(np.zeros(7), reachable_lower, reachable_upper)
    lower = np.where(increment_unreachable, braking, lower)
    upper = np.where(increment_unreachable, braking, upper)
    interior_lower = limits[:, 0] + margin - joints
    interior_upper = limits[:, 1] - margin - joints
    bounded_lower, bounded_upper = np.maximum(lower, interior_lower), np.minimum(upper, interior_upper)
    interior_unreachable = bounded_lower > bounded_upper
    restoring = np.where(interior_lower > upper, upper, lower)
    lower = np.where(interior_unreachable, restoring, bounded_lower)
    upper = np.where(interior_unreachable, restoring, bounded_upper)
    delta = np.clip(target - joints, lower, upper)
    raw = (delta - center) / radius
    if np.any(np.abs(raw) > 1 + 1e-10):
        raise RuntimeError("Joint posture action escaped the live EMA reachable interval.")
    raw = np.clip(raw, -1, 1)
    effective = center + radius * raw
    return raw, {
        "controller": "bounded_measured_joint_posture_v1",
        "maximum_joint_error_rad": float(np.max(np.abs(target - joints))),
        "effective_delta_rad": effective.tolist(),
        "joint_interior_unreachable": interior_unreachable.tolist(),
        "increment_unreachable_due_to_ema": increment_unreachable.tolist(),
        "restoration_or_braking_active": bool(interior_unreachable.any() or increment_unreachable.any()),
        "target_minimum_authored_joint_margin_rad": float(
            np.minimum(joints + effective - limits[:, 0], limits[:, 1] - joints - effective).min()
        ),
    }


class AssemblyTrajectory:
    """Produce targets from current physical measurements and confirmed grasps.

    Input poses are scene-local [m, XYZW]. ``hand_pose`` uses the measured TCP as
    its translation and the measured hand orientation. ``metrics`` contains the
    assembly geometry/state measurements; missing required gates fail closed.
    """

    def __init__(self) -> None:
        self.index = 0
        self.elapsed_s = 0.0
        self.stage_wall_s = 0.0
        self.hold_s = 0.0
        self.start: np.ndarray | None = None
        self.goal: np.ndarray | None = None
        self.joint_start: np.ndarray | None = None
        self.reference: dict[str, tuple[np.ndarray, Rotation]] = {}
        self.turn_lid_start: np.ndarray | None = None
        self.turn_cup: np.ndarray | None = None
        self.assembly_relative: tuple[np.ndarray, Rotation] | None = None
        self.complete = False
        self.release_support_hold_s = 0.0
        self.release_authorized = False
        self.cup_approach_bias = np.zeros(3)
        self.cup_approach_rotation_bias = np.zeros(3)
        self.cup_close_target: np.ndarray | None = None
        self.last_command: np.ndarray | None = None
        self.motor_lid_start: np.ndarray | None = None
        self.motor_bias = np.zeros(3)
        self.motor_rotation_bias = np.zeros(3)
        self.motor_preinsert_verified = False
        self.motor_entry_command: np.ndarray | None = None

    @property
    def stage(self) -> str:
        """Return the controller substage, independent of measured task milestones."""
        return STAGES[self.index][0]

    def _capture(self, name: str, hand: np.ndarray, obj: np.ndarray) -> None:
        rotation = Rotation.from_quat(obj[3:])
        self.reference[name] = (
            rotation.inv().apply(hand[:3] - obj[:3]),
            rotation.inv() * Rotation.from_quat(hand[3:]),
        )

    def _held_target(self, name: str, obj: np.ndarray) -> np.ndarray:
        if name not in self.reference:
            raise RuntimeError(f"No measured physical grasp reference for {name}.")
        local_position, local_rotation = self.reference[name]
        rotation = Rotation.from_quat(obj[3:])
        return _make_pose(obj[:3] + rotation.apply(local_position), rotation * local_rotation)

    def _grasp_target(self, name: str, obj: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_quat(obj[3:])
        hand = (
            Rotation.from_euler("z", OPTIONS["lid_hand_yaw_rad"]) * Rotation.from_euler("x", math.pi)
            if name == "lid"
            else Rotation.from_quat(OPTIONS["cup_hand_local_xyzw"])
        )
        return _make_pose(obj[:3] + rotation.apply(OPTIONS[name + "_grasp_local_m"]), rotation * hand)

    def _dock_cup_pose(self, clearance_m: float) -> np.ndarray:
        if self.assembly_relative is None:
            raise RuntimeError("Docking requires this episode's measured seated cup/lid transform.")
        relative_position, relative_rotation = self.assembly_relative
        cap_rotation = Rotation.from_euler("x", math.pi) * Rotation.from_euler("z", math.pi)
        cup_rotation = cap_rotation * relative_rotation.inv()
        cap_position = np.asarray(OPTIONS["motor_cap_target_m"]) + [0, 0, clearance_m]
        return _make_pose(cap_position - cup_rotation.apply(relative_position), cup_rotation)

    def _enter_motor(self, hand: np.ndarray, lid: np.ndarray) -> None:
        """Initialize measured lid feedback while preserving the prior command [m, rad]."""
        stage = self.stage
        if stage == "cup_lower_motor" and not self.motor_preinsert_verified:
            raise RuntimeError("Motor insertion requires the measured pre-insertion alignment dwell.")
        self.motor_lid_start = lid.copy()
        if stage == "cup_lower_motor":
            self.motor_lid_start[2] = OPTIONS["motor_preinsert_lid_z_m"]
        self.motor_entry_command = None if self.last_command is None else self.last_command.copy()
        if stage == "cup_preinsert_motor":
            self._capture("motor_lid", hand, lid)
            self.motor_bias[:] = 0.0
            self.motor_rotation_bias[:] = 0.0
            if self.last_command is not None:
                # Retain the existing actuator drive when changing controlled frames.
                change = Rotation.from_quat(self.last_command[3:]) * Rotation.from_quat(hand[3:]).inv()
                driven_lid = self.last_command[:3] + change.apply(lid[:3] - hand[:3])
                self.motor_bias[:] = driven_lid - lid[:3]
                self.motor_rotation_bias[:] = change.as_rotvec()
                if (
                    np.any(self.motor_bias < OPTIONS["motor_bias_lower_w_m"])
                    or np.any(self.motor_bias > OPTIONS["motor_bias_upper_w_m"])
                    or np.linalg.norm(self.motor_rotation_bias) > OPTIONS["motor_rotation_bias_limit_rad"]
                ):
                    raise RuntimeError("Existing carry drive exceeds the bounded motor alignment trim.")
        if self.last_command is not None:
            self.start = self.goal = self.last_command.copy()

    def _enter(
        self, hand: np.ndarray, lid: np.ndarray, cup: np.ndarray, joint_positions: np.ndarray | None = None
    ) -> None:
        self.start, self.goal = hand.copy(), hand.copy()
        self.joint_start = None
        self.release_support_hold_s = 0.0
        self.release_authorized = False
        stage = self.stage
        high = OPTIONS["transit_tcp_z_m"]
        if stage == "raise_empty_hand":
            self.goal[2] = max(OPTIONS["lid_empty_transit_tcp_z_m"], hand[2])
        elif stage == "lid_joint_posture":
            self.joint_start = _joint_array(joint_positions, (7,), "measured entry joints").copy()
        elif stage == "lid_approach":
            self.goal = self._grasp_target("lid", lid)
        elif stage in ("lid_lift", "lid_carry", "lid_lower_thread"):
            obj = lid.copy()
            if stage == "lid_lift":
                self._capture("lid", hand, lid)
                obj[2] += 0.14
            else:
                rotation = Rotation.from_quat(cup[3:])
                axial = (
                    OPTIONS["lid_seated_axial_m"] + 2 * OPTIONS["clockwise_turns_per_grasp"] * OPTIONS["thread_pitch_m"]
                )
                if stage == "lid_carry":
                    axial += 0.11
                obj[:3] = cup[:3] + rotation.apply([0, 0, axial])
                obj[3:] = rotation.as_quat()
            self.goal = self._held_target("lid", obj)
        elif stage in ("lid_turn_first", "lid_turn_second"):
            self._capture("lid", hand, lid)
            self.turn_lid_start, self.turn_cup = lid.copy(), cup.copy()
        elif stage == "lid_clear_regrasp":
            self.goal[2] += 0.10
        elif stage == "lid_retreat":
            # Grasp compression must not consume the measured release clearance.
            grasp = self._grasp_target("lid", lid)
            self.goal[2] = max(hand[2], grasp[2] + OPTIONS["lid_release_target_separation_m"])
        elif stage == "lid_unwind_hand":
            self.goal[3:] = (Rotation.from_euler("z", math.pi) * Rotation.from_quat(hand[3:])).as_quat()
        elif stage == "lid_lower_regrasp":
            self.goal[:3] = self._grasp_target("lid", lid)[:3]
        elif stage == "cup_raise_empty_high":
            self.goal[2] = max(OPTIONS["cup_empty_transit_tcp_z_m"], hand[2])
        elif stage == "cup_joint_posture":
            self.joint_start = _joint_array(joint_positions, (7,), "measured entry joints").copy()
        elif stage == "cup_pregrasp_descent":
            self.goal = self._grasp_target("cup", cup)
            self.goal[:3] -= Rotation.from_quat(self.goal[3:]).apply([0, 0, OPTIONS["cup_pregrasp_backoff_m"]])
        elif stage == "cup_approach":
            self.goal = self._grasp_target("cup", cup)
            self.cup_approach_bias[:] = 0.0
            self.cup_approach_rotation_bias[:] = 0.0
            self.cup_close_target = None
        elif stage == "cup_close":
            if self.cup_close_target is None:
                raise RuntimeError("Cup closing requires a precisely placed approach endpoint.")
            # Preserve corrective drive while the fingers begin to close.
            self.start = self.cup_close_target.copy()
            self.goal = self.cup_close_target.copy()
        elif stage == "cup_lift":
            self._capture("cup", hand, cup)
            cup_rotation = Rotation.from_quat(cup[3:])
            self.assembly_relative = (
                cup_rotation.inv().apply(lid[:3] - cup[:3]),
                cup_rotation.inv() * Rotation.from_quat(lid[3:]),
            )
            self.goal[2] = max(high, hand[2] + 0.20)
        elif stage == "cup_invert":
            obj = self._dock_cup_pose(0.20)
            self.goal = self._held_target("cup", obj)
            self.goal[:3] = hand[:3]
        elif stage == "cup_carry_motor":
            self.goal = self._held_target("cup", self._dock_cup_pose(0.20))
        elif stage in ("cup_preinsert_motor", "cup_lower_motor"):
            self._enter_motor(hand, lid)
        elif stage in ("cup_support_hold", "cup_release") and self.motor_preinsert_verified:
            if self.last_command is None:
                raise RuntimeError("Dock support must preserve the last corrected insertion command.")
            self.start = self.goal = self.last_command.copy()
        elif stage == "cup_retreat":
            # Retreat along the measured approach axis, keeping clear of the cup.
            self.goal[:3] -= Rotation.from_quat(hand[3:]).apply([0, 0, 0.12])

    def _sample(self, u: float) -> np.ndarray:
        if self.start is None or self.goal is None:
            raise RuntimeError("Stage targets have not been initialized.")
        blend = u * u * (3 - 2 * u)
        if self.stage in ("lid_turn_first", "lid_turn_second"):
            if self.turn_lid_start is None or self.turn_cup is None:
                raise RuntimeError("A thread motion needs the measured entry poses.")
            cup_r = Rotation.from_quat(self.turn_cup[3:])
            angle = -2 * math.pi * OPTIONS["clockwise_turns_per_grasp"] * blend
            rotation = (
                cup_r * Rotation.from_euler("z", angle) * cup_r.inv() * Rotation.from_quat(self.turn_lid_start[3:])
            )
            position = self.turn_lid_start[:3] - cup_r.apply(
                [0, 0, OPTIONS["thread_pitch_m"] * OPTIONS["clockwise_turns_per_grasp"] * blend]
            )
            return self._held_target("lid", _make_pose(position, rotation))
        if self.stage == "lid_unwind_hand":
            rotation = Rotation.from_euler("z", math.pi * blend) * Rotation.from_quat(self.start[3:])
        elif self.stage == "cup_invert":
            start = Rotation.from_quat(self.start[3:])
            delta = (Rotation.from_quat(self.goal[3:]) * start.inv()).as_rotvec()
            angle = np.linalg.norm(delta)
            # Near pi, measured tilt must not flip to the unreachable wrist arc.
            if angle > 1e-12 and delta[1] < 0:
                delta *= 1 - 2 * math.pi / angle
            rotation = Rotation.from_rotvec(delta * blend) * start
        else:
            rotation = Slerp([0, 1], Rotation.from_quat([self.start[3:], self.goal[3:]]))([blend])[0]
        return _make_pose((1 - blend) * self.start[:3] + blend * self.goal[:3], rotation)

    def _gates(self, metrics: dict, endpoint: bool, dt: float) -> tuple[bool, bool, str]:
        stage = self.stage

        def get(name: str) -> bool:
            return bool(metrics.get(name, False))

        opened = get("fingers_open")
        lid_motion = stage in (
            "lid_lift",
            "lid_carry",
            "lid_lower_thread",
            "lid_turn_first",
            "lid_turn_second",
            "lid_seat_hold",
        )
        cup_motion = stage in (
            "cup_lift",
            "cup_invert",
            "cup_carry_motor",
            "cup_preinsert_motor",
            "cup_lower_motor",
            "cup_support_hold",
        )
        close = lid_motion or cup_motion or stage in ("lid_close", "lid_close_second", "cup_close")
        gate = True
        if lid_motion:
            gate = get("lid_held")
        if cup_motion:
            gate = get("cup_held") and get("lid_seated") and not get("lid_retention_failed")
        if stage.startswith("lid_turn"):
            gate &= get("lid_near_thread")
        if stage in ("lid_clear_regrasp", "lid_unwind_hand", "lid_lower_regrasp"):
            supported = get("lid_near_thread") and get("lid_stable") and get("cup_stable")
            gate = supported and opened
            # These stages follow a verified open release. Settling pauses hand
            # motion without closing the fingers onto the supported lid again.
            close = False
        if stage in ("lid_open_regrasp", "lid_release"):
            retained = get("lid_near_thread")
            if stage == "lid_release":
                retained &= get("lid_twist_complete") and get("lid_seated")
            supported = retained and get("lid_stable") and get("cup_stable")
            if not retained:
                self.release_authorized = False
                self.release_support_hold_s = 0.0
            elif not self.release_authorized:
                self.release_support_hold_s = self.release_support_hold_s + dt if supported else 0.0
                self.release_authorized = self.release_support_hold_s >= OPTIONS["lid_release_support_hold_s"] - 1e-9
            # Opening can cause brief settling. Do not close again solely because
            # velocity changes after the measured support dwell authorized release.
            close = not self.release_authorized
            gate = self.release_authorized and retained
            if endpoint:
                gate &= supported
        if stage == "cup_release":
            ready = get("dock_support_ready")
            if not self.release_authorized:
                self.release_support_hold_s = self.release_support_hold_s + dt if ready else 0.0
                self.release_authorized = self.release_support_hold_s >= OPTIONS["cup_release_support_hold_s"] - 1e-9
            gate, close = ready and self.release_authorized, not self.release_authorized
        if stage in ("raise_empty_hand", "lid_joint_posture", "lid_approach"):
            gate = opened
        if stage in ("cup_raise_empty_high", "cup_joint_posture", "cup_pregrasp_descent", "cup_approach"):
            gate = get("lid_release_complete") and get("lid_seated") and opened
        if endpoint:
            if stage in ("lid_close", "lid_close_second"):
                gate &= get("lid_held")
            elif stage == "cup_close":
                gate &= get("cup_held")
            elif stage in ("lid_open_regrasp", "lid_release", "cup_release"):
                gate &= opened
            elif stage == "lid_seat_hold":
                gate &= get("lid_twist_complete") and get("lid_seated")
            elif stage == "lid_lower_thread":
                gate &= get("lid_near_thread")
            elif stage == "lid_retreat":
                gate &= get("lid_release_complete")
            elif stage == "cup_support_hold":
                gate &= get("dock_support_ready")
            elif stage in ("cup_retreat", "complete"):
                gate &= get("assembly_complete")
            elif not close:
                gate &= opened
        return gate, close, "measured_gates_passed" if gate else "waiting_for_physical_gate"

    def _cup_approach_endpoint(
        self, hand: np.ndarray, cup: np.ndarray, dt: float, allow_bias: bool
    ) -> tuple[np.ndarray, bool, dict]:
        """Refine the open-hand bridge placement with bounded target bias [m, rad]."""
        desired = self._grasp_target("cup", cup)
        cup_rotation = Rotation.from_quat(cup[3:])
        local_error = cup_rotation.inv().apply(hand[:3] - desired[:3])
        angular_error = cup_rotation.inv().apply(
            (Rotation.from_quat(desired[3:]) * Rotation.from_quat(hand[3:]).inv()).as_rotvec()
        )
        ready = bool(
            np.all(np.abs(local_error) <= OPTIONS["cup_placement_tolerance_m"])
            and np.linalg.norm(angular_error) <= OPTIONS["cup_placement_rotation_tolerance_rad"]
        )
        if not ready and allow_bias:
            for bias, residual, rate in (
                (self.cup_approach_bias, -local_error, "cup_approach_bias_rate_m_s"),
                (
                    self.cup_approach_rotation_bias,
                    angular_error,
                    "cup_approach_rotation_bias_rate_rad_s",
                ),
            ):
                increment = OPTIONS["cup_approach_integral_gain_s_inv"] * dt * residual
                increment *= min(1.0, OPTIONS[rate] * dt / max(np.linalg.norm(increment), 1e-15))
                bias += increment
            self.cup_approach_bias[:] = np.clip(
                self.cup_approach_bias, OPTIONS["cup_approach_bias_lower_b_m"], OPTIONS["cup_approach_bias_upper_b_m"]
            )
            self.cup_approach_rotation_bias *= min(
                1.0,
                OPTIONS["cup_approach_rotation_bias_limit_rad"]
                / max(np.linalg.norm(self.cup_approach_rotation_bias), 1e-15),
            )
        sample = desired.copy()
        sample[:3] += cup_rotation.apply(self.cup_approach_bias)
        sample[3:] = (
            Rotation.from_rotvec(cup_rotation.apply(self.cup_approach_rotation_bias)) * Rotation.from_quat(desired[3:])
        ).as_quat()
        self.cup_close_target = sample.copy()
        diagnostics = {
            "cup_placement_error_b_m": local_error.tolist(),
            "cup_placement_rotation_error_rad": float(np.linalg.norm(angular_error)),
            "cup_placement_ready": ready,
            "cup_approach_bias_b_m": self.cup_approach_bias.tolist(),
            "cup_approach_rotation_bias_b_rad": self.cup_approach_rotation_bias.tolist(),
        }
        return sample, ready, diagnostics

    def _motor_target(
        self, hand: np.ndarray, lid: np.ndarray, metrics: dict, u: float, endpoint: bool, dt: float
    ) -> tuple[np.ndarray, bool, dict]:
        """Control the measured lid frame with bounded world-frame trim [m, rad].

        The trim compensates actuator compliance, not permissible object error.
        The moving preinsert follows the measured key corridor above the free plane.
        Exact alignment remains mandatory at the preinsert endpoint and throughout insertion.
        """
        if self.motor_lid_start is None:
            raise RuntimeError("Motor alignment requires its measured entry pose.")
        preinsert = self.stage == "cup_preinsert_motor"
        position = np.asarray(OPTIONS["motor_cap_target_m"], dtype=float).copy()
        if preinsert:
            position[2] = OPTIONS["motor_preinsert_lid_z_m"]
        blend = u * u * (3 - 2 * u)
        position[2] = (1 - blend) * self.motor_lid_start[2] + blend * position[2]
        rotation = Rotation.from_euler("x", math.pi) * Rotation.from_euler("z", math.pi)
        lid_rotation = Rotation.from_quat(lid[3:])
        error = position - lid[:3]
        angular_error = (rotation * lid_rotation.inv()).as_rotvec()
        angle = float(np.linalg.norm(angular_error))
        aligned = bool(
            np.all(np.abs(error[:2]) <= OPTIONS["motor_alignment_xy_tolerance_m"])
            and angle <= OPTIONS["motor_alignment_rotation_tolerance_rad"]
        )
        key_fit = bool(metrics.get("dock_key_fit", False))
        held = bool(metrics.get("cup_held", False)) and bool(metrics.get("lid_seated", False))
        first = self.motor_entry_command is not None
        if held:
            if angle > OPTIONS["motor_alignment_free_rotation_limit_rad"]:
                raise RuntimeError("Measured lid rotation escaped the motor alignment envelope.")
            if lid[2] < OPTIONS["motor_alignment_free_min_z_m"]:
                if preinsert or not key_fit:
                    raise RuntimeError("Measured lid left the free-space/key corridor; refuse further insertion.")
            if not first:
                residual = error.copy()
                # Axial compensation is learned above the motor, never against contact.
                if not (preinsert and endpoint):
                    residual[2] = 0.0
                for bias, value, rate in (
                    (self.motor_bias, residual, OPTIONS["motor_bias_rate_m_s"]),
                    (self.motor_rotation_bias, angular_error, OPTIONS["motor_rotation_bias_rate_rad_s"]),
                ):
                    increment = OPTIONS["motor_integral_gain_s_inv"] * dt * value
                    increment *= min(1.0, rate * dt / max(np.linalg.norm(increment), 1e-15))
                    bias += increment
                self.motor_bias[:] = np.clip(
                    self.motor_bias, OPTIONS["motor_bias_lower_w_m"], OPTIONS["motor_bias_upper_w_m"]
                )
                self.motor_rotation_bias *= min(
                    1.0,
                    OPTIONS["motor_rotation_bias_limit_rad"] / max(np.linalg.norm(self.motor_rotation_bias), 1e-15),
                )
        desired_rotation = Rotation.from_rotvec(self.motor_rotation_bias) * rotation
        # Repeated recapture would chase roll around the weak jaw-contact axis.
        # The loaded entry reference stays fixed; only bounded measured-error trim adapts.
        if "motor_lid" not in self.reference:
            raise RuntimeError("Motor insertion requires the loaded alignment-entry grasp reference.")
        local_hand, relative = self.reference["motor_lid"]
        sample = _make_pose(
            position + self.motor_bias + desired_rotation.apply(local_hand), desired_rotation * relative
        )
        if first:
            sample = self.motor_entry_command.copy()
            self.motor_entry_command = None
        elif not held and self.last_command is not None:
            sample = self.last_command.copy()
        # Above the motor, follow the safe key corridor without endpoint-level alignment stops.
        moving_preinsert = preinsert and not endpoint
        ready = held and key_fit and not first and (moving_preinsert or aligned)
        if preinsert and endpoint:
            ready &= abs(error[2]) <= OPTIONS["motor_preinsert_z_tolerance_m"]
        elif not preinsert and endpoint:
            ready &= bool(metrics.get("dock_support_ready", False))
        else:
            ready &= abs(error[2]) <= OPTIONS["position_tolerance_m"]
        return (
            sample,
            bool(ready),
            {
                "motor_lid_position_error_w_m": error.tolist(),
                "motor_lid_rotation_error_rad": angle,
                "motor_alignment_ready": aligned,
                "motor_actual_key_fit": key_fit,
                "motor_bias_w_m": self.motor_bias.tolist(),
                "motor_rotation_bias_w_rad": self.motor_rotation_bias.tolist(),
                "motor_axial_bias_frozen": not preinsert,
                "motor_command_handoff": first,
            },
        )

    def step(self, state: dict, dt: float) -> dict:
        """Advance only with measured tracking and contact gates; timestep [s]."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Require a positive finite policy timestep [s].")
        hand, lid, cup = (_pose(state[name]) for name in ("hand_pose", "lid_pose", "cup_pose"))
        metrics = state["metrics"]
        if not bool(metrics.get("valid", False)) or bool(metrics.get("lid_retention_failed", False)):
            raise RuntimeError("Invalid assembly physical state or lost threaded lid.")
        if self.start is None:
            self._enter(hand, lid, cup, state.get("joint_positions"))
        duration = STAGES[self.index][1]
        u = min(self.elapsed_s / duration, 1.0)
        sample = self._sample(u)
        endpoint = self.elapsed_s >= duration - 1e-9
        placement_ready, placement_diagnostics = None, {}
        if endpoint and self.stage == "cup_approach":
            allow_bias = all(
                bool(metrics.get(key, False)) for key in ("fingers_open", "lid_release_complete", "lid_seated")
            )
            sample, placement_ready, placement_diagnostics = self._cup_approach_endpoint(hand, cup, dt, allow_bias)
        if self.stage in ("cup_preinsert_motor", "cup_lower_motor"):
            sample, placement_ready, placement_diagnostics = self._motor_target(hand, lid, metrics, u, endpoint, dt)
        joint_target = None
        joint_error = None
        if self.stage in ("lid_joint_posture", "cup_joint_posture"):
            joints = _joint_array(state.get("joint_positions"), (7,), "measured joints")
            if self.joint_start is None:
                raise RuntimeError("A joint posture requires measured entry joints.")
            blend = u * u * (3 - 2 * u)
            name = "lid" if self.stage == "lid_joint_posture" else "cup"
            waypoint = np.asarray(OPTIONS[name + "_empty_joint_waypoint_rad"])
            joint_target = self.joint_start + blend * (waypoint - self.joint_start)
            joint_error = float(np.max(np.abs(joint_target - joints)))
            sample = hand.copy()
        position_error = float(np.linalg.norm(sample[:3] - hand[:3]))
        rotation_error = float((Rotation.from_quat(sample[3:]) * Rotation.from_quat(hand[3:]).inv()).magnitude())
        tracking = (
            position_error <= OPTIONS["position_tolerance_m"] and rotation_error <= OPTIONS["rotation_tolerance_rad"]
        )
        if endpoint and self.stage in ("lid_lower_thread", "lid_lower_regrasp"):
            # Contact compliance can leave millimetres of TCP error even when the
            # measured lid is aligned. Physical support/grasp gates remain.
            tracking = position_error <= OPTIONS["position_tolerance_m"] and rotation_error <= 0.025
        elif endpoint and self.stage == "cup_lower_motor":
            tolerance = OPTIONS["position_tolerance_m"] if bool(metrics.get("dock_support_ready", False)) else 0.0015
            tracking = position_error <= tolerance and rotation_error <= 0.025
        if joint_target is not None:
            tolerance = OPTIONS["joint_endpoint_tolerance_rad" if endpoint else "joint_tracking_tolerance_rad"]
            tracking = joint_error <= tolerance
        if placement_ready is not None:
            # The physical nominal pose, not the compensating target, permits closure.
            tracking = placement_ready
        gate, close, reason = self._gates(metrics, endpoint, dt)
        self.stage_wall_s += dt
        if self.stage_wall_s > OPTIONS["maximum_stage_wall_sim_s"]:
            raise RuntimeError(f"Assembly substage {self.stage} exceeded its bounded physical time.")
        self.hold_s = self.hold_s + dt if endpoint and tracking and gate else 0.0
        result = {
            "phase": self.stage,
            "assembly_controller_stage_id": self.index,
            "tcp_position": sample[:3],
            "hand_xyzw": sample[3:],
            "raw_gripper_sign": -1.0 if close else 1.0,
            "elapsed_s": self.elapsed_s,
            "stage_wall_s": self.stage_wall_s,
            "position_error_m": position_error,
            "rotation_error_rad": rotation_error,
            "tracking": tracking,
            "gate_reason": reason,
            "hold_s": self.hold_s,
            "complete": self.complete,
            "release_support_hold_s": self.release_support_hold_s,
            "release_authorized": self.release_authorized,
            "control_mode": "joint_posture" if joint_target is not None else "cartesian_pose",
            "joint_position_target_rad": joint_target,
            "maximum_joint_error_rad": joint_error,
            **placement_diagnostics,
        }
        self.last_command = sample.copy()
        if tracking and gate:
            self.elapsed_s = min(duration, self.elapsed_s + dt)
        if self.hold_s >= OPTIONS["endpoint_hold_s"] - 1e-9:
            if self.stage == "cup_preinsert_motor":
                self.motor_preinsert_verified = True
            if self.index == len(STAGES) - 1:
                self.complete = True
                result["complete"] = True
            else:
                self.index += 1
                self.elapsed_s = self.stage_wall_s = self.hold_s = 0.0
                self.start = self.goal = None
        return result


class AssemblyController:
    """Adapt scene-local measured targets to ordinary bounded 8-D robot actions."""

    def __init__(self, env: Any, metrics_fn: Callable[[Any], dict] | None = None) -> None:
        from .bounded_return_pose import BoundedReturnPoseController

        if env.num_envs != 1:
            raise ValueError("Assembly candidate supports one uninterrupted physical world.")
        self.env = env
        self.metrics_fn = metrics_fn
        self.trajectory = AssemblyTrajectory()
        self.pose_controller = BoundedReturnPoseController(env)
        self.stage = self.trajectory.stage
        self.diagnostics: dict = {}
        self.previous_step: int | None = None

    def compute(self, step: int):
        """Return raw actions from current physical poses; global step is monotonic."""
        import torch

        if type(step) is not int or step < 0 or (self.previous_step is not None and step != self.previous_step + 1):
            raise ValueError("Assembly requires consecutive nonnegative global policy steps.")
        self.previous_step = step
        env = self.env
        origin = env.scene.env_origins[0].detach().cpu().numpy()
        hand = env.robot.data.body_link_pose_w.torch[0, env.hand_id].detach().cpu().numpy().copy()
        hand[:3] = env.tcp()[0].detach().cpu().numpy() - origin
        lid = env.pose("blade_cap")[0].detach().cpu().numpy().copy()
        cup = env.pose("cup")[0].detach().cpu().numpy().copy()
        lid[:3] -= origin
        cup[:3] -= origin
        metrics = self.metrics_fn(env) if self.metrics_fn is not None else env.assembly_measurements()
        metrics = {
            key: value[0].item() if torch.is_tensor(value) and value.numel() == 1 else value
            for key, value in metrics.items()
        }
        joints = env.robot.data.joint_pos.torch[:, self.pose_controller.joint_ids]
        sample = self.trajectory.step(
            {
                "hand_pose": hand,
                "lid_pose": lid,
                "cup_pose": cup,
                "joint_positions": joints[0].detach().cpu().numpy(),
                "metrics": metrics,
            },
            env.step_dt,
        )
        position = torch.as_tensor(sample["tcp_position"] + origin, device=env.device, dtype=env.tcp().dtype)[None]
        rotation = torch.as_tensor(sample["hand_xyzw"], device=env.device, dtype=env.tcp().dtype)[None]
        if sample["joint_position_target_rad"] is None:
            actions = self.pose_controller.compute(position, rotation, sample["raw_gripper_sign"] < 0)
            control_diagnostics = self.pose_controller.diagnostics
        else:
            self.pose_controller._check_action_contract()
            limits = env.robot.data.joint_pos_limits.torch[:, self.pose_controller.joint_ids]
            previous = env.action_manager.get_term("arm_action").processed_actions
            raw, control_diagnostics = bounded_joint_posture_step(
                sample["joint_position_target_rad"],
                joints[0].detach().cpu().numpy(),
                limits[0].detach().cpu().numpy(),
                previous[0].detach().cpu().numpy(),
            )
            actions = torch.zeros((1, 8), device=env.device, dtype=joints.dtype)
            actions[0, :7] = torch.as_tensor(raw, device=env.device, dtype=joints.dtype)
            actions[0, -1] = sample["raw_gripper_sign"]
            control_diagnostics = {
                "controller": control_diagnostics["controller"],
                "per_environment": [control_diagnostics],
            }
        self.stage, self.diagnostics = sample["phase"], sample
        self.diagnostics["bounded_pose_control"] = control_diagnostics
        if actions.shape != (1, 8) or not torch.isfinite(actions).all():
            raise RuntimeError("Assembly produced invalid raw actions.")
        return actions
