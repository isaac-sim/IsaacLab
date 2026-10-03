# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted basket-pour command geometry, without simulator or object-state writes.

The caller acquires the basket and owns every task milestone and return decision.
This helper emits TCP/hand references from live measured geometry. Its local clock
pauses when grasp or tracking is lost. CPU checks do not establish physical success.
The nominal pour follows the earlier physical FullBasketExpert recipe; measured
gates, live cup tracking, and clearance continuity make this a new candidate.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

FROZEN_ROOT = Path(__file__).parent
DEPENDENCY_PATHS = (FROZEN_ROOT / "basket_reference.py", FROZEN_ROOT / "grasp_feedback.py")
OPTIONS = {
    "schema": "scripted_measured_fruit_pour",
    "version": 11,
    "actions_only": True,
    "physical_validation_passed": False,
    "capture_gate": "caller requires strict upright lift complete and current held",
    "goal_fruit_count": 16,
    "return_owned_by_caller": True,
    "height_alignment_s": 3.0,
    "level_carry_s": 6.0,
    "rock_cycles": 2,
    "rock_cycle_s": 6.0,
    "rock_deg": 15.0,
    "tilt_s": 24.0,
    "tilt_deg": 130.0,
    "late_tilt_lift_start_deg": 115.0,
    "late_tilt_lift_m": 0.030,
    "terminal_tip_additional_deg": 20.0,
    "terminal_tip_s": 4.0,
    "terminal_tip_lift_m": 0.050,
    "terminal_tip_rationale": (
        "Preserve the selected initial tilt path through 115 degrees, then tip once from 130 to 150 degrees "
        "over 4 s and hold. "
        "Keep lip XY fixed; add 30 mm world-Z lift with cubic smoothstep from 115 to 130 degrees, "
        "then add the remaining 20 mm during the terminal tip for the same total 50 mm at 150 degrees. "
        "The late rise addresses sub-mm palm/cup clearance with the full04 peak measured grasp drift. "
        "Observed berry/strawberry releases precede the rise, but it changes late mango release height; "
        "actual fruit receipt requires fresh physical validation. "
        "Apply this offset after the original excess-height descent so its magnitude is not attenuated. "
        "Existing measured clock gates apply; actual hand clearance still requires geometry screening. "
        "This bounded continuation is an unvalidated candidate for fruit retained inside the basket. "
        "The caller still owns strict receipt, return and all task milestones."
    ),
    "tilt_timing_rationale": (
        "After measured carry alignment, perform two 0-to-15-to-0 degree cycles of 6 s each, "
        "then restore the original 24 s forward tilt on the same angle-dependent geometry. "
        "Each cubic 3 s half-cycle has zero endpoint velocity and peak angular speed 7.5 degrees/s. "
        "Earlier successful S19 motion included low-angle rocking; the full04 uniform 48 s tilt "
        "spilled fruit on different sides of the cup. This bounded settling candidate does not "
        "establish causality or physical success. Measured gates and physics remain unchanged."
    ),
    "align_grasp_radial": True,
    "grasp_alignment_rationale": (
        "Select the physical rim point opposite the captured grasp and turn upright during carry. "
        "The 135-degree pickup with a fixed original lip intersected the cup late in nominal tilt; "
        "in that prior 95 mm-height case, radial alignment gave 13.197 mm minimum exact hand-capsule clearance "
        "over the sampled nominal poses. This is historical S7 geometry evidence, not an S8 measurement. "
        "This does not bound actual tracking or grasp drift; physical validation is still required."
    ),
    "minimum_rim_clearance_m": 0.030,
    "clearance_rationale": (
        "S15 restores 30 mm commanded clearance to retain hand/cup separation with a larger inward aim. "
        "A bounded nominal screen using the S12 captured grasp found 18.595 mm whole-hand/cup separation "
        "and reachable key poses for 30 mm clearance and -25 mm cup-Y aim. This is sampled rigid geometry, "
        "not a guarantee under actual tracking, fruit contacts or grasp drift. "
        "Position-only grip, physics, tracking gates and completion criteria remain unchanged."
    ),
    "clearance_geometry": "outer basket boundary radius60.5mm; entire basket above highest actual cup rim",
    "cup_rim_local_z_m": 0.210,
    "cup_outer_rim_radius_m": 0.0515,
    "lip_aim_offset_cup_xy_m": [0.0, -0.025],
    "lip_aim_rationale": (
        "S15 applies -25 mm cup-frame Y aim to both the command and measured alignment gate. "
        "S14's -10 mm aim still spilled a blackberry beyond the positive-Y cup wall. "
        "A frozen-trajectory screen including the added flight height improved this berry's predicted "
        "margin from -3.270 to +3.964 mm, without making a baseline-positive footprint negative among "
        "16 comparable releases. Four already-below-rim mango releases were unassessed; two already-negative "
        "predictions worsened despite those fruits being received through contacts in the actual S14 run. "
        "These observations motivate a physical retry and do not prove receipt or contact-free execution. "
        "Actual cup rim clearance geometry, task gates and acceptance remain unchanged."
    ),
    "reference_recipe": {
        "controller": "historical fruit-pour reference",
        "episode": "demonstrations.hdf5/demo_16: world 11/reset 0/seed 20260918",
        "evidence": "Strict whole-hull count 20 for 2.4 s at 29.6-31.9667 s; fruit-only, not combined success.",
        "adaptation": (
            "Raise keeps captured rotation; carry turns upright with measured radial grasp alignment; "
            "pour yaw 0, cup-Y aim -25 mm, two 6 s/15 degree rocking cycles and 24 s/130 degree forward tilt "
            "with 30 mm base clearance; smoothly add 30 mm lift from 115 to 130 degrees, then 4 s to "
            "150 degrees with the remaining 20 mm lift for 50 mm total. "
            "Current strict-lift handoff, measured clock gates, grasp feedback and live cup remain; "
            "extra captured height descends continuously during tilt. Not an exact old-episode replay."
        ),
    },
    "tracking_position_tolerance_m": 0.008,
    "tracking_rotation_tolerance_rad": 0.08,
    "lip_alignment_tolerance_m": 0.010,
    "upright_cosine": 0.995,
    "alignment_hold_s": 0.4,
    "cup_reference_translation_rate_m_s": 0.02,
    "cup_reference_rotation_rate_rad_s": 0.15,
    "grasp_feedback": {
        "enabled": True,
        "time_constant_s": 0.5,
        "maximum_translation_rate_m_s": 0.004,
        "maximum_rotation_rate_rad_s": 0.15,
        "maximum_translation_from_capture_m": 0.015,
        "maximum_rotation_from_capture_rad": 1.1,
    },
}


from . import basket_reference as _geometry
from . import grasp_feedback as _feedback


def _pose(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    value = np.r_[position, quaternion].astype(np.float64)
    if value.shape != (7,) or not np.isfinite(value).all() or np.linalg.norm(value[3:]) < 1e-10:
        raise ValueError("Require a finite position [m] and nonzero XYZW quaternion.")
    value[3:] = Rotation.from_quat(value[3:]).as_quat()
    return value


class ControlledFruitPour:
    """Generate a measured, tracking-gated TCP pour trajectory after a stable lift.

    Args:
        basket_position: Measured basket origin [m], shape [3].
        basket_quaternion: Measured basket XYZW quaternion, shape [4].
        tcp_position: Measured tool centre position [m], shape [3].
        hand_quaternion: Measured hand XYZW quaternion, shape [4].
        cup_position: Measured cup origin [m], shape [3].
        cup_quaternion: Measured cup XYZW quaternion, shape [4].
        dt: Positive policy interval [s].
        options: Optional complete option mapping, recorded by the caller's manifest.
    """

    def __init__(
        self,
        basket_position: np.ndarray,
        basket_quaternion: np.ndarray,
        tcp_position: np.ndarray,
        hand_quaternion: np.ndarray,
        cup_position: np.ndarray,
        cup_quaternion: np.ndarray,
        dt: float,
        options: dict[str, Any] | None = None,
    ):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Require positive finite dt [s].")
        self.options = copy.deepcopy(OPTIONS if options is None else options)
        aim_offset = np.asarray(self.options["lip_aim_offset_cup_xy_m"], dtype=np.float64)
        if aim_offset.shape != (2,) or not np.isfinite(aim_offset).all():
            raise ValueError("Require a finite cup-frame XY lip offset [m], shape [2].")
        if type(self.options["rock_cycles"]) is not int or self.options["rock_cycles"] <= 0:
            raise ValueError("Require a positive integer rocking cycle count.")
        if not np.isfinite(self.options["rock_cycle_s"]) or self.options["rock_cycle_s"] <= 0:
            raise ValueError("Require a positive finite rocking cycle duration [s].")
        if not np.isfinite(self.options["rock_deg"]) or not 0 < self.options["rock_deg"] <= self.options["tilt_deg"]:
            raise ValueError("Rocking angle [deg] must be positive, finite, and within the forward tilt arc.")
        for key in ("terminal_tip_additional_deg", "terminal_tip_s"):
            if not np.isfinite(self.options[key]) or self.options[key] <= 0:
                raise ValueError("Terminal tip angle [deg] and duration [s] must be positive and finite.")
        if not np.isfinite(self.options["terminal_tip_lift_m"]) or self.options["terminal_tip_lift_m"] < 0:
            raise ValueError("Terminal tip lift [m] must be nonnegative and finite.")
        if (
            not np.isfinite(self.options["late_tilt_lift_start_deg"])
            or not 0 <= self.options["late_tilt_lift_start_deg"] < self.options["tilt_deg"]
        ):
            raise ValueError("Late lift start [deg] must be finite and within the forward tilt arc.")
        if (
            not np.isfinite(self.options["late_tilt_lift_m"])
            or not 0 <= self.options["late_tilt_lift_m"] <= self.options["terminal_tip_lift_m"]
        ):
            raise ValueError("Late lift [m] must be finite, nonnegative, and no larger than the final total lift.")
        self.dt = float(dt)
        basket = _pose(basket_position, basket_quaternion)
        hand_pose = _pose(tcp_position, hand_quaternion)
        self.cup = _pose(cup_position, cup_quaternion)
        self.geometry = _geometry.BasketPourReference(
            basket,
            hand_pose[:3],
            hand_pose[3:],
            self.cup,
            angle_deg=self.options["tilt_deg"],
            clearance_m=self.options["minimum_rim_clearance_m"],
            follow_clearance=True,
            align_grasp_radial=self.options["align_grasp_radial"],
        )
        self.feedback = _feedback.GraspFeedback(basket, hand_pose[:3], hand_pose[3:], self.options["grasp_feedback"])
        self.elapsed_s = 0.0
        self.alignment_hold_s = 0.0
        self.tilt_armed = False
        self._first_step = True
        self._level_rotation = self.geometry.upright
        self._level_slerp = Slerp([0.0, 1.0], Rotation.concatenate([self.geometry.start_r, self._level_rotation]))
        self._last_reference = self._command(0.0)
        self.diagnostics: dict[str, Any] = {}

    @property
    def tilt_start_s(self) -> float:
        """Return the gated rocking start on the local clock [s], excluding waiting."""
        return float(self.options["height_alignment_s"] + self.options["level_carry_s"])

    @property
    def forward_tilt_start_s(self) -> float:
        """Return the forward tilt start after the finite rocking schedule [s]."""
        return self.tilt_start_s + self.options["rock_cycles"] * float(self.options["rock_cycle_s"])

    @property
    def tilt_end_s(self) -> float:
        """Return the original tilt endpoint on the local clock [s]."""
        return self.forward_tilt_start_s + float(self.options["tilt_s"])

    @property
    def duration_s(self) -> float:
        """Return the bounded trajectory duration including the terminal tip [s]."""
        return self.tilt_end_s + float(self.options["terminal_tip_s"])

    def _rim(self, cup: np.ndarray) -> tuple[np.ndarray, float]:
        rotation = Rotation.from_quat(cup[3:])
        centre = cup[:3] + rotation.apply([0.0, 0.0, self.options["cup_rim_local_z_m"]])
        normal = rotation.apply([0.0, 0.0, 1.0])
        highest = centre[2] + self.options["cup_outer_rim_radius_m"] * np.linalg.norm(normal[:2])
        return centre, float(highest)

    def _command(self, time_s: float) -> dict[str, Any]:
        result = self.feedback.command(self._sample(time_s))
        result["tcp"] = result["tcp_position"]
        result["hand_quat"] = result["hand_xyzw"]
        return result

    def _lip_aim(self, cup: np.ndarray, rim_center: np.ndarray) -> np.ndarray:
        # Only aim XY moves; clearance still uses the unshifted actual rim geometry.
        offset = np.r_[self.options["lip_aim_offset_cup_xy_m"], 0.0]
        return rim_center + Rotation.from_quat(cup[3:]).apply(offset)

    def _sample(self, time_s: float) -> dict[str, Any]:
        geometry, options = self.geometry, self.options
        terminal_lift_m = 0.0
        rim, rim_top = self._rim(self.cup)
        aim = self._lip_aim(self.cup, rim)
        high = geometry.start_p.copy()
        clearance_floor = rim_top + options["minimum_rim_clearance_m"]
        captured_lowest = float(geometry.start_r.apply(geometry.boundary_b)[:, 2].min())
        high[2] = max(high[2], clearance_floor - min(0.0, captured_lowest))
        hover = np.r_[aim[:2], high[2] + geometry.lip_b[2]] - self._level_rotation.apply(geometry.lip_b)
        if time_s < options["height_alignment_s"]:
            fraction = float(_geometry.smooth(time_s / options["height_alignment_s"]))
            position = (1.0 - fraction) * geometry.start_p + fraction * high
            rotation = geometry.start_r
            phase = "height_alignment"
        elif time_s < self.tilt_start_s:
            fraction = float(_geometry.smooth((time_s - options["height_alignment_s"]) / options["level_carry_s"]))
            position = (1.0 - fraction) * high + fraction * hover
            rotation = self._level_slerp(fraction)
            position[2] = max(position[2], clearance_floor - float(rotation.apply(geometry.boundary_b)[:, 2].min()))
            phase = "level_carry"
        else:
            if time_s < self.forward_tilt_start_s:
                half_cycle_s = options["rock_cycle_s"] / 2.0
                cycle_time_s = (time_s - self.tilt_start_s) % options["rock_cycle_s"]
                rock_fraction = float(_geometry.smooth(cycle_time_s / half_cycle_s))
                if cycle_time_s >= half_cycle_s:
                    rock_fraction = 1.0 - float(_geometry.smooth((cycle_time_s - half_cycle_s) / half_cycle_s))
                # Reuse the original angle-dependent arc, including excess-height descent, in both directions.
                fraction = options["rock_deg"] / options["tilt_deg"] * rock_fraction
            else:
                fraction = float(_geometry.smooth((time_s - self.forward_tilt_start_s) / options["tilt_s"]))
            angle = geometry.angle * fraction
            late_fraction = float(
                _geometry.smooth(
                    (np.rad2deg(angle) - options["late_tilt_lift_start_deg"])
                    / (options["tilt_deg"] - options["late_tilt_lift_start_deg"])
                )
            )
            # This cumulative offset is applied after the original base-clearance height calculation.
            terminal_lift_m = options["late_tilt_lift_m"] * late_fraction
            if time_s >= self.tilt_end_s:
                tip_fraction = float(_geometry.smooth((time_s - self.tilt_end_s) / options["terminal_tip_s"]))
                angle += np.deg2rad(options["terminal_tip_additional_deg"]) * tip_fraction
                terminal_lift_m += (options["terminal_tip_lift_m"] - options["late_tilt_lift_m"]) * tip_fraction
            rotation = geometry.pour_yaw * Rotation.from_rotvec([-angle, 0.0, 0.0]) * geometry.grasp_alignment
            min_relative_z = float(rotation.apply(geometry.boundary_b - geometry.lip_b)[:, 2].min())
            # Preserve the carry endpoint, then remove excess height without a tilt-boundary jump.
            excess_height = (1.0 - fraction) * (high[2] - clearance_floor)
            lip = np.r_[aim[:2], clearance_floor - min_relative_z + excess_height]
            lip[2] += terminal_lift_m
            position = lip - rotation.apply(geometry.lip_b)
            if time_s < self.forward_tilt_start_s:
                phase = "rock"
            elif time_s < self.tilt_end_s:
                phase = "tilt"
            else:
                phase = "terminal_tip" if time_s < self.duration_s else "hold"
        tcp = position + rotation.apply(geometry.grip_b)
        hand = (rotation * geometry.hand_b).as_quat()
        return {
            "phase": phase,
            "basket_position": position,
            "basket_xyzw": rotation.as_quat(),
            "tcp_position": tcp,
            "hand_xyzw": hand,
            "tcp": tcp,
            "hand_quat": hand,
            "lip_position": position + rotation.apply(geometry.lip_b),
            "lowest_basket_z": float((position + rotation.apply(geometry.boundary_b))[:, 2].min()),
            "reference_cup_rim_highest_z": rim_top,
            "reference_lip_aim_position_m": aim,
            # Preserve the diagnostic key; its value includes the preceding late-forward lift.
            "terminal_tip_commanded_lift_m": terminal_lift_m,
        }

    def step(
        self,
        *,
        basket_position: np.ndarray,
        basket_quaternion: np.ndarray,
        tcp_position: np.ndarray,
        hand_quaternion: np.ndarray,
        cup_position: np.ndarray,
        cup_quaternion: np.ndarray,
        held: bool,
        valid: bool = True,
        delivered_count: int = 0,
    ) -> dict[str, Any]:
        """Return TCP [m]/hand XYZW command geometry plus gate diagnostics.

        Positions share a common world or environment-local frame. The return keys
        are ``tcp_position`` (alias ``tcp``), ``hand_xyzw`` (alias ``hand_quat``),
        ``basket_position``, ``basket_xyzw``, ``lip_position``, ``phase``, and
        ``diagnostics``. Receipt is diagnostic only;
        this helper neither changes task state nor starts the return trajectory.
        """
        try:
            basket = _pose(basket_position, basket_quaternion)
            hand = _pose(tcp_position, hand_quaternion)
            cup = _pose(cup_position, cup_quaternion)
        except ValueError:
            valid = False
        if not valid:
            self.alignment_hold_s = 0.0
            self.diagnostics = {
                "phase": self._last_reference["phase"],
                "local_elapsed_s": self.elapsed_s,
                "clock_advanced": False,
                "valid": False,
                "held": bool(held),
                "tilt_armed": self.tilt_armed,
                "alignment_hold_s": 0.0,
                "delivered_count": int(delivered_count),
                "goal_fruit_count": self.options["goal_fruit_count"],
            }
            return {**self._last_reference, "diagnostics": dict(self.diagnostics)}

        options = self.options
        if held:
            self.cup[:3] += _feedback.bounded_vector(
                cup[:3] - self.cup[:3], options["cup_reference_translation_rate_m_s"] * self.dt
            )
            rotation = Rotation.from_quat(self.cup[3:])
            change = _feedback.bounded_vector(
                (rotation.inv() * Rotation.from_quat(cup[3:])).as_rotvec(),
                options["cup_reference_rotation_rate_rad_s"] * self.dt,
            )
            self.cup[3:] = (rotation * Rotation.from_rotvec(change)).as_quat()
        self.feedback.update(basket, hand[:3], hand[3:], self.dt, held=held, valid=True)
        reference = self._command(self.elapsed_s)
        basket_rotation = Rotation.from_quat(basket[3:])
        errors = {
            "tcp_position_error_m": float(np.linalg.norm(reference["tcp_position"] - hand[:3])),
            "hand_rotation_error_rad": float(
                (Rotation.from_quat(reference["hand_xyzw"]).inv() * Rotation.from_quat(hand[3:])).magnitude()
            ),
            "basket_position_error_m": float(np.linalg.norm(reference["basket_position"] - basket[:3])),
            "basket_rotation_error_rad": float(
                (Rotation.from_quat(reference["basket_xyzw"]).inv() * basket_rotation).magnitude()
            ),
        }
        tracking = (
            max(errors["tcp_position_error_m"], errors["basket_position_error_m"])
            <= options["tracking_position_tolerance_m"]
            and max(errors["hand_rotation_error_rad"], errors["basket_rotation_error_rad"])
            <= options["tracking_rotation_tolerance_rad"]
        )
        rim, actual_rim_top = self._rim(cup)
        aim = self._lip_aim(cup, rim)
        measured_lip = basket[:3] + basket_rotation.apply(self.geometry.lip_b)
        lip_error = float(np.linalg.norm(measured_lip[:2] - aim[:2]))
        cup_center_error = float(np.linalg.norm(measured_lip[:2] - rim[:2]))
        aligned = lip_error <= options["lip_alignment_tolerance_m"]
        upright = float(basket_rotation.as_matrix()[2, 2]) >= options["upright_cosine"]
        cup_upright = float(Rotation.from_quat(cup[3:]).as_matrix()[2, 2]) >= options["upright_cosine"]
        gate = held and tracking and aligned and upright and cup_upright
        if self.elapsed_s >= self.tilt_start_s and not self.tilt_armed:
            self.alignment_hold_s = self.alignment_hold_s + self.dt if gate else 0.0
            self.tilt_armed = self.alignment_hold_s >= options["alignment_hold_s"] - 1e-10
        advance = held and tracking and cup_upright and not self._first_step
        if self.elapsed_s >= self.tilt_start_s:
            advance = advance and self.tilt_armed and aligned
        previous_time = self.elapsed_s
        if advance:
            if self.elapsed_s < self.tilt_start_s:
                boundary = self.tilt_start_s
            elif self.elapsed_s < self.forward_tilt_start_s:
                boundary = self.forward_tilt_start_s
            elif self.elapsed_s < self.tilt_end_s:
                boundary = self.tilt_end_s
            else:
                boundary = self.duration_s
            self.elapsed_s = min(self.elapsed_s + self.dt, boundary)
        self._first_step = False
        reference = self._command(self.elapsed_s)
        translation, angle, updated = self.feedback.diagnostics()
        self.diagnostics = {
            "phase": reference["phase"],
            "local_elapsed_s": self.elapsed_s,
            "clock_advanced": self.elapsed_s > previous_time,
            "valid": True,
            "held": bool(held),
            "tracking": bool(tracking),
            "tilt_armed": bool(self.tilt_armed),
            "alignment_gate": bool(gate),
            "alignment_hold_s": self.alignment_hold_s,
            "measured_lip_alignment_error_m": lip_error,
            "measured_lip_cup_center_error_m": cup_center_error,
            "actual_cup_rim_center_m": rim.tolist(),
            "lip_aim_position_m": aim.tolist(),
            "lip_aim_offset_cup_xy_m": list(options["lip_aim_offset_cup_xy_m"]),
            "lip_aim_offset_world_m": (aim - rim).tolist(),
            "reference_lip_aim_position_m": reference["reference_lip_aim_position_m"].tolist(),
            "measured_lip_position_m": measured_lip.tolist(),
            "measured_basket_upright_cosine": float(basket_rotation.as_matrix()[2, 2]),
            "measured_cup_upright": bool(cup_upright),
            "command_clearance_above_actual_rim_m": float(reference["lowest_basket_z"] - actual_rim_top),
            "terminal_tip_commanded_lift_m": reference["terminal_tip_commanded_lift_m"],
            "grasp_translation_correction_m": translation,
            "grasp_rotation_correction_rad": angle,
            "grasp_feedback_updated": bool(updated),
            "delivered_count": int(delivered_count),
            "goal_fruit_count": options["goal_fruit_count"],
            **errors,
        }
        self._last_reference = reference
        return {**reference, "diagnostics": dict(self.diagnostics)}
