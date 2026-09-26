# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded measured-grasp estimation; this changes TCP commands, never object states."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def bounded_vector(vector: np.ndarray, maximum: float) -> np.ndarray:
    """Limit a translation [m] or rotation vector [rad] without changing its direction."""
    norm = float(np.linalg.norm(vector))
    return vector * min(1.0, maximum / max(norm, 1e-15))


class GraspFeedback:
    """Estimate the basket-to-hand transform from measured poses with rate and total bounds."""

    def __init__(self, basket: np.ndarray, tcp: np.ndarray, hand_xyzw: np.ndarray, options: dict):
        self.options = dict(options)
        rotation = Rotation.from_quat(basket[3:])
        self.initial_grip = rotation.inv().apply(tcp - basket[:3])
        self.initial_hand = rotation.inv() * Rotation.from_quat(hand_xyzw)
        self.grip = self.initial_grip.copy()
        self.hand = self.initial_hand
        self.updated = False

    def update(self, basket, tcp, hand_xyzw, step_dt: float, *, held: bool, valid: bool) -> bool:
        """Update only while a finite, valid grasp is measured; rate limits use policy dt [s]."""
        self.updated = False
        if not held or not valid:
            return False
        if not np.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Require a positive finite policy timestep.")
        if not np.isfinite(np.r_[basket, tcp, hand_xyzw]).all():
            return False
        rotation = Rotation.from_quat(basket[3:])
        measured_grip = rotation.inv().apply(tcp - basket[:3])
        measured_hand = rotation.inv() * Rotation.from_quat(hand_xyzw)
        options = self.options
        fraction = -np.expm1(-step_dt / options["time_constant_s"])
        target_grip = self.initial_grip + bounded_vector(
            measured_grip - self.initial_grip, options["maximum_translation_from_capture_m"]
        )
        target_hand = self.initial_hand * Rotation.from_rotvec(
            bounded_vector(
                (self.initial_hand.inv() * measured_hand).as_rotvec(),
                options["maximum_rotation_from_capture_rad"],
            )
        )
        self.grip += bounded_vector(
            fraction * (target_grip - self.grip), options["maximum_translation_rate_m_s"] * step_dt
        )
        change = bounded_vector(
            fraction * (self.hand.inv() * target_hand).as_rotvec(),
            options["maximum_rotation_rate_rad_s"] * step_dt,
        )
        self.hand = self.hand * Rotation.from_rotvec(change)
        self.updated = True
        return True

    def command(self, reference: dict) -> dict:
        """Compensate measured grasp drift while preserving basket pose and any TCP retreat offset."""
        result = dict(reference)
        key = "basket_xyzw" if "basket_xyzw" in reference else "basket_reference_xyzw"
        rotation = Rotation.from_quat(reference[key])
        result["tcp_position"] = np.asarray(reference["tcp_position"]) + rotation.apply(self.grip - self.initial_grip)
        result["hand_xyzw"] = (rotation * self.hand).as_quat()
        return result

    def diagnostics(self) -> tuple[float, float, bool]:
        """Return bounded translation [m], rotation [rad], and the last update flag."""
        return (
            float(np.linalg.norm(self.grip - self.initial_grip)),
            float((self.initial_hand.inv() * self.hand).magnitude()),
            self.updated,
        )
