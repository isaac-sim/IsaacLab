# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""CPU-only basket-pour reference geometry and current-robot IK audit.

This constructs commands, not an attachment or a prescribed simulated object pose.
Only the TCP target is intended to be sent to PoseController.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent
LIP_B = np.array([0.0, 0.0605, 0.100])
CUP_RIM_LOCAL_Z = 0.210


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class BasketPourReference:
    """Preserve a measured TCP-to-basket transform through a slow reference path."""

    def __init__(
        self,
        basket_pose,
        tcp_position,
        hand_xyzw,
        cup_pose,
        yaw_deg=0.0,
        angle_deg=100.0,
        clearance_m=0.030,
        follow_clearance=False,
        align_grasp_radial=False,
    ):
        self.start_p = np.asarray(basket_pose[:3]).copy()
        self.start_r = Rotation.from_quat(basket_pose[3:])
        self.grip_b = self.start_r.inv().apply(np.asarray(tcp_position) - self.start_p)
        self.hand_b = self.start_r.inv() * Rotation.from_quat(hand_xyzw)
        self.cup = np.asarray(cup_pose).copy()
        self.lip_b = LIP_B.copy()
        self.pour_yaw = Rotation.from_euler("z", yaw_deg, degrees=True)
        self.grasp_alignment = Rotation.identity()
        if align_grasp_radial:
            radial_distance = np.linalg.norm(self.grip_b[:2])
            if radial_distance < 1e-6:
                raise ValueError("Radial alignment requires a grasp away from the basket axis.")
            self.lip_b[:2] = -self.grip_b[:2] * LIP_B[1] / radial_distance
            phi = np.arctan2(-self.grip_b[0], -self.grip_b[1])
            self.grasp_alignment = Rotation.from_rotvec([0.0, 0.0, phi])
        self.upright = self.pour_yaw * self.grasp_alignment
        self.to_upright = Slerp([0.0, 1.0], Rotation.concatenate([self.start_r, self.upright]))
        self.angle = np.deg2rad(angle_deg)
        self.clearance_m = clearance_m
        self.follow_clearance = follow_clearance
        angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        self.boundary_b = np.concatenate(
            [
                np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, z)], -1)
                for radius, z in [(0.039, 0.0), (0.0605, 0.1)]
            ]
        )

    def sample(self, time_s):
        # Relative to an already established grasp: raise3s, carry6s, tilt12s, hold6s.
        high = self.start_p.copy()
        high[2] = max(high[2], self.cup[2] + CUP_RIM_LOCAL_Z + (self.clearance_m if self.follow_clearance else 0.07))
        cup_xy = self.cup[:2]
        hover = np.r_[cup_xy, high[2] + 0.100] - self.upright.apply(self.lip_b)
        if time_s < 3.0:
            fraction = smooth(time_s / 3.0)
            position = (1 - fraction) * self.start_p + fraction * high
            rotation = self.start_r
            phase = "raise"
        elif time_s < 9.0:
            fraction = smooth((time_s - 3.0) / 6.0)
            position = (1 - fraction) * high + fraction * hover
            rotation = self.to_upright(float(fraction))
            phase = "carry"
        else:
            fraction = smooth((time_s - 9.0) / 12.0)
            rotation = self.pour_yaw * Rotation.from_rotvec([-self.angle * fraction, 0, 0]) * self.grasp_alignment
            # Keep the lowest basket point above the receiving rim throughout tilt.
            min_relative_z = rotation.apply(self.boundary_b - self.lip_b)[:, 2].min()
            floor = self.cup[2] + CUP_RIM_LOCAL_Z + self.clearance_m - min_relative_z
            initial_lip_z = high[2] + 0.100
            # Descend gradually from the high carry position, without a phase jump.
            proposed = initial_lip_z * (1 - fraction) + (self.cup[2] + CUP_RIM_LOCAL_Z + self.clearance_m) * fraction
            lip = np.r_[cup_xy, floor if self.follow_clearance else max(floor, proposed)]
            position = lip - rotation.apply(self.lip_b)
            phase = "tip" if time_s < 21.0 else "hold"
        tcp = position + rotation.apply(self.grip_b)
        hand = rotation * self.hand_b
        lip = position + rotation.apply(self.lip_b)
        minimum_z = (position + rotation.apply(self.boundary_b))[:, 2].min()
        return {
            "phase": phase,
            "basket_position": position,
            "basket_xyzw": rotation.as_quat(),
            "tcp_position": tcp,
            "hand_xyzw": hand.as_quat(),
            "lip_position": lip,
            "lowest_basket_z": float(minimum_z),
        }
