# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only measured-grasp basket placement candidate; no simulated object writes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent

from .basket_reference import BasketPourReference, smooth

HOME = np.array([0.36, 0.27, 0.002])


class ReturnCandidate:
    """Command poses [m, XYZW] through guarded height, upright return, and release."""

    def __init__(self, basket, tcp, hand, cup):
        self.p = basket[:3].copy()
        self.rotation = Rotation.from_quat(basket[3:])
        self.grip_b = self.rotation.inv().apply(tcp - self.p)
        self.hand_b = self.rotation.inv() * hand
        self.boundary = BasketPourReference(basket, tcp, hand.as_quat(), cup).boundary_b
        self.lip_b = np.array([0.0, 0.0605, 0.1])
        self.safe_z = cup[2] + 0.210 + 0.05
        self.raise_m = max(0.0, self.safe_z - (self.p + self.rotation.apply(self.boundary))[:, 2].min())
        self.lip = self.p + self.rotation.apply(self.lip_b) + [0.0, 0.0, self.raise_m]
        self.slerp = Slerp([0.0, 1.0], Rotation.concatenate([self.rotation, Rotation.identity()]))
        self.upright_p = self.untilt(1.0)[0]
        self.home_high = np.array([*HOME[:2], max(self.upright_p[2], self.safe_z)])
        self.home_pre = np.array([*HOME[:2], 0.025])
        self.home_rest = np.array([*HOME[:2], 0.0015])

    def untilt(self, fraction):
        rotation = self.slerp(float(fraction))
        lip = self.lip.copy()
        lip[2] = max(lip[2], self.safe_z - rotation.apply(self.boundary - self.lip_b)[:, 2].min())
        return lip - rotation.apply(self.lip_b), rotation

    def sample(self, t):
        closed = t < 34.0
        if t < 3.0:
            p, r, stage = self.p + np.array([0.0, 0.0, self.raise_m * smooth(t / 3.0)]), self.rotation, "clear_cup"
        elif t < 15.0:
            p, r = self.untilt(smooth((t - 3.0) / 12.0))
            stage = "upright"
        elif t < 23.0:
            u = smooth((t - 15.0) / 8.0)
            p = (1 - u) * self.upright_p + u * self.home_high
            r = Rotation.identity()
            stage = "return_above_home"
        elif t < 29.0:
            u = smooth((t - 23.0) / 6.0)
            p = (1 - u) * self.home_high + u * self.home_pre
            r = Rotation.identity()
            stage = "lower_coarse"
        elif t < 32.0:
            u = smooth((t - 29.0) / 3.0)
            p = (1 - u) * self.home_pre + u * self.home_rest
            r = Rotation.identity()
            stage = "lower_fine"
        else:
            p = self.home_rest.copy()
            r = Rotation.identity()
            stage = (
                "settle_closed" if t < 34.0 else ("open" if t < 37.0 else ("retreat" if t < 40.0 else "released_dwell"))
            )
        tcp = p + r.apply(self.grip_b)
        if t >= 37.0:
            tcp[2] += 0.12 * smooth((t - 37.0) / 3.0)
        return {
            "phase": stage,
            "basket_reference_position": p,
            "basket_reference_xyzw": r.as_quat(),
            "tcp_position": tcp,
            "hand_xyzw": (r * self.hand_b).as_quat(),
            "close": closed,
            "lowest_basket_reference_z": float((p + r.apply(self.boundary))[:, 2].min()),
        }
