# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Three-berry sorting demonstration; commands only, never edits tissue state."""

import numpy as np

from ..scene.tableware import BOWL, REJECT
from .motion import scripted_motion_time

# Task-frame center and usable radius [m] of the reject dish.
DISCARD = REJECT[:2]
DISCARD_RADIUS = REJECT[3]


def _ramp(t: float, start: float, duration: float) -> float:
    u = float(np.clip((t - start) / duration, 0, 1))
    return u * u * (3 - 2 * u)


class BerrySortSequence:
    """Crush the first berry and discard it in the reject dish, then gently place the other two in the glass bowl."""

    def __init__(self, speed: float, pick_gap: float | None = None):
        """Configure the arm speed multiplier and an optional grasp aperture [m] for the accepted berries."""
        self.speed = speed
        self.pick_gap = pick_gap
        # Each berry follows the place script, which contains 18 seconds of arm travel.
        self.cycle_seconds = 14 + 18 / speed
        self.reset()

    def reset(self) -> None:
        """Restart the command sequence without resetting any physical state."""
        self.index = -1
        self.center = None
        self.phase = "Settling in punnet"
        self.delay = 0.0
        self.last_elapsed = 0.0
        self.complete = False
        self.failed = False
        self.grip_adjustment = 0.0
        self.grip_offset = None
        self.holding = False

    def command(self, elapsed: float, berries: dict, tcp: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Return the desired TCP position [m] and gripper aperture [m].

        Args:
            elapsed: Simulation time since the sequence reset [s].
            berries: The three berries, in sorting order.
            tcp: Measured TCP position [m]; omit only for offline trajectory inspection.
        """
        dt = max(0.0, elapsed - self.last_elapsed)
        self.last_elapsed = elapsed
        sequence_time = elapsed - self.delay
        index = min(int(sequence_time / self.cycle_seconds), 2)
        t = scripted_motion_time(sequence_time - index * self.cycle_seconds, "place", self.speed)
        berry = list(berries.values())[index]
        if index != self.index:
            self.index = index
            self.center = None
            self.grip_adjustment = 0.0
            self.grip_offset = None
        if self.center is None or t < 1:
            tissue = berry.positions()
            self.center = (tissue.min(0) + tissue.max(0)) / 2 + berry.offset
            width = float(np.ptp(tissue[:, 1]))
            self.opening = float(np.clip(width + 0.008, 0.02, 0.08))
            self.gap = 0.001 if index == 0 else (self.pick_gap or float(np.clip(width * 0.7, 0.001, 0.08)))
        destination = np.array(DISCARD if index == 0 else BOWL[:2])
        # With the hand facing down, pad centers sit 3.6 mm above the TCP.
        grasp_z = max(0.009, self.center[2] - 0.0036)
        target = self.center.copy()
        target[2] = 0.09 - (0.09 - grasp_z) * _ramp(t, 1, 3)
        target[2] += 0.09 * _ramp(t, 9, 3)
        target[:2] += (destination - target[:2]) * _ramp(t, 13, 6)
        release_z = 0.028 if index == 0 else 0.035
        target[2] += (release_z - grasp_z - 0.09) * _ramp(t, 20, 3)
        target[2] += 0.08 * _ramp(t, 26, 3)
        self.holding = False
        if tcp is not None:
            # Freeze the clock at each arm-arrival boundary, especially before
            # release. A high speed multiplier must not open fingers in transit.
            gated = t < 1 or any(start <= t < start + 1 for start in (4, 12, 19, 23))
            self.holding = bool(gated and np.linalg.norm(tcp - target) > 0.003)
            if index > 0 and 8 <= t < 24:
                center_z = float(berry.positions()[:, 2].mean() + berry.offset[2])
                offset = center_z - tcp[2]
                if self.grip_offset is None:
                    self.grip_offset = offset
                slip = self.grip_offset - offset
                if slip > 0.002 and self.pick_gap is None:
                    # Bounded position feedback; no adhesive force or attachment.
                    self.grip_adjustment = min(self.grip_adjustment + dt * 0.001, self.gap * 0.18)
                if slip > 0.005:
                    self.holding = True
                    target = tcp.copy()
                if slip > 0.018:
                    self.failed = True
            if self.failed:
                self.holding = True
                target = tcp.copy()
            if self.holding:
                self.delay += dt
        gap = self.gap - self.grip_adjustment
        aperture = self.opening + (gap - self.opening) * np.clip((t - 4) / 4, 0, 1)
        aperture += (0.08 - gap) * np.clip((t - 24) / 2, 0, 1)
        phase = "Approach"
        if t >= 4:
            phase = "Crush" if index == 0 else "Gentle grasp"
        if t >= 9:
            phase = "Carry to discard" if index == 0 else "Carry to glass bowl"
        if t >= 24:
            phase = "Release and retreat"
        self.phase = f"Berry {index + 1}/3: {phase}"
        if self.holding:
            self.phase += " (waiting for arm / grip)"
        if self.failed:
            self.phase = f"Berry {index + 1}/3: grasp lost; press R to retry"
        self.complete = sequence_time >= 3 * self.cycle_seconds
        if self.complete:
            self.phase = "Sequence complete"
        return target, float(aperture)


def sorting_result(berries: dict) -> dict:
    """Measure the final physical outcome, without changing state or assuming success."""
    names = list(berries)
    states = {name: berry.metrics() for name, berry in berries.items()}
    rejected = berries[names[0]]
    world = rejected.positions() + rejected.offset
    fraction_discarded = float(
        (
            (np.linalg.norm(world[:, :2] - DISCARD, axis=1) < DISCARD_RADIUS)
            & (world[:, 2] >= REJECT[4] - 0.001)
            & (world[:, 2] < REJECT[5])
        ).mean()
    )
    damaged = states[names[0]]["mean_damage"] >= 0.02
    accepted = all(
        states[name]["fraction_in_bowl"] >= 0.95 and states[name]["mean_damage"] <= 0.01 for name in names[1:]
    )
    return dict(
        passed=damaged and fraction_discarded >= 0.9 and accepted,
        fraction_in_discard=fraction_discarded,
        berries=states,
        criteria=dict(
            min_reject_mean_damage=0.02,
            min_discard_fraction=0.9,
            min_accepted_bowl_fraction=0.95,
            max_accepted_mean_damage=0.01,
        ),
    )
