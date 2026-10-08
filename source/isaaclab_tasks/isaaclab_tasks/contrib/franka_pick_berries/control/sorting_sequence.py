# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted sorting of three raspberries: crush the first and discard it, gently place the others in the bowl.

:class:`SortingSequence` drives the robot like an operator would, through the environment's actions only: it never
edits the tissue. It watches the berries to aim the grasp, tightens a slipping grip and waits for the arm when it lags.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..scene.tableware import BOWL, REJECT_DISH
from .script_timing import elapsed_time, pause_speedup, script_time

# Task-frame center and usable radius [m] of the reject dish.
REJECT_DISH_CENTER = REJECT_DISH[:2]
REJECT_DISH_RADIUS = REJECT_DISH[3]
# Task-frame spots [m] in the bowl where the accepted berries are set down, side by side across the fingers' opening
# direction, so that the second is not lowered onto the first and the open fingers clear the bowl's wall.
BOWL_SPOTS = ((BOWL[0] - 0.022, BOWL[1]), (BOWL[0] + 0.022, BOWL[1]))


def _ramp(t: float, start: float, duration: float) -> float:
    u = float(np.clip((t - start) / duration, 0, 1))
    return u * u * (3 - 2 * u)


class SortingSequence:
    """Crush the first berry and discard it in the reject dish, then gently place the other two in the bowl."""

    def __init__(
        self,
        arm_speed: float,
        start_delay: float = 0.0,
        slow_closing: tuple[int, ...] = (),
    ):
        """Configure the sequence.

        Args:
            arm_speed: Arm speed multiplier.
            start_delay: Time [s] the arm waits above the first berry before the sequence starts, for example for
                an establishing shot.
            slow_closing: Indices of the berries whose grasps keep the gripper's original closing pace, which shows
                their deformation longer.
        """
        self.arm_speed = arm_speed
        self.pauses = pause_speedup(arm_speed)
        self.start_delay = start_delay
        # Each berry follows the place script.
        self.closing = [1.0 if index in slow_closing else None for index in range(3)]
        self.cycle_ends = np.cumsum([elapsed_time(32.0, arm_speed, self.pauses, closing) for closing in self.closing])
        self.reset()

    def reset(self) -> None:
        """Restart the command sequence without resetting any physical state."""
        self.index = -1
        self.center = None
        self.script_time = 0.0
        self.phase = "Settling in punnet"
        self.delay = 0.0
        self.last_elapsed = 0.0
        self.complete = False
        self.failed = False
        self.grip_adjustment = 0.0
        self.grip_offset = None
        self.hang = None
        self.holding = False
        self.tcp = None

    def action(self, env, elapsed: float) -> torch.Tensor:
        """Return the environment action toward this step's command.

        Args:
            env: The berry environment, with three berries.
            elapsed: Simulation time since the sequence reset [s].
        """
        position, orientation = env.action_manager.get_term("arm_action").tcp_pose()
        self.tcp = position
        target, aperture = self.command(elapsed, env.berries, position)
        # Per-step limits: 8 mm and 0.03 rad at the default speed 2, scaled with it so that the arm keeps up.
        step_limit, turn_limit = 0.004 * self.arm_speed, 0.015 * self.arm_speed
        # Turn the hand back to pointing straight down.
        turn = (Rotation.from_euler("x", np.pi) * Rotation.from_quat(orientation).inv()).as_rotvec()
        action = torch.zeros((1, 7))
        action[0, :3] = torch.from_numpy(np.clip((target - position) * 0.6, -step_limit, step_limit))
        action[0, 3:6] = torch.from_numpy(np.clip(turn * 0.2, -turn_limit, turn_limit))
        action[0, 6] = aperture / 0.04 - 1
        return action

    def command(self, elapsed: float, berries: dict, tcp: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Return the desired TCP position [m] and gripper aperture [m].

        Args:
            elapsed: Simulation time since the sequence reset [s].
            berries: The three berries, in sorting order.
            tcp: Measured TCP position [m]; omit only for offline trajectory inspection.
        """
        dt = max(0.0, elapsed - self.last_elapsed)
        self.last_elapsed = elapsed
        sequence_time = max(0.0, elapsed - self.start_delay - self.delay)
        index = min(int(np.searchsorted(self.cycle_ends, sequence_time, side="right")), 2)
        cycle_start = self.cycle_ends[index - 1] if index else 0.0
        t = script_time(sequence_time - cycle_start, self.arm_speed, self.pauses, self.closing[index])
        # Script time [s] of the current berry, for observers such as a camera.
        self.script_time = t
        berry = list(berries.values())[index]
        if index != self.index:
            self.index = index
            self.center = None
            self.grip_adjustment = 0.0
            self.grip_offset = None
            self.hang = None
        if self.center is None or t < 1:
            tissue = berry.positions()
            self.center = (tissue.min(0) + tissue.max(0)) / 2 + berry.offset
            width = float(np.ptp(tissue[:, 1]))
            self.opening = float(np.clip(width + 0.008, 0.02, 0.08))
            self.gap = 0.001 if index == 0 else float(np.clip(width * 0.7, 0.001, 0.08))
        destination = np.array(REJECT_DISH_CENTER if index == 0 else BOWL_SPOTS[index - 1])
        # With the hand facing down, pad centers sit 3.6 mm above the TCP.
        grasp_z = max(0.009, self.center[2] - 0.0036)
        target = self.center.copy()
        target[2] = 0.09 - (0.09 - grasp_z) * _ramp(t, 1, 3)
        target[2] += 0.09 * _ramp(t, 9, 3)
        target[:2] += (destination - target[:2]) * _ramp(t, 13, 6)
        # The accepted berries are set down, not dropped: lowered until they hang just above the bowl's floor, however
        # far they sit below the fingers. A fall would also smear them on screen.
        release_z = 0.028
        if index > 0:
            hang = self.hang if self.hang is not None else grasp_z - BOWL[4]
            release_z = max(0.009, BOWL[4] + 0.0025 + hang)
        target[2] += (release_z - grasp_z - 0.09) * _ramp(t, 20, 3)
        target[2] += 0.08 * _ramp(t, 26, 3)
        self.holding = False
        if tcp is not None:
            # Freeze the clock at each arm-arrival boundary, especially before
            # release. A high speed multiplier must not open fingers in transit.
            # The grasp (4 s) and the release (23 s) need the arm in place; the start and the transit checkpoints
            # after lifting (12 s) and over the destination (19 s) only need it close.
            tolerance = None
            if t < 1:
                tolerance = 0.006
            elif any(start <= t < start + 1 for start in (4, 23)):
                tolerance = 0.003
            elif any(start <= t < start + 1 for start in (12, 19)):
                tolerance = 0.010
            self.holding = bool(tolerance is not None and np.linalg.norm(tcp - target) > tolerance)
            tissue_z = berry.positions()[:, 2] + berry.offset[2] if index > 0 and 8 <= t < 24 else None
            if tissue_z is not None and 12 <= t < 20:
                # Height [m] of the TCP above the berry's bottom, measured over the bowl before setting it down. A low
                # percentile ignores a particle left behind in the punnet.
                self.hang = float(tcp[2] - np.percentile(tissue_z, 2))
            if tissue_z is not None:
                center_z = float(tissue_z.mean())
                offset = center_z - tcp[2]
                if self.grip_offset is None:
                    self.grip_offset = offset
                slip = self.grip_offset - offset
                if slip > 0.002:
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
            if self.holding and elapsed >= self.start_delay:
                self.delay += dt
        gap = self.gap - self.grip_adjustment
        aperture = self.opening + (gap - self.opening) * np.clip((t - 4) / 4, 0, 1)
        # Inside the bowl the fingers open only to the approach opening, which keeps them clear of its wall.
        aperture += ((0.08 if index == 0 else self.opening) - gap) * np.clip((t - 24) / 2, 0, 1)
        phase = "Approach"
        if t >= 4:
            phase = "Crush" if index == 0 else "Gentle grasp"
        if t >= 9:
            phase = "Carry to reject dish" if index == 0 else "Carry to bowl"
        if t >= 24:
            phase = "Release and retreat"
        self.phase = f"Berry {index + 1}/3: {phase}"
        if self.holding:
            self.phase += " (waiting for arm / grip)"
        if self.failed:
            self.phase = f"Berry {index + 1}/3: grasp lost; press R to retry"
        if elapsed < self.start_delay:
            self.phase = "Opening"
        self.complete = sequence_time >= self.cycle_ends[-1]
        if self.complete:
            self.phase = "Sequence complete"
        return target, float(aperture)


def evaluate_sorting(berries: dict) -> dict:
    """Measure the final physical outcome, without changing state or assuming success."""
    names = list(berries)
    states = {name: berry.metrics() for name, berry in berries.items()}
    rejected = berries[names[0]]
    world = rejected.positions() + rejected.offset
    fraction_rejected = float(
        (
            (np.linalg.norm(world[:, :2] - REJECT_DISH_CENTER, axis=1) < REJECT_DISH_RADIUS)
            & (world[:, 2] >= REJECT_DISH[4] - 0.001)
            & (world[:, 2] < REJECT_DISH[5])
        ).mean()
    )
    damaged = states[names[0]]["mean_damage"] >= 0.02
    accepted = all(
        states[name]["fraction_in_bowl"] >= 0.95 and states[name]["mean_damage"] <= 0.01 for name in names[1:]
    )
    return dict(
        passed=damaged and fraction_rejected >= 0.9 and accepted,
        fraction_in_reject_dish=fraction_rejected,
        berries=states,
        criteria=dict(
            min_reject_mean_damage=0.02,
            min_reject_dish_fraction=0.9,
            min_accepted_bowl_fraction=0.95,
            max_accepted_mean_damage=0.01,
        ),
    )


def sorting_summary(result: dict) -> str:
    """Describe the outcome measured by :func:`evaluate_sorting` in one sentence."""
    crushed, *picked = result["berries"].values()
    in_bowl = " and ".join(f"{berry['fraction_in_bowl']:.0%}" for berry in picked)
    damage = " and ".join(f"{berry['mean_damage']:.1%}" for berry in picked)
    return (
        f"Sorting done: the crushed berry ({crushed['mean_damage']:.1%} damaged) is "
        f"{result['fraction_in_reject_dish']:.0%} in the reject dish; the two picked berries are {in_bowl} in the "
        f"bowl, {damage} damaged."
    )
