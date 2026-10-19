# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured-pose IK demonstration of one scoop and optional stirring."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from .assets import BOWL_POSITION, CUP_HEIGHT, CUP_POSITION, SPOON_BOWL_WIDTH
from .robot import HAND_HOLD_POSITIONS

if TYPE_CHECKING:
    from .demo_env import CoffeeDemoEnv


class CoffeeScoopDemo:
    """Generate absolute palm and finger targets for the demo's IK controller.

    Waypoints describe the spoon cavity in the environment frame. Each command
    is converted through the measured robot root and palm/tool transform, so IK
    controls the physical arm through its configured rigid tool offset.
    Phase transitions require both elapsed motion time and measured arrival.
    """

    def __init__(self, env: CoffeeDemoEnv) -> None:
        self._env = env
        self._offset_pos = torch.tensor(env.cfg.spoon_offset_pos, device=env.device).expand(env.num_envs, -1)
        self._offset_quat = torch.tensor(env.cfg.spoon_offset_rot, device=env.device).expand(env.num_envs, -1)
        self._hand = torch.tensor(HAND_HOLD_POSITIONS, device=env.device).expand(env.num_envs, -1)
        self._phases: list[tuple[str, float, torch.Tensor, float]] = []
        self.reset()

    @property
    def phase(self) -> str:
        """Current motion phase, or ``complete`` after returning upright."""
        return "complete" if self.finished else self._phases[self._phase_index][0]

    @property
    def finished(self) -> bool:
        """Whether all waypoints were reached with the measured physical spoon."""
        return self._phase_index >= len(self._phases)

    def reset(self) -> None:
        """Restart from the current physical spoon pose after an environment reset."""
        initial = self._env.spoon_pose()
        source_x, source_y, source_z = BOWL_POSITION
        # Follow the coffee pile independently of the surrounding bowl size.
        scoop_x = source_x + min(0.015, 0.030 * (1.0 - self._env.cfg.coffee_fill_scale))
        scoop_y = source_y - 0.010 * self._env.cfg.coffee_fill_scale
        entry_x = scoop_x - 0.031
        lift_x = scoop_x - 0.016
        cup_x, cup_y, cup_z = CUP_POSITION
        level = (0.0, 0.0, 0.0, 1.0)
        steep = (0.0, math.sin(math.pi / 6.0), 0.0, math.cos(math.pi / 6.0))
        tipped = (math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0))
        # Curl upward while leveling so the original flat handle clears the
        # ceramic rim before the loaded spoon lifts out of the powder.
        scoop_z = source_z + 0.0335
        lift_position = (lift_x + 0.005, scoop_y, source_z + 0.075)
        transfer_z = max(source_z + 0.090, cup_z + CUP_HEIGHT + 0.030)
        pour_z = cup_z + CUP_HEIGHT + 0.025
        # Positive X-axis roll releases over the -Y lip. Center that lip over
        # the mouth rather than centering the unrotated spoon cavity.
        pour_y = cup_y + SPOON_BOWL_WIDTH * 0.40

        def pose(position, quaternion=level):
            return (
                torch.tensor((*position, *quaternion), dtype=torch.float32, device=self._env.device)
                .expand(self._env.num_envs, -1)
                .clone()
            )

        # First raise the current tool before leveling it, keeping the hand and
        # the long handle clear of the containers during the large wrist turn.
        clearance = initial.clone()
        clearance[:, 2] = torch.maximum(clearance[:, 2], torch.full_like(clearance[:, 2], transfer_z + 0.08))
        approach = pose((entry_x, scoop_y, source_z + 0.090), steep)
        # Enter steeply, then curl level inside the powder during a short
        # forward draw. Keep the lateral position fixed until the level lift.
        # Nonzero interior velocities keep the motion continuous;
        # zero knot accelerations make the quintic segments C2 continuous.
        # Coordinates are X/Y/Z and pitch, followed by derivatives per s.
        self._scoop_knots = [
            (0.0, (entry_x, scoop_y, source_z + 0.090, math.radians(60)), (0.0, 0.0, 0.0, 0.0)),
            (0.65, (entry_x, scoop_y, source_z + 0.070, math.radians(60)), (0.0, 0.0, -0.015, 0.0)),
            (
                1.95,
                (entry_x, scoop_y, source_z + 0.030, math.radians(60)),
                (0.001, 0.0, -0.002, math.radians(-5)),
            ),
            (
                3.35,
                (scoop_x - 0.026, scoop_y, source_z + 0.030, math.radians(40)),
                (0.0025, 0.0, 0.001, math.radians(-12)),
            ),
            (6.35, (lift_x, scoop_y, scoop_z, 0.0), (0.001, 0.0, 0.0, 0.0)),
            (7.75, (*lift_position, 0.0), (0.0, 0.0, 0.025, 0.0)),
        ]
        # Continue rising over the bowl so loose rim grains fall back before
        # the loaded spoon crosses its edge. Keep one continuous carry arc.
        self._transfer_knots = [
            (0.0, self._scoop_knots[-1][1], self._scoop_knots[-1][2]),
            (
                0.6,
                (lift_position[0] + 0.006, scoop_y + 0.004, source_z + 0.089, 0.0),
                (0.015, 0.015, 0.020, 0.0),
            ),
            (
                1.8,
                (
                    lift_position[0] + 0.42 * (cup_x - lift_position[0]),
                    scoop_y + 0.42 * (pour_y - scoop_y),
                    transfer_z + 0.008,
                    0.0,
                ),
                (0.035, 0.080, 0.0, 0.0),
            ),
            (3.4, (cup_x, pour_y, pour_z, 0.0), (0.0, 0.0, 0.0, 0.0)),
        ]
        ready_distance = torch.linalg.vector_norm(initial[:, :3] - approach[:, :3], dim=-1)
        ready_angle = math_utils.quat_error_magnitude(initial[:, 3:], approach[:, 3:])
        already_ready = bool(((ready_distance <= 0.007) & (ready_angle <= math.radians(6.0))).all())
        self._phases = [("settle", 0.45, initial, 0.010)]
        if not already_ready:
            self._phases += [
                ("clearance", 1.4, clearance, 0.010),
                ("approach", 2.5, approach, 0.007),
            ]
        self._phases += [
            (
                "scoop_1",
                self._scoop_knots[-1][0],
                pose(lift_position),
                0.007,
            ),
            ("transfer_1", self._transfer_knots[-1][0], pose((cup_x, pour_y, pour_z)), 0.006),
            ("tip_1", 1.4, pose((cup_x, pour_y, pour_z), tipped), 0.006),
            ("empty_1", 0.45, pose((cup_x, pour_y, pour_z), tipped), 0.006),
        ]
        if self._env.cfg.water_enabled:
            # An inclined spoon keeps the original flat handle above the cup
            # rim while its front is immersed. Lift before leveling again.
            pitch = math.radians(65.0)
            # Face along +Y to keep the arm away from its wrist limit.
            yaw = math.pi / 2.0
            stirring = (
                -math.sin(yaw / 2.0) * math.sin(pitch / 2.0),
                math.cos(yaw / 2.0) * math.sin(pitch / 2.0),
                math.sin(yaw / 2.0) * math.cos(pitch / 2.0),
                math.cos(yaw / 2.0) * math.cos(pitch / 2.0),
            )
            stir_clearance_z = pour_z
            self._stir_center = (cup_x, cup_y, cup_z + 0.031)
            self._stir_radius = 0.009
            self._stir_turns = 2
            start = (cup_x + self._stir_radius, cup_y, self._stir_center[2])
            self._phases += [
                ("stir_align", 2.4, pose((start[0], start[1], stir_clearance_z), stirring), 0.006),
                ("stir_lower", 1.25, pose(start, stirring), 0.004),
                ("stir", 5.0, pose(start, stirring), 0.004),
                ("stir_raise", 1.25, pose((start[0], start[1], stir_clearance_z), stirring), 0.006),
            ]
        self._phases.append(("retreat", 1.5, pose((cup_x + 0.015, pour_y, transfer_z + 0.04)), 0.010))
        self._phase_index = 0
        self._elapsed = 0.0
        self._start_pose = initial.clone()
        self._command_pose = initial.clone()

    def compute(self) -> torch.Tensor:
        """Return absolute robot-base palm pose and hand positions, shape ``(N,29)``.

        Raises:
            RuntimeError: The real spoon fails to reach a phase endpoint within
                six additional seconds or its state becomes invalid.
        """
        actual = self._env.spoon_pose()
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"Coffee demo has an invalid spoon pose in phase {self.phase!r}.")
        if not self.finished:
            name, duration, endpoint, tolerance = self._phases[self._phase_index]
            self._elapsed += self._env.step_dt
            progress = min(self._elapsed / duration, 1.0)
            blend = progress * progress * progress * (10.0 + progress * (-15.0 + 6.0 * progress))
            if name in ("scoop_1", "transfer_1"):
                knots = self._scoop_knots if name == "scoop_1" else self._transfer_knots
                self._command_pose[:] = self._curve_pose(knots, self._elapsed)
            elif name == "stir":
                self._command_pose[:] = endpoint
                angle = 2.0 * math.pi * self._stir_turns * blend
                self._command_pose[:, 0] = self._stir_center[0] + self._stir_radius * math.cos(angle)
                self._command_pose[:, 1] = self._stir_center[1] + self._stir_radius * math.sin(angle)
            else:
                self._command_pose[:, :3] = torch.lerp(self._start_pose[:, :3], endpoint[:, :3], blend)
                self._command_pose[:, 3:] = self._slerp(self._start_pose[:, 3:], endpoint[:, 3:], blend)
            if progress >= 1.0:
                distance = torch.linalg.vector_norm(actual[:, :3] - endpoint[:, :3], dim=-1)
                angle = math_utils.quat_error_magnitude(actual[:, 3:], endpoint[:, 3:])
                arrived = (distance <= tolerance) & (angle <= math.radians(6.0))
                if bool(arrived.all()):
                    self._phase_index += 1
                    self._elapsed = 0.0
                    self._start_pose = actual.clone()
                elif self._elapsed > duration + 6.0:
                    raise RuntimeError(
                        f"Coffee demo phase {name!r} timed out: spoon position error "
                        f"{float(distance.max()):.4f} m, rotation error {math.degrees(float(angle.max())):.1f} deg."
                    )
        return self._palm_action(self._command_pose)

    def _curve_pose(
        self, knots: list[tuple[float, tuple[float, ...], tuple[float, ...]]], elapsed: float
    ) -> torch.Tensor:
        """Evaluate a C2 continuous position/pitch curve in the environment frame."""
        elapsed = min(elapsed, knots[-1][0])
        for start, end in zip(knots[:-1], knots[1:]):
            if elapsed <= end[0]:
                duration = end[0] - start[0]
                fraction = max(0.0, (elapsed - start[0]) / duration)
                s3, s4, s5 = fraction**3, fraction**4, fraction**5
                weights = (
                    1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5,
                    fraction - 6.0 * s3 + 8.0 * s4 - 3.0 * s5,
                    10.0 * s3 - 15.0 * s4 + 6.0 * s5,
                    -4.0 * s3 + 7.0 * s4 - 3.0 * s5,
                )
                coordinates = [
                    weights[0] * p0 + weights[1] * duration * v0 + weights[2] * p1 + weights[3] * duration * v1
                    for p0, v0, p1, v1 in zip(start[1], start[2], end[1], end[2])
                ]
                pose = (*coordinates[:3], 0.0, math.sin(coordinates[3] / 2.0), 0.0, math.cos(coordinates[3] / 2.0))
                return torch.tensor(pose, dtype=torch.float32, device=self._env.device).expand(self._env.num_envs, -1)
        raise RuntimeError("The coffee motion curve has no endpoint.")

    def _palm_action(self, spoon_pose: torch.Tensor) -> torch.Tensor:
        # T_world_palm = T_world_spoon * inverse(T_palm_spoon).
        robot = self._env.scene["robot"].data
        spoon_pos_w = spoon_pose[:, :3] + self._env.scene.env_origins
        palm_quat_w = math_utils.quat_mul(spoon_pose[:, 3:], math_utils.quat_inv(self._offset_quat))
        palm_pos_w = spoon_pos_w - math_utils.quat_apply(palm_quat_w, self._offset_pos)
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            robot.root_pos_w.torch, robot.root_quat_w.torch, palm_pos_w, palm_quat_w
        )
        return torch.cat((pos_b, quat_b, self._hand), dim=-1)

    @staticmethod
    def _slerp(start: torch.Tensor, end: torch.Tensor, fraction: float) -> torch.Tensor:
        """Interpolate a batch of XYZW quaternions along the shortest rotation."""
        dot = torch.sum(start * end, dim=-1, keepdim=True)
        end = torch.where(dot < 0.0, -end, end)
        cosine = dot.abs().clamp(max=1.0)
        angle = torch.acos(cosine)
        denominator = torch.sin(angle).clamp_min(1.0e-6)
        spherical = (torch.sin((1.0 - fraction) * angle) * start + torch.sin(fraction * angle) * end) / denominator
        linear = torch.lerp(start, end, fraction)
        result = torch.where(cosine > 0.9995, linear, spherical)
        return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1.0e-6)
