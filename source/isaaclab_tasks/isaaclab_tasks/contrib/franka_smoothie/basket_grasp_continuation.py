# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Acquire basket grasp evidence from coupled motion, then track measured continuation."""

from __future__ import annotations

import math
from collections import deque

import torch

OPTIONS = {
    "version": 3,
    "initialization": "One second of closed bilateral deflection, raised upright basket and coupled measured motion.",
    "minimum_each_deflection_m": 0.0009,
    "minimum_sustained_each_deflection_m": 0.0008,
    "maximum_closed_command_m": 0.0001,
    "maximum_relative_speed_m_s": 0.02,
    "minimum_finger_width_m": 0.001,
    "maximum_finger_width_m": 0.065,
    "stability_window_s": 1.0,
    "maximum_relative_translation_drift_m": 0.002,
    "maximum_relative_rotation_drift_rad": math.radians(3),
    "acquisition_minimum_clearance_m": 0.08,
    "acquisition_minimum_upright_cosine": 0.9,
    "acquisition_minimum_each_path_m": 0.005,
    "acquisition_minimum_each_displacement_m": 0.005,
    "acquisition_minimum_motion_correlation": 0.95,
    "interface_tcp_radial_range_m": [0.035, 0.075],
    "interface_tcp_height_range_m": [0.060, 0.110],
    "loss_each_deflection_below_m": 0.0002,
    "loss_reference_translation_drift_m": 0.01,
    "loss_reference_rotation_drift_rad": math.radians(30),
    "maximum_frame_consistency_error_m": 0.00002,
    "window_rule": (
        "Every sample qualifies across the full1s stability window. Acquisition requires each finger>=0.9mm; "
        "only an existing acquisition credential permits sustained each-finger deflection>=0.8mm. "
        "Captured basket point and TCP motion must agree for acquisition."
    ),
    "scope": "Alternate basket azimuth allowed; nominal-point strict held and strict lift remain separate telemetry.",
    "evidence": (
        "v2 acquisition used coupled travel120mm with .401mm/1.86deg drift. Closed v4 probe retained its "
        "acquisition credential but late0.893mm deflection samples interrupted current-held qualification "
        "despite stable relative geometry. Sustained hysteresis requires a new physical validation."
    ),
    "limitation": "Deflection, material-interface proximity and coupled-motion evidence; no recorded force assertion.",
}


def rotate(rotation, vector):
    """Rotate vectors [m] with normalized XYZW quaternions."""
    uv = torch.linalg.cross(rotation[..., :3], vector)
    return vector + 2 * (rotation[..., 3:] * uv + torch.linalg.cross(rotation[..., :3], uv))


def stable_window(qualified, positions, rotations, span_s):
    """Evaluate a measured relative-pose window [m, rad, s]."""
    translation = torch.linalg.vector_norm(positions - positions[0], dim=-1).amax(0)
    dots = (rotations * rotations[0]).sum(-1).abs().clamp(0, 1)
    rotation = (2 * torch.acos(dots)).amax(0)
    accepted = (
        qualified.all(0)
        & (span_s >= OPTIONS["stability_window_s"] - 1e-6)
        & (translation <= OPTIONS["maximum_relative_translation_drift_m"])
        & (rotation <= OPTIONS["maximum_relative_rotation_drift_rad"])
    )
    return accepted, translation, rotation


class BasketGraspContinuation:
    """Track one world's grasp evidence without modifying physical state.

    Args:
        device: Tensor device.
        step_dt: Policy interval [s], required to be 1/30.
    """

    def __init__(self, device: str, step_dt: float):
        self.device, self.step_dt = device, step_dt
        self.window_steps = round(OPTIONS["stability_window_s"] / step_dt)
        self.history = deque(maxlen=self.window_steps + 1)
        self.reset()

    def reset(self) -> None:
        """Discard all acquisition credentials and sample coverage on reset."""
        self.history.clear()
        self.last_step = None
        self.current = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.acquired = torch.zeros_like(self.current)
        self.acquisition_step = torch.full((1,), -1, dtype=torch.long, device=self.device)
        self.reference_position = torch.zeros(1, 3, device=self.device)
        self.reference_rotation = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device)
        for name in (
            "window_s",
            "translation_drift_m",
            "rotation_drift_rad",
            "hand_path_m",
            "basket_path_m",
            "hand_displacement_m",
            "basket_displacement_m",
            "motion_correlation",
        ):
            setattr(self, name, torch.zeros(1, device=self.device))

    @torch.no_grad()
    def update(
        self,
        step: int,
        *,
        valid: torch.Tensor,
        command_m: torch.Tensor,
        deflection_m: torch.Tensor,
        finger_width_m: torch.Tensor,
        relative_speed_m_s: torch.Tensor,
        grip_position_b_m: torch.Tensor,
        hand_rotation_b_xyzw: torch.Tensor,
        tcp_position_w_m: torch.Tensor,
        basket_pose_w: torch.Tensor,
        clearance_m: torch.Tensor,
        upright: torch.Tensor,
    ) -> torch.Tensor:
        """Return current grasp evidence from copied measured inputs [m, rad, s].

        Acquisition requires nonzero coupled motion and strong bilateral deflection.
        Later stationary holding permits the lower sustained deflection threshold
        only with that credential and a full rolling stable window.
        Opening, invalid state, clear loss, reset or missing cadence clears it.
        """
        if type(step) is not int or step < 0:
            raise ValueError("Require a nonnegative integer global policy step.")
        if self.last_step == step:
            return self.current
        if self.last_step is not None and step != self.last_step + 1:
            self.reset()
        expected = (
            (valid, (1,)),
            (command_m, (1, 1)),
            (deflection_m, (1, 2)),
            (finger_width_m, (1,)),
            (relative_speed_m_s, (1,)),
            (grip_position_b_m, (1, 3)),
            (hand_rotation_b_xyzw, (1, 4)),
            (tcp_position_w_m, (1, 3)),
            (basket_pose_w, (1, 7)),
            (clearance_m, (1,)),
            (upright, (1,)),
        )
        if any(value.shape != shape for value, shape in expected):
            raise ValueError("Require one-world measured tensors.")
        finite = torch.stack([torch.isfinite(v).reshape(1, -1).all(-1) for v, _ in expected]).all(0)
        hand_norm = torch.linalg.vector_norm(hand_rotation_b_xyzw, dim=-1, keepdim=True)
        body_norm = torch.linalg.vector_norm(basket_pose_w[:, 3:], dim=-1, keepdim=True)
        rotation = hand_rotation_b_xyzw / hand_norm.clamp_min(1e-8)
        body_rotation = basket_pose_w[:, 3:] / body_norm.clamp_min(1e-8)
        radial = torch.linalg.vector_norm(grip_position_b_m[:, :2], dim=-1)
        radius_range, height_range = OPTIONS["interface_tcp_radial_range_m"], OPTIONS["interface_tcp_height_range_m"]
        interface = (
            (radial >= radius_range[0])
            & (radial <= radius_range[1])
            & (grip_position_b_m[:, 2] >= height_range[0])
            & (grip_position_b_m[:, 2] <= height_range[1])
        )
        closed = (command_m[:, 0] >= 0) & (command_m[:, 0] <= OPTIONS["maximum_closed_command_m"])
        frame_error = torch.linalg.vector_norm(
            rotate(body_rotation, grip_position_b_m) + basket_pose_w[:, :3] - tcp_position_w_m, dim=-1
        )
        measured_valid = (
            valid
            & finite
            & (hand_norm[:, 0] > 1e-8)
            & (body_norm[:, 0] > 1e-8)
            & (frame_error <= OPTIONS["maximum_frame_consistency_error_m"])
        )
        reference_distance = torch.linalg.vector_norm(grip_position_b_m - self.reference_position, dim=-1)
        reference_angle = 2 * torch.acos((rotation * self.reference_rotation).sum(-1).abs().clamp(0, 1))
        lost = (
            ~measured_valid
            | ~closed
            | ~interface
            | (deflection_m.amin(-1) < OPTIONS["loss_each_deflection_below_m"])
            | (
                self.acquired
                & (
                    (reference_distance > OPTIONS["loss_reference_translation_drift_m"])
                    | (reference_angle > OPTIONS["loss_reference_rotation_drift_rad"])
                )
            )
        )
        if bool(lost.any()):
            self.reset()
        minimum_deflection = torch.where(
            self.acquired,
            OPTIONS["minimum_sustained_each_deflection_m"],
            OPTIONS["minimum_each_deflection_m"],
        )
        qualified = (
            measured_valid
            & closed
            & interface
            & (deflection_m.amin(-1) >= minimum_deflection)
            & (finger_width_m > OPTIONS["minimum_finger_width_m"])
            & (finger_width_m < OPTIONS["maximum_finger_width_m"])
            & (relative_speed_m_s >= 0)
            & (relative_speed_m_s < OPTIONS["maximum_relative_speed_m_s"])
        )
        raised = (clearance_m >= OPTIONS["acquisition_minimum_clearance_m"]) & (
            upright >= OPTIONS["acquisition_minimum_upright_cosine"]
        )
        self.history.append(
            (
                step,
                qualified.clone(),
                torch.nan_to_num(grip_position_b_m).clone(),
                torch.nan_to_num(rotation).clone(),
                torch.nan_to_num(tcp_position_w_m).clone(),
                torch.nan_to_num(basket_pose_w[:, :3]).clone(),
                torch.nan_to_num(body_rotation).clone(),
                raised.clone(),
            )
        )
        self.last_step = step
        self.window_s.fill_((step - self.history[0][0]) * self.step_dt)
        positions = torch.stack([row[2] for row in self.history])
        rotations = torch.stack([row[3] for row in self.history])
        stable, self.translation_drift_m, self.rotation_drift_rad = stable_window(
            torch.stack([row[1] for row in self.history]), positions, rotations, self.window_s
        )
        hand_path = torch.stack([row[4] for row in self.history])
        body_positions = torch.stack([row[5] for row in self.history])
        body_rotations = torch.stack([row[6] for row in self.history])
        object_path = rotate(body_rotations, positions[0].expand_as(body_positions)) + body_positions
        dh, db = torch.diff(hand_path, dim=0), torch.diff(object_path, dim=0)
        self.hand_path_m = torch.linalg.vector_norm(dh, dim=-1).sum(0)
        self.basket_path_m = torch.linalg.vector_norm(db, dim=-1).sum(0)
        self.hand_displacement_m = torch.linalg.vector_norm(hand_path[-1] - hand_path[0], dim=-1)
        self.basket_displacement_m = torch.linalg.vector_norm(object_path[-1] - object_path[0], dim=-1)
        denominator = torch.sqrt((dh.square().sum((0, 2))) * (db.square().sum((0, 2))))
        self.motion_correlation = (dh * db).sum((0, 2)) / denominator.clamp_min(1e-16)
        acquisition = (
            stable
            & torch.stack([row[7] for row in self.history]).all(0)
            & (self.hand_path_m >= OPTIONS["acquisition_minimum_each_path_m"])
            & (self.basket_path_m >= OPTIONS["acquisition_minimum_each_path_m"])
            & (self.hand_displacement_m >= OPTIONS["acquisition_minimum_each_displacement_m"])
            & (self.basket_displacement_m >= OPTIONS["acquisition_minimum_each_displacement_m"])
            & (self.motion_correlation >= OPTIONS["acquisition_minimum_motion_correlation"])
        )
        newly_acquired = acquisition & ~self.acquired
        self.reference_position = torch.where(newly_acquired[:, None], grip_position_b_m, self.reference_position)
        self.reference_rotation = torch.where(newly_acquired[:, None], rotation, self.reference_rotation)
        self.acquisition_step = torch.where(newly_acquired, step, self.acquisition_step)
        self.acquired |= acquisition
        self.current = self.acquired & stable & ~lost
        return self.current
