# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured lid threading and inverted cup docking; no simulation state writes."""

from __future__ import annotations

import copy
import math
from enum import IntEnum

import torch


class AssemblyStage(IntEnum):
    """Ordered physical milestones within the lid and docking phases."""

    LID_PICKUP = 0
    LID_ALIGN = 1
    LID_THREAD = 2
    LID_RELEASE = 3
    CUP_PICKUP = 4
    CUP_INVERT = 5
    CUP_DOCK = 6
    FINAL_RELEASE = 7
    COMPLETE = 8


CRITERIA = {
    "lid_seated_axial_m": 0.220,
    "lid_seated_tolerance_m": 0.0025,
    "lid_radial_tolerance_m": 0.004,
    "lid_axis_cosine": 0.985,
    "lid_near_axial_min_m": 0.217,
    "lid_near_axial_max_m": 0.232,
    "lid_clockwise_turns": 1.0,
    "lid_engaged_descent_m": 0.003,
    "maximum_turn_increment_rad": 0.2,
    "lid_lift_height_m": 0.04,
    "cup_lift_height_m": 0.08,
    "grasp_distance_m": 0.015,
    "grasp_relative_speed_m_s": 0.08,
    "grasp_relative_angular_speed_rad_s": 0.5,
    "contact_deflection_m": 0.0005,
    "stable_linear_speed_m_s": 0.01,
    "stable_angular_speed_rad_s": 0.1,
    "finger_open_min_m": 0.0395,
    "hand_separation_m": 0.09,
    "dock_position_tolerance_m": 0.006,
    "dock_yaw_tolerance_rad": 0.06,
    "dock_key_surface_clearance_m": 0.0003,
    "lid_lift_hold_s": 0.25,
    "lid_align_hold_s": 0.10,
    "lid_seat_hold_s": 0.25,
    "lid_release_hold_s": 0.50,
    "cup_lift_hold_s": 0.25,
    "cup_inverted_hold_s": 0.25,
    "dock_support_hold_s": 0.50,
    "final_release_hold_s": 1.0,
    "minimum_fruit_count": 16,
    "minimum_fill_fraction": 1.0,
}
OBSERVATION_SCALES = (
    ("assembly_stage", 8.0),
    ("lid_clockwise_turns", 1.0),
    ("lid_engaged_descent_m", 0.012),
    ("lid_lift_hold_s", 0.25),
    ("lid_align_hold_s", 0.10),
    ("lid_seat_hold_s", 0.25),
    ("lid_release_hold_s", 0.50),
    ("cup_lift_hold_s", 0.25),
    ("cup_inverted_hold_s", 0.25),
    ("dock_support_hold_s", 0.50),
    ("final_release_hold_s", 1.0),
    ("lid_twist_complete", 1.0),
    ("lid_release_complete", 1.0),
    ("cup_lift_complete", 1.0),
    ("lid_retention_failed", 1.0),
    ("assembly_complete", 1.0),
)


def assembly_contract() -> dict:
    """Return independent geometry, chronology and observation contracts [SI units]."""
    return {
        "schema": "measured_threaded_lid_and_inverted_dock_v1",
        "criteria": copy.deepcopy(CRITERIA),
        "stages": {stage.name: int(stage) for stage in AssemblyStage},
        "observation_names": [name for name, _ in OBSERVATION_SCALES],
        "observation_scales": [scale for _, scale in OBSERVATION_SCALES],
        "observation_size": len(OBSERVATION_SCALES),
        "cup_home_m": [0.57, -0.07, 0.013],
        "lid_home_m": [0.29, -0.075, 0.078],
        "lid_grasp_local_m": [0.0, 0.0, 0.018],
        "cup_grasp_local_m": [0.0, -0.061, 0.150],
        "cup_grasp_feature": {
            "prim_path": "/Asset/Cup/Handle/Bridge1",
            "bounds_cup_m": [[-0.011, -0.075, 0.143], [0.011, -0.047, 0.157]],
            "anchor": "Authored upper bridge center, independent of the commanded TCP offset.",
        },
        "dock_lid_root_motor_local_m": [0.0, 0.0, 0.145],
        "lid_tab_half_widths_m": [0.0135, 0.0175],
        "socket_key_half_gaps_m": [0.015, 0.019],
        "seated_lid_floor_underside_cup_local_m": 0.210,
        "cup_receipt_mouth_cup_local_m": 0.210,
        "thread_semantics": (
            "Signed clockwise credit only between consecutive aligned, near, held samples; "
            "counterclockwise motion subtracts credit. Regrasp gives no unheld rotation credit. "
            "Require a measured engaged axial descent and stable seating."
        ),
        "support_semantics": (
            "Supported placement inferred from authored support geometry and measured velocities; "
            "this gate does not claim independent contact-force sensing."
        ),
        "retention_semantics": (
            "Lid seating must remain true after screw-on completion through transport and final release. "
            "Any observed loss is latched as failure. "
            "Current whole-fruit receipt and scalar tap fill gate final success."
        ),
        "state_write_policy": "Only milestone tensors change; never object, robot, particle or joint state.",
    }


def _rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz, w = q[..., :3], q[..., 3:]
    cross = torch.linalg.cross(xyz, v)
    return v + 2 * (w * cross + torch.linalg.cross(xyz, cross))


def _conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def _multiply(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            q[..., 3:] * r[..., :3] + r[..., 3:] * q[..., :3] + torch.linalg.cross(q[..., :3], r[..., :3]),
            q[..., 3:] * r[..., 3:] - (q[..., :3] * r[..., :3]).sum(-1, keepdim=True),
        ),
        -1,
    )


def assembly_geometry(
    *,
    cup_pose: torch.Tensor,
    cap_pose: torch.Tensor,
    motor_pose: torch.Tensor,
    hand_pose: torch.Tensor,
    cup_velocity: torch.Tensor,
    cap_velocity: torch.Tensor,
    hand_velocity: torch.Tensor,
    tcp_position: torch.Tensor,
    finger_positions: torch.Tensor,
    commanded_positions: torch.Tensor,
    contact_deflection: torch.Tensor,
    bilateral_contact: torch.Tensor,
    origins: torch.Tensor,
    fruit_count: torch.Tensor,
    all_types: torch.Tensor,
    fill_fraction: torch.Tensor,
    failed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure batched physical geometry without changing any simulator state.

    Poses use world position [m] and XYZW quaternion, shape [N, 7]. Velocities
    contain world linear [m/s] and angular [rad/s] values, shape [N, 6]. TCP and
    origins are [m], shape [N, 3]. Finger positions, commands and deflections
    are per-finger [m], shape [N, 2]. Remaining inputs have shape [N].
    """
    n = cup_pose.shape[0]
    poses = (cup_pose, cap_pose, motor_pose, hand_pose)
    velocities = (cup_velocity, cap_velocity, hand_velocity)
    groups = ((poses, 7), (velocities, 6), ((tcp_position, origins), 3))
    groups += (((finger_positions, commanded_positions, contact_deflection), 2),)
    for values, width in groups:
        if any(value.shape != (n, width) or value.device != cup_pose.device for value in values):
            raise ValueError(f"Expected same-device physical tensors with shape [N, {width}].")
    scalars = (bilateral_contact, fruit_count, all_types, fill_fraction, failed)
    if any(value.shape != (n,) or value.device != cup_pose.device for value in scalars):
        raise ValueError("Expected same-device scalar measurements with shape [N].")
    if any(value.dtype != torch.bool for value in (bilateral_contact, all_types, failed)):
        raise ValueError("Contact, fruit-type and failure flags must be Boolean tensors.")
    finite = torch.ones(n, dtype=torch.bool, device=cup_pose.device)
    for values, _ in groups:
        for value in values:
            finite &= torch.isfinite(value).all(-1)
    for pose in poses:
        finite &= (pose[:, 3:].norm(dim=-1) - 1.0).abs() <= 0.001
    finite &= torch.isfinite(fruit_count) & torch.isfinite(fill_fraction)
    finite &= (fruit_count >= 0) & (fruit_count <= 16) & (fill_fraction >= 0) & (fill_fraction <= 1)
    finite &= ((finger_positions >= -0.001) & (finger_positions <= 0.041)).all(-1)
    finite &= ((commanded_positions >= 0) & (commanded_positions <= 0.04)).all(-1)
    valid = finite & ~failed
    up = cup_pose.new_tensor((0.0, 0.0, 1.0)).expand(n, -1)
    relative_position = _rotate(_conjugate(cup_pose[:, 3:]), cap_pose[:, :3] - cup_pose[:, :3])
    relative_rotation = _multiply(_conjugate(cup_pose[:, 3:]), cap_pose[:, 3:])
    relative_up = _rotate(relative_rotation, up)[:, 2]
    q = relative_rotation
    yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1].square() + q[:, 2].square()))
    aligned = (relative_position[:, :2].norm(dim=-1) <= CRITERIA["lid_radial_tolerance_m"]) & (
        relative_up > CRITERIA["lid_axis_cosine"]
    )
    axial = relative_position[:, 2]
    seated = aligned & ((axial - CRITERIA["lid_seated_axial_m"]).abs() <= CRITERIA["lid_seated_tolerance_m"])
    near = aligned & (axial > CRITERIA["lid_near_axial_min_m"]) & (axial < CRITERIA["lid_near_axial_max_m"])
    finger_width = finger_positions.sum(-1)
    closing_contact = (
        bilateral_contact
        & (contact_deflection >= CRITERIA["contact_deflection_m"]).all(-1)
        & ((finger_positions - commanded_positions) >= CRITERIA["contact_deflection_m"]).all(-1)
        & (finger_width > 0.001)
        & (finger_width < 0.065)
    )
    result = {
        "finite": finite,
        "valid": valid,
        "failed": ~valid,
        "lid_near_thread": near & valid,
        "lid_seated": seated & valid,
    }
    for name, pose, velocity, offset in (
        ("lid", cap_pose, cap_velocity, (0.0, 0.0, 0.018)),
        ("cup", cup_pose, cup_velocity, (0.0, -0.061, 0.150)),
    ):
        grasp = pose[:, :3] + _rotate(pose[:, 3:], pose.new_tensor(offset).expand(n, -1))
        distance = (tcp_position - grasp).norm(dim=-1)
        source_speed = velocity[:, :3] + torch.linalg.cross(velocity[:, 3:], grasp - pose[:, :3])
        hand_speed = hand_velocity[:, :3] + torch.linalg.cross(hand_velocity[:, 3:], grasp - hand_pose[:, :3])
        relative_speed = (source_speed - hand_speed).norm(dim=-1)
        relative_angular_speed = (velocity[:, 3:] - hand_velocity[:, 3:]).norm(dim=-1)
        held = (
            closing_contact
            & (distance <= CRITERIA["grasp_distance_m"])
            & (relative_speed <= CRITERIA["grasp_relative_speed_m_s"])
            & (relative_angular_speed <= CRITERIA["grasp_relative_angular_speed_rad_s"])
            & valid
        )
        result.update(
            {
                f"{name}_held": held,
                f"{name}_grasp_position_w_m": grasp,
                f"{name}_hand_separation_m": distance,
                f"{name}_grasp_relative_speed_m_s": relative_speed,
                f"{name}_grasp_relative_angular_speed_rad_s": relative_angular_speed,
                f"{name}_grip_position_b_m": _rotate(_conjugate(pose[:, 3:]), tcp_position - pose[:, :3]),
                f"{name}_hand_rotation_b_xyzw": _multiply(_conjugate(pose[:, 3:]), hand_pose[:, 3:]),
                f"{name}_stable": (velocity[:, :3].norm(dim=-1) < CRITERIA["stable_linear_speed_m_s"])
                & (velocity[:, 3:].norm(dim=-1) < CRITERIA["stable_angular_speed_rad_s"])
                & valid,
            }
        )
    cup_up, cap_up = _rotate(cup_pose[:, 3:], up)[:, 2], _rotate(cap_pose[:, 3:], up)[:, 2]
    motor_local = _rotate(_conjugate(motor_pose[:, 3:]), cap_pose[:, :3] - motor_pose[:, :3])
    dock_error = (motor_local - cup_pose.new_tensor((0.0, 0.0, 0.145))).norm(dim=-1)
    motor_relative = _multiply(_conjugate(motor_pose[:, 3:]), cap_pose[:, 3:])
    tab_x = _rotate(motor_relative, cup_pose.new_tensor((1.0, 0.0, 0.0)).expand(n, -1))
    tab_yaw = torch.atan2(tab_x[:, 1], tab_x[:, 0])
    yaw_error = 0.5 * torch.atan2(torch.sin(2 * tab_yaw), torch.cos(2 * tab_yaw)).abs()
    corners = cup_pose.new_tensor(
        [[x, y, z] for x in (-0.0135, 0.0135) for y in (-0.0175, 0.0175) for z in (0.0, 0.036)]
    )
    motor_corners = motor_local[:, None] + _rotate(
        motor_relative[:, None].expand(-1, 8, -1), corners[None].expand(n, -1, -1)
    )
    key_extent = motor_corners[:, :, :2].abs().amax(1)
    key_fit = (key_extent + CRITERIA["dock_key_surface_clearance_m"] <= cup_pose.new_tensor((0.015, 0.019))).all(-1)
    dock = (
        (dock_error <= CRITERIA["dock_position_tolerance_m"])
        & (cup_up < -CRITERIA["lid_axis_cosine"])
        & seated
        & key_fit
        & (yaw_error <= CRITERIA["dock_yaw_tolerance_rad"])
        & result["cup_stable"]
        & result["lid_stable"]
        & valid
    )
    opened = (finger_positions >= CRITERIA["finger_open_min_m"]).all(-1) & valid
    cup_home = cup_pose[:, :3] - origins
    cup_supported = (
        ((cup_home[:, :2] - cup_pose.new_tensor((0.57, -0.07))).norm(dim=-1) <= 0.006)
        & (cup_home[:, 2] >= 0.011)
        & (cup_home[:, 2] <= 0.015)
        & (cup_up > CRITERIA["lid_axis_cosine"])
        & result["cup_stable"]
    )
    result.update(
        lid_relative_yaw_rad=yaw,
        lid_axial_m=axial,
        lid_radial_error_m=relative_position[:, :2].norm(dim=-1),
        lid_axis_cosine=relative_up,
        lid_upright=cap_up,
        cup_up=cup_up,
        lid_lift_height_m=cap_pose[:, 2] - origins[:, 2] - 0.078,
        cup_lift_height_m=cup_pose[:, 2] - origins[:, 2] - 0.013,
        cup_home_supported=cup_supported,
        fingers_open=opened,
        dock_position_error_m=dock_error,
        dock_yaw_error_rad=yaw_error,
        dock_key_fit=key_fit & valid,
        dock_key_projected_half_extent_m=key_extent,
        dock_support_ready=dock,
        dock_released=dock
        & opened
        & (result["cup_hand_separation_m"] >= CRITERIA["hand_separation_m"])
        & ~result["cup_held"],
        ingredients_retained=(fruit_count == CRITERIA["minimum_fruit_count"])
        & all_types
        & (fill_fraction >= CRITERIA["minimum_fill_fraction"])
        & valid,
    )
    return result


class AssemblyGeometryState:
    """Ordered measured state, with no object attachment or pose mutation."""

    def __init__(self, num_envs: int, device: str | torch.device, step_dt: float):
        if type(num_envs) is not int or num_envs < 1 or not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Require positive world count and timestep [s].")
        self.step_dt = step_dt
        self.state: dict[str, torch.Tensor] = {}
        floats = [name for name, _ in OBSERVATION_SCALES if name.endswith("_s")]
        floats += ["lid_clockwise_turns", "lid_engaged_descent_m", "previous_yaw_rad", "maximum_engaged_axial_m"]
        for name in floats:
            self.state[name] = torch.zeros(num_envs, device=device)
        for name in (
            "lid_twist_complete",
            "lid_release_complete",
            "cup_lift_complete",
            "cup_inverted",
            "dock_complete",
            "lid_retention_failed",
            "assembly_failed",
            "assembly_complete",
            "previous_turn_eligible",
            "engagement_seen",
        ):
            self.state[name] = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.state["assembly_stage"] = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.state["last_step"] = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.measurements: dict[str, torch.Tensor] = self.state.copy()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset only selected episode milestones, without modifying physics."""
        for name, value in self.state.items():
            value[env_ids] = -1 if name == "last_step" else 0
        self.measurements = self.state.copy()

    def observations(self) -> torch.Tensor:
        """Return the documented 16 normalized state features, shape [N, 16]."""
        return torch.stack([self.state[name].float() / scale for name, scale in OBSERVATION_SCALES], -1)

    def update(
        self,
        measurements: dict[str, torch.Tensor],
        active: torch.Tensor,
        receipt_complete: torch.Tensor,
        *,
        step: int,
    ) -> dict[str, torch.Tensor]:
        """Advance once per observed policy boundary, in physical milestone order.

        Args:
            measurements: Output of :func:`assembly_geometry` from this boundary.
            active: Boolean worlds currently in the assembly phase, shape [N].
            receipt_complete: Boolean prior fruit/tap completion proof, shape [N].
            step: Nonnegative global policy step; first activation may occur at any step.
        """
        if type(step) is not int or step < 0:
            raise ValueError("Require a nonnegative integer global policy step.")
        s, m = self.state, measurements
        if any(
            value.shape != s["assembly_stage"].shape or value.dtype != torch.bool
            for value in (active, receipt_complete)
        ):
            raise ValueError("Require one Boolean active and receipt flag per world.")
        if bool((active & (s["last_step"] > step)).any()):
            raise ValueError("Assembly policy steps cannot move backward without reset.")
        fresh = active & (s["last_step"] != step)
        if not bool(fresh.any()):
            return self.measurements
        consecutive = fresh & (s["last_step"] == step - 1)
        gap = fresh & (s["last_step"] >= 0) & ~consecutive
        for name in s:
            if name.endswith("_hold_s"):
                s[name][gap] = 0
        s["previous_turn_eligible"][gap] = False
        s["last_step"][fresh] = step
        s["assembly_failed"] |= fresh & ~m["valid"]
        valid = fresh & receipt_complete & m["valid"] & ~s["assembly_failed"]
        stage = s["assembly_stage"].clone()
        retention_required = stage >= AssemblyStage.LID_RELEASE
        s["lid_retention_failed"] |= fresh & retention_required & ~m["lid_seated"]
        s["assembly_failed"] |= s["lid_retention_failed"]
        valid &= ~s["lid_retention_failed"]

        def hold(name: str, condition: torch.Tensor) -> torch.Tensor:
            updated = torch.where(valid & condition, s[name] + self.step_dt, 0.0)
            s[name].copy_(torch.where(fresh, updated, s[name]))
            return s[name] >= CRITERIA[name] - 1e-6

        lifted_lid = hold(
            "lid_lift_hold_s",
            (stage == AssemblyStage.LID_PICKUP)
            & m["lid_held"]
            & (m["lid_lift_height_m"] >= CRITERIA["lid_lift_height_m"])
            & (m["lid_upright"] > CRITERIA["lid_axis_cosine"]),
        )
        s["assembly_stage"][valid & lifted_lid & (stage == AssemblyStage.LID_PICKUP)] = AssemblyStage.LID_ALIGN
        aligned = hold("lid_align_hold_s", (stage == AssemblyStage.LID_ALIGN) & m["lid_near_thread"] & m["lid_held"])
        s["assembly_stage"][valid & aligned & (stage == AssemblyStage.LID_ALIGN)] = AssemblyStage.LID_THREAD
        eligible = valid & (stage == AssemblyStage.LID_THREAD) & m["lid_near_thread"] & m["lid_held"]
        delta = torch.atan2(
            torch.sin(m["lid_relative_yaw_rad"] - s["previous_yaw_rad"]),
            torch.cos(m["lid_relative_yaw_rad"] - s["previous_yaw_rad"]),
        )
        pair = eligible & s["previous_turn_eligible"] & consecutive
        plausible = delta.abs() <= CRITERIA["maximum_turn_increment_rad"]
        s["assembly_failed"] |= pair & ~plausible
        increment = torch.where(pair & plausible, -delta / (2 * math.pi), 0.0)
        s["lid_clockwise_turns"] += increment
        first_engagement = eligible & ~s["engagement_seen"]
        s["maximum_engaged_axial_m"].copy_(
            torch.where(first_engagement, m["lid_axial_m"], s["maximum_engaged_axial_m"])
        )
        s["maximum_engaged_axial_m"].copy_(
            torch.where(
                eligible, torch.maximum(s["maximum_engaged_axial_m"], m["lid_axial_m"]), s["maximum_engaged_axial_m"]
            )
        )
        descent = (s["maximum_engaged_axial_m"] - m["lid_axial_m"]).clamp_min(0)
        s["lid_engaged_descent_m"].copy_(
            torch.where(eligible, torch.maximum(s["lid_engaged_descent_m"], descent), s["lid_engaged_descent_m"])
        )
        s["engagement_seen"] |= eligible
        s["previous_turn_eligible"].copy_(torch.where(fresh, eligible, s["previous_turn_eligible"]))
        s["previous_yaw_rad"].copy_(torch.where(fresh, m["lid_relative_yaw_rad"], s["previous_yaw_rad"]))
        valid &= ~s["assembly_failed"]
        twisted = hold(
            "lid_seat_hold_s",
            (stage == AssemblyStage.LID_THREAD)
            & eligible
            & m["lid_seated"]
            & m["lid_stable"]
            & m["cup_stable"]
            & (s["lid_clockwise_turns"] >= CRITERIA["lid_clockwise_turns"] - 1e-6)
            & (s["lid_engaged_descent_m"] >= CRITERIA["lid_engaged_descent_m"]),
        )
        s["lid_twist_complete"] |= valid & twisted
        s["assembly_stage"][valid & twisted & (stage == AssemblyStage.LID_THREAD)] = AssemblyStage.LID_RELEASE
        released_lid = hold(
            "lid_release_hold_s",
            (stage == AssemblyStage.LID_RELEASE)
            & s["lid_twist_complete"]
            & m["cup_home_supported"]
            & m["lid_stable"]
            & m["lid_seated"]
            & m["fingers_open"]
            & (m["lid_hand_separation_m"] >= CRITERIA["hand_separation_m"])
            & ~m["lid_held"],
        )
        s["lid_release_complete"] |= valid & released_lid
        s["assembly_stage"][valid & released_lid & (stage == AssemblyStage.LID_RELEASE)] = AssemblyStage.CUP_PICKUP
        lifted_cup = hold(
            "cup_lift_hold_s",
            (stage == AssemblyStage.CUP_PICKUP)
            & s["lid_release_complete"]
            & m["cup_held"]
            & m["lid_seated"]
            & (m["cup_up"] > CRITERIA["lid_axis_cosine"])
            & (m["cup_lift_height_m"] >= CRITERIA["cup_lift_height_m"]),
        )
        s["cup_lift_complete"] |= valid & lifted_cup
        s["assembly_stage"][valid & lifted_cup & (stage == AssemblyStage.CUP_PICKUP)] = AssemblyStage.CUP_INVERT
        inverted = hold(
            "cup_inverted_hold_s",
            (stage == AssemblyStage.CUP_INVERT)
            & s["cup_lift_complete"]
            & m["cup_held"]
            & m["lid_seated"]
            & (m["cup_up"] < -CRITERIA["lid_axis_cosine"]),
        )
        s["cup_inverted"] |= valid & inverted
        s["assembly_stage"][valid & inverted & (stage == AssemblyStage.CUP_INVERT)] = AssemblyStage.CUP_DOCK
        supported = hold(
            "dock_support_hold_s",
            (stage == AssemblyStage.CUP_DOCK) & s["cup_inverted"] & m["cup_held"] & m["dock_support_ready"],
        )
        s["dock_complete"] |= valid & supported
        s["assembly_stage"][valid & supported & (stage == AssemblyStage.CUP_DOCK)] = AssemblyStage.FINAL_RELEASE
        finished = hold(
            "final_release_hold_s",
            (stage >= AssemblyStage.FINAL_RELEASE)
            & s["dock_complete"]
            & m["dock_released"]
            & m["ingredients_retained"]
            & s["lid_twist_complete"],
        )
        s["assembly_complete"].copy_(torch.where(fresh, finished & valid, s["assembly_complete"]))
        s["assembly_stage"][valid & finished] = AssemblyStage.COMPLETE
        self.measurements = {**m, **s}
        return self.measurements
