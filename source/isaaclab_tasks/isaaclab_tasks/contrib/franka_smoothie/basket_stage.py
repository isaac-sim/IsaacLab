# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Measured fruit delivery and basket return gates."""

import math

import torch

from isaaclab.utils import math as math_utils

from .basket_geometry import BASKET_POSITION

CRITERIA = {
    "receipt_metric": "whole_authored_collision_hulls_in_actual_cup_zero_tolerance",
    "geometry_tolerance_m": 0.0,
    "fruit_count": 16,
    "minimum_delivered_fruit": 16,
    "all_four_fruit_types_required": True,
    "legacy_center_region_for_comparison_only": {"radial_lt_m": 0.035, "z_gt_m": 0.01, "z_lt_m": 0.185},
    "held_upright_lift_m": 0.08,
    "held_lift_upright_cosine": 0.9,
    "held_lift_hold_s": 1.0,
    "delivery_hold_s": 1.0,
    "held_return_handoff_s": 0.25,
    "home_position_m": list(BASKET_POSITION),
    "home_xy_tolerance_m": 0.02,
    "upright_cosine": math.cos(math.radians(5.0)),
    "floor_radius_m": 0.039,
    "floor_height_m": 0.003,
    "table_height_m": 0.0,
    "controlled_descent_floor_max_m": 0.015,
    "supported_floor_min_m": -0.001,
    "supported_floor_max_m": 0.003,
    "maximum_linear_speed_m_s": 0.01,
    "maximum_angular_speed_rad_s": 0.1,
    "support_hold_before_open_s": 0.5,
    "minimum_open_fraction_each_finger": 0.85,
    "minimum_open_command_fraction": 0.9,
    "released_tcp_to_grasp_m": 0.10,
    "released_stable_hold_s": 2.0,
    "support_semantics": (
        "Basket floor at the authored table plane, upright and stationary; no direct contact-force assertion."
    ),
    "failure": "Unchanged maintained mdp.failure; nonfinite or failed samples never advance milestones.",
}

FLOAT_STATES = ("lift_hold_s", "delivery_hold_s", "handoff_hold_s", "support_hold_s", "released_hold_s")

BOOL_STATES = ("lift_complete", "delivery_complete", "controlled_descent", "support_complete", "finished")


def new_milestones(num_envs: int, device: str) -> dict[str, torch.Tensor]:
    """Create resettable per-world holds [s] and ordered milestone flags."""
    return {
        **{name: torch.zeros(num_envs, device=device) for name in FLOAT_STATES},
        **{name: torch.zeros(num_envs, device=device, dtype=torch.bool) for name in BOOL_STATES},
    }


def reset_milestones(state: dict[str, torch.Tensor], env_ids: torch.Tensor) -> None:
    """Clear only the resetting worlds' task history."""
    for value in state.values():
        value[env_ids] = 0


def placement_geometry(pose, velocity, tcp, fingers, commanded, origin, open_position, grasp):
    """Measure support, opening and hand separation from raw SI-unit state tensors."""
    up = math_utils.quat_apply(pose[:, 3:], pose.new_tensor([0.0, 0.0, 1.0]).expand(len(pose), -1))[:, 2]
    local_position = pose[:, :3] - origin
    # Exact lower support height of the authored floor cylinder [m].
    low = (
        local_position[:, 2]
        + 0.5 * CRITERIA["floor_height_m"] * (up - up.abs())
        - CRITERIA["floor_radius_m"] * (1 - up.square()).clamp_min(0).sqrt()
    )
    home_xy = (local_position[:, :2] - pose.new_tensor(BASKET_POSITION[:2])).norm(dim=-1)
    linear, angular = velocity[:, :3].norm(dim=-1), velocity[:, 3:].norm(dim=-1)
    separation = (tcp - grasp).norm(dim=-1)
    near_home = (home_xy <= CRITERIA["home_xy_tolerance_m"]) & (up >= CRITERIA["upright_cosine"])
    supported = (
        near_home
        & (low >= CRITERIA["supported_floor_min_m"])
        & (low <= CRITERIA["supported_floor_max_m"])
        & (linear <= CRITERIA["maximum_linear_speed_m_s"])
        & (angular <= CRITERIA["maximum_angular_speed_rad_s"])
    )
    opened = (fingers >= open_position * CRITERIA["minimum_open_fraction_each_finger"]).all(-1) & (
        commanded >= open_position * CRITERIA["minimum_open_command_fraction"]
    ).all(-1)
    finite = torch.isfinite(torch.cat((pose, velocity, tcp, fingers, commanded, grasp), -1)).all(-1)
    return {
        "floor_min_z_m": low,
        "home_xy_error_m": home_xy,
        "basket_upright": up,
        "basket_linear_speed_m_s": linear,
        "basket_angular_speed_rad_s": angular,
        "tcp_to_grasp_m": separation,
        "near_home": near_home & finite,
        "supported": supported & finite,
        "open": opened & finite,
        "separated": (separation > CRITERIA["released_tcp_to_grasp_m"]) & finite,
        "finite": finite,
    }


def advance_milestones(state: dict[str, torch.Tensor], measurements: dict[str, torch.Tensor], step_dt: float) -> None:
    """Advance valid ordered basket milestones with a policy timestep [s]."""
    m = measurements
    valid = ~m["failed"] & m["finite"]

    def hold(name: str, condition: torch.Tensor) -> None:
        state[name].copy_(torch.where(condition & valid, state[name] + step_dt, 0.0))

    hold(
        "lift_hold_s",
        m["held"]
        & (m["clearance_m"] >= CRITERIA["held_upright_lift_m"])
        & (m["basket_upright"] >= CRITERIA["held_lift_upright_cosine"]),
    )
    state["lift_complete"] |= state["lift_hold_s"] >= CRITERIA["held_lift_hold_s"] - 1e-6
    delivered = (m["fruit_count"] >= CRITERIA["minimum_delivered_fruit"]) & m["all_types"]
    hold("delivery_hold_s", state["lift_complete"] & delivered)
    state["delivery_complete"] |= state["delivery_hold_s"] >= CRITERIA["delivery_hold_s"] - 1e-6
    hold("handoff_hold_s", state["delivery_complete"] & delivered & m["held"])
    state["controlled_descent"] |= (
        valid
        & state["delivery_complete"]
        & m["held"]
        & m["near_home"]
        & (m["floor_min_z_m"] >= CRITERIA["supported_floor_min_m"])
        & (m["floor_min_z_m"] <= CRITERIA["controlled_descent_floor_max_m"])
    )
    hold("support_hold_s", state["controlled_descent"] & state["delivery_complete"] & m["supported"])
    state["support_complete"] |= state["support_hold_s"] >= CRITERIA["support_hold_before_open_s"] - 1e-6
    release = state["delivery_complete"] & state["support_complete"] & m["supported"] & m["open"] & m["separated"]
    hold("released_hold_s", release)
    state["finished"] |= valid & (state["released_hold_s"] >= CRITERIA["released_stable_hold_s"] - 1e-6)
