#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA, collision, reset-clearance, and throughput preflight for YAM cable routing."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import warp as wp

# Warp captures this setting when kernel modules are imported.
wp.config.enable_backward = False

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_newton.physics.newton_manager import NewtonManager  # noqa: E402

from isaaclab.utils.string import string_to_callable  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.contrib.cable_routing.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    CableRoutingPPORunnerCfg,
)
from isaaclab_tasks.contrib.cable_routing.cable_routing_env_cfg import (  # noqa: E402
    CABLE_RADIUS,
    CABLE_SEGMENT_LENGTH,
    PEG_BASE_POSITIONS_B,
    PEG_RADIUS,
)
from isaaclab_tasks.contrib.cable_routing.mdp.events import (  # noqa: E402
    BENCHMARK_GRID_DIRECTIONS,
    cable_capsule_clearance_mask,
    cable_capsule_self_clearance_mask,
)
from isaaclab_tasks.contrib.cable_routing.mdp.reset_curve_xpbd import (  # noqa: E402
    CableResetCurveXPBDCfg,
    relax_open_cable_curve_xpbd,
)
from isaaclab_tasks.contrib.cable_routing.mdp.reset_replay import (  # noqa: E402
    active_step_progress_from_route_progress,
)
from isaaclab_tasks.contrib.cable_routing.mdp.route_metrics import benchmark_winding_angle  # noqa: E402
from isaaclab_tasks.core.lift.mdp.events import SuccessMonitor  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

YAM_OPEN_MIRROR_TOLERANCE_M = 2.0e-3
"""Maximum unloaded finger-pair mismatch after opening."""

YAM_CABLE_LOADED_MIRROR_TOLERANCE_M = 5.0e-3
"""Maximum finger-pair mismatch while closing around a cable."""


def _assert_policy_cfg() -> dict[str, object]:
    """Construct and exercise bounded exploration before allocating the simulator."""
    cfg = CableRoutingPPORunnerCfg()
    distribution_cfg = cfg.actor.distribution_cfg
    expected_distribution = {
        "class_name": "isaaclab_tasks.contrib.cable_routing.agents.models:BoundedGaussianDistribution",
        "init_std": 0.25,
        "std_type": "log",
        "std_range": (0.02, 0.5),
    }
    actual_distribution = {
        "class_name": distribution_cfg.class_name,
        "init_std": distribution_cfg.init_std,
        "std_type": distribution_cfg.std_type,
        "std_range": distribution_cfg.std_range,
    }
    if actual_distribution != expected_distribution:
        raise AssertionError(
            "Cable-routing exploration must use the bounded log-standard-deviation policy: "
            f"expected={expected_distribution}, actual={actual_distribution}"
        )

    serialized_distribution = distribution_cfg.to_dict()
    distribution_class = string_to_callable(serialized_distribution.pop("class_name"))
    distribution = distribution_class(output_dim=4, **serialized_distribution)
    with torch.no_grad():
        distribution.log_std_param.fill_(10.0)
    distribution.update(torch.zeros(128, 4))
    expected_maximum_std = expected_distribution["std_range"][1]
    if not bool(torch.isfinite(distribution.std).all()) or float(distribution.std.max().item()) > expected_maximum_std:
        raise AssertionError(
            "Cable-routing exploration distribution did not enforce its finite standard-deviation maximum: "
            f"maximum={float(distribution.std.max().item())}, expected<={expected_maximum_std}"
        )
    expected_algorithm = {"entropy_coef": 0.0, "learning_rate": 1.0e-4, "schedule": "fixed"}
    actual_algorithm = {
        "entropy_coef": cfg.algorithm.entropy_coef,
        "learning_rate": cfg.algorithm.learning_rate,
        "schedule": cfg.algorithm.schedule,
    }
    if actual_algorithm != expected_algorithm:
        raise AssertionError(
            "Cable-routing PPO must disable entropy-driven scale growth and adaptive LR: "
            f"expected={expected_algorithm}, actual={actual_algorithm}"
        )
    return {
        "distribution": actual_distribution,
        "constructed_maximum_std": float(distribution.std.max().item()),
        **actual_algorithm,
    }


def _assert_reward_cfg(env_cfg) -> dict[str, object]:
    """Require sparse route success with penalties and no dense shaping rewards."""
    expected_weights = {
        "success": 20.0,
        "stretch": -0.25,
        "action_rate": -0.002,
        "left_joint_velocity": -0.0001,
        "right_joint_velocity": -0.0001,
    }
    actual_weights = {
        name: reward_cfg.weight
        for name, reward_cfg in vars(env_cfg.rewards).items()
        if not name.startswith("_") and reward_cfg is not None
    }
    if actual_weights != expected_weights:
        raise AssertionError(
            "Cable-routing rewards must contain only sparse success and penalties: "
            f"expected={expected_weights}, actual={actual_weights}"
        )
    return {
        "success_terms": ["success"],
        "penalty_terms": [name for name, weight in actual_weights.items() if weight < 0.0],
        "weights": actual_weights,
    }


def _assert_xpbd_reset_projection(iterations: int) -> dict[str, float | int]:
    """Compile and exercise the production CUDA reset-curve projector."""
    device = torch.device("cuda:0")
    rest_length = CABLE_SEGMENT_LENGTH
    board_bounds = ((-0.03, 0.03), (-0.02, 0.02))
    peg_radius = 0.004
    vertices = torch.tensor(
        [[[-0.015, 0.0], [0.0, 0.0], [0.035, 0.0]]],
        device=device,
        dtype=torch.float32,
    )
    peg_centers = torch.zeros((1, 1, 2), device=device, dtype=torch.float32)
    lower = torch.tensor((board_bounds[0][0], board_bounds[1][0]), device=device)
    upper = torch.tensor((board_bounds[0][1], board_bounds[1][1]), device=device)

    initial_edge_error = float(
        (torch.linalg.vector_norm(vertices[:, 1:] - vertices[:, :-1], dim=-1) - rest_length).abs().amax().item()
    )
    initial_peg_penetration = float(
        torch.nn.functional.relu(peg_radius - torch.linalg.vector_norm(vertices - peg_centers[:, :1], dim=-1))
        .amax()
        .item()
    )
    initial_bounds_penetration = float(
        torch.maximum(
            torch.nn.functional.relu(lower - vertices),
            torch.nn.functional.relu(vertices - upper),
        )
        .amax()
        .item()
    )
    if min(initial_edge_error, initial_peg_penetration, initial_bounds_penetration) <= 0.0:
        raise AssertionError("XPBD reset smoke input does not require spacing, peg, and bounds repair")

    projection_cfg = CableResetCurveXPBDCfg(
        self_separation_distance=2.0 * CABLE_RADIUS + 0.001,
        bend_radius=max(1.2 * rest_length, 2.0 * CABLE_RADIUS),
        iterations=iterations,
    )
    if projection_cfg.taubin_smoothing_passes != 5:
        raise AssertionError(
            "XPBD reset projection must use the five-pass Taubin smoothing tail; "
            f"got {projection_cfg.taubin_smoothing_passes}"
        )
    projected, diagnostics = relax_open_cable_curve_xpbd(
        vertices,
        rest_length=rest_length,
        board_bounds=board_bounds,
        cfg=projection_cfg,
        peg_centers=peg_centers,
        peg_radii=peg_radius,
    )

    residuals = torch.stack(
        (
            diagnostics.maximum_edge_length_error,
            diagnostics.maximum_waypoint_error,
            diagnostics.maximum_peg_penetration,
            diagnostics.maximum_bounds_penetration,
            diagnostics.maximum_turning_angle,
        ),
        dim=-1,
    )
    if not bool(torch.isfinite(projected).all()) or not bool(torch.isfinite(residuals).all()):
        raise AssertionError("XPBD reset projection produced non-finite positions or diagnostics")

    maximum_edge_error = float(diagnostics.maximum_edge_length_error.item())
    maximum_peg_penetration = float(diagnostics.maximum_peg_penetration.item())
    maximum_bounds_penetration = float(diagnostics.maximum_bounds_penetration.item())
    maximum_turning_angle = float(diagnostics.maximum_turning_angle.item())
    if maximum_edge_error >= initial_edge_error or maximum_edge_error > 1.0e-4:
        raise AssertionError(
            "XPBD reset projection did not repair cable spacing: "
            f"initial={initial_edge_error:.6g} m, projected={maximum_edge_error:.6g} m"
        )
    if maximum_peg_penetration > 1.0e-6:
        raise AssertionError(f"XPBD reset projection retained {maximum_peg_penetration:.6g} m peg penetration")
    if maximum_bounds_penetration > 1.0e-6:
        raise AssertionError(f"XPBD reset projection retained {maximum_bounds_penetration:.6g} m bounds penetration")

    # Keep independent geometry checks beside the solver diagnostics so a
    # diagnostics-only regression cannot make this preflight pass accidentally.
    peg_distance = torch.linalg.vector_norm(projected - peg_centers[:, :1], dim=-1)
    if not bool((peg_distance >= peg_radius - 1.0e-6).all()):
        raise AssertionError("XPBD reset projection failed the direct peg-clearance check")
    if not bool(((projected >= lower - 1.0e-6) & (projected <= upper + 1.0e-6)).all()):
        raise AssertionError("XPBD reset projection failed the direct board-bounds check")

    return {
        "iterations": iterations,
        "taubin_smoothing_passes": projection_cfg.taubin_smoothing_passes,
        "vertex_count": vertices.shape[1],
        "initial_maximum_edge_length_error_m": initial_edge_error,
        "maximum_edge_length_error_m": maximum_edge_error,
        "initial_maximum_peg_penetration_m": initial_peg_penetration,
        "maximum_peg_penetration_m": maximum_peg_penetration,
        "initial_maximum_bounds_penetration_m": initial_bounds_penetration,
        "maximum_bounds_penetration_m": maximum_bounds_penetration,
        "maximum_turning_angle_rad": maximum_turning_angle,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="IsaacContrib-CableRouting-YAM")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--reset_rounds", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=16)
    parser.add_argument("--benchmark_steps", type=int, default=64)
    parser.add_argument("--contact_stress_steps", type=int, default=32)
    parser.add_argument("--min_env_steps_per_second", type=float, default=0.0)
    parser.add_argument("--max_zero_action_transition_success_fraction", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _assert_collision_cfg(env_cfg) -> dict[str, object]:
    """Verify that both coupled solvers receive all required colliders."""
    solver_cfg = env_cfg.sim.physics.solver_cfg
    entries = {entry.name: entry for entry in solver_cfg.entries}
    if set(entries) != {"rigid", "cable"}:
        raise AssertionError(f"Expected rigid/cable coupled entries, got {sorted(entries)}")

    rigid_selectors = tuple(entries["rigid"].bodies)
    required_rigid = ("YamLeft", "YamRight", "Table", "Board", "Peg0", "Peg1")
    missing_rigid = [name for name in required_rigid if not any(name in selector for selector in rigid_selectors)]
    if missing_rigid:
        raise AssertionError(f"Rigid entry is missing collision owners: {missing_rigid}")

    proxy_selectors = tuple(selector for proxy in solver_cfg.proxies for selector in proxy.bodies)
    required_proxy = ("link_6", "Table", "Board", "Peg")
    missing_proxy = [name for name in required_proxy if not any(name in selector for selector in proxy_selectors)]
    if missing_proxy:
        raise AssertionError(f"Cable entry is missing rigid collision proxies: {missing_proxy}")

    action_types = {
        name: type(action_cfg).__name__
        for name, action_cfg in vars(env_cfg.actions).items()
        if not name.startswith("_") and action_cfg is not None
    }
    if action_types.get("left_arm") != "RelativeJointPositionActionCfg":
        raise AssertionError(f"Left arm is not relative JointPos: {action_types.get('left_arm')}")
    if action_types.get("right_arm") != "RelativeJointPositionActionCfg":
        raise AssertionError(f"Right arm is not relative JointPos: {action_types.get('right_arm')}")
    if not env_cfg.sim.physics.use_cuda_graph:
        raise AssertionError("Cable-routing config disabled Newton CUDA graph capture")

    cable_solver_cfg = entries["cable"].solver_cfg
    rigid_contact_capacity = getattr(cable_solver_cfg, "rigid_body_contact_buffer_size", 0)
    if rigid_contact_capacity < 256:
        raise AssertionError(
            f"Cable VBD requires at least 256 per-body rigid contacts; configured capacity is {rigid_contact_capacity}"
        )

    return {
        "rigid_body_selectors": rigid_selectors,
        "proxy_body_selectors": proxy_selectors,
        "action_types": action_types,
        "configured_rigid_contact_capacity_per_body": rigid_contact_capacity,
    }


def _assert_reset_replay_cfg(env_cfg) -> dict[str, object]:
    """Require the success-conditioned route replay used by production training."""
    replay_cfg = env_cfg.commands.route.reset_replay
    monitor_cfg = replay_cfg.success_monitor
    monitor_type = monitor_cfg.class_type
    resolved_monitor_type = string_to_callable(str(monitor_type)) if isinstance(monitor_type, str) else monitor_type
    if not replay_cfg.enabled:
        raise AssertionError("Cable reset replay is disabled")
    if replay_cfg.buffer_size != 4096:
        raise AssertionError(f"Expected a 4096-state reset bank, got {replay_cfg.buffer_size}")
    if replay_cfg.curve_projection_iterations != 50:
        raise AssertionError(
            f"Reset replay must use 50 XPBD curve-projection iterations; got {replay_cfg.curve_projection_iterations}"
        )
    if replay_cfg.max_curve_attempts < 512:
        raise AssertionError(
            "Reset replay must retain at least 512 row-local curve attempts for strict-bend tail safety; "
            f"got {replay_cfg.max_curve_attempts}"
        )
    if resolved_monitor_type is not SuccessMonitor:
        raise AssertionError(f"Reset replay does not use Lift's exact SuccessMonitor: {monitor_type}")
    expected_monitor = (10, 0.5, 1.0, 1.0)
    actual_monitor = (
        monitor_cfg.monitored_history_len,
        monitor_cfg.target_success_rate,
        monitor_cfg.kappa,
        monitor_cfg.temperature,
    )
    if actual_monitor != expected_monitor:
        raise AssertionError(f"Unexpected SuccessMonitor configuration: {actual_monitor}")
    if not replay_cfg.robot_targets.enabled:
        raise AssertionError("Route-conditioned robot reset targets are disabled")
    if replay_cfg.maximum_settled_active_progress != 0.92:
        raise AssertionError(
            f"Reset replay must preserve an 8% completion margin; got {replay_cfg.maximum_settled_active_progress}"
        )
    if replay_cfg.restore_clearance != 5.0e-6:
        raise AssertionError(
            f"Reset replay must preserve a 5 um cross-clone relocation reserve; got {replay_cfg.restore_clearance}"
        )
    return {
        "enabled": True,
        "buffer_size": replay_cfg.buffer_size,
        "curve_projection_iterations": replay_cfg.curve_projection_iterations,
        "max_curve_attempts": replay_cfg.max_curve_attempts,
        "monitor": "SuccessMonitor",
        "monitored_history_len": monitor_cfg.monitored_history_len,
        "target_success_rate": monitor_cfg.target_success_rate,
        "kappa": monitor_cfg.kappa,
        "temperature": monitor_cfg.temperature,
        "robot_targets_enabled": True,
        "maximum_settled_active_progress": replay_cfg.maximum_settled_active_progress,
        "restore_clearance_m": replay_cfg.restore_clearance,
    }


def _assert_runtime_shapes() -> dict[str, int]:
    """Verify that imported Newton shapes exist for robot and fixture collision groups."""
    model = NewtonManager._model
    if model is None:
        raise AssertionError("Newton model was not finalized")
    labels = [str(label) for label in (model.shape_label or ())]
    shape_counts = {
        token: sum(token in label for label in labels)
        for token in ("YamLeft", "YamRight", "Table", "Board", "Peg0", "Peg1")
    }
    missing = [token for token, count in shape_counts.items() if count == 0]
    if missing:
        raise AssertionError(f"Newton model has no collision shapes for: {missing}")
    return shape_counts


def _assert_yam_runtime_contract(env) -> dict[str, object]:
    """Verify the Robot Menagerie joint/body contract and 14-D policy interface."""
    expected_joints = {
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "left_finger",
        "right_finger",
    }
    required_bodies = {"arm", "link_6", "link_left_finger", "link_right_finger"}
    robots: dict[str, dict[str, list[str]]] = {}
    for asset_name in ("yam_left", "yam_right"):
        robot = env.unwrapped.scene[asset_name]
        joint_names = list(robot.joint_names)
        body_names = list(robot.body_names)
        if set(joint_names) != expected_joints:
            raise AssertionError(
                f"{asset_name} does not match the Robot Menagerie YAM joints: "
                f"expected={sorted(expected_joints)}, actual={joint_names}"
            )
        missing_bodies = sorted(required_bodies - set(body_names))
        if missing_bodies:
            raise AssertionError(f"{asset_name} is missing Robot Menagerie bodies: {missing_bodies}")
        robots[asset_name] = {"joint_names": joint_names, "body_names": body_names}

    if env.action_space.shape[-1] != 14:
        raise AssertionError(f"Expected a 14-D bimanual policy action, got {env.action_space.shape}")
    return {"action_dim": env.action_space.shape[-1], "robots": robots}


def _assert_yam_finger_mimic(
    env,
    seed: int,
    open_steps: int = 16,
    close_steps: int = 16,
) -> dict[str, float]:
    """Open then close both grippers and verify driven travel and passive -1 gearing."""
    env.reset(seed=seed)
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    # Action order is left arm (6), left gripper (1), right arm (6), right gripper (1).
    actions[:, 6] = 1.0
    actions[:, 13] = 1.0
    with torch.inference_mode():
        for _ in range(open_steps):
            env.step(actions)

    opened_positions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for asset_name in ("yam_left", "yam_right"):
        robot = env.unwrapped.scene[asset_name]
        left_index = robot.joint_names.index("left_finger")
        right_index = robot.joint_names.index("right_finger")
        opened_positions[asset_name] = (
            robot.data.joint_pos.torch[:, left_index].clone(),
            robot.data.joint_pos.torch[:, right_index].clone(),
        )

    actions[:, 6] = -1.0
    actions[:, 13] = -1.0
    with torch.inference_mode():
        for _ in range(close_steps):
            env.step(actions)

    minimum_open_position = float("inf")
    minimum_driven_travel = float("inf")
    maximum_open_mirror_error = 0.0
    maximum_closed_mirror_error = 0.0
    for asset_name in ("yam_left", "yam_right"):
        robot = env.unwrapped.scene[asset_name]
        left_index = robot.joint_names.index("left_finger")
        right_index = robot.joint_names.index("right_finger")
        opened_left, opened_right = opened_positions[asset_name]
        closed_left = robot.data.joint_pos.torch[:, left_index]
        closed_right = robot.data.joint_pos.torch[:, right_index]
        finger_states = {
            "opened_left": opened_left,
            "opened_right": opened_right,
            "closed_left": closed_left,
            "closed_right": closed_right,
        }
        for state_name, state in finger_states.items():
            if not bool(torch.isfinite(state).all()):
                raise AssertionError(f"Non-finite YAM finger position in {asset_name}.{state_name}")
        minimum_open_position = min(minimum_open_position, float(opened_left.amin().item()))
        minimum_driven_travel = min(
            minimum_driven_travel,
            float((opened_left - closed_left).amin().item()),
        )
        maximum_open_mirror_error = max(
            maximum_open_mirror_error,
            float((opened_left + opened_right).abs().amax().item()),
        )
        maximum_closed_mirror_error = max(
            maximum_closed_mirror_error,
            float((closed_left + closed_right).abs().amax().item()),
        )

    if minimum_open_position < 0.02:
        raise AssertionError(
            f"Robot Menagerie YAM driven fingers did not open: minimum position={minimum_open_position:.6f} m"
        )
    if minimum_driven_travel < 0.01:
        raise AssertionError(
            "Robot Menagerie YAM driven fingers did not close from the measured open position: "
            f"minimum travel={minimum_driven_travel:.6f} m"
        )
    if maximum_open_mirror_error > YAM_OPEN_MIRROR_TOLERANCE_M:
        raise AssertionError(
            "Robot Menagerie YAM unloaded finger mimic violated q_right=-q_left after opening: "
            f"maximum error={maximum_open_mirror_error:.6f} m"
        )
    if maximum_closed_mirror_error > YAM_CABLE_LOADED_MIRROR_TOLERANCE_M:
        raise AssertionError(
            "Robot Menagerie YAM cable-loaded finger mimic deflected beyond tolerance after closing: "
            f"maximum error={maximum_closed_mirror_error:.6f} m"
        )
    return {
        "open_steps": open_steps,
        "close_steps": close_steps,
        "minimum_driven_finger_open_position_m": minimum_open_position,
        "minimum_driven_finger_travel_m": minimum_driven_travel,
        "maximum_open_finger_mirror_error_m": maximum_open_mirror_error,
        "maximum_cable_loaded_finger_mirror_error_m": maximum_closed_mirror_error,
        "open_finger_mirror_tolerance_m": YAM_OPEN_MIRROR_TOLERANCE_M,
        "cable_loaded_finger_mirror_tolerance_m": YAM_CABLE_LOADED_MIRROR_TOLERANCE_M,
    }


def _cable_vbd_contact_snapshot() -> dict[str, int]:
    """Read the cable subsolver's current per-body contact occupancy and overflow state."""
    coupled_solver = NewtonManager._solver
    if coupled_solver is None or not callable(getattr(coupled_solver, "solver", None)):
        raise AssertionError("Newton coupled solver does not expose its named cable subsolver")
    cable_solver = coupled_solver.solver("cable")
    required = (
        "body_body_contact_buffer_pre_alloc",
        "body_body_contact_counts",
        "body_body_contact_overflow_max",
        "body_body_contact_indices",
    )
    missing = [name for name in required if not hasattr(cable_solver, name)]
    if missing:
        raise AssertionError(f"Cable VBD subsolver is missing contact diagnostics: {missing}")

    capacity = int(cable_solver.body_body_contact_buffer_pre_alloc)
    counts = cable_solver.body_body_contact_counts.numpy()
    overflow_max = int(cable_solver.body_body_contact_overflow_max.numpy()[0])
    observed_max = int(counts.max()) if counts.size else 0
    if overflow_max > capacity:
        raise AssertionError(
            f"Cable VBD per-body rigid contacts overflowed during preflight: {overflow_max} > {capacity}"
        )
    return {
        "capacity_per_body": capacity,
        "observed_contacts_per_body": observed_max,
        "overflow_contacts_per_body": overflow_max,
        "index_buffer_bytes": int(cable_solver.body_body_contact_indices.size) * 4,
    }


def _assert_contact_stress(env, steps: int, seed: int) -> dict[str, int]:
    """Exercise bounded arm/gripper motion and reject any VBD contact-list overflow."""
    if steps < 1:
        raise AssertionError("Contact stress test requires at least one policy step")
    observations, _ = env.reset(seed=seed)
    _assert_finite_tree(observations)
    device = env.unwrapped.device
    env_phase = torch.arange(env.action_space.shape[0], device=device, dtype=torch.float32)[:, None] * 0.17
    action_phase = torch.arange(env.action_space.shape[1], device=device, dtype=torch.float32)[None] * 0.31
    maximum_contacts = 0
    maximum_overflow = 0
    snapshot: dict[str, int] | None = None

    with torch.inference_mode():
        for step in range(steps):
            actions = 0.75 * torch.sin(env_phase + action_phase + 0.23 * (seed + step))
            # Alternate both grippers between close/open while the arm targets move.
            gripper_command = -1.0 if (step // 8) % 2 == 0 else 1.0
            actions[:, 6] = gripper_command
            actions[:, 13] = gripper_command
            observations, rewards = env.step(actions)[:2]
            _assert_finite_tree(observations)
            if not bool(torch.isfinite(rewards).all()):
                raise AssertionError(f"Non-finite rewards during contact stress step {step + 1}")
            snapshot = _cable_vbd_contact_snapshot()
            maximum_contacts = max(maximum_contacts, snapshot["observed_contacts_per_body"])
            maximum_overflow = max(maximum_overflow, snapshot["overflow_contacts_per_body"])

    assert snapshot is not None
    return {
        "policy_steps": steps,
        "capacity_per_body": snapshot["capacity_per_body"],
        "maximum_observed_contacts_per_body": maximum_contacts,
        "maximum_overflow_contacts_per_body": maximum_overflow,
        "index_buffer_bytes": snapshot["index_buffer_bytes"],
    }


def _assert_live_reset_clearance(env, reset_rounds: int, seed: int) -> dict[str, float | int]:
    """Exercise replay resets and validate their geometry, attribution, and diversity."""
    unwrapped = env.unwrapped
    command = unwrapped.command_manager.get_term("route")
    allowed_directions = torch.tensor(
        BENCHMARK_GRID_DIRECTIONS,
        device=unwrapped.device,
        dtype=torch.int64,
    )
    base_xy = torch.tensor(PEG_BASE_POSITIONS_B, device=unwrapped.device)[:, :2]
    min_clearance = float("inf")
    max_abs_winding = 0.0
    distinct_reset_count = 0
    sampled_sources: set[int] = set()
    sampled_phases: set[int] = set()
    min_sampled_progress = float("inf")
    max_sampled_progress = float("-inf")
    min_sampled_active_progress = float("inf")
    max_sampled_active_progress = float("-inf")
    min_requested_active_progress = float("inf")
    max_requested_active_progress = float("-inf")
    max_progress_metadata_drift = 0.0
    previous_local_cable = None

    for reset_index in range(reset_rounds):
        env.reset(seed=seed + reset_index)
        segment_poses = unwrapped.scene["cable"].data.segment_pose_w.torch
        peg_positions = torch.stack(
            (
                unwrapped.scene["peg_0"].data.root_pos_w.torch,
                unwrapped.scene["peg_1"].data.root_pos_w.torch,
            ),
            dim=1,
        )
        env_origins = unwrapped.scene.env_origins
        clear = cable_capsule_clearance_mask(
            segment_poses,
            peg_positions,
            env_origins,
            rest_length=CABLE_SEGMENT_LENGTH,
            cable_radius=CABLE_RADIUS,
            peg_radius=PEG_RADIUS,
            # Route-conditioned snapshots intentionally rest on fixtures and
            # may overhang the board onto the table. Match the bank's
            # post-settle validity; zero extra margin still rejects geometric
            # interpenetration with fixtures or the tabletop boundary.
            fixture_clearance=0.0,
            board_bounds_b=command.cfg.settled_cable_bounds_b,
            board_clearance=0.0,
        )
        if not bool(clear.all()):
            bad = (~clear).nonzero(as_tuple=False).squeeze(-1).tolist()
            raise AssertionError(f"Cable reset penetrated a randomized fixture or tabletop boundary in envs {bad}")
        self_clear = cable_capsule_self_clearance_mask(
            segment_poses,
            rest_length=CABLE_SEGMENT_LENGTH,
            cable_radius=CABLE_RADIUS,
        )
        if not bool(self_clear.all()):
            bad = (~self_clear).nonzero(as_tuple=False).squeeze(-1).tolist()
            raise AssertionError(f"Cable reset has non-neighbor capsule self-contact in envs {bad}")

        replay = command.reset_replay
        if replay is None or not replay.built or replay.state_buffer is None:
            raise AssertionError("Initial reset did not build the success-conditioned scene replay bank")
        if replay.state_buffer.capacity != command.cfg.reset_replay.buffer_size:
            raise AssertionError("Reset replay state-buffer capacity does not match its configuration")
        if reset_index == 0:
            bank_routes = replay.route_id
            bank_steps = replay.active_step
            bank_phases = replay.interaction_phase
            bank_progress = replay.requested_active_progress
            for route_id in command.cfg.allowed_route_ids:
                route_rows = bank_routes == route_id
                if not bool(route_rows.any()):
                    raise AssertionError(f"Reset replay bank contains no rows for route {route_id}")
                route_length = len(command.cfg.route_options[route_id])
                for active_step in range(route_length):
                    stage_rows = route_rows & (bank_steps == active_step)
                    if not bool(stage_rows.any()):
                        raise AssertionError(
                            f"Reset replay bank contains no rows for route {route_id}, active step {active_step}"
                        )
                    if set(bank_phases[stage_rows].tolist()) != {0, 1, 2}:
                        raise AssertionError(
                            f"Reset replay bank lacks an interaction phase for route {route_id}, "
                            f"active step {active_step}"
                        )
                    if len(torch.unique(bank_progress[stage_rows])) != 16:
                        raise AssertionError(
                            f"Reset replay bank does not cover all 16 progress bins for route {route_id}, "
                            f"active step {active_step}"
                        )
        source = replay.env_source
        valid_source = (source >= 0) & (source < replay.cfg.buffer_size)
        if not bool(valid_source.all()):
            bad = (~valid_source).nonzero(as_tuple=False).squeeze(-1).tolist()
            raise AssertionError(f"Reset replay did not attribute valid bank rows to envs {bad}")
        rows = source.to(torch.long)
        sampled_sources.update(rows.tolist())
        sampled_phase = replay.interaction_phase[rows]
        if not bool(((sampled_phase >= 0) & (sampled_phase < 3)).all()):
            raise AssertionError("Reset replay sampled an invalid robot interaction phase")
        sampled_phases.update(sampled_phase.tolist())
        sampled_progress = replay.start_progress[rows]
        sampled_requested_progress = replay.requested_active_progress[rows]
        sampled_route_ids = replay.route_id[rows]
        sampled_active_steps = replay.active_step[rows]
        progress_min, progress_max = replay.cfg.active_progress_range
        if not bool(torch.isfinite(sampled_requested_progress).all()) or not bool(
            ((sampled_requested_progress >= progress_min) & (sampled_requested_progress <= progress_max)).all()
        ):
            raise AssertionError("Reset replay sampled requested active-step progress outside its authored interval")
        allowed_route_ids = torch.tensor(command.cfg.allowed_route_ids, device=unwrapped.device)
        if not bool(torch.isin(sampled_route_ids, allowed_route_ids).all()):
            raise AssertionError("Reset replay sampled a route outside the task's allowed route IDs")
        route_lengths = tuple(len(route) for route in command.cfg.route_options)
        sampled_active_progress = active_step_progress_from_route_progress(
            sampled_progress,
            sampled_route_ids,
            sampled_active_steps,
            route_lengths,
        )
        progress_error = (sampled_active_progress - sampled_requested_progress).abs()
        valid_settled_progress = (
            torch.isfinite(sampled_progress)
            & torch.isfinite(sampled_active_progress)
            & (sampled_active_progress > 0.02)
            & (sampled_active_progress < replay.cfg.maximum_settled_active_progress)
            & (progress_error < replay.cfg.post_settle_progress_tolerance)
        )
        if not bool(valid_settled_progress.all()):
            bad = (~valid_settled_progress).nonzero(as_tuple=False).squeeze(-1).tolist()
            raise AssertionError(f"Reset replay sampled invalid settled active-step progress in envs {bad}")
        min_sampled_progress = min(min_sampled_progress, float(sampled_progress.amin().item()))
        max_sampled_progress = max(max_sampled_progress, float(sampled_progress.amax().item()))
        min_sampled_active_progress = min(
            min_sampled_active_progress,
            float(sampled_active_progress.amin().item()),
        )
        max_sampled_active_progress = max(
            max_sampled_active_progress,
            float(sampled_active_progress.amax().item()),
        )
        min_requested_active_progress = min(
            min_requested_active_progress,
            float(sampled_requested_progress.amin().item()),
        )
        max_requested_active_progress = max(
            max_requested_active_progress,
            float(sampled_requested_progress.amax().item()),
        )
        if not torch.equal(command.route_id, sampled_route_ids):
            raise AssertionError("Live command route IDs do not match restored replay metadata")
        winding = benchmark_winding_angle(
            segment_poses[..., :3],
            peg_positions,
            command.cfg.radial_cutoff,
            command.cfg.axial_cutoff,
        )
        max_abs_winding = max(max_abs_winding, float(winding.abs().amax().item()))
        command.refresh_route_state()
        if not torch.equal(command.active_step, sampled_active_steps):
            raise AssertionError("Live command active steps do not match restored replay metadata")
        live_progress = command.metrics["route_progress"]
        live_active_progress = active_step_progress_from_route_progress(
            live_progress,
            sampled_route_ids,
            sampled_active_steps,
            route_lengths,
        )
        live_requested_error = (live_active_progress - sampled_requested_progress).abs()
        valid_live_progress = (
            torch.isfinite(live_progress)
            & torch.isfinite(live_active_progress)
            & (live_active_progress > 0.02)
            & (live_active_progress < replay.cfg.maximum_settled_active_progress)
            & (live_requested_error < replay.cfg.post_settle_progress_tolerance)
        )
        if not bool(valid_live_progress.all()):
            bad = (~valid_live_progress).nonzero(as_tuple=False).squeeze(-1)
            details = [
                {
                    "env": int(env_id),
                    "source": int(rows[env_id]),
                    "route_id": int(sampled_route_ids[env_id]),
                    "active_step": int(sampled_active_steps[env_id]),
                    "requested_active_progress": float(sampled_requested_progress[env_id]),
                    "stored_progress": float(sampled_progress[env_id]),
                    "live_progress": float(live_progress[env_id]),
                    "live_active_progress": float(live_active_progress[env_id]),
                    "requested_error": float(live_requested_error[env_id]),
                }
                for env_id in bad[:8].tolist()
            ]
            raise AssertionError(
                f"Live restored route progress violates the replay acceptance bounds: first_mismatches={details}"
            )
        metadata_drift = (live_progress - sampled_progress).abs()
        max_progress_metadata_drift = max(max_progress_metadata_drift, float(metadata_drift.amax().item()))
        if bool(command.succeeded.any()):
            bad = command.succeeded.nonzero(as_tuple=False).squeeze(-1).tolist()
            raise AssertionError(f"Cable reset already completes the sampled route command in envs {bad}")

        offsets = peg_positions[..., :2] - env_origins[:, None, :2] - base_xy[None]
        grid_offsets = torch.round(offsets / 0.01).to(torch.int64)
        exact_grid = torch.isclose(offsets, 0.01 * grid_offsets, atol=1.0e-5, rtol=0.0).all(dim=-1)
        allowed = (grid_offsets[:, :, None, :] == allowed_directions[None, None]).all(dim=-1).any(dim=-1)
        if not bool((exact_grid & allowed).all()):
            raise AssertionError("Live fixture reset did not use an exact nonzero benchmark-grid offset")

        local_cable = segment_poses[..., :3] - env_origins[:, None, :]
        if previous_local_cable is not None and not torch.equal(local_cable, previous_local_cable):
            distinct_reset_count += 1
        previous_local_cable = local_cable.clone()

        # Report the observed planar centerline-to-center distance margin. The exact
        # capsule test above is authoritative; this scalar is useful for dashboards.
        center_distance = torch.cdist(segment_poses[..., :2], peg_positions[..., :2])
        observed = float(center_distance.amin().item() - CABLE_RADIUS - PEG_RADIUS)
        min_clearance = min(min_clearance, observed)

    if reset_rounds > 1 and distinct_reset_count == 0:
        raise AssertionError("Repeated cable resets were not heterogeneous")
    if len(sampled_sources) < 2:
        raise AssertionError("Repeated replay resets did not sample distinct scene-state rows")
    replay = command.reset_replay
    assert replay is not None
    return {
        "minimum_planar_surface_clearance_m": min_clearance,
        "maximum_abs_winding_rad": max_abs_winding,
        "distinct_reset_transitions": distinct_reset_count,
        "distinct_sampled_sources": len(sampled_sources),
        "distinct_sampled_phases": len(sampled_phases),
        "minimum_sampled_progress": min_sampled_progress,
        "maximum_sampled_progress": max_sampled_progress,
        "minimum_sampled_active_progress": min_sampled_active_progress,
        "maximum_sampled_active_progress": max_sampled_active_progress,
        "minimum_requested_active_progress": min_requested_active_progress,
        "maximum_requested_active_progress": max_requested_active_progress,
        "maximum_live_progress_metadata_drift": max_progress_metadata_drift,
        "bank_capacity": replay.cfg.buffer_size,
        "build_candidates": replay.build_candidate_count,
        "build_rejections": replay.build_rejection_count,
        "build_max_attempts": replay.build_max_attempts,
    }


def _assert_finite_tree(value, path: str = "observation") -> None:
    """Recursively require finite floating-point observation tensors."""
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise AssertionError(f"Non-finite values in {path}")


def main() -> None:
    args = _parse_args()
    if args.num_envs < 2:
        raise SystemExit("--num_envs must be at least 2 to test heterogeneous resets")
    if args.reset_rounds < 2:
        raise SystemExit("--reset_rounds must be at least 2")
    if args.contact_stress_steps < 1:
        raise SystemExit("--contact_stress_steps must be at least 1")
    if not 0.0 <= args.max_zero_action_transition_success_fraction <= 1.0:
        raise SystemExit("--max_zero_action_transition_success_fraction must lie in [0, 1]")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(f"Preflight requires exactly one isolated CUDA device, found {torch.cuda.device_count()}")

    policy_cfg = _assert_policy_cfg()
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.commands.route.reset_replay.seed = args.seed
    reward_cfg = _assert_reward_cfg(env_cfg)
    collision_cfg = _assert_collision_cfg(env_cfg)
    reset_replay_cfg = _assert_reset_replay_cfg(env_cfg)
    xpbd_reset_projection = _assert_xpbd_reset_projection(
        env_cfg.commands.route.reset_replay.curve_projection_iterations
    )
    env_cfg.commands.route.debug_vis = False
    env = gym.make(args.task, cfg=env_cfg)
    try:
        yam_runtime_contract = _assert_yam_runtime_contract(env)
        finger_mimic = _assert_yam_finger_mimic(env, args.seed)
        reset_replay_runtime = _assert_live_reset_clearance(
            env,
            args.reset_rounds,
            args.seed,
        )
        runtime_shape_counts = _assert_runtime_shapes()

        observations, _ = env.reset(seed=args.seed + args.reset_rounds)
        _assert_finite_tree(observations)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        zero_action_steps = 0
        zero_action_route_successes = 0
        zero_action_success_sources: set[int] = set()
        first_zero_action_success_step: int | None = None

        def record_zero_action_outcome(source_before_step: torch.Tensor) -> None:
            """Record the bounded open-gripper diagnostic after one all-zero action."""
            nonlocal zero_action_steps, zero_action_route_successes, first_zero_action_success_step
            zero_action_steps += 1
            success = env.unwrapped.termination_manager.get_term("success")
            if bool(success.any()):
                count = int(success.sum().item())
                zero_action_route_successes += count
                if first_zero_action_success_step is None:
                    first_zero_action_success_step = zero_action_steps
                zero_action_success_sources.update(source_before_step[success].cpu().tolist())

        command = env.unwrapped.command_manager.get_term("route")

        with torch.inference_mode():
            for _ in range(args.warmup_steps):
                source_before_step = command.reset_replay.env_source.clone()
                observations, rewards = env.step(actions)[:2]
                record_zero_action_outcome(source_before_step)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.benchmark_steps):
                source_before_step = command.reset_replay.env_source.clone()
                observations, rewards = env.step(actions)[:2]
                record_zero_action_outcome(source_before_step)
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - start

        # Binary gripper action zero selects the open target, so this is a
        # commanded-release stress test rather than a passive hold test. Its
        # event count scales with both the vector width and rollout horizon;
        # gate the transition hazard while keeping rare frontier releases
        # observable.
        zero_action_transitions = args.num_envs * zero_action_steps
        zero_action_transition_success_fraction = zero_action_route_successes / zero_action_transitions
        max_zero_action_route_successes = math.floor(
            args.max_zero_action_transition_success_fraction * zero_action_transitions
        )
        if zero_action_route_successes > max_zero_action_route_successes:
            raise AssertionError(
                f"All-zero/open-gripper actions completed {zero_action_route_successes} routes over "
                f"{zero_action_transitions} environment transitions "
                f"({zero_action_transition_success_fraction:.8f}); allowed at most "
                f"{max_zero_action_route_successes} ({args.max_zero_action_transition_success_fraction:.8f})"
            )

        _assert_finite_tree(observations)
        if not bool(torch.isfinite(rewards).all()):
            raise AssertionError("Non-finite rewards after CUDA benchmark")
        if NewtonManager._graph is None:
            raise AssertionError("Newton CUDA graph was requested but no graph was captured")

        contact_stress = _assert_contact_stress(
            env,
            args.contact_stress_steps,
            args.seed + args.reset_rounds + 1,
        )

        env_steps_per_second = args.num_envs * args.benchmark_steps / elapsed_s
        if env_steps_per_second < args.min_env_steps_per_second:
            raise AssertionError(
                f"Throughput {env_steps_per_second:.1f} env-steps/s is below {args.min_env_steps_per_second:.1f}"
            )

        report = {
            "task": args.task,
            "device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "num_envs": args.num_envs,
            "action_shape": list(env.action_space.shape),
            "policy_cfg": policy_cfg,
            "reward_cfg": reward_cfg,
            "cuda_graph_captured": True,
            "runtime_shape_counts": runtime_shape_counts,
            "yam_runtime_contract": yam_runtime_contract,
            "finger_mimic": finger_mimic,
            "reset_rounds": args.reset_rounds,
            "reset_replay_cfg": reset_replay_cfg,
            "reset_replay_runtime": reset_replay_runtime,
            "xpbd_reset_projection": xpbd_reset_projection,
            "non_neighbor_self_clearance_verified": True,
            "benchmark_policy_steps": args.benchmark_steps,
            "contact_stress": contact_stress,
            "zero_action_route_successes": zero_action_route_successes,
            "zero_action_distinct_success_sources": len(zero_action_success_sources),
            "zero_action_successes_per_initial_env": zero_action_route_successes / args.num_envs,
            "zero_action_policy_steps": zero_action_steps,
            "zero_action_transitions": zero_action_transitions,
            "zero_action_transition_success_fraction": zero_action_transition_success_fraction,
            "max_zero_action_transition_success_fraction": args.max_zero_action_transition_success_fraction,
            "max_zero_action_route_successes": max_zero_action_route_successes,
            "first_zero_action_success_step": first_zero_action_success_step,
            "benchmark_elapsed_s": elapsed_s,
            "policy_steps_per_second": args.benchmark_steps / elapsed_s,
            "env_steps_per_second": env_steps_per_second,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            **collision_cfg,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
