# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless runtime integration for batched Franka Pour MPM."""

from __future__ import annotations

import math
import os
from unittest import mock

import pytest

_RUNTIME_UNAVAILABLE_REASON = "Isaac Sim runtime is unavailable because EXP_PATH is not set."
_RUNTIME_AVAILABLE = bool(os.environ.get("EXP_PATH"))

if _RUNTIME_AVAILABLE:
    from isaaclab.app import AppLauncher

    # Launch Kit before importing simulation-dependent modules.
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import newton
    import torch
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import MPM_ENTRY, PANDA_ARM_JOINT_LIMITS
    from isaaclab_tasks.contrib.franka_pour.reset_utils import (
        asymmetric_reset_offset_samples,
        polar_workspace_cells,
    )
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.newton_ci]

_TASK_ID = "Isaac-Pour-Franka-v0"


def _require_cuda() -> None:
    """Require a CUDA device for the Franka Pour runtime integration."""
    if not wp.is_cuda_available():
        pytest.skip("Franka Pour runtime integration requires a CUDA device.")


def _assert_scene_solver_roles(model) -> None:
    """Check exact per-world task bodies and solver-only collision roles."""
    body_world = model.body_world.numpy()
    body_mass = model.body_mass.numpy()
    body_inv_mass = model.body_inv_mass.numpy()
    body_flags = model.body_flags.numpy()
    shape_body = model.shape_body.numpy()
    shape_flags = model.shape_flags.numpy()
    collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
    collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    visible = int(newton.ShapeFlags.VISIBLE)

    for world in range(2):
        bodies_by_name: dict[str, list[int]] = {}
        for body_id, label in enumerate(model.body_label):
            if int(body_world[body_id]) == world:
                bodies_by_name.setdefault(str(label).rsplit("/", 1)[-1], []).append(body_id)
        for name in ("SourceCup", "TargetCup", "SpillFloor"):
            assert len(bodies_by_name.get(name, [])) == 1, (world, name, bodies_by_name.get(name))
        assert "TargetCupRigid" not in bodies_by_name

        target_body = bodies_by_name["TargetCup"][0]
        assert int(body_flags[target_body]) & int(newton.BodyFlags.KINEMATIC)
        assert float(body_mass[target_body]) == 0.0
        assert float(body_inv_mass[target_body]) == 0.0

        expected_shapes = (
            ("SourceCup", "/SourceCup/ParticleCollider", False, True, False),
            ("TargetCup", "/TargetCup/ParticleCollider", False, True, False),
            ("TargetCup", "/TargetCup/Collision", True, False, False),
            ("SpillFloor", "/SpillFloor/Collision", False, True, False),
        )
        for body_name, suffix, rigid, particles, is_visible in expected_shapes:
            body_id = bodies_by_name[body_name][0]
            matches = [
                shape_id
                for shape_id, label in enumerate(model.shape_label)
                if int(shape_body[shape_id]) == body_id and str(label).endswith(suffix)
            ]
            assert len(matches) == 1, (world, body_name, matches)
            flags = int(shape_flags[matches[0]])
            assert bool(flags & collide_shapes) is rigid
            assert bool(flags & collide_particles) is particles
            assert bool(flags & visible) is is_visible


def _make_runtime_cfg(
    *,
    use_cuda_graph: bool = False,
    env_spacing: float = 2.5,
    mpm_iterations: int = 2,
):
    cfg = parse_env_cfg(_TASK_ID, device="cuda:0", num_envs=2)
    cfg.seed = 37
    # Keep the existing solver/reset regressions on the full-task open-hand approach contract.
    # The dedicated curriculum test below exercises mixed easy/full stages explicitly.
    cfg.curriculum_start_stage = cfg.curriculum_stage_names.index("full")
    cfg.curriculum_freeze = True
    cfg.scene.env_spacing = env_spacing
    cfg.decimation = 1
    cfg.physics_substeps = 1
    cfg.mpm_iterations = mpm_iterations
    cfg.use_cuda_graph = use_cuda_graph
    cfg.sim.render_interval = 1

    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}
    assert entries[MPM_ENTRY].in_place
    return cfg


def _assert_full_task_reset_state(task, *, arm_atol: float = 1.0e-5, gripper_atol: float = 1.0e-5) -> None:
    """Check that a full-task reset is finite, open, and begins from a real approach."""
    arm_action = task.action_manager.get_term("arm_action")
    arm_q = task._robot.data.joint_pos.torch[:, task._arm_joint_ids]
    assert bool(torch.isfinite(arm_q).all())
    joint_limits = torch.as_tensor(PANDA_ARM_JOINT_LIMITS, device=task.device)
    assert bool(torch.all(arm_q >= joint_limits[:, 0] - arm_atol))
    assert bool(torch.all(arm_q <= joint_limits[:, 1] + arm_atol))
    torch.testing.assert_close(
        arm_action.processed_actions,
        torch.zeros_like(arm_action.processed_actions),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        arm_action.raw_actions,
        torch.zeros_like(arm_action.raw_actions),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        task.gripper_width(),
        torch.full((task.num_envs,), 0.08, device=task.device),
        rtol=0.0,
        atol=gripper_atol,
    )
    tcp_distance = torch.linalg.vector_norm(task.tcp_pos_e() - task.cup_grasp_point_e(), dim=-1)
    minimum_distance = task.cfg.curriculum_randomized_reset_tcp_min_grasp_distance - 0.005
    assert bool(torch.all(tcp_distance >= minimum_distance)), (
        f"Full-task TCP reset is not a real approach: distance={tcp_distance.tolist()}"
    )


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_reset_uses_direct_actions_and_zero_arm_action_holds():
    """The task exposes direct 7+1 actions and zero arm increments hold the reset pose."""
    _require_cuda()
    sim_utils.create_new_stage()
    env = None
    try:
        env = gym.make(_TASK_ID, cfg=_make_runtime_cfg())
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        assert task.action_manager.active_terms == ["arm_action", "gripper_action"]
        assert task.action_manager.action_term_dim == [7, 1]
        assert task.action_manager.total_action_dim == 8
        _assert_full_task_reset_state(task)
        arm_action = task.action_manager.get_term("arm_action")
        arm_before = task._robot.data.joint_pos.torch[:, task._arm_joint_ids].clone()
        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        actions[:, -1] = 1.0
        env.step(actions)
        torch.testing.assert_close(
            arm_action.processed_actions,
            torch.zeros_like(arm_action.processed_actions),
            rtol=0.0,
            atol=0.0,
        )
        arm_after = task._robot.data.joint_pos.torch[:, task._arm_joint_ids]
        torch.testing.assert_close(arm_after, arm_before, rtol=0.0, atol=2.0e-3)
        assert bool(torch.all(task.state_finite()))
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_direct_arm_action_moves_without_hidden_reference():
    """A bounded arm increment moves its joint while all other direct coordinates remain zero."""
    _require_cuda()
    sim_utils.create_new_stage()
    env = None
    try:
        cfg = _make_runtime_cfg(mpm_iterations=24)
        full_stage = cfg.curriculum_stage_names.index("full")
        cfg.curriculum_start_stage = full_stage
        cfg.curriculum_freeze = True
        cfg.decimation = 2
        cfg.physics_substeps = 2
        env = gym.make(_TASK_ID, cfg=cfg)
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None

        env.reset()
        assert bool(torch.all(task.curriculum_stage == full_stage))
        assert task.cfg.curriculum_freeze is True
        _assert_full_task_reset_state(task)

        arm_action = task.action_manager.get_term("arm_action")
        arm_before = task._robot.data.joint_pos.torch[:, task._arm_joint_ids].clone()
        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        actions[:, 0] = 0.25
        actions[:, -1] = 1.0
        for _ in range(3):
            env.step(actions)
        wp.synchronize_device(NewtonManager.get_model().device)
        torch.testing.assert_close(
            arm_action.processed_actions[:, 0],
            torch.full((task.num_envs,), 0.25 * task.cfg.actions.arm_action.scale, device=task.device),
            rtol=0.0,
            atol=1.0e-7,
        )
        assert bool(torch.all(task._robot.data.joint_pos.torch[:, task._arm_joint_ids[0]] > arm_before[:, 0]))
        assert bool(torch.all(task.state_finite()))
        assert bool(torch.all(task.rigid_state_in_bounds()))
        assert bool(torch.all(task.particles_in_workspace()))
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_randomized_stage_builds_safe_ik_resets_and_resets_selected_world():
    """Randomized full-task resets stay reachable, separated, bounded, and world-selective."""
    _require_cuda()
    sim_utils.create_new_stage()
    env = None
    try:
        cfg = _make_runtime_cfg()
        randomized_stage = cfg.curriculum_stage_names.index("randomized")
        cfg.curriculum_start_stage = randomized_stage
        cfg.curriculum_randomization_start_level = len(cfg.curriculum_randomization_extent_levels) - 1
        cfg.curriculum_freeze = True
        env = gym.make(_TASK_ID, cfg=cfg)
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        source_bank = task._randomized_source_pos_bank_t
        source_yaw_bank = task._randomized_source_yaw_bank_t
        source_quat_bank = task._randomized_source_quat_bank_t
        target_bank = task._randomized_target_pos_bank_t
        tcp_start_bank = task._randomized_tcp_pos_bank_t
        tcp_start_quat_bank = task._randomized_tcp_quat_bank_t
        tcp_rotation_vector_bank = task._randomized_tcp_rotation_vector_bank_t
        arm_bank = task._randomized_arm_q_bank_t
        assert source_bank.shape == target_bank.shape
        assert source_bank.shape == tcp_start_bank.shape
        assert source_yaw_bank.shape == (source_bank.shape[0],)
        assert source_quat_bank.shape == tcp_start_quat_bank.shape == (source_bank.shape[0], 4)
        assert tcp_rotation_vector_bank.shape == (source_bank.shape[0], 3)
        assert source_bank.shape[0] == arm_bank.shape[0] > 1
        rows_per_extent = (
            task.cfg.curriculum_randomized_reset_ik_grid_size**2
            * task.cfg.curriculum_randomized_reset_ik_samples_per_source
        )
        extent_levels = task.cfg.curriculum_randomization_extent_levels
        assert source_bank.shape[0] == len(extent_levels) * rows_per_extent
        source_center = torch.as_tensor(task.cfg.cup_reset_pos[:2], device=task.device)
        source_range = torch.as_tensor(task.cfg.curriculum_randomized_source_position_range, device=task.device)
        extent_pools = task._randomized_extent_index_pools
        extent_weights = task._randomized_extent_index_weights
        source_cell_counts = task._randomized_extent_source_cell_counts
        minimum_variant_counts = task._randomized_extent_minimum_variant_counts
        assert len(extent_pools) == len(extent_levels)
        assert len(extent_weights) == len(extent_pools)
        assert len(source_cell_counts) == len(extent_pools)
        assert len(minimum_variant_counts) == len(extent_pools)
        collision_free_counts = task._randomized_collision_free_candidate_count_t
        assert collision_free_counts.shape == (source_bank.shape[0],)
        joint6_within_open_branch = arm_bank[:, 5] <= task.cfg.curriculum_randomized_reset_joint6_max
        assert all(0 < pool.numel() <= rows_per_extent for pool in extent_pools)
        assert torch.unique(torch.cat(extent_pools)).numel() == sum(pool.numel() for pool in extent_pools)
        target_center = torch.as_tensor(task.cfg.curriculum_randomized_target_center_xy, device=task.device)
        target_range = torch.as_tensor(task.cfg.curriculum_randomized_target_position_range, device=task.device)
        tcp_standoff = torch.as_tensor(task.cfg.curriculum_randomized_reset_tcp_standoff, device=task.device)
        reset_offset_lower = torch.as_tensor(
            task.cfg.curriculum_randomized_reset_tcp_offset_lower,
            device=task.device,
        )
        reset_offset_upper = torch.as_tensor(
            task.cfg.curriculum_randomized_reset_tcp_offset_upper,
            device=task.device,
        )
        grasp_points = source_bank.clone()
        grasp_points[:, 2] += task.cfg.cup_grasp_height
        tcp_displacements = tcp_start_bank - grasp_points
        cos_yaw = torch.cos(source_yaw_bank)
        sin_yaw = torch.sin(source_yaw_bank)
        tcp_displacements_c = torch.stack(
            (
                cos_yaw * tcp_displacements[:, 0] + sin_yaw * tcp_displacements[:, 1],
                -sin_yaw * tcp_displacements[:, 0] + cos_yaw * tcp_displacements[:, 1],
                tcp_displacements[:, 2],
            ),
            dim=-1,
        )
        tcp_jitter = tcp_displacements_c - tcp_standoff
        grasp_tcp_quat_c = torch.as_tensor(task.cfg.cup_grasp_tcp_quat_c, device=task.device)
        aligned_tcp_quat_bank = math_utils.quat_mul(
            source_quat_bank,
            grasp_tcp_quat_c.expand(source_quat_bank.shape[0], -1),
        )
        tcp_rotation_angles = torch.linalg.vector_norm(tcp_rotation_vector_bank, dim=-1)
        tcp_quat_error = math_utils.quat_mul(
            math_utils.quat_conjugate(aligned_tcp_quat_bank),
            tcp_start_quat_bank,
        )
        tcp_quat_error_angles = torch.linalg.vector_norm(math_utils.axis_angle_from_quat(tcp_quat_error), dim=-1)
        torch.testing.assert_close(tcp_quat_error_angles, tcp_rotation_angles, rtol=0.0, atol=1.0e-5)
        samples_per_source = task.cfg.curriculum_randomized_reset_ik_samples_per_source
        grid_size = task.cfg.curriculum_randomized_reset_ik_grid_size
        nominal_source = torch.as_tensor(task.cfg.cup_reset_pos, device=task.device)
        full_source_cells = polar_workspace_cells(
            nominal_source,
            radius_range=task.cfg.curriculum_randomized_source_radius_range,
            azimuth_half_range=task.cfg.curriculum_randomized_source_azimuth_range,
            grid_size=grid_size,
        )
        full_reset_offsets = asymmetric_reset_offset_samples(
            reset_offset_lower,
            reset_offset_upper,
            samples_per_source,
        )
        nominal_source_cell = int(torch.argmin(torch.linalg.vector_norm(full_source_cells - nominal_source, dim=-1)))
        zero_jitter_slot = int(torch.argmin(torch.linalg.vector_norm(full_reset_offsets, dim=-1)))
        zero_bank_row = nominal_source_cell * samples_per_source + zero_jitter_slot
        minimum_source_cells = math.ceil(grid_size**2 * task.cfg.curriculum_randomized_min_source_cell_fraction)
        source_outer_half_x = task.cfg.source_cup_inner_width / 2.0 + task.cfg.source_cup_wall_thickness
        source_outer_half_y = task.cfg.source_cup_inner_depth / 2.0 + task.cfg.source_cup_wall_thickness
        target_outer_half_y = task.cfg.target_cup_inner_depth / 2.0 + task.cfg.target_cup_wall_thickness
        for level, (extent, pool, pool_weights) in enumerate(
            zip(extent_levels, extent_pools, extent_weights, strict=True)
        ):
            level_indices = torch.arange(
                level * rows_per_extent,
                (level + 1) * rows_per_extent,
                device=task.device,
            )
            all_source_cell_ids = torch.arange(rows_per_extent, device=task.device) // samples_per_source
            branch_continuous = (
                task._randomized_reference_branch_delta_t[level_indices]
                <= task._randomized_reference_branch_limit_t[level_indices]
            )
            raw_valid = (
                (collision_free_counts[level_indices] > 0)
                & joint6_within_open_branch[level_indices]
                & branch_continuous
            )
            raw_variants_per_cell = torch.bincount(
                all_source_cell_ids[raw_valid],
                minlength=grid_size**2,
            )
            retained_cell = raw_variants_per_cell >= task.cfg.curriculum_randomized_min_reset_variants_per_source
            if extent == 0.0:
                expected_pool = torch.as_tensor((zero_bank_row,), device=task.device)
            else:
                eligible = raw_valid & retained_cell[all_source_cell_ids]
                variant_limit = max(1, math.ceil(extent * samples_per_source))
                score_by_cell = torch.where(
                    eligible,
                    task._randomized_reference_branch_delta_t[level_indices],
                    torch.full_like(task._randomized_reference_branch_delta_t[level_indices], torch.inf),
                ).reshape(grid_size**2, samples_per_source)
                ordered_slots = torch.argsort(score_by_cell, dim=1, stable=True)
                selected_by_cell = torch.zeros_like(score_by_cell, dtype=torch.bool)
                selected_by_cell.scatter_(1, ordered_slots[:, :variant_limit], True)
                expected_pool = level_indices[eligible & selected_by_cell.reshape(-1)]
            torch.testing.assert_close(pool, expected_pool)
            assert bool(torch.all(collision_free_counts[pool] > 0))
            assert bool(torch.all(joint6_within_open_branch[pool]))
            assert pool_weights.shape == pool.shape
            assert pool_weights.device == pool.device
            assert bool(torch.all(torch.isfinite(pool_weights) & (pool_weights > 0.0)))
            source_cell_ids = (pool - level * rows_per_extent) // samples_per_source
            unique_cells, inverse_cells = torch.unique(source_cell_ids, sorted=True, return_inverse=True)
            total_weight_per_cell = torch.zeros(unique_cells.numel(), device=task.device).scatter_add_(
                0,
                inverse_cells,
                pool_weights,
            )
            torch.testing.assert_close(
                total_weight_per_cell,
                torch.ones_like(total_weight_per_cell),
                rtol=1.0e-6,
                atol=1.0e-6,
            )
            assert unique_cells.numel() == source_cell_counts[level]
            assert source_cell_counts[level] >= (1 if extent == 0.0 else minimum_source_cells)
            if level == len(extent_levels) - 1:
                radial_rings = torch.div(unique_cells, grid_size, rounding_mode="floor")
                azimuth_slots = unique_cells.remainder(grid_size)
                torch.testing.assert_close(
                    torch.unique(radial_rings, sorted=True),
                    torch.arange(grid_size, device=task.device),
                )
                assert int(azimuth_slots.amin()) < grid_size // 2
                assert int(azimuth_slots.amax()) > grid_size // 2
                assert int(azimuth_slots.amax() - azimuth_slots.amin()) >= grid_size // 2
            rows_per_cell = torch.bincount(inverse_cells, minlength=unique_cells.numel())
            assert int(rows_per_cell.min()) == minimum_variant_counts[level]
            expected_minimum_variants = min(
                task.cfg.curriculum_randomized_min_reset_variants_per_source,
                max(1, math.ceil(extent * samples_per_source)),
            )
            assert minimum_variant_counts[level] >= (1 if extent == 0.0 else expected_minimum_variants)

            eligible_arm_q = arm_bank[pool]
            if extent == 0.0:
                torch.testing.assert_close(
                    eligible_arm_q,
                    eligible_arm_q[:1].expand_as(eligible_arm_q),
                    rtol=0.0,
                    atol=0.0,
                )
            else:
                arm_span = eligible_arm_q.amax(dim=0) - eligible_arm_q.amin(dim=0)
                assert int((arm_span > 1.0e-3).sum()) >= 4
                arm_rank = int(torch.linalg.matrix_rank(eligible_arm_q - eligible_arm_q.mean(dim=0), tol=1.0e-4))
                assert arm_rank >= 3
                if level == len(extent_levels) - 1:
                    # The final task must contain genuinely different whole-arm postures, not only
                    # Cartesian cup motion with numerically distinct IK rows. The validated default
                    # bank spans 0.7--1.7 rad over all seven joints; keep a conservative margin for
                    # solver/platform variation while still enforcing substantial diversity.
                    assert int((arm_span > 0.5).sum()) >= 6
                    assert arm_rank == eligible_arm_q.shape[1]
                quantized_arm_q = torch.round(eligible_arm_q * 1.0e4).to(dtype=torch.int64)
                assert torch.unique(quantized_arm_q, dim=0).shape[0] >= source_cell_counts[level]
                for cell_slot in range(unique_cells.numel()):
                    cell_arm_q = quantized_arm_q[inverse_cells == cell_slot]
                    assert torch.unique(cell_arm_q, dim=0).shape[0] >= expected_minimum_variants

            source_offsets = source_bank[level_indices, :2] - source_center
            expected_source_range = source_range * extent
            source_cells = source_offsets.reshape(-1, samples_per_source, 2)
            source_positions_by_cell = source_bank[level_indices, :2].reshape(-1, samples_per_source, 2)
            torch.testing.assert_close(
                source_cells,
                source_cells[:, :1].expand_as(source_cells),
                rtol=0.0,
                atol=0.0,
            )
            expected_cells = nominal_source + extent * (full_source_cells - nominal_source)
            torch.testing.assert_close(source_cells[:, 0], expected_cells[:, :2] - source_center, rtol=0.0, atol=1.0e-6)
            assert bool(torch.all(torch.abs(source_offsets) <= expected_source_range + 1.0e-6))

            nominal_source_rows = torch.isclose(
                source_bank[level_indices, :2], source_center, atol=1.0e-7, rtol=0.0
            ).all(dim=-1)
            nominal_source_yaws = source_yaw_bank[level_indices][nominal_source_rows]
            expected_nominal_source_rows = rows_per_extent if extent == 0.0 else samples_per_source
            assert nominal_source_yaws.shape == (expected_nominal_source_rows,)
            expected_yaw_range = task.cfg.curriculum_randomized_source_yaw_range * extent
            assert float(nominal_source_yaws.amin()) == pytest.approx(-expected_yaw_range)
            assert float(nominal_source_yaws.amax()) == pytest.approx(expected_yaw_range)

            expected_target_range = target_range * extent
            assert bool(
                torch.all(torch.abs(target_bank[level_indices, :2] - target_center) <= expected_target_range + 1.0e-6)
            )
            target_positions_by_cell = target_bank[level_indices, :2].reshape(
                grid_size**2,
                samples_per_source,
                2,
            )
            # Receiver X is cell-local. Receiver Y may vary across source-yaw variants because the
            # exact rotated source support changes the minimum collision-free cup separation.
            torch.testing.assert_close(
                target_positions_by_cell[:, :, 0],
                target_positions_by_cell[:, :1, 0].expand_as(target_positions_by_cell[:, :, 0]),
                rtol=0.0,
                atol=0.0,
            )
            expected_unique_targets = 1 if extent == 0.0 else grid_size**2
            assert torch.unique(target_positions_by_cell[:, 0, 0]).numel() == expected_unique_targets
            cell_index = torch.arange(grid_size**2, device=task.device, dtype=target_bank.dtype)
            center_cell = torch.argmin(torch.linalg.vector_norm(full_source_cells - nominal_source, dim=-1)).to(
                dtype=target_bank.dtype
            )
            target_unit_x = torch.remainder((cell_index - center_cell) * 0.754877666 + 0.5, 1.0)
            target_unit_y = torch.remainder((cell_index - center_cell) * 0.569840296 + 0.5, 1.0)
            expected_target_x = (
                target_center[0] - expected_target_range[0] + 2.0 * expected_target_range[0] * target_unit_x
            )
            torch.testing.assert_close(
                target_positions_by_cell[:, 0, 0],
                expected_target_x,
                rtol=0.0,
                atol=1.0e-6,
            )
            level_minimum_y_separation = (
                source_outer_half_x * torch.abs(torch.sin(source_yaw_bank[level_indices]))
                + source_outer_half_y * torch.abs(torch.cos(source_yaw_bank[level_indices]))
                + target_outer_half_y
                + task.cfg.curriculum_randomized_cup_clearance
            )
            allowed_target_y_upper = torch.minimum(
                target_center[1] + expected_target_range[1],
                source_bank[level_indices, 1] - level_minimum_y_separation,
            )
            target_y_lower = target_center[1] - expected_target_range[1]
            expected_target_y = target_y_lower + target_unit_y.repeat_interleave(samples_per_source) * (
                allowed_target_y_upper - target_y_lower
            )
            torch.testing.assert_close(
                target_bank[level_indices, 1],
                expected_target_y,
                rtol=0.0,
                atol=1.0e-6,
            )
            expected_offsets = (full_reset_offsets * extent).repeat(grid_size**2, 1)
            torch.testing.assert_close(
                tcp_jitter[level_indices],
                expected_offsets,
                rtol=0.0,
                atol=1.0e-6,
            )
            assert bool(torch.all(tcp_jitter[level_indices] >= reset_offset_lower * extent - 1.0e-6))
            assert bool(torch.all(tcp_jitter[level_indices] <= reset_offset_upper * extent + 1.0e-6))

            level_rotation_angles = tcp_rotation_angles[level_indices]
            rotation_lower, rotation_upper = task.cfg.curriculum_randomized_reset_tcp_rotation_angle_range
            if extent == 0.0:
                torch.testing.assert_close(
                    level_rotation_angles,
                    torch.zeros_like(level_rotation_angles),
                    rtol=0.0,
                    atol=0.0,
                )
            else:
                assert float(level_rotation_angles.amin()) == pytest.approx(rotation_lower * extent, abs=1.0e-6)
                assert float(level_rotation_angles.amax()) == pytest.approx(rotation_upper * extent, abs=1.0e-6)

            yaw_by_source = source_yaw_bank[level_indices].reshape(
                grid_size**2,
                samples_per_source,
            )
            source_bearing = torch.atan2(source_positions_by_cell[:, 0, 1], source_positions_by_cell[:, 0, 0])
            local_yaw_by_source = yaw_by_source - source_bearing.unsqueeze(-1)
            expected_yaw_marginal = torch.sort(local_yaw_by_source[0]).values.expand_as(local_yaw_by_source)
            torch.testing.assert_close(torch.sort(local_yaw_by_source, dim=-1).values, expected_yaw_marginal)
            assert float(local_yaw_by_source.amin()) == pytest.approx(-expected_yaw_range)
            assert float(local_yaw_by_source.amax()) == pytest.approx(expected_yaw_range)

        assert bool(torch.all(task.curriculum_stage == randomized_stage))
        torch.testing.assert_close(
            torch.linalg.vector_norm(source_quat_bank, dim=-1),
            torch.ones(source_bank.shape[0], device=task.device),
            rtol=0.0,
            atol=1.0e-6,
        )
        expected_source_quat = torch.zeros_like(source_quat_bank)
        expected_source_quat[:, 2] = torch.sin(0.5 * source_yaw_bank)
        expected_source_quat[:, 3] = torch.cos(0.5 * source_yaw_bank)
        torch.testing.assert_close(source_quat_bank, expected_source_quat, rtol=0.0, atol=1.0e-6)

        final_pool = extent_pools[-1]
        final_rotation_angles = tcp_rotation_angles[final_pool]
        assert float(final_rotation_angles.amin()) >= (
            task.cfg.curriculum_randomized_reset_tcp_rotation_angle_range[0] - 1.0e-6
        )
        assert int(torch.linalg.matrix_rank(tcp_rotation_vector_bank[final_pool], tol=1.0e-5)) == 3
        final_tcp_jitter = tcp_jitter[final_pool]
        final_tcp_jitter_span = final_tcp_jitter.amax(dim=0) - final_tcp_jitter.amin(dim=0)
        assert bool(torch.all(final_tcp_jitter_span > torch.tensor((0.04, 0.08, 0.06), device=task.device)))

        assert bool(
            torch.all(
                torch.linalg.vector_norm(tcp_displacements, dim=-1)
                >= task.cfg.curriculum_randomized_reset_tcp_min_grasp_distance - 1.0e-6
            )
        )
        assert bool(torch.all(task._reach_source_yaw_bank_t == 0.0))
        reach_count = task._reach_arm_q_bank_t.shape[0]
        assert task._reach_tcp_pos_bank_t.shape == (reach_count, 3)
        assert task._reach_source_yaw_bank_t.shape == (reach_count,)
        assert task._reach_pregrasp_arm_q_bank_t.shape == task._reach_arm_q_bank_t.shape
        assert task._reach_midgrasp_arm_q_bank_t.shape == task._reach_arm_q_bank_t.shape
        assert task._reach_grasp_arm_q_bank_t.shape == task._reach_arm_q_bank_t.shape
        assert task._reach_reset_ik_cost_t.shape == (reach_count,)
        assert task._reach_reset_ik_margin_t.shape == (reach_count,)
        assert task._reach_collision_free_candidate_count_t.shape == (reach_count,)
        assert bool(torch.all(task._reach_collision_free_candidate_count_t > 0))
        assert task.cfg.curriculum_randomized_min_reset_variants_per_source <= reach_count <= samples_per_source
        assert bool(torch.all(task._reach_reset_ik_cost_t <= task.cfg.curriculum_randomized_reset_ik_max_cost))
        assert bool(torch.all(task._reach_reset_ik_margin_t >= task.cfg.curriculum_randomized_reset_ik_joint_margin))
        quantized_reach_arm_q = torch.round(task._reach_arm_q_bank_t * 1.0e4).to(dtype=torch.int64)
        assert torch.unique(quantized_reach_arm_q, dim=0).shape[0] >= (
            task.cfg.curriculum_randomized_min_reset_variants_per_source
        )

        minimum_y_separation = (
            source_outer_half_x * torch.abs(torch.sin(source_yaw_bank))
            + source_outer_half_y * torch.abs(torch.cos(source_yaw_bank))
            + target_outer_half_y
            + task.cfg.curriculum_randomized_cup_clearance
        )
        assert bool(torch.all(source_bank[:, 1] - target_bank[:, 1] >= minimum_y_separation - 1.0e-6))
        waypoint_costs = (
            task._randomized_reset_ik_cost_t,
            task._randomized_pregrasp_ik_cost_t,
            task._randomized_midgrasp_ik_cost_t,
            task._randomized_grasp_ik_cost_t,
            task._randomized_carry_ik_cost_t,
            task._randomized_pour_ik_cost_t,
            task._randomized_tilt_ik_cost_t,
        )
        waypoint_margins = (
            task._randomized_reset_ik_margin_t,
            task._randomized_pregrasp_ik_margin_t,
            task._randomized_midgrasp_ik_margin_t,
            task._randomized_grasp_ik_margin_t,
            task._randomized_carry_ik_margin_t,
            task._randomized_pour_ik_margin_t,
            task._randomized_tilt_ik_margin_t,
        )
        assert all(cost.shape == (source_bank.shape[0],) for cost in waypoint_costs)
        assert all(margin.shape == (source_bank.shape[0],) for margin in waypoint_margins)
        eligible_indices = torch.cat(extent_pools)
        assert all(
            bool(torch.all(cost[eligible_indices] <= task.cfg.curriculum_randomized_reset_ik_max_cost))
            for cost in waypoint_costs
        )
        assert all(
            bool(torch.all(margin[eligible_indices] >= task.cfg.curriculum_randomized_reset_ik_joint_margin))
            for margin in waypoint_margins
        )
        waypoint_bank = torch.stack(
            (
                task._randomized_arm_q_bank_t,
                task._randomized_pregrasp_arm_q_bank_t,
                task._randomized_midgrasp_arm_q_bank_t,
                task._randomized_grasp_arm_q_bank_t,
                task._randomized_carry_arm_q_bank_t,
                task._randomized_pour_arm_q_bank_t,
                task._randomized_tilt_arm_q_bank_t,
            ),
            dim=1,
        )
        joint_limits = torch.as_tensor(PANDA_ARM_JOINT_LIMITS, device=task.device)
        joint_lower = joint_limits[:, 0]
        joint_upper = joint_limits[:, 1]
        eligible_waypoint_bank = waypoint_bank[eligible_indices]
        assert bool(torch.isfinite(eligible_waypoint_bank).all())
        assert bool(torch.all(eligible_waypoint_bank >= joint_lower))
        assert bool(torch.all(eligible_waypoint_bank <= joint_upper))

        # The zero-amplitude frontier is the exact behavioral continuation of the mastered full
        # task: source/receiver geometry and zero-jitter insertion path are nominal, and the
        # transport/pour tail stays on the authored safe branch.
        assert extent_levels[0] == 0.0
        zero_pool = extent_pools[0]
        torch.testing.assert_close(
            source_bank[zero_pool],
            nominal_source.expand(zero_pool.numel(), -1),
            rtol=0.0,
            atol=0.0,
        )
        nominal_target = torch.as_tensor(task.cfg.target_cup_reset_pos, device=task.device)
        torch.testing.assert_close(
            target_bank[zero_pool],
            nominal_target.expand(zero_pool.numel(), -1),
            rtol=0.0,
            atol=0.0,
        )
        reach_waypoint_prefix = torch.stack(
            (
                task._reach_arm_q_bank_t,
                task._reach_pregrasp_arm_q_bank_t,
                task._reach_midgrasp_arm_q_bank_t,
                task._reach_grasp_arm_q_bank_t,
            ),
            dim=1,
        )
        torch.testing.assert_close(
            waypoint_bank[zero_pool, :4],
            reach_waypoint_prefix[0].expand(zero_pool.numel(), -1, -1),
            rtol=0.0,
            atol=0.0,
        )
        nominal_waypoint_tail = torch.as_tensor(
            (
                task.cfg.curriculum_carry_arm_q,
                task.cfg.curriculum_pour_arm_q,
                task.cfg.curriculum_pour_target_arm_q,
            ),
            device=task.device,
            dtype=waypoint_bank.dtype,
        )
        torch.testing.assert_close(
            waypoint_bank[zero_pool, 4:],
            nominal_waypoint_tail.expand(zero_pool.numel(), -1, -1),
            rtol=0.0,
            atol=0.0,
        )

        # Every row in the first nonzero frontier must stay on the nominal local IK branch. The
        # production filter permits 0.04 rad plus three times the physical extent, or 0.07 rad at
        # the one-percent frontier. Assert the actual configured contract rather than a loose
        # order-one bound that could admit the elbow/wrist jump this curriculum is designed to avoid.
        assert len(extent_levels) > 1 and extent_levels[1] > 0.0
        level_one_pool = extent_pools[1]
        maximum_level_one_joint_delta = torch.abs(
            waypoint_bank[level_one_pool] - waypoint_bank[zero_pool[0]].unsqueeze(0)
        ).amax()
        maximum_level_one_limit = task._randomized_reference_branch_limit_t[level_one_pool].amax()
        assert float(maximum_level_one_joint_delta) <= float(maximum_level_one_limit) + 1.0e-6

        selected = torch.tensor([0], device=task.device, dtype=torch.long)
        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        actions[:, -1] = 1.0
        fixed_robot_root = task._robot.data.root_link_pose_w.torch[0].clone()
        # The full-amplitude bank remains the physical safety census. Probe both boundaries of the
        # preceding levels as well, which verifies that level-local random slots map to their own
        # disjoint global bank rows rather than accidentally sampling the full-amplitude block.
        for level, pool in enumerate(extent_pools):
            task.set_curriculum_randomization_level(selected, level)
            local_slots = range(pool.numel()) if level == len(extent_pools) - 1 else (0, pool.numel() - 1)
            for local_slot in local_slots:
                bank_index = int(pool[local_slot])
                sampled_slot = torch.tensor([local_slot], device=task.device, dtype=torch.long)
                with mock.patch.object(torch, "multinomial", return_value=sampled_slot):
                    task.reset_pour_scene(selected)
                torch.testing.assert_close(
                    task._robot.data.root_link_pose_w.torch[0],
                    fixed_robot_root,
                    rtol=0.0,
                    atol=0.0,
                )
                achieved_tcp_pose = task.tcp_pose_e()[0]
                arm_index = int(task._last_arm_bank_index[0])
                assert int(task._last_source_bank_index[0]) == bank_index
                expected_tcp_position = tcp_start_bank[arm_index]
                tcp_position_error = torch.linalg.vector_norm(achieved_tcp_pose[:3] - expected_tcp_position)
                tcp_quat_alignment = torch.abs(torch.dot(achieved_tcp_pose[3:7], tcp_start_quat_bank[arm_index]))
                assert float(tcp_position_error) < 0.005, (
                    f"Reset bank row {bank_index} TCP position error is {float(tcp_position_error):.6g} m."
                )
                assert float(tcp_quat_alignment) > 1.0 - 1.0e-5, (
                    f"Reset bank row {bank_index} TCP quaternion alignment is {float(tcp_quat_alignment):.6g}."
                )
                _, _, terminated, truncated, _ = env.step(actions)
                assert not bool(terminated[0]), f"Reset bank row {bank_index} terminated after one step."
                assert not bool(truncated[0]), f"Reset bank row {bank_index} timed out after one step."
                assert bool(torch.all(task.state_finite())), f"Reset bank row {bank_index} produced non-finite state."
                assert bool(torch.all(task.rigid_state_in_bounds())), (
                    f"Reset bank row {bank_index} exceeded rigid bounds."
                )
                assert bool(torch.all(task.particles_in_workspace())), f"Reset bank row {bank_index} lost particles."
                torch.testing.assert_close(
                    task._robot.data.root_link_pose_w.torch[0],
                    fixed_robot_root,
                    rtol=0.0,
                    atol=0.0,
                )
        env.reset()

        arm_action = task.action_manager.get_term("arm_action")
        world_0_root = task._robot.data.root_link_pose_w.torch[0].clone()
        world_1_q = task._robot.data.joint_pos.torch[1].clone()
        world_1_root = task._robot.data.root_link_pose_w.torch[1].clone()
        world_1_source = task._source_cup.data.root_link_pose_w.torch[1].clone()
        world_1_target = task._target_cup.data.root_link_pose_w.torch[1].clone()
        world_1_media = task._media.data.particle_pos_w.torch[1].clone()
        world_1_raw_action = arm_action.raw_actions[1].clone()
        world_1_command = arm_action.processed_actions[1].clone()

        task.set_curriculum_randomization_level(selected, len(extent_pools) - 1)
        bank_index = int(extent_pools[-1][-1])
        sampled_slot = torch.tensor([extent_pools[-1].numel() - 1], device=task.device, dtype=torch.long)
        with mock.patch.object(torch, "multinomial", return_value=sampled_slot):
            task.reset_pour_scene(selected)
        wp.synchronize_device(NewtonManager.get_model().device)

        source_index = int(task._last_source_bank_index[0])
        arm_index = int(task._last_arm_bank_index[0])
        target_index = int(task._last_target_bank_index[0])
        assert source_index == arm_index == target_index == bank_index
        torch.testing.assert_close(task.cup_pose_e()[0, :3], source_bank[source_index], rtol=0.0, atol=1.0e-6)
        torch.testing.assert_close(task.cup_pose_e()[0, 3:7], source_quat_bank[source_index], rtol=0.0, atol=1.0e-6)
        torch.testing.assert_close(task.target_pose_e()[0, :3], target_bank[target_index], rtol=0.0, atol=1.0e-6)
        torch.testing.assert_close(
            task.target_pose_e()[0, 3:7],
            torch.tensor((0.0, 0.0, 0.0, 1.0), device=task.device),
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            task._robot.data.joint_pos.torch[0, task._arm_joint_ids],
            arm_bank[arm_index],
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            task.gripper_width()[0],
            torch.tensor(task.gripper_open_width, device=task.device),
            rtol=0.0,
            atol=1.0e-6,
        )
        tcp_start_error = torch.linalg.vector_norm(task.tcp_pos_e()[0] - tcp_start_bank[arm_index])
        assert float(tcp_start_error) < 0.005, f"Randomized reset TCP-start error is {float(tcp_start_error):.6g} m."
        tcp_quat_alignment = torch.abs(torch.dot(task.tcp_pose_e()[0, 3:7], tcp_start_quat_bank[arm_index]))
        assert float(tcp_quat_alignment) > 1.0 - 1.0e-5
        tcp_grasp_distance = torch.linalg.vector_norm(task.tcp_pos_e()[0] - task.cup_grasp_point_e()[0])
        assert float(tcp_grasp_distance) >= task.cfg.curriculum_independent_arm_min_tcp_distance - 0.005

        torch.testing.assert_close(task._robot.data.root_link_pose_w.torch[0], world_0_root, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task._robot.data.joint_pos.torch[1], world_1_q, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task._robot.data.root_link_pose_w.torch[1], world_1_root, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task._source_cup.data.root_link_pose_w.torch[1], world_1_source, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task._target_cup.data.root_link_pose_w.torch[1], world_1_target, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task._media.data.particle_pos_w.torch[1], world_1_media, rtol=0.0, atol=0.0)
        torch.testing.assert_close(arm_action.raw_actions[1], world_1_raw_action, rtol=0.0, atol=0.0)
        torch.testing.assert_close(arm_action.processed_actions[1], world_1_command, rtol=0.0, atol=0.0)

        randomized_target = task._target_cup.data.root_link_pose_w.torch[0].clone()
        env.step(actions)
        wp.synchronize_device(NewtonManager.get_model().device)
        torch.testing.assert_close(
            task._target_cup.data.root_link_pose_w.torch[0],
            randomized_target,
            rtol=0.0,
            atol=0.0,
        )
        assert bool(torch.all(task.state_finite()))
        assert bool(torch.all(task.particles_in_workspace()))

        assert "reach" not in task.cfg.curriculum_stage_names
        approach_distances = []
        for approach_index, stage_name in enumerate(
            ("approach_1", "approach_2", "approach_3", "approach_4", "approach_5", "approach_6")
        ):
            approach_stage = task.cfg.curriculum_stage_names.index(stage_name)
            task.set_curriculum_stage(selected, approach_stage)
            task.reset_pour_scene(selected)
            wp.synchronize_device(NewtonManager.get_model().device)
            torch.testing.assert_close(
                task._robot.data.joint_pos.torch[0, task._arm_joint_ids],
                task._curriculum_approach_arm_q_t[approach_index],
                rtol=0.0,
                atol=1.0e-6,
            )
            approach_distances.append(
                float(torch.linalg.vector_norm(task.tcp_pos_e()[0] - task.cup_grasp_point_e()[0]))
            )
        assert all(
            near_distance < far_distance
            for near_distance, far_distance in zip(approach_distances, approach_distances[1:])
        )
        assert 0.005 < approach_distances[0] < task.cfg.rewards.grasp_lift_progress.params["grasp_reach_std"]
        expected_pre_grasp_distance = 0.5 * torch.linalg.vector_norm(
            torch.tensor(task.cfg.curriculum_randomized_reset_tcp_standoff, device=task.device)
        )
        torch.testing.assert_close(
            torch.tensor(approach_distances[-1], device=task.device),
            expected_pre_grasp_distance,
            rtol=0.0,
            atol=0.005,
        )

        full_stage = task.cfg.curriculum_stage_names.index("full")
        task.set_curriculum_stage(selected, full_stage)
        task.reset_pour_scene(selected)
        wp.synchronize_device(NewtonManager.get_model().device)
        torch.testing.assert_close(
            task.cup_pose_e()[0, :3],
            torch.tensor(task.cfg.cup_reset_pos, device=task.device),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            task.target_pose_e()[0, :3],
            torch.tensor(task.cfg.target_cup_reset_pos, device=task.device),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            task._robot.data.joint_pos.torch[0, task._arm_joint_ids],
            task._reach_pregrasp_arm_q_bank_t[0],
            rtol=0.0,
            atol=1.0e-6,
        )
        pregrasp_distance = torch.linalg.vector_norm(task.tcp_pos_e()[0] - task.cup_grasp_point_e()[0])
        expected_pregrasp_distance = torch.linalg.vector_norm(
            torch.tensor(task.cfg.curriculum_randomized_reset_tcp_standoff, device=task.device)
        )
        torch.testing.assert_close(pregrasp_distance, expected_pregrasp_distance, rtol=0.0, atol=0.005)
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_extreme_finite_rigid_state_resets_before_returned_observation():
    """A finite dynamics outlier must be reset before RSL-RL can normalize the next observation."""
    _require_cuda()
    sim_utils.create_new_stage()
    env = None
    try:
        env = gym.make(_TASK_ID, cfg=_make_runtime_cfg())
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        real_scene_update = task.scene.update
        arm_joint_id = task._arm_joint_ids[0]

        def update_then_inject_outlier(dt: float) -> None:
            real_scene_update(dt)
            task._robot.data.joint_vel.torch[0, arm_joint_id] = 1.0e6

        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        with mock.patch.object(task.scene, "update", side_effect=update_then_inject_outlier):
            observations, _, terminated, truncated, _ = env.step(actions)

        assert terminated.tolist() == [True, False]
        assert truncated.tolist() == [False, False]
        assert task.termination_manager.get_term("extreme_rigid_state").tolist() == [True, False]
        assert bool(torch.isfinite(observations["policy"]).all())
        assert bool(torch.isfinite(observations["privileged"]).all())
        assert bool(torch.all(torch.abs(observations["policy"][0, 7:14]) <= task.cfg.state_bound_max_joint_velocity))
        assert float(task._success_dwell_count[0]) == 0.0
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_curriculum_mixed_stage_reset_uses_direct_actions_and_stays_isolated():
    """Reset two worlds at different backward stages, command both directly, and step eagerly."""
    _require_cuda()
    sim_utils.create_new_stage()
    env = None
    try:
        env = gym.make(_TASK_ID, cfg=_make_runtime_cfg())
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        observations, info = env.reset()

        assert task.curriculum_manager.active_terms == ["stage"]
        assert set(observations) == {"policy", "privileged"}
        assert observations["policy"].shape == (2, 62)
        assert observations["privileged"].shape == (2, 20)
        assert bool(torch.isfinite(observations["policy"]).all())
        assert bool(torch.isfinite(observations["privileged"]).all())
        full_stage = task.cfg.curriculum_stage_names.index("full")
        grasp_stage = task.cfg.curriculum_stage_names.index("grasp")
        assert info["log"]["Curriculum/stage/stage"] == float(full_stage)
        assert "Curriculum/stage/success_rate" in info["log"]
        assert "Curriculum/stage/completed_episodes" in info["log"]
        assert "Curriculum/stage/required_completed_episodes" in info["log"]
        assert "Curriculum/stage/mastered" in info["log"]

        selected = torch.tensor([0], device=task.device, dtype=torch.long)
        transport_distance = []
        for stage_name in ("pour", "near_carry", "mid_carry", "carry"):
            stage_index = task.cfg.curriculum_stage_names.index(stage_name)
            task.set_curriculum_stage(selected, stage_index)
            task.reset_pour_scene(selected)
            distance = torch.linalg.vector_norm(task.cup_pose_e()[0, :2] - task.target_pose_e()[0, :2])
            transport_distance.append(float(distance))
            torch.testing.assert_close(
                task._robot.data.joint_pos.torch[0, task._arm_joint_ids],
                task._curriculum_arm_q_t[stage_index],
                rtol=0.0,
                atol=1.0e-6,
            )
        assert all(left < right for left, right in zip(transport_distance, transport_distance[1:], strict=True))

        env_ids = torch.tensor([0, 1], device=task.device, dtype=torch.long)
        task.set_curriculum_stage(torch.tensor([0], device=task.device), 0)
        task.set_curriculum_stage(torch.tensor([1], device=task.device), full_stage)
        task.reset_pour_scene(env_ids)

        arm_action = task.action_manager.get_term("arm_action")
        arm_q = task._robot.data.joint_pos.torch[:, task._arm_joint_ids]
        torch.testing.assert_close(arm_q[0], task._curriculum_arm_q_t[0], rtol=0.0, atol=1.0e-6)
        assert bool(torch.isfinite(arm_q[1]).all())
        torch.testing.assert_close(
            arm_action.processed_actions,
            torch.zeros_like(arm_action.processed_actions),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            task.gripper_width(),
            torch.tensor([task.gripper_grasp_width, task.gripper_open_width], device=task.device),
            rtol=0.0,
            atol=1.0e-6,
        )
        gripper_action = task.action_manager.get_term("gripper_action")
        torch.testing.assert_close(
            gripper_action.action_offset[:, 0],
            torch.full((2,), task.cfg.gripper_preload_pos, device=task.device),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            task.pour_target_frac,
            torch.tensor([0.05, 0.30], device=task.device),
            rtol=0.0,
            atol=0.0,
        )
        grasp_error = torch.linalg.vector_norm(task.tcp_pos_e() - task.cup_grasp_point_e(), dim=-1)
        assert float(grasp_error[0]) < 2.0e-5
        assert float(grasp_error[1]) >= task.cfg.curriculum_randomized_reset_tcp_min_grasp_distance - 0.005
        assert float(task.cup_pose_e()[0, 2]) > 0.16
        source_target_distance = torch.linalg.vector_norm(task.cup_pose_e()[0, :2] - task.target_pose_e()[0, :2])
        assert float(source_target_distance) < task.cfg.rewards.task_progress.params["alignment_radius"]
        torch.testing.assert_close(
            task.cup_pose_e()[1, :3],
            torch.tensor(task.cfg.cup_reset_pos, device=task.device),
            rtol=0.0,
            atol=0.0,
        )

        source_pose = task._source_cup.data.root_link_pose_w.torch
        expected_media = task._sample_cup_media(source_pose[:, :3], source_pose[:, 3:7])
        torch.testing.assert_close(task._media.data.particle_pos_w.torch, expected_media, rtol=0.0, atol=0.0)
        assert bool(torch.all(task.particles_in_workspace()))

        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        gripper_before = gripper_action.processed_actions.clone()
        task.action_manager.process_action(actions)
        torch.testing.assert_close(
            arm_action.processed_actions,
            torch.zeros_like(arm_action.processed_actions),
            rtol=0.0,
            atol=0.0,
        )
        gripper_target = torch.full_like(gripper_before, task.cfg.gripper_preload_pos)
        expected_gripper = torch.lerp(gripper_before, gripper_target, task.cfg.actions.gripper_action.alpha)
        torch.testing.assert_close(
            gripper_action.processed_actions,
            expected_gripper,
            rtol=0.0,
            atol=0.0,
        )

        actions[1, -1] = 1.0
        env.step(actions)
        for _ in range(9):
            env.step(actions)
        wp.synchronize_device(NewtonManager.get_model().device)
        assert bool(torch.all(task.state_finite()))
        assert bool(torch.all(task.particles_in_workspace()))
        assert float(task.cup_pose_e()[0, 2]) > 0.12
        assert float(torch.linalg.vector_norm(task.tcp_pos_e()[0] - task.cup_grasp_point_e()[0])) < 0.03

        world_1_q = task._robot.data.joint_pos.torch[1].clone()
        world_1_cup = task._source_cup.data.root_link_pose_w.torch[1].clone()
        world_1_media = task._media.data.particle_pos_w.torch[1].clone()
        world_1_raw_action = arm_action.raw_actions[1].clone()
        world_1_command = arm_action.processed_actions[1].clone()
        world_1_threshold = task.pour_target_frac[1].clone()

        selected = torch.tensor([0], device=task.device, dtype=torch.long)
        task.set_curriculum_stage(selected, 1)
        task.reset_pour_scene(selected)
        wp.synchronize_device(NewtonManager.get_model().device)

        torch.testing.assert_close(task._robot.data.joint_pos.torch[1], world_1_q, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task._source_cup.data.root_link_pose_w.torch[1], world_1_cup, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task._media.data.particle_pos_w.torch[1], world_1_media, rtol=0.0, atol=0.0)
        torch.testing.assert_close(arm_action.raw_actions[1], world_1_raw_action, rtol=0.0, atol=0.0)
        torch.testing.assert_close(arm_action.processed_actions[1], world_1_command, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task.pour_target_frac[1], world_1_threshold, rtol=0.0, atol=0.0)

        torch.testing.assert_close(
            task._robot.data.joint_pos.torch[0, task._arm_joint_ids],
            task._curriculum_arm_q_t[1],
            rtol=0.0,
            atol=1.0e-6,
        )
        assert (
            float(torch.linalg.vector_norm(task.cup_pose_e()[0, :2] - task.target_pose_e()[0, :2]))
            < (task.cfg.rewards.task_progress.params["alignment_radius"])
        )
        assert float(task.cup_pose_e()[0, 2]) > 0.16
        for _ in range(3):
            env.step(actions)
        assert bool(torch.all(task.state_finite()))
        assert float(task.cup_pose_e()[0, 2]) > 0.12

        task.set_curriculum_stage(selected, grasp_stage)
        task.reset_pour_scene(selected)
        torch.testing.assert_close(
            task._robot.data.joint_pos.torch[0, task._arm_joint_ids],
            task._reach_grasp_arm_q_bank_t[0],
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            task.cup_pose_e()[0, :3],
            torch.tensor(task.cfg.cup_reset_pos, device=task.device),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(task.gripper_width()[0], torch.tensor(task.gripper_open_width, device=task.device))
        torch.testing.assert_close(task.pour_target_frac[0], torch.tensor(0.3, device=task.device))
        for _ in range(3):
            env.step(actions)
        assert bool(torch.all(task.state_finite()))
        assert float(task.cup_pose_e()[0, 2]) > -0.02

        # Exercise the real manager lifecycle: consume grasp-stage success, advance globally, then
        # let the reset event apply the near-grasp approach and clear only the selected world's latches.
        curriculum_term = task.curriculum_manager._term_cfgs[0].func
        curriculum_term.stage = grasp_stage
        curriculum_term.success_rate = 0.0
        curriculum_term.resets_in_stage = 0
        task.cfg.curriculum_freeze = False
        task.cfg.curriculum_min_resets_per_stage = 1
        task.cfg.curriculum_min_reset_cohorts_per_stage = 0.0
        task.cfg.curriculum_previous_stage_replay_fraction = 0.0
        task.cfg.curriculum_success_threshold = 0.5
        task.episode_length_buf[0] = 10
        task.episode_succeeded[0] = True
        task.ep_max_target_frac[0] = 0.4
        task.episode_succeeded[1] = True
        task.ep_max_target_frac[1] = 0.25
        delivered_term = task.reward_manager.get_term_cfg("delivered").func
        task._target_entry_seen[:, 0] = True
        task._held_delivered[:, 0] = True
        delivered_term._previous_credit[:] = 0.1
        world_1_target_entry_seen = task._target_entry_seen[1].clone()
        world_1_held_delivered = task._held_delivered[1].clone()
        world_1_delivery_credit = delivered_term._previous_credit[1].clone()

        task._reset_idx(selected)

        first_approach_stage = task.cfg.curriculum_stage_names.index("approach_1")
        assert curriculum_term.stage == first_approach_stage
        assert int(task.curriculum_stage[0]) == first_approach_stage
        assert not bool(task.episode_succeeded[0])
        assert float(task.ep_max_target_frac[0]) == 0.0
        assert not bool(torch.any(task._target_entry_seen[0]))
        assert not bool(torch.any(task._held_delivered[0]))
        assert float(delivered_term._previous_credit[0]) == 0.0
        assert bool(task.episode_succeeded[1])
        assert float(task.ep_max_target_frac[1]) == pytest.approx(0.25)
        torch.testing.assert_close(task._target_entry_seen[1], world_1_target_entry_seen, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task._held_delivered[1], world_1_held_delivered, rtol=0.0, atol=0.0)
        torch.testing.assert_close(delivered_term._previous_credit[1], world_1_delivery_credit, rtol=0.0, atol=0.0)
        assert bool(torch.isfinite(task._robot.data.joint_pos.torch[0, task._arm_joint_ids]).all())
        torch.testing.assert_close(
            task._robot.data.joint_pos.torch[0, task._arm_joint_ids],
            task._curriculum_approach_arm_q_t[0],
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(task.gripper_width()[0], torch.tensor(0.08, device=task.device))
        torch.testing.assert_close(task.pour_target_frac[0], torch.tensor(0.30, device=task.device))
        assert task.extras["log"]["Curriculum/stage/stage"] == float(first_approach_stage)
        env.step(actions)
        assert bool(torch.all(task.state_finite()))
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_scene_owned_cups_use_public_state_and_leave_caller_cfg_unmodified():
    """The task resolves cloned cup assets without mutating its caller-owned config."""
    _require_cuda()
    sim_utils.create_new_stage()
    env = None
    caller_cfg = _make_runtime_cfg(use_cuda_graph=False, env_spacing=2.5)
    # Keep public views on the authoritative state after the eager step below.
    caller_cfg.physics_substeps = 2
    assert caller_cfg.scene.source_cup is None
    assert caller_cfg.scene.target_cup is None
    assert caller_cfg.scene.media is None
    try:
        env = gym.make(_TASK_ID, cfg=caller_cfg)
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        assert caller_cfg.scene.source_cup is None
        assert caller_cfg.scene.target_cup is None
        assert caller_cfg.scene.media is None
        assert task.cfg is not caller_cfg

        source_cup = task.scene["source_cup"]
        target_cup = task.scene["target_cup"]
        assert source_cup.num_instances == 2
        assert target_cup.num_instances == 2

        source_pose_e = source_cup.data.root_link_pose_w.torch.clone()
        source_pose_e[:, :3] -= task.scene.env_origins
        target_pose_e = target_cup.data.root_link_pose_w.torch.clone()
        target_pose_e[:, :3] -= task.scene.env_origins
        torch.testing.assert_close(task.cup_pose_e(), source_pose_e)
        torch.testing.assert_close(task.target_pose_e(), target_pose_e)
        _assert_scene_solver_roles(NewtonManager.get_model())
        assert NewtonManager.get_model().particle_max_velocity == pytest.approx(task.cfg.particle_max_velocity)

        for legacy_attribute in (
            "_cup_body_ids",
            "_target_body_ids",
            "_cup_joint_q",
            "_cup_joint_qd",
            "_cup_articulation_ids",
        ):
            assert not hasattr(task, legacy_attribute)

        target_pose_before = target_cup.data.root_link_pose_w.torch.clone()
        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        actions[:, -1] = 1.0
        env.step(actions)
        wp.synchronize_device(NewtonManager.get_model().device)
        torch.testing.assert_close(
            target_cup.data.root_link_pose_w.torch,
            target_pose_before,
            rtol=0.0,
            atol=0.0,
        )
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_offset_world_tables_support_both_cups_without_reset():
    """Supported cups retain their media beyond the former delayed-leak horizon."""
    _require_cuda()
    sim_utils.create_new_stage()
    env = None
    try:
        runtime_cfg = _make_runtime_cfg(use_cuda_graph=False, env_spacing=2.5, mpm_iterations=24)
        runtime_cfg.physics_substeps = 2
        env = gym.make(_TASK_ID, cfg=runtime_cfg)
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        entries = {entry.name: entry for entry in task.cfg.sim.physics.solver_cfg.entries}
        assert entries[MPM_ENTRY].solver_cfg.project_outside_colliders is False
        assert task.num_envs == 2
        assert not torch.equal(task.scene.env_origins[0], task.scene.env_origins[1])
        torch.testing.assert_close(task.cup_pose_e()[:, 2], torch.zeros(2, device=task.device), rtol=0.0, atol=0.0)

        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        actions[:, -1] = 1.0  # keep the gripper open so this tests table support only
        for step in range(180):
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            assert not bool(torch.any(done)), (
                f"A resting environment reset at step {step}: "
                f"terminated={terminated.tolist()}, truncated={truncated.tolist()}"
            )

        wp.synchronize_device(NewtonManager.get_model().device)
        cup_z = task.cup_pose_e()[:, 2]
        assert bool(torch.all(cup_z > -0.02)), f"A source cup fell through its local table: z={cup_z.tolist()}"
        expected_count = torch.full_like(task.count_in_source(), task.particle_pos_e().shape[1])
        torch.testing.assert_close(task.count_in_source(), expected_count, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task.count_spilled(), torch.zeros_like(expected_count), rtol=0.0, atol=0.0)
        torch.testing.assert_close(cup_z[0], cup_z[1], rtol=0.0, atol=2.0e-3)
        source_fraction = task.count_in_source() / float(task.num_particles)
        assert bool(torch.all(source_fraction >= 0.99)), (
            f"PIC27 failed to retain resting media in the source cup: fraction={source_fraction.tolist()}"
        )
    finally:
        if env is not None:
            env.close()
