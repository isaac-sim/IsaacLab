# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for conveyor-transfer state, curriculum, and success geometry."""

from collections import Counter
from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.conveyor_franka.agents.rsl_rl_ppo_cfg import (
    ConveyorFrankaPPORunnerCfg,
    ConveyorGaussianBernoulliDistribution,
)
from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import ConveyorFrankaEnvCfg
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import (
    BELT_CENTER_X,
    BELT_INNER_STRAIGHT_Y,
    BELT_OUTER_STRAIGHT_Y,
)
from isaaclab_tasks.contrib.conveyor_franka.mdp.actions import (
    ConveyorRelativeJointPositionAction,
    ResetBufferedGripperAction,
)
from isaaclab_tasks.contrib.conveyor_franka.mdp.commands import ConveyorTransferCommand, transfer_success_mask
from isaaclab_tasks.contrib.conveyor_franka.mdp.curriculums import (
    deployment_probability_from_progress,
    reset_sampling_probabilities,
)
from isaaclab_tasks.contrib.conveyor_franka.mdp.observations import classify_cube_conveyors
from isaaclab_tasks.contrib.conveyor_franka.mdp.reset_events import (
    CUBE_COUNT,
    ConveyorResetRecipe,
    _balanced_cube_slots,
    _sample_collision_free_active_x,
    build_reset_rows,
    franka_tool_position,
    reset_variant_counts,
    select_next_transfer_cube,
)
from isaaclab_tasks.contrib.conveyor_franka.mdp.rewards import transfer_potential
from isaaclab_tasks.contrib.conveyor_franka.mdp.terminations import (
    invalid_action,
    subgoal_time_out,
    transfer_sequence_time_out,
)


def _make_arm_action_term() -> ConveyorRelativeJointPositionAction:
    """Build the tensor-only portion of the arm action term without a simulator."""
    workspace_lower = torch.tensor((-0.75, -0.45, -0.55, -2.75, -0.45, 1.85, -0.10))
    workspace_upper = torch.tensor((0.85, 0.85, 0.35, -1.75, 0.45, 3.05, 1.65))
    positions = ((workspace_lower + workspace_upper) * 0.5).repeat(2, 1)
    limits = torch.tensor((-4.0, 4.0)).repeat(2, 7, 1)
    action = object.__new__(ConveyorRelativeJointPositionAction)
    action.cfg = SimpleNamespace(max_delta=0.12, joint_limit_margin=0.02, clip=None)
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=positions),
            soft_joint_pos_limits=SimpleNamespace(torch=limits),
        )
    )
    action._joint_ids = slice(None)
    action._scale = 0.12
    action._offset = 0.0
    action._workspace_lower = workspace_lower
    action._workspace_upper = workspace_upper
    action._raw_actions = torch.zeros((2, 7))
    action._processed_actions = torch.zeros((2, 7))
    action._position_targets = positions.clone()
    action._invalid_actions = torch.zeros(2, dtype=torch.bool)
    return action


def _make_gripper_action_term() -> ResetBufferedGripperAction:
    """Build the tensor-only portion of the binary gripper action term."""
    action = object.__new__(ResetBufferedGripperAction)
    action.cfg = SimpleNamespace(clip=None, command_name="transfer", force_close_steps=2)
    action._raw_actions = torch.zeros((2, 1))
    action._processed_actions = torch.zeros((2, 2))
    action._open_command = torch.full((2,), 0.04)
    action._close_command = torch.zeros(2)
    action._invalid_actions = torch.zeros(2, dtype=torch.bool)
    command = SimpleNamespace(held_cube_ids=torch.full((2,), -1, dtype=torch.long))
    action._env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        episode_length_buf=torch.zeros(2, dtype=torch.long),
    )
    return action


def test_arm_action_sanitizes_nonfinite_values_and_clamps_normalized_input():
    """Invalid policy outputs cannot reach joint targets or exceed one normalized unit."""
    action = _make_arm_action_term()
    policy_actions = torch.tensor(((float("nan"), float("inf"), -float("inf"), 5.0, -5.0, 0.5, -0.5), (-2.0,) * 7))

    action.process_actions(policy_actions)

    expected = torch.tensor(((0.0, 1.0, -1.0, 1.0, -1.0, 0.5, -0.5), (-1.0,) * 7))
    torch.testing.assert_close(action.raw_actions, expected)
    assert action.invalid_actions.tolist() == [True, False]
    assert torch.isfinite(action.processed_actions).all()
    expected_targets = action._asset.data.joint_pos.torch + expected * 0.12
    torch.testing.assert_close(action.processed_actions, expected_targets)


def test_gripper_action_sanitizes_nonfinite_values_before_binary_mapping():
    """The eighth policy dimension cannot silently turn a NaN into an open command."""
    action = _make_gripper_action_term()

    action.process_actions(torch.tensor(((float("nan"),), (float("inf"),))))

    torch.testing.assert_close(action.raw_actions, torch.tensor(((-1.0,), (1.0,))))
    assert action.invalid_actions.tolist() == [True, True]
    torch.testing.assert_close(action.processed_actions[0], action._close_command)
    torch.testing.assert_close(action.processed_actions[1], action._open_command)


def test_invalid_action_termination_and_reset_are_per_environment():
    """Arm and gripper failures aggregate per environment, and reset clears the arm flag."""
    arm_action = _make_arm_action_term()
    gripper_action = _make_gripper_action_term()
    arm_action._invalid_actions[:] = torch.tensor((True, False))
    gripper_action._invalid_actions[:] = torch.tensor((False, True))
    actions = {"arm_action": arm_action, "gripper_action": gripper_action}
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=actions.__getitem__))

    assert invalid_action(env).tolist() == [True, True]
    arm_action.reset([0])
    gripper_action._invalid_actions[1] = False

    assert invalid_action(env).tolist() == [False, False]


def test_final_config_validation_catches_overridden_arm_contracts():
    """Top-level validation runs after overrides and protects workspace-to-joint alignment."""
    cfg = ConveyorFrankaEnvCfg()
    assert cfg.seed is None
    cfg.validate()

    cfg.actions.arm_action.preserve_order = False
    try:
        cfg.validate()
    except ValueError as exc:
        assert "preserve" in str(exc)
    else:
        raise AssertionError("Expected invalid arm ordering to fail configuration validation.")


def test_production_solver_and_viewer_defaults_are_bounded():
    """The task keeps CUDA graphs enabled and avoids scene-wide over-allocation or rendering."""
    cfg = ConveyorFrankaEnvCfg()
    physics = cfg.sim.physics
    solver = physics.solver_cfg

    assert cfg.conveyor_force.speed == 0.35
    assert physics.use_cuda_graph is True
    assert physics.load_visual_shapes is None
    assert solver.njmax == 300
    assert solver.nconmax == 200
    assert solver.impratio == 1.0
    assert cfg.sim.default_visualizer_cfg.max_visible_envs == 1
    assert cfg.sim.default_visualizer_cfg.randomly_sample_visible_envs is False


def test_reset_rows_cover_every_cube_direction_and_phase_once():
    """The reset bank is the complete command and physical-phase cross product."""
    rows = build_reset_rows()

    assert len(rows) == sum(reset_variant_counts()) * CUBE_COUNT * 2
    assert Counter((row.recipe, row.variant_id, row.target_cube_id, row.source_side_id) for row in rows) == Counter(
        (recipe, variant_id, cube_id, side_id)
        for recipe in ConveyorResetRecipe
        for variant_id in range(reset_variant_counts()[int(recipe)])
        for cube_id in range(CUBE_COUNT)
        for side_id in range(2)
    )
    for row in rows:
        expected_held = row.recipe in {
            ConveyorResetRecipe.LIFT,
            ConveyorResetRecipe.CARRY,
            ConveyorResetRecipe.PLACE,
        } or (
            row.recipe == ConveyorResetRecipe.GRASP
            and row.variant_id == reset_variant_counts()[int(ConveyorResetRecipe.GRASP)] - 1
        )
        assert row.held == expected_held

    grasp_rows = [
        row
        for row in rows
        if row.recipe == ConveyorResetRecipe.GRASP and row.target_cube_id == 0 and row.source_side_id == 0
    ]
    assert all(first.finger_position > second.finger_position for first, second in zip(grasp_rows, grasp_rows[1:]))
    assert grasp_rows[-1].held


def test_reset_arm_anchors_reach_expected_transfer_waypoints():
    """IK anchors place the tool over source, transit, or destination waypoints."""
    anchor_variants = {
        ConveyorResetRecipe.GOAL: 0,
        ConveyorResetRecipe.PLACE: 3,
        ConveyorResetRecipe.CARRY: 2,
        ConveyorResetRecipe.LIFT: 3,
        ConveyorResetRecipe.GRASP: 0,
        ConveyorResetRecipe.PREGRASP: 0,
    }
    rows = [
        row
        for row in build_reset_rows()
        if row.target_cube_id == 0 and anchor_variants.get(row.recipe) == row.variant_id
    ]
    joints = torch.tensor([row.arm_positions for row in rows], dtype=torch.float64)
    positions = franka_tool_position(joints)

    for row, position in zip(rows, positions, strict=True):
        source_y = 0.27 if row.source_side_id == 0 else -0.27
        target_y = -source_y
        if row.recipe == ConveyorResetRecipe.BELT:
            continue
        expected = {
            ConveyorResetRecipe.PREGRASP: (0.52, source_y, 0.14),
            ConveyorResetRecipe.GRASP: (0.52, source_y, 0.06),
            ConveyorResetRecipe.LIFT: (0.52, source_y, 0.22),
            ConveyorResetRecipe.CARRY: (0.52, 0.0, 0.25),
            ConveyorResetRecipe.PLACE: (0.52, target_y, 0.105),
            ConveyorResetRecipe.GOAL: (0.52, target_y, 0.14),
        }[row.recipe]
        torch.testing.assert_close(position, torch.tensor(expected, dtype=position.dtype), atol=3.0e-4, rtol=0.0)


def test_cube_conveyor_state_has_stable_three_way_encoding():
    """Cube side observations distinguish both belts from the transfer corridor."""
    positions = torch.tensor([[[0.5, 0.27, 0.06], [0.5, 0.0, 0.20], [0.5, -0.27, 0.06]]])

    encoded = classify_cube_conveyors(positions)

    torch.testing.assert_close(
        encoded,
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
    )


def test_released_goal_state_succeeds_only_on_commanded_conveyor():
    """The same physical cube placement is successful for exactly one direction."""
    cube_positions = torch.tensor([[0.58, -0.27, 0.06], [0.58, -0.27, 0.06]])
    cube_velocities = torch.zeros((2, 3))
    tool_positions = torch.tensor([[0.58, -0.27, 0.14], [0.58, -0.27, 0.14]])
    finger_positions = torch.full((2, 2), 0.04)
    target_side_ids = torch.tensor([1, 0])

    successful = transfer_success_mask(
        cube_positions,
        cube_velocities,
        tool_positions,
        finger_positions,
        target_side_ids,
    )

    assert successful.tolist() == [True, False]


def test_transfer_potential_increases_through_release():
    """Dense shaping must not punish lowering and releasing at the destination."""
    source_side = torch.zeros(5, dtype=torch.long)
    cube_positions = torch.tensor(
        [
            [0.52, 0.27, 0.06],
            [0.52, 0.27, 0.20],
            [0.52, 0.00, 0.20],
            [0.52, -0.27, 0.105],
            [0.52, -0.27, 0.06],
        ]
    )
    tool_positions = cube_positions.clone()
    tool_positions[-1, 2] += 0.10
    finger_positions = torch.full((5, 2), 0.019)
    finger_positions[-1] = 0.04

    potentials = transfer_potential(cube_positions, tool_positions, finger_positions, source_side)

    assert torch.all(potentials[1:] > potentials[:-1])


def test_transfer_potential_rewards_closing_only_near_cube():
    """The acquisition bridge credits a close command only around the object."""
    cube_positions = torch.tensor([[0.52, 0.27, 0.06], [0.52, 0.27, 0.06]])
    tool_positions = torch.tensor([[0.52, 0.27, 0.07], [0.52, 0.27, 0.20]])
    source_side = torch.zeros(2, dtype=torch.long)
    open_fingers = torch.full((2, 2), 0.04)
    closed_fingers = torch.full((2, 2), 0.019)

    open_potential = transfer_potential(cube_positions, tool_positions, open_fingers, source_side)
    closed_potential = transfer_potential(cube_positions, tool_positions, closed_fingers, source_side)

    assert closed_potential[0] - open_potential[0] > 0.5
    assert closed_potential[1] - open_potential[1] < 0.03


def test_policy_distribution_samples_exact_binary_gripper_and_finite_kl():
    """PPO likelihoods match the gripper command that reaches physics."""
    distribution = ConveyorGaussianBernoulliDistribution(output_dim=8)
    distribution.update(torch.zeros((4096, 8)))

    samples = distribution.sample()
    old_params = tuple(parameter.clone() for parameter in distribution.params)
    distribution.update(torch.full((4096, 8), 0.2))
    divergence = distribution.kl_divergence(old_params, distribution.params)

    assert set(torch.unique(samples[:, -1]).tolist()) == {-1.0, 1.0}
    assert torch.isfinite(distribution.log_prob(samples)).all()
    assert torch.isfinite(divergence).all()
    assert torch.all(divergence >= 0.0)


def test_reset_sampling_guarantees_deployment_mass_and_tracks_frontier():
    """Sampling reserves deployment starts and favors intermediate frontier rows."""
    rows = build_reset_rows()
    recipe_ids = torch.tensor([row.recipe for row in rows], dtype=torch.long)
    variant_ids = torch.tensor([row.variant_id for row in rows], dtype=torch.long)
    target_cube_ids = torch.tensor([row.target_cube_id for row in rows], dtype=torch.long)
    source_side_ids = torch.tensor([row.source_side_id for row in rows], dtype=torch.long)
    place_stratum = (recipe_ids == int(ConveyorResetRecipe.PLACE)) & (target_cube_ids == 0) & (source_side_ids == 0)
    place_ids = torch.nonzero(place_stratum, as_tuple=False).flatten()
    monitor_cfg = ConveyorFrankaEnvCfg().curriculum.reset_sampling.params["success_monitor"]
    monitor = monitor_cfg.class_type(monitor_cfg, num_partitions=1, partition_size=len(rows), device="cpu")
    monitor.success_update(
        torch.cat((place_ids[0].repeat(50), place_ids[1].repeat(50))),
        torch.cat((torch.ones(50, dtype=torch.bool), torch.arange(50) % 2 == 0)),
    )

    deployment_rows = (recipe_ids == int(ConveyorResetRecipe.BELT)) & (
        variant_ids == reset_variant_counts()[int(ConveyorResetRecipe.BELT)] - 1
    )
    probabilities = reset_sampling_probabilities(
        recipe_ids,
        variant_ids,
        target_cube_ids,
        source_side_ids,
        monitor.target_weights(),
        deployment_probability=0.35,
    )

    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    torch.testing.assert_close(probabilities[deployment_rows].sum(), torch.tensor(0.35))
    torch.testing.assert_close(probabilities[~deployment_rows].sum(), torch.tensor(0.65))
    assert probabilities[place_ids[0]] < probabilities[place_ids[1]]
    for recipe in ConveyorResetRecipe:
        for cube_id in range(CUBE_COUNT):
            for side_id in range(2):
                stratum_rows = (recipe_ids == int(recipe)) & (target_cube_ids == cube_id) & (source_side_ids == side_id)
                torch.testing.assert_close(
                    probabilities[stratum_rows & ~deployment_rows].sum(),
                    torch.tensor(0.65 / (len(ConveyorResetRecipe) * CUBE_COUNT * 2)),
                )
    torch.testing.assert_close(
        probabilities[deployment_rows & (source_side_ids == 0)].sum(),
        torch.tensor(0.35 / 2),
    )
    torch.testing.assert_close(
        probabilities[deployment_rows & (source_side_ids == 1)].sum(),
        torch.tensor(0.35 / 2),
    )


def test_deployment_probability_increases_with_rolling_readiness():
    """Mastered, well-covered reset rows shift sampling toward deployment starts."""
    kwargs = {
        "initial_probability": 0.35,
        "final_probability": 0.90,
        "progress_start": 0.45,
        "progress_end": 0.80,
        "coverage_target": 0.50,
    }

    initial = deployment_probability_from_progress(torch.tensor(0.30), torch.tensor(1.0), **kwargs)
    middle = deployment_probability_from_progress(torch.tensor(0.625), torch.tensor(0.50), **kwargs)
    final = deployment_probability_from_progress(torch.tensor(0.90), torch.tensor(1.0), **kwargs)

    torch.testing.assert_close(initial, torch.tensor(0.35))
    torch.testing.assert_close(middle, torch.tensor(0.625))
    torch.testing.assert_close(final, torch.tensor(0.90))


def test_next_transfer_cube_is_random_among_eligible_alternatives():
    """Continuing commands use the one-hot target instead of a cyclic identity shortcut."""
    count = 4096
    positions = torch.zeros((count, CUBE_COUNT, 3))
    positions[:, :, 1] = torch.tensor((-0.27, -0.27, -0.27, 0.27))
    current_cube_ids = torch.zeros(count, dtype=torch.long)
    source_side_ids = torch.ones(count, dtype=torch.long)
    torch.manual_seed(7)

    selected = select_next_transfer_cube(positions, current_cube_ids, source_side_ids)

    assert set(selected.tolist()) == {1, 2}
    frequencies = torch.bincount(selected, minlength=CUBE_COUNT).float() / count
    assert torch.all(torch.abs(frequencies[1:3] - 0.5) < 0.05)


def test_manual_transfer_goal_uses_selected_cube_current_side():
    """Viewer goal changes preserve cube identity and infer the opposite destination."""

    class _Scene(dict):
        pass

    origins = torch.tensor(((0.0, 1.0, 0.0), (0.0, -2.0, 0.0)))
    selected_cube_positions = torch.tensor(((0.2, 1.3, 0.06), (0.7, -2.3, 0.06)))
    cubes = tuple(
        SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(
                    torch=selected_cube_positions if cube_id == 2 else torch.zeros_like(selected_cube_positions)
                )
            )
        )
        for cube_id in range(CUBE_COUNT)
    )
    scene = _Scene()
    scene.env_origins = origins
    command = object.__new__(ConveyorTransferCommand)
    command._env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=scene,
        episode_length_buf=torch.tensor((11, 19)),
    )
    command._cubes = cubes
    command.target_cube_ids = torch.tensor((0, 1))
    command.source_side_ids = torch.tensor((1, 0))
    command.held_cube_ids = torch.tensor((0, 1))
    command.subgoal_start_steps = torch.tensor((2, 3))
    command._stable_steps = torch.ones(2, dtype=torch.long)
    command.is_success = torch.ones(2, dtype=torch.bool)
    command.new_success = torch.ones(2, dtype=torch.bool)
    command.pending_success = torch.ones(2, dtype=torch.bool)
    command._last_evaluation_steps = torch.tensor((11, 19))

    command.set_goal(2)

    assert command.target_cube_ids.tolist() == [2, 2]
    assert command.source_side_ids.tolist() == [0, 1]
    assert command.held_cube_ids.tolist() == [-1, -1]
    assert command.subgoal_start_steps.tolist() == [11, 19]
    torch.testing.assert_close(
        command.command,
        torch.tensor(((0, 0, 1, 0, 0, 1), (0, 0, 1, 0, 1, 0)), dtype=torch.float32),
    )


def test_continuing_training_truncates_only_stalled_or_long_sequences():
    """Subgoal and sequence limits bound training without ending successful transfers."""
    assert not ConveyorFrankaPPORunnerCfg().init_at_random_ep_len
    command = SimpleNamespace(
        evaluate=lambda: None,
        subgoal_start_steps=torch.tensor((0, 0, 300)),
        pending_success=torch.zeros(3, dtype=torch.bool),
        transfer_counts=torch.tensor((0, 7, 8)),
    )
    env = SimpleNamespace(
        step_dt=0.1,
        episode_length_buf=torch.tensor((199, 200, 450)),
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )

    assert subgoal_time_out(env, timeout_s=20.0).tolist() == [False, True, False]
    assert transfer_sequence_time_out(env, maximum_transfers=8).tolist() == [False, False, True]


def test_active_cube_sampling_avoids_inactive_source_lane_cubes():
    """Random deployment starts cannot begin with interpenetrating parcels."""
    count = 2048
    base_x = torch.tensor((0.26, 0.42, 0.72, 0.88)).expand(count, -1).clone()
    cube_sides = torch.tensor((0, 0, 1, 1)).expand(count, -1).clone()
    target_cube_ids = torch.arange(count) % CUBE_COUNT
    source_side_ids = torch.arange(count) % 2
    cube_sides.scatter_(1, target_cube_ids.unsqueeze(1), source_side_ids.unsqueeze(1))

    sampled = _sample_collision_free_active_x(
        base_x,
        cube_sides,
        target_cube_ids,
        source_side_ids,
        torch.full((count,), 0.30),
        torch.full((count,), 0.82),
    )

    cube_ids = torch.arange(CUBE_COUNT).expand(count, -1)
    inactive_on_source = (cube_sides == source_side_ids.unsqueeze(1)) & (cube_ids != target_cube_ids.unsqueeze(1))
    separation = torch.abs(sampled.unsqueeze(1) - base_x)
    assert torch.all(separation[inactive_on_source] >= 0.055)


def test_deployment_layout_uses_each_racetrack_straight_run_once():
    """Every command keeps its target reachable and distributes all four cubes evenly."""
    target_cube_ids = torch.arange(CUBE_COUNT).repeat_interleave(2)
    source_side_ids = torch.arange(2).repeat(CUBE_COUNT)

    slots, cube_sides, cube_x, cube_y = _balanced_cube_slots(target_cube_ids, source_side_ids)

    expected_slots = torch.arange(CUBE_COUNT).expand(target_cube_ids.numel(), -1)
    torch.testing.assert_close(torch.sort(slots, dim=1).values, expected_slots)
    active_slots = slots.gather(1, target_cube_ids.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(active_slots, 2 * source_side_ids)
    assert torch.all(torch.sum(cube_sides == 0, dim=1) == 2)
    assert torch.all(torch.sum(cube_sides == 1, dim=1) == 2)

    expected_y = torch.tensor(
        (-BELT_OUTER_STRAIGHT_Y, -BELT_INNER_STRAIGHT_Y, BELT_INNER_STRAIGHT_Y, BELT_OUTER_STRAIGHT_Y)
    )
    torch.testing.assert_close(torch.sort(cube_y, dim=1).values, expected_y.expand_as(cube_y))
    for side_id in (0, 1):
        side_x = torch.where(cube_sides == side_id, cube_x, torch.nan)
        expected_sum = torch.full((source_side_ids.numel(),), 2 * BELT_CENTER_X, dtype=cube_x.dtype)
        torch.testing.assert_close(torch.nansum(side_x, dim=1), expected_sum)
