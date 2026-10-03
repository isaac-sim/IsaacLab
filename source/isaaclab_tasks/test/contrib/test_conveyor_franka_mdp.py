# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conveyor policy and sorting contracts exercised through real CPU managers and assets."""

from contextlib import closing

import gymnasium as gym
import pytest
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.conveyor_franka import mdp
from isaaclab_tasks.contrib.conveyor_franka.agents.rsl_rl_ppo_cfg import ConveyorGaussianBernoulliDistribution
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import (
    BELT_CENTER_X,
    BELT_INNER_STRAIGHT_Y,
    BELT_OUTER_STRAIGHT_Y,
)
from isaaclab_tasks.contrib.conveyor_franka.mdp.commands import select_next_transfer_cube
from isaaclab_tasks.contrib.conveyor_franka.mdp.curriculums import (
    deployment_probability_from_progress,
    reset_sampling_probabilities,
)
from isaaclab_tasks.contrib.conveyor_franka.mdp.kinematics import end_effector_pose
from isaaclab_tasks.contrib.conveyor_franka.mdp.reset_events import (
    CUBE_COUNT,
    ConveyorResetRecipe,
    _balanced_cube_slots,
    _sample_collision_free_active_x,
    build_reset_rows,
    franka_tool_position,
    reset_variant_counts,
)
from isaaclab_tasks.contrib.conveyor_franka.mdp.rewards import transfer_potential
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

_BASE = "IsaacContrib-Conveyor-Racetrack-Transfer-v0"
_SORTER = "IsaacContrib-Conveyor-Warehouse-Sorting-v0"


def _environment(task):
    cfg = parse_env_cfg(task, device="cpu", num_envs=2)
    cfg.sim.physics.use_cuda_graph = False
    cfg.sim.visualizer_cfgs = []
    cfg.sim.use_fabric = False
    cfg.sim.save_logs_to_file = False
    cfg.seed = 41
    return gym.make(task, cfg=cfg).unwrapped


def _parcels(env):
    return tuple(env.scene[f"cube_{i}"] for i in range(env.cfg.conveyor_force.transported_body_count_per_env))


def _waiting_positions(env):
    positions = torch.tensor([[[2.5, 0.8, 0.5]] * len(_parcels(env))] * 2)
    ids = torch.arange(positions.shape[1])
    positions[:, :, 0] = 2.4 + 0.06 * (ids % 6)
    positions[:, :, 1] += 0.06 * (ids // 6)
    return positions


def _place(env, positions):
    for i, parcel in enumerate(_parcels(env)):
        pose = parcel.data.default_root_pose.torch.clone()
        pose[:, :3] = positions[:, i] + env.scene.env_origins
        parcel.write_root_pose_to_sim_index(root_pose=pose, skip_forward=True)
        parcel.write_root_velocity_to_sim_index(root_velocity=torch.zeros(2, 6), skip_forward=True)
    env.sim.forward()
    env.scene.update(env.step_dt)


@pytest.mark.parametrize("task", [_BASE, _SORTER])
def test_checkpoint_interface_and_invalid_action_recovery(task):
    """Both real tasks preserve learned feature ordering, finite rewards and independent reset flags."""
    with closing(_environment(task)) as env:
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, 123)
        assert env.action_manager.total_action_dim == 8
        assert env.physics_dt == 1 / 120 and env.step_dt == 1 / 60
        result = env.step(torch.zeros(2, 8))
        assert torch.isfinite(result[0]["policy"]).all() and torch.isfinite(result[1]).all()
        pool = getattr(env, "conveyor_cube_pool", None)
        if pool is not None:
            assert not env.curriculum_manager.active_terms, "Warehouse training must not use phase reset rows."
            env.command_manager.get_term("transfer").has_target[:] = torch.tensor([False, True])
            env.step(torch.full((2, 8), 0.5))
            torch.testing.assert_close(env.action_manager.action, torch.full((2, 8), 0.5))
            env.command_manager.get_term("transfer").has_target[:] = torch.tensor([False, True])
            env.cfg.park_when_idle = True
            env.step(torch.full((2, 8), 0.5))
            torch.testing.assert_close(env.action_manager.action[1], torch.full((8,), 0.5))
            assert env.action_manager.action[0, :7].abs().max() <= 0.25
            assert env.action_manager.action[0, 7] == 0
        positions = _waiting_positions(env)
        positions[:, :4] = torch.tensor([[0.4, 0.27, 0.06], [0.6, -0.27, 0.06], [0.8, 0, 0.2], [0.9, -0.27, 0.06]])
        positions[1, 2] = torch.tensor([0.8, -0.27, 0.2])
        if pool is not None:
            pool.slot_ids[:] = torch.tensor([[4, 1, 2, 3], [0, 5, 2, 3]])
            positions[0, 4] = torch.tensor([0.52, 0.27, 0.06])
            positions[1, 5] = torch.tensor([0.72, -0.27, 0.06])
        _place(env, positions)
        if pool is not None:
            env.sim.step(render=False)
            model = env.sim.physics_manager.get_model()
            contacts = env.sim.physics_manager.get_contacts()
            count = int(contacts.rigid_contact_count.numpy()[0])
            assert count > 0
            bodies = model.shape_body.numpy()
            first = bodies[contacts.rigid_contact_shape0.numpy()[:count]]
            second = bodies[contacts.rigid_contact_shape1.numpy()[:count]]
            assert ((first >= 0) | (second >= 0)).all()
            _place(env, positions)
        command = env.command_manager.get_term("transfer")
        env.episode_length_buf[:] = torch.tensor([11, 19])
        command.set_goal(2)
        if pool is not None:
            command.has_target.fill_(True)
        if pool is not None:
            pose = pool.assets[4].data.root_pose_w.torch.clone()
            pose[0, 3:] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
            velocity = torch.zeros(2, 6)
            velocity[0] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            pool.assets[4].write_root_pose_to_sim_index(root_pose=pose, skip_forward=True)
            pool.assets[4].write_root_velocity_to_sim_index(root_velocity=velocity, skip_forward=True)
            env.sim.forward()
            env.scene.update(env.step_dt)
        policy = env.observation_manager.compute()["policy"]
        expected = positions[:, :4] if pool is None else positions[torch.arange(2)[:, None], pool.slot_ids]
        torch.testing.assert_close(policy[:, 16:28], expected.flatten(1))
        torch.testing.assert_close(policy[:, 76:79], positions[:, 2])
        torch.testing.assert_close(policy[:, 85:89], torch.tensor([[0, 0, 1, 0]] * 2).float())
        torch.testing.assert_close(policy[:, 101:103], torch.tensor([[0, 1], [1, 0]]).float())
        torch.testing.assert_close(
            policy[:, 89:101],
            torch.tensor([[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]]).float(),
        )
        if pool is not None:
            torch.testing.assert_close(policy[0, 40:43], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6, rtol=0)
            torch.testing.assert_close(policy[0, 52:58], velocity[0])
            remote = positions.clone()
            remote[:, 2:4] = torch.tensor([[2.8, 0.1, 0.56], [1.2, -0.59, 0.16]])
            _place(env, remote)
            adapted = env.observation_manager.compute()["policy"][:, 16:28].reshape(2, 4, 3)
            torch.testing.assert_close(adapted[:, 2:, 1:], torch.tensor([[[0.75, 0.06], [-0.75, 0.06]]] * 2))
            physical = (
                torch.stack([p.data.root_pos_w.torch for p in pool.assets], dim=1) - env.scene.env_origins[:, None]
            )
            torch.testing.assert_close(physical, remote)
            _place(env, positions)
        assert command.held_cube_ids.tolist() == [-1, -1]
        assert command.subgoal_start_steps.tolist() == [11, 19]
        if pool is None:
            command.subgoal_start_steps[:] = env.episode_length_buf - torch.tensor([1199, 1200])
            assert mdp.subgoal_time_out(env, timeout_s=20).tolist() == [False, True]
            command.pending_success[1] = True
            assert not mdp.subgoal_time_out(env, timeout_s=20).any()
            command.transfer_counts[:] = torch.tensor([7, 8])
            assert mdp.transfer_sequence_time_out(env, maximum_transfers=8).tolist() == [False, True]
            command.pending_success.zero_()
        arm = env.action_manager.get_term("arm_action")
        gripper = env.action_manager.get_term("gripper_action")
        previous = torch.tensor([[0.1] * 7 + [-1.0], [-0.2] * 7 + [1.0]])
        current = torch.tensor([[-0.3] * 7 + [1.0], [0.4] * 7 + [-1.0]])
        env.action_manager.process_action(previous)
        env.action_manager.process_action(current)
        torch.testing.assert_close(mdp.finite_action_rate_l2(env), (current - previous).square().sum(1))
        bad = torch.tensor(
            [[float("nan"), float("inf"), -float("inf"), 5, -5, 0.5, -0.5, float("nan")], [0.5] * 7 + [float("inf")]]
        )
        robot = env.scene["robot"]
        joint_ids, _ = robot.find_joints("panda_joint[1-7]", preserve_order=True)
        joints = robot.data.joint_pos.torch.clone()
        # Interior joint positions make the learned 0.12-rad residual independently observable.
        initial = torch.tensor([0.05, 0.2, -0.1, -2.25, 0, 2.45, 0.775]).repeat(2, 1)
        joints[:, joint_ids] = initial
        robot.write_joint_position_to_sim_index(position=joints)
        env.sim.forward()
        env.scene.update(env.step_dt)
        env.action_manager.process_action(bad)
        torch.testing.assert_close(
            arm.processed_actions, initial + torch.tensor([[0, 0.12, -0.12, 0.12, -0.12, 0.06, -0.06], [0.06] * 7])
        )
        torch.testing.assert_close(arm.raw_actions, torch.tensor([[0, 1, -1, 1, -1, 0.5, -0.5], [0.5] * 7]))
        torch.testing.assert_close(gripper.raw_actions, torch.tensor([[-1.0], [1.0]]))
        torch.testing.assert_close(gripper.processed_actions, torch.tensor([[0.0, 0.0], [0.04, 0.04]]))
        assert torch.isfinite(arm.processed_actions).all() and torch.isfinite(gripper.processed_actions).all()
        assert torch.isfinite(mdp.finite_action_rate_l2(env)).all()
        # Isolate an arm-only failure from a gripper-only failure.
        bad[0, 7] = -1
        env.action_manager.process_action(bad)
        assert mdp.invalid_action(env).tolist() == [True, True]
        env.action_manager.reset(env_ids=[0])
        assert mdp.invalid_action(env).tolist() == [False, True]
        env.action_manager.reset()
        assert not mdp.invalid_action(env).any()
        if pool is not None:
            command.has_target.zero_()
        terminated = env.step(bad)
        assert terminated[2].tolist() == [True, True]
        assert torch.isfinite(terminated[0]["policy"]).all() and torch.isfinite(terminated[1]).all()
        cfg = env.cfg.copy()
        cfg.actions.arm_action.preserve_order = False
        with pytest.raises(ValueError, match="preserve"):
            cfg.validate()


def test_sorter_dispatch_inventory_and_selected_resets():
    """Actual parcels are shuffled, individually dispatched, pinned while grasped, and left sorted."""
    from isaaclab_tasks.contrib.conveyor_franka.conveyor_warehouse_geometry import warehouse_parcel_positions

    with closing(_environment(_SORTER)) as env:
        env.reset()
        pool = env.conveyor_cube_pool
        before_assignments = pool.assignment_counts.clone()
        before_transfers = pool.transfer_counts.clone()
        before_slots = pool.slot_ids[0].clone()
        before_velocity = torch.stack([p.data.root_vel_w.torch.clone() for p in pool.assets], dim=1)
        before = torch.stack([p.data.root_pose_w.torch.clone() for p in pool.assets], dim=1)
        repeated = None
        for ids in (torch.tensor([1]), slice(1, 2)):
            pool.assignment_counts.fill_(7)
            pool.transfer_counts.fill_(3)
            velocity = pool.assets[-1].data.root_vel_w.torch.clone()
            velocity[1].fill_(0.2)
            pool.assets[-1].write_root_velocity_to_sim_index(root_velocity=velocity)
            torch.manual_seed(42)
            env._reset_idx(ids)
            assert not pool.assets[-1].data.root_vel_w.torch[1].any()
            torch.testing.assert_close(
                pool.assignment_counts[1], torch.tensor([1] * CUBE_COUNT + [0] * (len(pool.assets) - CUBE_COUNT))
            )
            assert not pool.transfer_counts[1].any()
            assert (pool.assignment_counts[0] == 7).all() and (pool.transfer_counts[0] == 3).all()
            actual = torch.stack([p.data.root_pose_w.torch.clone() for p in pool.assets], dim=1)
            torch.testing.assert_close(actual[0], before[0])
            torch.testing.assert_close(pool.slot_ids[0], before_slots)
            torch.testing.assert_close(pool.slot_ids[1], torch.arange(4))
            torch.testing.assert_close(
                torch.stack([p.data.root_vel_w.torch[0] for p in pool.assets]), before_velocity[0]
            )
            local = actual[1, :, :3] - env.scene.env_origins[1]
            destinations = torch.tensor(warehouse_parcel_positions())
            distance = torch.linalg.vector_norm(local[:, None] - destinations[None], dim=-1)
            assert distance.min(1).values.max() < 1e-5
            assert distance.argmin(1).unique().numel() == len(pool.assets)
            assert not torch.allclose(local, destinations)
            if repeated is not None:
                torch.testing.assert_close(actual, repeated)
            repeated = actual.clone()
        pool.assignment_counts.copy_(before_assignments)
        pool.transfer_counts.copy_(before_transfers)
        positions = _waiting_positions(env)
        positions[:, 0] = torch.tensor([0.7, 0.27, 0.06])  # Correct class, so leave it alone.
        positions[0, 2:4] = torch.tensor([[0.6, 0.27, 0.06], [0.6, -0.27, 0.06]])
        positions[0, 7] = torch.tensor([0.9, 0.27, 0.06])
        pool.assignment_counts[0, 7] = 100  # A repeatedly assigned, nearer arrival must not starve a new one.
        positions[0, 5] = torch.tensor([0.8, 0.27, 0.06])  # Arrival outside the initial four policy slots.
        _place(env, positions)
        command = env.command_manager.get_term("transfer")
        command.has_target.zero_()
        command._update_command()
        assert command.has_target.tolist() == [True, False]
        slot = int(command.target_cube_ids[0])
        assert pool.slot_ids[0, slot] == 5
        assert command.command[0, -2:].tolist() == [0, 1]
        assert pool.slot_ids[0, 0] == 0
        assert command.metrics["sorted_parcels"].tolist() == [3, 1]
        torch.testing.assert_close(pool.assets[5].data.root_pos_w.torch[0] - env.scene.env_origins[0], positions[0, 5])
        # Move the assigned parcel into a real closed-gripper grasp, even outside the pickup lane.
        robot = env.scene["robot"]
        fingers, _ = robot.find_joints("panda_finger_joint[1-2]", preserve_order=True)
        joints = robot.data.joint_pos.torch.clone()
        joints[:, fingers] = 0.019
        robot.write_joint_position_to_sim_index(position=joints)
        arms, _ = robot.find_joints("panda_joint[1-7]", preserve_order=True)
        lift = next(
            r
            for r in build_reset_rows()
            if r.recipe == ConveyorResetRecipe.LIFT
            and r.variant_id == 3
            and r.target_cube_id == 0
            and r.source_side_id == 0
        )
        joints[:, arms] = torch.tensor(lift.arm_positions)
        robot.write_joint_position_to_sim_index(position=joints)
        env.sim.forward()
        env.scene.update(env.step_dt)
        tool, _ = end_effector_pose(env)
        positions[0, 5] = tool[0] - env.scene.env_origins[0]
        _place(env, positions)
        assert mdp.physical_cube_acquisition_mask(env)[0]
        command._update_command()
        assert pool.slot_ids[0, slot] == 5 and command.has_target[0]
        positions[0, 5] = torch.tensor([0.8, 0.27, 0.06])
        joints[:, fingers] = 0.04
        robot.write_joint_position_to_sim_index(position=joints)
        _place(env, positions)
        env.episode_length_buf += command.cfg.minimum_subgoal_steps
        command.evaluate()
        assert not command.pending_success[0]
        positions[0, 5] = torch.tensor([0.8, -0.27, 0.06])
        joints[:, fingers] = 0.04
        robot.write_joint_position_to_sim_index(position=joints)
        _place(env, positions)
        env.episode_length_buf += command.cfg.minimum_subgoal_steps
        for _ in range(command.cfg.hold_steps):
            env.episode_length_buf += 1
            command.evaluate()
        assert pool.transfer_counts[0, 5] == 1 and pool.transfer_counts[1].sum() == 0
        # Completed classes stay circulating and cannot keep paying a previous success reward.
        positions[:, :, 0] = 0.6
        positions[:, :, 1] = torch.tensor([0.27, -0.27] * (len(pool.assets) // 2))
        positions[:, :, 2] = 0.06
        _place(env, positions)
        command._update_command()
        assert not command.has_target.any()
        assert command.metrics["batch_complete"].tolist() == [1, 1]
        command.new_success.fill_(True)
        command.is_success.fill_(True)
        command.evaluate()
        assert not command.new_success.any() and not command.is_success.any()
        assert not mdp.cube_out_of_workspace(env, **env.cfg.terminations.cube_out_of_workspace.params).any()
        # Present every identity as a separate arrival and let the real dispatcher refill its slots.
        for parcel_id in range(len(pool.assets)):
            positions[0] = _waiting_positions(env)[0]
            source_y = -0.27 if command.parcel_destinations[parcel_id] == 0 else 0.27
            positions[0, parcel_id] = torch.tensor([0.8, source_y, 0.06])
            _place(env, positions)
            command.pending_success[0] = True
            command._update_command()
            assert pool.slot_ids[0, command.target_cube_ids[0]] == parcel_id
        assert (pool.assignment_counts[0] > 0).all()
        unassigned = next(i for i in range(len(pool.assets)) if i not in pool.slot_ids[0])
        positions[0, unassigned, 2] = -1
        _place(env, positions)
        assert mdp.cube_out_of_workspace(env, **env.cfg.terminations.cube_out_of_workspace.params).tolist() == [
            True,
            False,
        ]


def test_curriculum_checkpoint_restores_progress_and_deployment_outcomes(tmp_path):
    """Actual RSL-RL saves preserve reset evidence; phase progress alone cannot advance deployment."""
    with closing(_environment(_BASE)) as env:
        reset_cfg = env.event_manager.get_term_cfg("reset_from_state_table")
        reset_cfg.params.update(
            fixed_recipe=int(ConveyorResetRecipe.BELT), fixed_variant_id=mdp.BELT_DEPLOYMENT_VARIANT
        )
        env.reset()
        command = env.command_manager.get_term("transfer")
        # Supply partial-progress outcomes at the curriculum's episode boundary.
        env.episode_length_buf.fill_(5)
        command.progress_ever_success.fill_(True)
        env.reset()
        log = env.extras["log"]
        assert log["Curriculum/reset_sampling/overall_progress_rate"] == 1
        assert log["Curriculum/reset_sampling/deployment_transfer_success_rate"] == 0
        torch.testing.assert_close(log["Curriculum/reset_sampling/deployment_probability"], torch.tensor(0.35))
        env.episode_length_buf.fill_(5)
        command.ever_success[0] = True
        env.reset()

        agent = load_cfg_from_registry(_BASE, "rsl_rl_cfg_entry_point")
        agent.device = "cpu"
        agent.actor.hidden_dims = agent.critic.hidden_dims = [16]
        runner = OnPolicyRunner(RslRlVecEnvWrapper(env), to_dict(agent), log_dir=str(tmp_path), device=agent.device)
        curriculum = env.curriculum_manager.cfg.reset_sampling.func
        expected = curriculum.get_state()
        checkpoint = tmp_path / "model.pt"
        runner.save(str(checkpoint), infos={"purpose": "resume"})
        assert "conveyor_reset_curriculum" in torch.load(checkpoint, weights_only=False)
        env.episode_length_buf.fill_(5)
        env.reset()
        assert curriculum.get_state()["attempts"].sum() > expected["attempts"].sum()
        assert runner.load(str(checkpoint), map_location="cpu") == {"purpose": "resume"}
        for name, value in expected.items():
            torch.testing.assert_close(curriculum.get_state()[name], value)
        # A policy-only load must not replace a fine-tuning run's curriculum.
        env.episode_length_buf.fill_(5)
        env.reset()
        attempts = curriculum.get_state()["attempts"].sum()
        runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cpu")
        assert curriculum.get_state()["attempts"].sum() == attempts


def test_reset_bank_is_complete_and_physically_calibrated():
    """Every command/phase is represented once; tool anchors and held metadata agree with physical poses."""
    rows = build_reset_rows()
    counts = reset_variant_counts()
    keys = {(r.recipe, r.variant_id, r.target_cube_id, r.source_side_id) for r in rows}
    assert len(keys) == len(rows) == sum(counts) * 8
    assert all(
        recipe in ConveyorResetRecipe and 0 <= variant < counts[recipe] and 0 <= cube < 4 and side in (0, 1)
        for recipe, variant, cube, side in keys
    )
    for row in rows:
        held = row.recipe in (ConveyorResetRecipe.LIFT, ConveyorResetRecipe.CARRY, ConveyorResetRecipe.PLACE)
        held |= row.recipe == ConveyorResetRecipe.GRASP and row.variant_id == counts[ConveyorResetRecipe.GRASP] - 1
        assert row.held == held
    closure = [
        r.finger_position
        for r in rows
        if r.recipe == ConveyorResetRecipe.GRASP and r.target_cube_id == 0 and r.source_side_id == 0
    ]
    assert all(a > b for a, b in zip(closure, closure[1:]))
    # Independently calibrated approach, lift, transit and release tool positions.
    anchors = {
        ConveyorResetRecipe.GOAL: (0, 0.14),
        ConveyorResetRecipe.PLACE: (3, 0.105),
        ConveyorResetRecipe.CARRY: (2, 0.25),
        ConveyorResetRecipe.LIFT: (3, 0.22),
        ConveyorResetRecipe.GRASP: (0, 0.06),
        ConveyorResetRecipe.PREGRASP: (0, 0.14),
    }
    for row in rows:
        if row.target_cube_id or row.recipe not in anchors or row.variant_id != anchors[row.recipe][0]:
            continue
        y = 0.27 * (1 - 2 * row.source_side_id)
        if row.recipe in (ConveyorResetRecipe.GOAL, ConveyorResetRecipe.PLACE):
            y = -y
        elif row.recipe == ConveyorResetRecipe.CARRY:
            y = 0
        actual = franka_tool_position(torch.tensor([row.arm_positions], dtype=torch.float64))[0]
        torch.testing.assert_close(
            actual, torch.tensor([0.52, y, anchors[row.recipe][1]], dtype=actual.dtype), atol=3e-4, rtol=0
        )


def test_transfer_shaping_and_binary_policy_likelihoods():
    """Release improves progress; remote closing earns no grasp credit; sampled gripper likelihoods remain finite."""
    cubes = torch.tensor(
        [[0.52, 0.27, 0.06], [0.52, 0.27, 0.20], [0.52, 0, 0.20], [0.52, -0.27, 0.105], [0.52, -0.27, 0.06]]
    )
    tool = cubes.clone()
    tool[-1, 2] += 0.1
    fingers = torch.full((5, 2), 0.019)
    fingers[-1] = 0.04
    potential = transfer_potential(cubes, tool, fingers, torch.zeros(5, dtype=torch.long))
    assert (potential[1:] > potential[:-1]).all()
    cubes = torch.tensor([[0.52, 0.27, 0.06]] * 2)
    tool = torch.tensor([[0.52, 0.27, 0.07], [0.52, 0.27, 0.20]])
    credit = transfer_potential(cubes, tool, torch.full((2, 2), 0.019), torch.zeros(2, dtype=torch.long))
    credit -= transfer_potential(cubes, tool, torch.full((2, 2), 0.04), torch.zeros(2, dtype=torch.long))
    assert credit[0] > 0.5 and credit[1] < 0.03
    distribution = ConveyorGaussianBernoulliDistribution(output_dim=8)
    distribution.update(torch.zeros(4096, 8))
    samples, previous = distribution.sample(), tuple(p.clone() for p in distribution.params)
    distribution.update(torch.full((4096, 8), 0.2))
    divergence = distribution.kl_divergence(previous, distribution.params)
    assert set(samples[:, -1].unique().tolist()) == {-1, 1}
    assert torch.isfinite(distribution.log_prob(samples)).all()
    assert torch.isfinite(divergence).all() and (divergence >= 0).all()


def test_curriculum_reserves_deployment_mass_and_balances_commands():
    """Adaptive weights cannot starve any recipe/cube/direction, or the moving-belt deployment starts."""
    rows = build_reset_rows()
    recipe, variant, cube, side = torch.tensor(
        [[r.recipe, r.variant_id, r.target_cube_id, r.source_side_id] for r in rows]
    ).T
    deployment = (recipe == ConveyorResetRecipe.BELT) & (
        variant == reset_variant_counts()[ConveyorResetRecipe.BELT] - 1
    )
    place = torch.where((recipe == ConveyorResetRecipe.PLACE) & (cube == 0) & (side == 0))[0]
    weights = torch.ones(len(rows))
    weights[place[:2]] = torch.tensor([0.2, 0.8])
    probabilities = reset_sampling_probabilities(recipe, variant, cube, side, weights, deployment_probability=0.35)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    assert probabilities[place[0]] < probabilities[place[1]]
    for r, c, s in {(r.recipe, r.target_cube_id, r.source_side_id) for r in rows}:
        mask = (recipe == r) & (cube == c) & (side == s) & ~deployment
        torch.testing.assert_close(probabilities[mask].sum(), torch.tensor(0.65 / (len(ConveyorResetRecipe) * 8)))
    for s in (0, 1):
        torch.testing.assert_close(probabilities[deployment & (side == s)].sum(), torch.tensor(0.35 / 2))
    for progress, coverage, expected in ((0.30, 1.0, 0.35), (0.625, 0.50, 0.625), (0.90, 1.0, 0.90)):
        actual = deployment_probability_from_progress(torch.tensor(progress), torch.tensor(coverage))
        torch.testing.assert_close(actual, torch.tensor(expected))


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
