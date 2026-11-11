# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical receipt counting across target handoffs and independent environment resets."""

import math
from contextlib import closing

import gymnasium as gym
import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.conveyor_franka.mdp.kinematics import end_effector_pose
from isaaclab_tasks.contrib.conveyor_suction.actions import solve_tool_ik
from isaaclab_tasks.contrib.conveyor_suction.scene import PARCEL_SIZES, OutfeedLayout, OutfeedsCfg
from isaaclab_tasks.utils import parse_env_cfg

pytestmark = pytest.mark.kitless


def _check_receipt_pulse(env, actions, env_id):
    pulses = 0
    for _ in range(15):
        _, _, terminated, timed_out, extras = env.step(actions)
        assert not terminated.any() and not timed_out.any()
        pulse = extras["successes"][env_id].item()
        pulses += pulse
        assert dict(env.reward_manager.get_active_iterable_terms(env_id))["delivery"][0] == 600.0 * pulse
    assert pulses == 1


def _check_delayed_arrival_after_miss(env, rows, actions, hold_at_goal, place_in_bin, place_on_intake):
    command = env.command_manager.get_term("sort")
    reset = env.event_manager.get_term_cfg("reset_parcels").func
    monitor = reset.reset_sampler.success_monitor
    impact_ids = torch.tensor([0], device=env.device)
    # A miss stays ready until an arrival, even after the original parcel leaves the miss boundary.
    env.reset(env_ids=impact_ids)
    history_size = monitor.success_size[rows[0]].item()
    history_success = monitor.success_buf[rows[0]].sum().item()
    first = env.active_ids[0].item()
    second = (first + 1) % len(env.parcels)
    env.gripper.reset(impact_ids)
    command.lifted[0] = command.placed[0] = False
    actions[0] = 0.0
    actions[0, 3] = 1.0
    for parcel_id in range(len(env.parcels)):
        place_on_intake(0, parcel_id, (0.60 + 0.20 * parcel_id, -0.75))
    place_on_intake(0, first, (0.92, -0.19))
    for _ in range(4):
        env.step(actions)
        assert env.active_ids[0] == first and not command.delivered[0].any()
        assert command.metrics["missed_pickups"][0] == 1
    place_on_intake(0, first, (0.60 + 0.20 * first, -0.75))
    place_on_intake(0, second, (0.30, -0.72))
    lengths = env.episode_length_buf.clone()
    env.step(actions)
    assert env.active_ids[0] == second, "A delayed arrival must consume the previously observed miss."
    torch.testing.assert_close(env.episode_length_buf, lengths + 1)
    assert command.metrics["missed_pickups"][0] == 1

    # Receiving that replacement rewards it once without labeling the missed sampled parcel successful.
    hold_at_goal(0)
    place_in_bin(0, second, command.destination_ids[0].item())
    _check_receipt_pulse(env, actions, 0)
    assert env.episode_success[0]
    assert command.delivered[0, second] and not command.delivered[0, first]
    assert command.delivered[0].sum() == command.metrics["correct_deliveries"][0] == 1
    assert monitor.success_size[rows[0]] == history_size
    env.reset(env_ids=impact_ids)
    assert monitor.success_size[rows[0]] == history_size + 1
    assert monitor.success_rate[rows[0]].item() == pytest.approx(history_success / (history_size + 1))


def _check_full_batch_receipts(env, rows, actions, hold_at_goal, place_in_bin, place_on_intake):
    command = env.command_manager.get_term("sort")
    reset = env.event_manager.get_term_cfg("reset_parcels").func
    monitor = reset.reset_sampler.success_monitor
    cfg = env.cfg
    batch_ids = torch.tensor([1], device=env.device)
    # Full batches keep all five receipt pulses, and only the fifth receipt terminates.
    full_bucket = 4 * cfg.events.reset_parcels.params["samples_per_path"]
    rows.fill_(full_bucket)
    command.cfg.randomize_destination = False
    cfg.commands.sort.randomize_destination = False
    env.reset(env_ids=batch_ids)
    actions[1] = 0.0
    actions[1, 3] = 1.0
    for batch_size in (2, 5):
        history_size = monitor.success_size[full_bucket].item()
        for receipt in range(1, batch_size + 1):
            parcel_id = env.active_ids[1].item()
            assert not command.delivered[1, parcel_id]
            assert command.destination_ids[1] == env.parcel_classes[parcel_id]
            hold_at_goal(1)
            place_in_bin(1, parcel_id, command.destination_ids[1].item())
            for _ in range(15):
                lengths = env.episode_length_buf.clone()
                _, _, terminated, timed_out, extras = env.step(actions)
                delivery = dict(env.reward_manager.get_active_iterable_terms(1))["delivery"][0]
                if delivery:
                    assert delivery == 600.0 and not timed_out[1]
                    if receipt == 5:
                        assert terminated[1] and env.episode_length_buf[1] == 0
                        assert extras["log"]["Metrics/sort/correct_deliveries"] == 5
                        assert extras["log"]["Metrics/sort/full_batch_success_rate"] == 1
                        assert extras["log"]["Metrics/sort/second_delivery_success_rate"] == 1
                    else:
                        assert not terminated[1]
                        torch.testing.assert_close(env.episode_length_buf[1], lengths[1] + 1)
                        assert command.delivered[1].sum() == receipt
                    break
                assert not terminated[1] and not timed_out[1]
            else:
                pytest.fail("A qualified parcel did not earn its native bin receipt.")
            if receipt < batch_size:
                if env.active_ids[1] == parcel_id:
                    candidate = (~command.delivered[1]).nonzero()[0, 0].item()
                    place_on_intake(1, candidate, (0.30, -0.72))
                    _, _, terminated, timed_out, extras = env.step(actions)
                    assert not terminated[1] and not timed_out[1]
                    assert not extras["successes"][1]
                assert env.active_ids[1] != parcel_id
                assert not command.delivered[1, env.active_ids[1]]
        assert monitor.success_size[full_bucket] == history_size + 1
        if batch_size == 2:
            _, extras = env.reset(env_ids=batch_ids)
            assert extras["log"]["Metrics/sort/correct_deliveries"] == 2
            assert extras["log"]["Metrics/sort/second_delivery_success_rate"] == 1
            assert extras["log"]["Metrics/sort/full_batch_success_rate"] == 0


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_receipts_record_success_and_continue_without_reset(device, monkeypatch):
    """Receipts reward once, record the sampled row once, and preserve history through handoffs and failures."""
    task = "IsaacContrib-Conveyor-Suction-Sorting"
    cfg = parse_env_cfg(task, device=device, num_envs=2)
    cfg.sim.visualizer_cfgs = []
    cfg.sim.save_logs_to_file = False
    cfg.seed = 41
    cfg.events.reset_parcels.params["samples_per_path"] = 16
    cfg.commands.sort.randomize_destination = True
    cfg.parcel_count = 5
    # Keep the measured bin-impact fixture independent of randomized task defaults.
    angles = tuple(math.radians(value) for value in (10, 35, 60, 85, 110, 135))
    cfg.scene.outfeeds.spawn.assets_cfg = [
        OutfeedsCfg(layout=OutfeedLayout(angles[:count], (0.0,) * count, (0.0,) * count)) for count in (5, 6)
    ]
    cfg.scene.outfeeds.spawn.assets_cfg[1].layout.radial_offsets = (0.035,) * 6
    cfg.scene.outfeeds.spawn.assets_cfg[1].layout.height_offsets = (0.015,) * 6
    cfg.commands.sort.goal_radial_range = (0.0, 0.0)
    for index in range(len(PARCEL_SIZES)):
        setattr(cfg.events, f"parcel_{index}_mass", None)
        setattr(cfg.events, f"parcel_{index}_material", None)
    cfg.episode_length_s = 240.0
    cfg.terminations.failure = None
    cfg.terminations.invalid_action = None
    with closing(gym.make(task, cfg=cfg).unwrapped) as env:
        assert len(env.parcels) == 5, "The completion boundary must use five physical parcels."
        assert env.layout_ids.tolist() == [0, 1], "The impact fixture uses the original five-lane world."
        term = env.event_manager.get_term_cfg("reset_parcels")
        reset = term.func
        generate = type(reset)._generate_recipes
        for field, value, condition in (
            ("pick_positions", float("nan"), "finite"),
            ("arm_positions", 10.0, "joint_limits"),
        ):
            with monkeypatch.context() as patch:

                def invalid_recipe(self, ids):
                    state = generate(self, ids)
                    state[field][0, 0] = value
                    return state

                patch.setattr(type(reset), "_generate_recipes", invalid_recipe)
                # Invalid canonical rows must fail, rather than duplicate or renumber valid recipes.
                with pytest.raises(RuntimeError, match=f"after 1 batches.*{condition}") as error:
                    type(reset)(term, env)
                assert f"'rejected/{condition}': 1" in str(error.value)
        samples = cfg.events.reset_parcels.params["samples_per_path"]
        rows = torch.tensor([samples // 2 + 1, samples // 2 + 2], device=device)
        reset.reset_sampler.sampler.sample = lambda probabilities, count: rows[:count]
        env.reset()
        initial_parcels = env.active_ids.tolist()
        assert env.gripper.engaged.all(), "Held resets must restore physical seals before the first policy action."
        env.gripper.set_command(torch.tensor([True, False], device=device), env.active_ids)
        positions = env.scene["robot"].data.joint_pos.torch.clone()
        env.gripper.refresh(torch.tensor([0], device=device))
        assert env.gripper.engaged.all(), "Refreshing one world must preserve another world's pending attachment."
        torch.testing.assert_close(env.scene["robot"].data.joint_pos.torch, positions, rtol=0, atol=0)
        env.gripper.set_command(torch.ones(2, dtype=torch.bool, device=device), env.active_ids)
        command = env.command_manager.get_term("sort")
        monitor = reset.reset_sampler.success_monitor
        parcel_heights = torch.stack(
            [
                env.parcels[parcel].data.root_link_pose_w.torch[env_id, 2]
                for env_id, parcel in enumerate(initial_parcels)
            ]
        )
        assert (
            parcel_heights - env.scene.env_origins[:, 2] - command.goal_positions[:, 2]
            > cfg.commands.sort.placement_max_height
        ).all(), "Descent resets must require movement before a release can qualify."
        assert not command.delivered.any(), "Every reset must start with all parcels unsorted."
        assert not command.new_success.any() and not env.episode_success.any()
        assert command.metrics["correct_deliveries"].tolist() == [0, 0]
        assert monitor.success_size[rows].tolist() == [0, 0]
        actions = torch.zeros((2, 4), device=device)

        def place_in_bin(env_id, parcel_id, bin_id, lateral_offset=0.0):
            ids = torch.tensor([env_id], device=device)
            parcel = env.parcels[parcel_id]
            pose = parcel.data.default_root_pose.torch[ids].clone()
            pose[:, :3] = env.bin_positions[env_id, bin_id] + env.scene.env_origins[ids]
            direction = env.bin_directions[env_id, bin_id]
            pose[:, 0] -= lateral_offset * direction[1]
            pose[:, 1] += lateral_offset * direction[0]
            pose[:, 2] -= 0.28 - env.parcel_sizes[parcel_id, 2] / 2 - 0.006
            parcel.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            parcel.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device), env_ids=ids)
            env.sim.forward()
            env.scene.update(env.step_dt)

        def place_on_intake(env_id, parcel_id, position):
            ids = torch.tensor([env_id], device=device)
            parcel = env.parcels[parcel_id]
            pose = parcel.data.default_root_pose.torch[ids].clone()
            pose[:, :2] = pose.new_tensor(position) + env.scene.env_origins[ids, :2]
            pose[:, 2] = env.scene.env_origins[ids, 2] + 0.04 + env.parcel_sizes[parcel_id, 2] / 2 + 0.006
            parcel.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            parcel.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device), env_ids=ids)
            env.sim.forward()
            env.scene.update(env.step_dt)

        def hold_at_goal(env_id, release=True):
            ids = torch.tensor([env_id], device=device)
            parcel_id = env.active_ids[env_id].item()
            goal = command.goal_positions[ids].clone()
            goal[:, 2] += env.parcel_sizes[parcel_id, 2] / 2.0 + 0.010
            robot = env.scene["robot"]
            joints = robot.data.default_joint_pos.torch[ids].clone()
            joints[:, reset._arm_ids] = solve_tool_ik(goal)
            robot.write_joint_position_to_sim_index(position=joints, env_ids=ids)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints), env_ids=ids)
            robot.actuators.target_command.set_position_index(value=joints, env_ids=ids)
            pose = env.parcels[parcel_id].data.default_root_pose.torch[ids].clone()
            pose[:, :3] = goal + env.scene.env_origins[ids]
            pose[:, 2] -= env.parcel_sizes[parcel_id, 2] / 2 + 0.002
            env.parcels[parcel_id].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.parcels[parcel_id].write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=device), env_ids=ids
            )
            env.action_manager.reset(ids)
            env.gripper.reset(ids)
            command.lifted[ids] = True
            env.sim.forward()
            env.scene.update(env.step_dt)
            actions[env_id, 3] = -1.0
            for _ in range(10):
                env.step(actions)
            assert env.gripper.engaged[env_id]
            if release:
                actions[env_id, 3] = 1.0
                env.step(actions)
                assert command.placed[env_id] and not command.failed[env_id]

        actions[:, 3] = -1.0
        for env_id, destination in enumerate((0, 5)):
            goal_row = (
                (reset.layout_ids == env.layout_ids[env_id])
                & (reset.parcel_ids == initial_parcels[env_id])
                & (reset.phase_ids == 0)
                & (reset.destination_ids == destination)
            ).nonzero()[0, 0]
            command.destination_ids[env_id] = destination
            command.goal_positions[env_id] = reset.goal_positions[goal_row]
            hold_at_goal(env_id, release=False)
        observations = env.observation_manager.compute()["policy"]
        assert observations.shape == (2, 312)
        assert torch.isfinite(observations).all()
        terms = env.observation_manager.active_terms["policy"]
        dimensions = env.observation_manager.group_obs_term_dim["policy"]
        goal_start = sum(size[0] for size in dimensions[: terms.index("goal")])
        torch.testing.assert_close(observations[:, goal_start : goal_start + 3], command.command)
        assert command.destination_ids[1] != env.parcel_classes[initial_parcels[1]], (
            "The commanded outlet must be independent of size."
        )
        assert command.destination_ids[1] == 5, "The sixth physical lane must accept a controlled release and receipt."
        assert command.goal_positions[1, :2].norm() > 0.73, "The release must account for its translated receiving pad."
        actions[:, 3] = 1.0
        env.step(actions)
        assert command.placed.all() and not command.failed.any()
        assert command.metrics["controlled_placements"].tolist() == [1, 1]
        # Clearance alone must not replace the observed target before its physical receipt.
        original_goals = command.goal_positions.clone()
        actions[:, 2] = 1.0
        for _ in range(60):
            env.step(actions)
        separation = end_effector_pose(env)[0][1] - env.parcels[initial_parcels[1]].data.root_link_pose_w.torch[1, :3]
        assert separation.norm() > 0.20, "The tool must physically clear the released parcel."
        assert env.active_ids.tolist() == initial_parcels
        torch.testing.assert_close(command.goal_positions, original_goals)
        actions[:, 2] = 0.0
        assert not command.delivered.any() and not env.episode_success.any()
        assert monitor.success_size[rows].tolist() == [0, 0]
        assert dict(env.reward_manager.get_active_iterable_terms(1))["delivery"][0] == 0.0
        # An outfeed-speed impact must settle in the bin without crossing its floor.
        impact_ids = torch.tensor([0], device=device)
        pose = torch.tensor(
            [[2.07563591, 0.28751278, -0.71146984, -0.97748345, -0.07764439, -0.19566739, -0.01455657]],
            device=device,
        )
        vertical = math_utils.matrix_from_quat(pose[:, 3:7])[0, 2].abs()
        pose[:, 2] += (
            vertical * (env.parcel_sizes[initial_parcels[0]] - pose.new_tensor((0.09, 0.075, 0.055)))
        ).sum() / 2
        pose[:, :3] += env.scene.env_origins[impact_ids]
        velocity = torch.tensor(
            [[0.31988335, 0.05606911, -3.89049959, -1.84165192, 10.48158455, 0.01690909]], device=device
        )
        env.parcels[initial_parcels[0]].write_root_pose_to_sim_index(root_pose=pose, env_ids=impact_ids)
        env.parcels[initial_parcels[0]].write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=impact_ids)
        env.sim.forward()
        env.scene.update(env.step_dt)
        received = [False, False]
        for _ in range(720):
            lengths = env.episode_length_buf.clone()
            _, rewards, terminated, timed_out, extras = env.step(actions)
            assert not terminated.any() and not timed_out.any(), "One receipt must not reset the physical scene."
            torch.testing.assert_close(env.episode_length_buf, lengths + 1)
            assert torch.isfinite(rewards).all()
            for env_id, parcel_id in enumerate(initial_parcels):
                if command.delivered[env_id, parcel_id] and not received[env_id]:
                    assert env.episode_success[env_id] and command.metrics["success_rate"][env_id] == 1
                    assert extras["successes"][env_id]
                    assert monitor.success_size[rows[env_id]] == monitor.success_rate[rows[env_id]] == 1
                    delivery = dict(env.reward_manager.get_active_iterable_terms(env_id))["delivery"][0]
                    assert delivery == 600.0
                    received[env_id] = True
                elif not received[env_id]:
                    assert env.active_ids[env_id] == parcel_id
                    torch.testing.assert_close(command.goal_positions[env_id], original_goals[env_id])
            if all(received) and env.active_ids[1] != initial_parcels[1]:
                break
        else:
            pytest.fail("The released parcels never reached their physical bins.")
        assert not command.failed.any()
        assert env.active_ids[1] != initial_parcels[1], "The release fixture must also reach the next-target handoff."
        expected = torch.zeros(len(env.parcels), dtype=torch.bool, device=device)
        expected[initial_parcels[1]] = True
        torch.testing.assert_close(command.delivered[1], expected)
        next_parcel = env.active_ids[1].item()
        next_goal = command.goal_positions[1].clone()
        hold_at_goal(1, release=False)

        # A handoff and an unrelated reset cannot repeat the receipt or its sampled-row outcome.
        _, _, _, _, extras = env.step(actions)
        assert not extras["successes"].any()
        assert dict(env.reward_manager.get_active_iterable_terms(1))["delivery"][0] == 0.0
        env.reset(env_ids=impact_ids)
        command.evaluate()
        torch.testing.assert_close(command.delivered[1], expected)
        assert command.metrics["correct_deliveries"][1] == 1
        assert env.active_ids[1] == next_parcel and env.gripper.engaged[1]
        torch.testing.assert_close(command.goal_positions[1], next_goal)
        assert env.episode_success[1]
        assert monitor.success_size[rows].tolist() == [1, 1]

        # A positive-length attempt without a receipt records one failure when it resets.
        env.step(actions)
        assert not command.delivered[0].any() and not env.episode_success[0]
        env.reset(env_ids=impact_ids)
        assert monitor.success_size[rows].tolist() == [2, 1]
        assert monitor.success_rate[rows].tolist() == [0.5, 1.0]
        actions[0, 3] = -1.0

        # Identical released poses require observable qualification before they can earn a receipt.
        second = env.active_ids[1].item()
        second_destination = command.destination_ids[1].item()
        env.gripper.reset(torch.tensor([1], device=device))
        actions[1, 3] = 1.0
        place_in_bin(1, second, second_destination)
        command.lifted[1] = False
        unqualified = env.observation_manager.compute()["policy"][1].clone()
        command.lifted[1] = True
        qualified = env.observation_manager.compute()["policy"][1]
        assert not torch.equal(unqualified, qualified), "Pickup qualification must be observable to the policy."
        assert qualified.shape == env.observation_manager.group_obs_dim["policy"]
        command.placed[1] = True
        placement_qualified = env.observation_manager.compute()["policy"][1]
        assert not torch.equal(qualified, placement_qualified), "Controlled-placement qualification must be observable."
        command.placed[1] = False
        command.lifted[1] = False
        for _ in range(15):
            env.step(actions)
        assert command.failed[1] and not command.delivered[1, second]
        command.failed[1] = False
        hold_at_goal(1)
        place_in_bin(1, second, second_destination)
        for _ in range(2):
            env.step(actions)
        assert not command.delivered[1, second]

        # An invalid action on a would-be receipt step earns no credit, without erasing earlier success.
        actions[1, 0] = torch.nan
        _, rewards, _, _, extras = env.step(actions)
        assert command.failed[1] and not command.delivered[1, second]
        assert not extras["successes"][1]
        assert dict(env.reward_manager.get_active_iterable_terms(1))["delivery"][0] == 0.0
        assert env.episode_success[1] and command.metrics["success_rate"][1] == 1
        assert monitor.success_size[rows[1]] == monitor.success_rate[rows[1]] == 1
        assert torch.isfinite(rewards).all()

        actions[1, 0] = 0.0
        command.failed[1] = False
        place_in_bin(1, second, second_destination)
        _check_receipt_pulse(env, actions, 1)
        assert env.episode_success[1]
        assert command.delivered[1].sum() == command.metrics["correct_deliveries"][1] == 2
        assert monitor.success_size[rows[1]] == monitor.success_rate[rows[1]] == 1

        # An unselected parcel cannot disappear into a wrong bin without failing its world.
        inactive = next(
            i for i in range(len(env.parcels)) if i != env.active_ids[0].item() and not command.delivered[0, i]
        )
        assert not command.failed[0]
        place_in_bin(0, inactive, (env.parcel_classes[inactive].item() + 1) % 5)
        for _ in range(15):
            env.step(actions)
        assert command.failed.tolist() == [True, False]
        assert env.episode_success.tolist() == [False, True]
        env.reset(env_ids=impact_ids)
        assert monitor.success_size[rows].tolist() == [3, 1]
        assert monitor.success_rate[rows[0]].item() == pytest.approx(1.0 / 3.0)

        # A fast or elevated release is rejected before a parcel can coast into a valid bin.
        for elevated in (False, True):
            env.reset(env_ids=impact_ids)
            hold_at_goal(0, release=False)
            parcel = env.parcels[env.active_ids[0].item()]
            if elevated:
                pose = parcel.data.root_link_pose_w.torch[impact_ids].clone()
                pose[:, 2] += 0.05
                parcel.write_root_pose_to_sim_index(root_pose=pose, env_ids=impact_ids)
            else:
                actions[0, 3] = 1.0
                env.step(actions)
                assert command.placed[0]
                actions[0, 3] = -1.0
                for _ in range(8):
                    env.step(actions)
                    if env.gripper.engaged[0]:
                        break
                assert env.gripper.engaged[0] and not command.placed[0], "Regrasping must revoke placement."
                velocity = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
                parcel.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=impact_ids)
            env.sim.forward()
            env.scene.update(env.step_dt)
            actions[0, 3] = 1.0
            env.step(actions)
            assert command.failed[0] and not command.placed[0] and not command.delivered[0].any()
            assert command.metrics["invalid_releases"][0] == 1

        env.reset(env_ids=impact_ids)
        hold_at_goal(0)
        place_in_bin(0, env.active_ids[0].item(), command.destination_ids[0])
        for _ in range(2):
            env.step(actions)
        assert not command.delivered[0].any()
        robot = env.scene["robot"]
        body = robot.find_bodies("panda_link3")[0][0]
        obstacle = env.parcels[(env.active_ids[0].item() + 1) % len(env.parcels)]
        pose = obstacle.data.default_root_pose.torch[impact_ids].clone()
        link_pose = robot.data.body_link_pose_w.torch[impact_ids, body]
        capsule_center = torch.tensor([[0.0, 0.0, -0.145]], device=device)
        pose[:, :3] = link_pose[:, :3] + math_utils.quat_apply(link_pose[:, 3:7], capsule_center)
        obstacle.write_root_pose_to_sim_index(root_pose=pose, env_ids=impact_ids)
        obstacle.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device), env_ids=impact_ids)
        env.sim.forward()
        env.scene.update(env.step_dt)
        env.step(actions)
        assert command.failed[0] and command.metrics["arm_collisions"][0] == 1
        forces = env.scene["arm_contacts"].data.net_normal_forces_w_history.torch[0]
        assert forces.norm(dim=-1).max() > cfg.commands.sort.arm_contact_force_threshold
        assert not command.delivered[0].any() and not command.new_success[0]

        _check_delayed_arrival_after_miss(env, rows, actions, hold_at_goal, place_in_bin, place_on_intake)

        # Different physical layouts contribute to the same path-progress history.
        history_size = monitor.success_size[rows[0]].item()
        history_success = monitor.success_buf[rows[0]].sum().item()
        env.reset(env_ids=torch.tensor([1], device=device))
        assert reset.reset_sampler.row_ids[1] == rows[0]
        hold_at_goal(1)
        place_in_bin(1, env.active_ids[1].item(), command.destination_ids[1])
        for _ in range(15):
            env.step(actions)
        assert command.metrics["correct_deliveries"][1] == 1
        assert monitor.success_size[rows[0]] == history_size + 1
        assert monitor.success_buf[rows[0]].sum() == history_success + 1

        # The held carry boundary connects to placement without pre-crediting a delivery.
        rows.fill_(2 * samples - 1)
        env.reset()
        active_positions = (
            torch.stack(
                [
                    env.parcels[parcel].data.root_link_pose_w.torch[env_id, :2]
                    for env_id, parcel in enumerate(env.active_ids.tolist())
                ]
            )
            - env.scene.env_origins[:, :2]
        )
        assert (active_positions - command.goal_positions[:, :2]).norm(dim=-1).max() < 0.02, (
            "The nearest carry reset must connect to the placement region."
        )
        assert env.gripper.engaged.all(), "The carry boundary must retain real suction seals."
        assert not command.delivered.any() and not env.episode_success.any()

        # A fallen parcel fails the batch whether or not it is currently selected.
        rows.fill_(3 * samples)
        env.reset()
        actions.zero_()
        actions[:, 3] = 1.0

        def drop_parcel(env_id, parcel_id):
            ids = torch.tensor([env_id], device=device)
            parcel = env.parcels[parcel_id]
            pose = parcel.data.root_link_pose_w.torch[ids].clone()
            pose[:, 2] = env.scene.env_origins[env_id, 2] - 0.8
            parcel.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            parcel.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device), env_ids=ids)

        drop_parcel(0, (env.active_ids[0].item() + 1) % len(env.parcels))
        drop_parcel(1, env.active_ids[1].item())
        env.sim.forward()
        env.scene.update(env.step_dt)
        env.step(actions)
        assert command.failed.all()

        _check_full_batch_receipts(env, rows, actions, hold_at_goal, place_in_bin, place_on_intake)

        # Return resets start empty-handed away from home, with an earlier incoming parcel and no credit.
        full_bucket = 4 * cfg.events.reset_parcels.params["samples_per_path"]
        rows.fill_(full_bucket - 1)
        env.reset(env_ids=torch.tensor([1], device=device))
        tool = end_effector_pose(env)[0][1] - env.scene.env_origins[1]
        active = env.parcels[env.active_ids[1].item()].data.root_link_pose_w.torch[1, :3] - env.scene.env_origins[1]
        assert tool[1] > 0 and tool[2] > 0.25, "The return start must restore the tool toward an outfeed."
        assert active[0] < 0.5 and active[1] < -0.47, "The next parcel must approach from the early left turn."
        assert not env.gripper.engaged[1] and not command.lifted[1] and not command.placed[1]
        assert not command.delivered[1].any() and not command.new_success[1]
        assert command.metrics["correct_deliveries"][1] == 0
        observations = env.observation_manager.compute()["policy"]
        assert observations.shape == (2, 312) and torch.isfinite(observations).all()
