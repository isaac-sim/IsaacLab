# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""G1 capture and motion contracts, including stationary padding beyond recorded clip ends."""

import os
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import h5py
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.app import launch_simulation
from isaaclab.managers import DatasetExportMode, RewardTermCfg, TerminationTermCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.string import resolve_matching_names_values

from isaaclab_tasks.contrib.g1_tracking.g1_tracking_env_cfg import TransitionRecorderCfg
from isaaclab_tasks.contrib.g1_tracking.kinematics import TRACKED_BODIES
from isaaclab_tasks.contrib.g1_tracking.motion import MotionLib
from isaaclab_tasks.contrib.g1_tracking.tracking import unflatten_state
from isaaclab_tasks.utils import resolve_task_config


def test_motion_rates_and_stationary_endpoint_padding(tmp_path):
    raw = np.zeros((121, 36), dtype=np.float32)
    raw[:, 2], raw[:, 6] = 0.8, 1.0
    raw[:, 0] = np.linspace(0, 0.2, len(raw))
    roll = np.linspace(0, 0.2, len(raw))
    raw[:, 3], raw[:, 6] = np.sin(roll / 2), np.cos(roll / 2)
    raw[:, 7] = np.linspace(0, 0.1, len(raw))
    for directory in ("a", "b"):
        (tmp_path / directory).mkdir()
        raw[:, 0] = np.linspace(0, 0.2 if directory == "a" else 0.4, len(raw))
        np.savetxt(tmp_path / directory / "walk.csv", raw, delimiter=",")
    # Kinematics is an independent dependency here; this test checks temporal pose/rate semantics.
    kin = SimpleNamespace(forward=lambda p, R, v, w, q, qd: (p[:, None], R[:, None], v[:, None], w[:, None]))
    motion = MotionLib(str(tmp_path), kin, "cpu", src_fps=120)
    assert motion.names == ["a/walk", "b/walk"]
    assert motion.clip_len.tolist() == [51, 51] and motion.num_bins == 2
    expected_v = torch.tensor([[0.2, 0, 0]] * 51 + [[0.4, 0, 0]] * 51)
    torch.testing.assert_close(motion.root_v, expected_v, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(motion.velocity_command[:, :2], expected_v[:, :2], atol=2e-6, rtol=1e-5)
    assert motion.future_frames(torch.tensor([49, 100]), 3).tolist() == [[50, 50, 50], [101, 101, 101]]
    goals = motion.goal(torch.tensor([47, 49, 50, 100]), 3).reshape(4, 3, 73)
    # Recorded endpoint rates remain factual; only slots beyond the clip describe a stationary held pose.
    expected_velocity = torch.tensor([[0.2, 0.2, 0.2], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    torch.testing.assert_close(goals[:, :, 9], expected_velocity, atol=2e-6, rtol=1e-5)
    expected_velocity[-1, 0] = 0.2
    torch.testing.assert_close(goals[:, :, 12], expected_velocity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(goals[:, :, 44], expected_velocity * 0.05, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(goals[1:, :, 15], torch.full((3, 3), 0.1), atol=2e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="frame rate"):
        MotionLib(str(tmp_path), kin, "cpu", src_fps=0)


def never(env):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def task(tmp_path_factory, request):
    torch.backends.cuda.matmul.allow_tf32 = False
    output = tmp_path_factory.mktemp("native_task")
    cfg, _ = resolve_task_config(
        "IsaacContrib-Motion-Tracking-G1", None, overrides=["env.scene.num_envs=40", "env.seed=123"]
    )
    cfg.sim.device = request.param
    data_dir = os.environ.get("ISAACLAB_G1_TRACKING_DATA")
    if not data_dir:
        pytest.skip("Set ISAACLAB_G1_TRACKING_DATA to the prepared G1 tracking directory.")
    cfg.scene.robot.spawn.usd_path = str(Path(data_dir) / "g1_usd/g1_29dof/g1_29dof.usda")
    cfg.commands.motion.kinematics_path = str(Path(data_dir) / "g1_description/g1_29dof.xml")
    cfg.commands.motion.motion_dir = str(Path(data_dir) / "lafan1_g1")
    cfg.recorders.transition = TransitionRecorderCfg()
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.history_length = 3
    cfg.commands.motion.goal_frames = 2
    cfg.commands.motion.reset_noise = False
    cfg.commands.motion.velocity.fraction = 0.5
    cfg.commands.motion.velocity.resampling_steps = 2
    cfg.recorders.transition.record_to_episodes = True
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    cfg.recorders.dataset_export_dir_path = str(output)
    cfg.recorders.buffer_initial_capacity = 2
    for name in ("failed", "clip_end", "time_out"):
        getattr(cfg.terminations, name).func = never
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Motion-Tracking-G1", cfg=cfg).unwrapped
        try:
            yield env, output
        finally:
            env.close()


def test_real_capture_observations_context_and_offline_export(task):
    env, output = task
    command = env.command_manager.get_term("motion")
    recorder = env.recorder_manager.get_term("transition")
    observations, _ = env.reset()
    assert env.cfg.sim.physics.use_cuda_graph
    robot = command.robot
    model = mujoco.MjModel.from_xml_path(command.cfg.kinematics_path)
    expected_com = torch.tensor(model.body("torso_link").ipos, device=env.device, dtype=torch.float32)
    torch.testing.assert_close(
        robot.data.body_com_pos_b.torch[:, command.torso_id], expected_com + command.xi[:, 30:33]
    )
    mu = wp.to_torch(command.robot_view.get_attribute("shape_material_mu", command.physics.get_model()))
    torch.testing.assert_close(mu, command.xi[:, :1, None].expand_as(mu))
    previous_action = torch.zeros(env.num_envs, 29, device=env.device)
    records = []
    for step in range(3):
        state = torch.cat(
            (
                robot.data.root_link_pose_w.torch,
                robot.data.root_link_vel_w.torch,
                robot.data.joint_pos.torch,
                robot.data.joint_vel.torch,
            ),
            -1,
        ).clone()
        frame = command.frame.clone()
        velocity = command.velocity_command.clone()
        source = command.velocity_reference.clone()
        action = torch.full_like(previous_action, 0.01 * (step + 1))
        observations, reward, terminated, truncated, _ = env.step(action)
        transition = recorder.latest
        assert not (terminated | truncated).any()
        torch.testing.assert_close(transition["s"], state, rtol=0, atol=0)
        torch.testing.assert_close(transition["a_prev"], previous_action, rtol=0, atol=0)
        torch.testing.assert_close(transition["a"], action, rtol=0, atol=0)
        torch.testing.assert_close(transition["r"], reward, rtol=0, atol=0)
        torch.testing.assert_close(transition["f"], frame, rtol=0, atol=0)
        torch.testing.assert_close(transition["velocity_command"], velocity, rtol=0, atol=0)
        torch.testing.assert_close(transition["velocity_reference"], source, rtol=0, atol=0)
        torch.testing.assert_close(transition["is_velocity"], command.is_velocity)
        analytical = command.mdp.obs(command.state(), action, command.xi, command.frame)
        analytical[command.is_velocity, -6:] = 0
        torch.testing.assert_close(observations["policy"][:, -1], analytical, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(observations["goal"], command.motion.goal(command.frame, 2))
        torch.testing.assert_close(observations["interface"][:, 1:], command.velocity_command)
        if step == 0:
            # Non-reference commands are held until the second control step.
            torch.testing.assert_close(command.velocity_command[~source], velocity[~source])
        if step == 1:
            assert not torch.equal(command.velocity_command, velocity)
        if source.any():
            torch.testing.assert_close(velocity[source], command.motion.velocity_command[frame[source]])
        successor = unflatten_state(transition["s1"])
        expected_reward = (
            torch.where(
                command.is_velocity,
                command.mdp.velocity_reward(successor, velocity, action, previous_action),
                command.mdp.reward(successor, frame + 1, action, previous_action),
            )
            * env.step_dt
        )
        torch.testing.assert_close(reward, expected_reward)
        action_term = env.action_manager.get_term("joint_pos")
        _, _, values = resolve_matching_names_values(env.cfg.actions.joint_pos.scale, robot.joint_names)
        scale = torch.tensor(values, device=env.device)
        expected_target = command.mdp.default_q + command.xi[:, 1:30] + scale * action
        torch.testing.assert_close(action_term.processed_actions, expected_target)
        records.append({key: value.clone() for key, value in transition.items()})
        previous_action = action
    body_ids = [robot.body_names.index(name) for name in TRACKED_BODIES]
    positions, rotations, velocities, angular = command.kin.forward(*command.state())
    torch.testing.assert_close(positions, robot.data.body_link_pos_w.torch[:, body_ids], atol=5e-5, rtol=1e-5)
    torch.testing.assert_close(
        rotations, matrix_from_quat(robot.data.body_link_quat_w.torch[:, body_ids]), atol=5e-5, rtol=1e-5
    )
    torch.testing.assert_close(velocities, robot.data.body_link_lin_vel_w.torch[:, body_ids], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(angular, robot.data.body_link_ang_vel_w.torch[:, body_ids], atol=1e-4, rtol=1e-4)
    # Velocity failures depend on actual falls, not deviation from a reference motion.
    upright = list(command.state())
    upright[0] = torch.zeros_like(upright[0])
    upright[0][:, 2] = 0.8
    upright[1] = torch.eye(3, device=env.device).expand(env.num_envs, -1, -1)
    upright[2], upright[3], upright[5] = (torch.zeros_like(upright[i]) for i in (2, 3, 5))
    target = torch.zeros(env.num_envs, 3, device=env.device)
    action = torch.zeros(env.num_envs, 29, device=env.device)
    matched = command.mdp.velocity_reward(upright, target, action, action)
    target[:, 0] = 0.5
    error = matched - command.mdp.velocity_reward(upright, target, action, action)
    torch.testing.assert_close(error, torch.full_like(error, 2 * (1 - np.exp(-1))))
    assert not command.mdp.velocity_termination(upright).any()
    upright[0][:, 2] = 0.2
    assert command.mdp.velocity_termination(upright).all()
    assert (command.mdp.velocity_termination(upright, soft=True) > 0.99).all()
    snapshot = env.recorder_manager.get_episode(0).data["transition"]
    for key in records[0]:
        torch.testing.assert_close(snapshot[key], torch.stack([record[key][0] for record in records]), rtol=0, atol=0)
    env.recorder_manager.export_episodes([0])
    with h5py.File(output / "dataset.hdf5", "r") as dataset:
        saved = dataset["data/demo_0/transition"]
        for key in snapshot:
            torch.testing.assert_close(torch.from_numpy(saved[key][:]), snapshot[key].cpu(), rtol=0, atol=0)


def test_terminal_capture_and_return_accounting(task):
    env, _ = task
    command = env.command_manager.get_term("motion")
    capture = env.recorder_manager.get_term("transition")
    reward_cfg = env.reward_manager.get_term_cfg("tracking")
    term_cfgs = {name: env.termination_manager.get_term_cfg(name) for name in ("failed", "clip_end", "time_out")}
    gamma = command.cfg.gamma
    try:
        env.reward_manager.set_term_cfg(
            "tracking",
            RewardTermCfg(
                func=lambda env: (env.episode_length_buf + 10 * command.is_velocity) / env.step_dt, weight=1.0
            ),
        )
        env.termination_manager.set_term_cfg(
            "failed",
            TerminationTermCfg(
                func=lambda env: (
                    (env.episode_length_buf == 3) & (torch.arange(env.num_envs, device=env.device) % 2 == 0)
                )
            ),
        )
        env.termination_manager.set_term_cfg(
            "time_out",
            TerminationTermCfg(
                func=lambda env: (
                    (env.episode_length_buf == 3) & (torch.arange(env.num_envs, device=env.device) % 2 == 1)
                ),
                time_out=True,
            ),
        )
        command.cfg.gamma = 0.9
        env.reset()
        command.bin_return.zero_()
        command.start_bin.zero_()
        for _ in range(3):
            applied_command = command.velocity_command.clone()
            applied_source = command.velocity_reference.clone()
            env.step(torch.zeros(env.num_envs, 29, device=env.device))
        torch.testing.assert_close(capture.latest["velocity_command"], applied_command, rtol=0, atol=0)
        torch.testing.assert_close(capture.latest["velocity_reference"], applied_source, rtol=0, atol=0)
        assert capture.latest["done"].all()
        assert capture.latest["term"].sum() == env.num_envs // 2
        assert (capture.latest["ep_step"] == 2).all()
        assert (env.episode_length_buf == 0).all()
        assert (capture.latest["s1"][:, :3] != command.state()[0]).any()
        # Motion returns: 1,2,3 -> 5.23; timeout adds 14.58. Larger velocity returns must not affect these bins.
        torch.testing.assert_close(command.bin_return[0], torch.tensor(0.2 * (5.23 + 19.81) / 2, device=env.device))
        before = command.bin_return.clone()
        env.reset()
        torch.testing.assert_close(command.bin_return, before, rtol=0, atol=0)
    finally:
        command.cfg.gamma = gamma
        env.reward_manager.set_term_cfg("tracking", reward_cfg)
        for name, cfg in term_cfgs.items():
            env.termination_manager.set_term_cfg(name, cfg)
