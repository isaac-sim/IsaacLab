# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for restoring Franka Pour reset-dataset rows into simulation assets."""

from types import SimpleNamespace

import torch

import isaaclab_tasks.contrib.franka_pour.pour_env as pour_env_module
from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv


class _RobotRecorder:
    def __init__(self):
        self.position_writes = []
        self.velocity_writes = []
        self.position_targets = []
        self.data = SimpleNamespace(body_link_pose_w=torch.empty(0))

    def write_joint_position_to_sim_index(self, *, position, **_kwargs):
        self.position_writes.append(position.clone())

    def write_joint_velocity_to_sim_index(self, *, velocity, **_kwargs):
        self.velocity_writes.append(velocity.clone())

    def set_joint_position_target_index(self, *, target, **_kwargs):
        self.position_targets.append(target.clone())


class _RigidRecorder:
    def __init__(self):
        self.root_pose = None
        self.root_velocity = None

    def write_root_pose_to_sim_index(self, *, root_pose, **_kwargs):
        self.root_pose = root_pose.clone()

    def write_root_velocity_to_sim_index(self, *, root_velocity, **_kwargs):
        self.root_velocity = root_velocity.clone()


class _MediaRecorder:
    def __init__(self):
        self.position = None
        self.velocity = None

    def write_particle_pos_to_sim_index(self, position, **_kwargs):
        self.position = position.clone()

    def write_particle_velocity_to_sim_index(self, velocity, **_kwargs):
        self.velocity = velocity.clone()


def test_reset_dataset_restores_exact_state_and_clears_solver_history(monkeypatch):
    robot = _RobotRecorder()
    source_cup = _RigidRecorder()
    target_cup = _RigidRecorder()
    media = _MediaRecorder()
    gripper_reset = []
    gripper_action = SimpleNamespace(
        set_reset_position=lambda position, **_kwargs: gripper_reset.append(position.clone())
    )
    rows = torch.tensor((1, 0))
    identity = torch.tensor((0.0, 0.0, 0.0, 1.0))
    states = {
        "category": torch.tensor((0, 1), dtype=torch.int8),
        "arm_joint_position": torch.arange(14, dtype=torch.float32).reshape(2, 7),
        "arm_joint_velocity": torch.arange(14, dtype=torch.float32).reshape(2, 7) * 0.01,
        "finger_joint_position": torch.tensor(((0.01, 0.01), (0.028, 0.028))),
        "finger_joint_velocity": torch.tensor(((0.1, 0.1), (0.2, 0.2))),
        "finger_joint_target": torch.tensor(((0.012, 0.012), (0.024, 0.024))),
        "source_root_pose": torch.stack(
            (
                torch.cat((torch.tensor((0.4, 0.1, 0.0)), identity)),
                torch.cat((torch.tensor((0.6, -0.2, 0.3)), identity)),
            )
        ),
        "source_root_velocity": torch.arange(12, dtype=torch.float32).reshape(2, 6) * 0.01,
        "target_root_pose": torch.stack(
            (
                torch.cat((torch.tensor((0.5, -0.2, 0.0)), identity)),
                torch.cat((torch.tensor((0.7, 0.2, 0.0)), identity)),
            )
        ),
        "target_root_velocity": torch.arange(12, dtype=torch.float32).reshape(2, 6) * 0.02,
        "particle_layout_id": torch.zeros(2, dtype=torch.int32),
    }
    solver_resets = []
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        reset_dataset_row_id=rows,
        _reset_dataset_states=states,
        _reset_dataset_particle_local_position=torch.tensor((((0.0, 0.0, 0.01), (0.01, 0.0, 0.02)),)),
        _reset_dataset_particle_local_velocity=torch.zeros((1, 2, 3)),
        _robot=robot,
        _arm_joint_ids=torch.arange(7),
        _finger_joint_ids=torch.tensor((7, 8)),
        action_manager=SimpleNamespace(
            get_term=lambda name: gripper_action if name == "gripper_action" else SimpleNamespace()
        ),
        env_origins=torch.tensor(((10.0, 0.0, 0.0), (20.0, 0.0, 0.0))),
        _source_cup=source_cup,
        _target_cup=target_cup,
        _media=media,
        _last_source_bank_index=torch.zeros(2, dtype=torch.long),
        _last_arm_bank_index=torch.zeros(2, dtype=torch.long),
        _last_target_bank_index=torch.zeros(2, dtype=torch.long),
        _particle_region_cache=object(),
        _particle_region_cache_step=1,
        episode_succeeded=torch.ones(2, dtype=torch.bool),
        ep_max_target_frac=torch.ones(2),
        _success_dwell_count=torch.ones(2, dtype=torch.long),
        _lost_grasp_dwell_count=torch.ones(2, dtype=torch.long),
        _lifted_grasp_seen=torch.ones(2, dtype=torch.bool),
        _target_entry_seen=torch.ones((2, 2), dtype=torch.bool),
        _held_delivered=torch.ones((2, 2), dtype=torch.bool),
        _held_delivery_tracker_step=1,
    )
    monkeypatch.setattr(
        pour_env_module.NewtonManager,
        "reset_solver_state",
        lambda **kwargs: solver_resets.append(kwargs),
    )

    FrankaPourEnv._reset_from_dataset(env, torch.arange(2), torch.ones(2, dtype=torch.bool))

    torch.testing.assert_close(robot.position_writes[0], states["arm_joint_position"][rows])
    torch.testing.assert_close(robot.velocity_writes[0], states["arm_joint_velocity"][rows])
    torch.testing.assert_close(robot.position_writes[1], states["finger_joint_position"][rows])
    torch.testing.assert_close(robot.position_targets[1], states["finger_joint_target"][rows])
    torch.testing.assert_close(gripper_reset[0], states["finger_joint_target"][rows, :1])
    torch.testing.assert_close(
        source_cup.root_pose[:, :3],
        states["source_root_pose"][rows, :3] + env.env_origins,
    )
    torch.testing.assert_close(target_cup.root_velocity, states["target_root_velocity"][rows])
    expected_particles = env._reset_dataset_particle_local_position.expand(2, -1, -1).clone()
    expected_particles += source_cup.root_pose[:, None, :3]
    torch.testing.assert_close(media.position, expected_particles)
    torch.testing.assert_close(media.velocity, torch.zeros_like(media.velocity))
    assert len(solver_resets) == 1
    assert (
        solver_resets[0]["flags"] == pour_env_module.newton.StateFlags.BODY | pour_env_module.newton.StateFlags.PARTICLE
    )
    assert env._particle_region_cache is None
    assert not bool(env.episode_succeeded.any())
    assert env._lifted_grasp_seen.tolist() == [True, False]
