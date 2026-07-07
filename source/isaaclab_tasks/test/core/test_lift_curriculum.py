# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the adaptive rigid-object lift curriculum."""

from types import SimpleNamespace

import torch

from isaaclab.managers import CurriculumTermCfg, TerminationTermCfg

from isaaclab_tasks.core.lift.mdp import (
    LiftDifficultyScheduler,
    ObjectPoseHeld,
    curriculum_object_below_reset_height,
    object_goal_pose_accuracy,
)


class _TerminationManager:
    def __init__(self, success: torch.Tensor):
        self.success = success

    def get_term(self, _: str) -> torch.Tensor:
        return self.success


def test_lift_difficulty_requires_success_termination() -> None:
    """Only the task's success termination should advance the curriculum."""
    termination_manager = _TerminationManager(torch.tensor([False, True]))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        termination_manager=termination_manager,
    )
    cfg = CurriculumTermCfg(
        func=LiftDifficultyScheduler,
        params={
            "success_termination_name": "success",
            "max_difficulty": 20,
            "successes_to_promote": 3,
        },
    )
    scheduler = LiftDifficultyScheduler(cfg, env)
    env_ids = torch.arange(2)

    for _ in range(3):
        scheduler(env, env_ids, **cfg.params)

    assert scheduler.difficulties.tolist() == [0, 1]
    assert scheduler.success_streak.tolist() == [0, 0]


def test_lift_difficulty_resets_streak_after_failure() -> None:
    """Successes separated by a failed episode should not promote difficulty."""
    termination_manager = _TerminationManager(torch.tensor([True]))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        termination_manager=termination_manager,
    )
    cfg = CurriculumTermCfg(
        func=LiftDifficultyScheduler,
        params={
            "success_termination_name": "success",
            "max_difficulty": 20,
            "successes_to_promote": 3,
        },
    )
    scheduler = LiftDifficultyScheduler(cfg, env)

    scheduler(env, torch.tensor([0]), **cfg.params)
    termination_manager.success[0] = False
    scheduler(env, torch.tensor([0]), **cfg.params)
    termination_manager.success[0] = True
    scheduler(env, torch.tensor([0]), **cfg.params)

    assert scheduler.difficulties.item() == 0
    assert scheduler.success_streak.item() == 1


def test_object_pose_held_requires_continuous_position_and_orientation_accuracy() -> None:
    """Success should require every tolerance to hold for the configured duration."""
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    rotated = identity.clone()
    rotated[1] = torch.tensor([0.0, 0.0, 0.1, 0.994987])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.zeros(2, 3)),
            root_quat_w=SimpleNamespace(torch=identity),
        )
    )
    object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]])),
            root_quat_w=SimpleNamespace(torch=rotated),
        )
    )
    command = torch.tensor(
        [
            [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        scene={"robot": robot, "object": object},
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    cfg = TerminationTermCfg(
        func=ObjectPoseHeld,
        params={
            "command_name": "object_pose",
            "position_threshold": 0.02,
            "orientation_threshold": 0.15,
            "hold_time": 1.0,
        },
    )
    accuracy = object_goal_pose_accuracy(
        env,
        command_name="object_pose",
        position_threshold=0.02,
        orientation_threshold=0.15,
    )
    assert accuracy.tolist() == [1.0, 0.0]
    success = ObjectPoseHeld(cfg, env)

    for _ in range(49):
        assert not success(env, **cfg.params).any()

    assert success(env, **cfg.params).tolist() == [True, False]
    object.data.root_pos_w.torch[0, 0] = 0.1
    assert not success(env, **cfg.params).any()
    assert success.consecutive_steps.tolist() == [0, 0]


def test_curriculum_drop_height_rejects_failed_pregrasp_without_changing_final_floor() -> None:
    """Early lifted resets should fail above the final task's below-table floor."""
    scheduler_env = SimpleNamespace(num_envs=2, device="cpu")
    scheduler_cfg = CurriculumTermCfg(
        func=LiftDifficultyScheduler,
        params={"initial_difficulty": 0, "max_difficulty": 40},
    )
    scheduler = LiftDifficultyScheduler(scheduler_cfg, scheduler_env)
    scheduler.difficulties[:] = torch.tensor([0, 40])
    object = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.24], [0.0, 0.0, -0.04]])))
    )
    scene = type("Scene", (), {"env_origins": torch.zeros(2, 3), "__getitem__": lambda _, name: object})()
    env = SimpleNamespace(
        scene=scene,
        curriculum_manager=SimpleNamespace(
            cfg=SimpleNamespace(lift_difficulty=SimpleNamespace(func=scheduler)),
        ),
    )

    dropped = curriculum_object_below_reset_height(
        env,
        high_object_height=0.349140,
        low_object_height=0.029296,
        transition_start=0.30,
        transition_end=0.55,
        height_margin=0.10,
        minimum_height=-0.05,
    )
    assert dropped.tolist() == [True, False]

    object.data.root_pos_w.torch[1, 2] = -0.06
    assert curriculum_object_below_reset_height(
        env,
        high_object_height=0.349140,
        low_object_height=0.029296,
        transition_start=0.30,
        transition_end=0.55,
        height_margin=0.10,
        minimum_height=-0.05,
    ).tolist() == [True, True]
