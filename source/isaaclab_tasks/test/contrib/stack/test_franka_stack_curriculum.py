# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab.managers import CurriculumManager, CurriculumTermCfg, EventManager, EventTermCfg

from isaaclab_tasks.contrib.stack.mdp.curriculums import StackResetTableCurriculum
from isaaclab_tasks.contrib.stack.mdp.reset_events import StackResetRecipe, StackResetStateTable
from isaaclab_tasks.contrib.stack.mdp.runtime_state import get_stack_reset_runtime_state
from isaaclab_tasks.utils import SuccessMonitorCfg

pytestmark = pytest.mark.kitless


def test_curriculum_credits_the_completed_row_before_reset() -> None:
    num_envs = 4
    robot = Mock()
    robot.data.default_joint_pos.torch = torch.zeros(num_envs, 9)
    robot.data.default_root_pose.torch = torch.zeros(num_envs, 7)
    robot.data.default_root_vel.torch = torch.zeros(num_envs, 6)
    robot.find_joints.side_effect = [(list(range(7)), []), ([7, 8], [])]
    cubes = [Mock() for _ in range(3)]
    for cube in cubes:
        cube.data.default_root_pose.torch = torch.zeros(num_envs, 7)
    entities = {"robot": robot, **{f"cube_{i}": cube for i, cube in enumerate(cubes, start=1)}}
    # The asset mocks record backend writes. The event and curriculum are the real manager terms.
    origins = torch.zeros(num_envs, 3)
    origins[:, 0] = torch.arange(num_envs)
    scene = _Scene(entities, origins)
    outcome = SimpleNamespace(ever_success=torch.tensor([True, False, False, True]))
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        sim=SimpleNamespace(is_playing=lambda: True),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        reset_buf=torch.zeros(num_envs, dtype=torch.bool),
        termination_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=outcome)),
    )
    env.event_manager = EventManager(
        {
            "reset_from_state_buffer": EventTermCfg(
                func=StackResetStateTable, mode="reset", params={"fixed_recipe": int(StackResetRecipe.SECOND_TRANSPORT)}
            )
        },
        env,
    )
    curriculum = CurriculumManager(
        {
            "reset_sampling": CurriculumTermCfg(
                func=StackResetTableCurriculum, params={"success_monitor": SuccessMonitorCfg(monitored_history_len=4)}
            )
        },
        env,
    )
    curriculum.compute(slice(None))
    env.event_manager.apply("reset", env_ids=slice(None), global_env_step_count=0)
    initial = curriculum.reset(slice(None))
    assert "Curriculum/reset_sampling/batch_success_rate" not in initial

    state = get_stack_reset_runtime_state(env)
    old_row = state.row_ids[-1].clone()
    env.episode_length_buf[:] = torch.tensor([4, 4, 0, 4])
    # Environment 0 completed, 1 was reset manually, and 2 has no elapsed steps.
    env.reset_buf[:] = torch.tensor([True, False, True, False])
    env_ids = torch.tensor([0, 1, 2])
    term = env.event_manager.get_term_cfg("reset_from_state_buffer")
    term.params["fixed_recipe"] = int(StackResetRecipe.TABLE)
    curriculum.compute(env_ids)
    env.event_manager.apply("reset", env_ids=env_ids, global_env_step_count=4)
    metrics = curriculum.reset(env_ids)
    assert metrics["Curriculum/reset_sampling/full_task_attempts"].item() == 1
    assert metrics["Curriculum/reset_sampling/recipe_second_transport_attempts"].item() == 1
    assert metrics["Curriculum/reset_sampling/recipe_table_attempts"].item() == 0
    assert metrics["Curriculum/reset_sampling/batch_success_rate"].item() == 1
    assert "Curriculum/reset_sampling/batch_table_full_task_success_rate" not in metrics
    assert state.row_ids[-1] == old_row
    assert (state.recipes[env_ids] == StackResetRecipe.TABLE).all()
    assert torch.equal(robot.write_joint_position_to_sim_index.call_args.kwargs["env_ids"], env_ids)
    root_write = robot.write_root_pose_to_sim_index.call_args.kwargs
    assert torch.equal(root_write["env_ids"], env_ids)
    assert torch.equal(root_write["root_pose"][:, :3], origins[env_ids])

    curriculum.cfg["reset_sampling"].params["table_sampling_probability"] = 0.60
    curriculum.compute(slice(0, 0))
    updated = curriculum.reset(slice(0, 0))
    assert updated["Curriculum/reset_sampling/table_probability"].item() == pytest.approx(0.60)


class _Scene(dict):
    def __init__(self, entities: dict[str, Mock], origins: torch.Tensor) -> None:
        super().__init__(entities)
        self.env_origins = origins
