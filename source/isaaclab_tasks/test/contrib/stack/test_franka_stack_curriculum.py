# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab.managers import CurriculumManager, CurriculumTermCfg, EventManager, EventTermCfg

from isaaclab_tasks.contrib.stack.constants import FRANKA_STACK_ARM_WORKSPACE_LOWER, FRANKA_STACK_ARM_WORKSPACE_UPPER
from isaaclab_tasks.contrib.stack.mdp.curriculums import StackResetTableCurriculum
from isaaclab_tasks.contrib.stack.mdp.reset_events import StackResetRecipe, StackResetStateTable
from isaaclab_tasks.contrib.stack.mdp.runtime_state import get_stack_reset_runtime_state
from isaaclab_tasks.utils import SuccessMonitorCfg

pytestmark = pytest.mark.kitless


def test_curriculum_credits_the_completed_row_before_reset() -> None:
    env = _make_env({"fixed_recipe": int(StackResetRecipe.SECOND_TRANSPORT)})
    robot = env.scene["robot"]
    origins = env.scene.env_origins
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
    assert metrics["Curriculum/reset_sampling/table_probability"].item() == pytest.approx(0.35)
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

    term.params["fixed_recipe"] = None
    env.reset_buf.zero_()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        curriculum.compute(env_ids)
        env.event_manager.apply("reset", env_ids=env_ids, global_env_step_count=5)
        sampled_rows = state.row_ids.clone()
        sampled_roles = state.role_to_cube.clone()

        torch.manual_seed(42)
        curriculum.compute(env_ids)
        env.event_manager.apply("reset", env_ids=env_ids, global_env_step_count=6)

    torch.testing.assert_close(state.row_ids, sampled_rows)
    torch.testing.assert_close(state.role_to_cube, sampled_roles)
    assert curriculum.reset(env_ids)["Curriculum/reset_sampling/full_task_attempts"].item() == 1


def test_reset_rejects_arm_noise_that_pushes_a_held_cube_into_its_support() -> None:
    env = _make_env({"fixed_recipe": int(StackResetRecipe.FINAL_RELEASE), "fixed_role_permutation": 0})
    event = env.event_manager.get_term_cfg("reset_from_state_buffer")
    state = get_stack_reset_runtime_state(env)
    robot = env.scene["robot"]
    top_cube = env.scene["cube_3"]
    support_cube = env.scene["cube_2"]

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        env.event_manager.apply("reset", env_ids=slice(None), global_env_step_count=0)
        initial_rows = state.row_ids.clone()
        held = state.held_cube_ids == 2
        assert held.any() and not held.all()
        authored_arm = robot.write_joint_position_to_sim_index.call_args.kwargs["position"]
        authored_top = top_cube.write_root_pose_to_sim_index.call_args.kwargs["root_pose"]

        event.params["arm_joint_noise_range"] = 0.01
        torch.manual_seed(1)
        env.event_manager.apply("reset", env_ids=slice(None), global_env_step_count=1)

    torch.testing.assert_close(state.row_ids, initial_rows)
    arm = robot.write_joint_position_to_sim_index.call_args.kwargs["position"]
    target = robot.actuators.target_command.set_position_index.call_args.kwargs["value"]
    top = top_cube.write_root_pose_to_sim_index.call_args.kwargs["root_pose"]
    support = support_cube.write_root_pose_to_sim_index.call_args.kwargs["root_pose"]
    # With this draw the held cube penetrates its support by 4 mm without validation.
    assert (top[held, 2] - support[held, 2] >= 0.04 - 1.0e-4).all()
    torch.testing.assert_close(top[held], authored_top[held], rtol=0.0, atol=0.0)
    torch.testing.assert_close(arm[held], authored_arm[held], rtol=0.0, atol=0.0)
    torch.testing.assert_close(target, arm, rtol=0.0, atol=0.0)
    assert not torch.equal(arm[~held], authored_arm[~held])


def test_reset_uses_the_configured_action_workspace() -> None:
    # Exclude table starts on one side of the default controller workspace.
    workspace_upper = (0.0, *FRANKA_STACK_ARM_WORKSPACE_UPPER[1:])
    env = _make_env(
        {"fixed_recipe": int(StackResetRecipe.TABLE), "table_arm_joint_noise_range": 0.05},
        workspace_upper=workspace_upper,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        env.event_manager.apply("reset", env_ids=slice(None), global_env_step_count=0)
    robot = env.scene["robot"]
    arm = robot.write_joint_position_to_sim_index.call_args.kwargs["position"][:, :7]
    assert (arm[:, 0] <= workspace_upper[0]).all()


def _make_env(
    reset_params: dict[str, int | float], workspace_upper: tuple[float, ...] = FRANKA_STACK_ARM_WORKSPACE_UPPER
) -> SimpleNamespace:
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
    arm_action = SimpleNamespace(workspace_lower=FRANKA_STACK_ARM_WORKSPACE_LOWER, workspace_upper=workspace_upper)
    env = SimpleNamespace(
        cfg=SimpleNamespace(actions=SimpleNamespace(arm_action=arm_action)),
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        sim=SimpleNamespace(is_playing=lambda: True),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        reset_buf=torch.zeros(num_envs, dtype=torch.bool),
        termination_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=outcome)),
    )
    env.event_manager = EventManager(
        {"reset_from_state_buffer": EventTermCfg(func=StackResetStateTable, mode="reset", params=reset_params)},
        env,
    )
    return env


class _Scene(dict):
    def __init__(self, entities: dict[str, Mock], origins: torch.Tensor) -> None:
        super().__init__(entities)
        self.env_origins = origins
