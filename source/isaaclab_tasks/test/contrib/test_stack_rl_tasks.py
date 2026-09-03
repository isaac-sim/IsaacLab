# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public-contract tests for the reset-oriented Franka and KUKA stack tasks."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_distillation_cfg import (
    StackDistillationAlgorithmCfg,
    StackVisualDistillationModelCfg,
)
from isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_ppo_cfg import (
    StackGaussianDistribution,
)
from isaaclab_tasks.contrib.stack.config.kuka_allegro.agents.rsl_rl_ppo_cfg import (
    KukaAllegroGaussianDistribution,
)
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import (
    KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
    KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,
)
from isaaclab_tasks.contrib.stack.mdp.runtime_state import (
    create_stack_reset_runtime_state,
    get_stack_reset_runtime_state,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

FRANKA_STATE_TASK = "IsaacContrib-Stack-Cube-Franka-RL"
FRANKA_CAMERA_TASK = "IsaacContrib-Stack-Cube-Franka-RL-Camera"
FRANKA_DISTILLATION_TASK = "IsaacContrib-Stack-Cube-Franka-RL-Camera-Distillation"
KUKA_STATE_TASK = "IsaacContrib-Stack-Cube-KukaAllegro-RL"


@pytest.mark.parametrize(
    ("task_name", "env_cfg_name", "runner_cfg_name"),
    (
        (FRANKA_STATE_TASK, "FrankaCubeStackRLEnvCfg", "FrankaStackPPORunnerCfg"),
        (FRANKA_CAMERA_TASK, "FrankaCubeStackCameraRLEnvCfg", "FrankaStackCameraPPORunnerCfg"),
        (
            FRANKA_DISTILLATION_TASK,
            "FrankaCubeStackCameraDistillationEnvCfg",
            "FrankaStackCameraDistillationRunnerCfg",
        ),
        (KUKA_STATE_TASK, "KukaAllegroCubeStackRLEnvCfg", "KukaAllegroStackPPORunnerCfg"),
    ),
)
def test_supported_tasks_are_registered(task_name: str, env_cfg_name: str, runner_cfg_name: str):
    """Each supported task resolves one environment and one RSL-RL configuration."""
    spec = gym.spec(task_name)

    assert spec.kwargs["env_cfg_entry_point"].endswith(f":{env_cfg_name}")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(f":{runner_cfg_name}")
    assert load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point") is not None


def test_obsolete_camera_fine_tune_task_is_not_registered():
    with pytest.raises(gym.error.Error):
        gym.spec("IsaacContrib-Stack-Cube-Franka-RL-Camera-Finetune")


@pytest.fixture(scope="module")
def stack_cfgs():
    cfgs = {
        task_name: parse_env_cfg(task_name, device="cuda:0", num_envs=8)
        for task_name in (FRANKA_STATE_TASK, FRANKA_CAMERA_TASK, FRANKA_DISTILLATION_TASK, KUKA_STATE_TASK)
    }
    for cfg in cfgs.values():
        cfg.validate_config()
    return cfgs


def test_franka_state_task_exposes_the_training_contract(stack_cfgs):
    cfg = stack_cfgs[FRANKA_STATE_TASK]

    assert cfg.scene.num_envs == 8
    assert cfg.scene.replicate_physics
    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert cfg.sim.physics.num_substeps == 2
    assert cfg.sim.physics.collision_decimation == 1
    assert cfg.decimation == cfg.sim.render_interval == 2
    assert cfg.actions.arm_action.gravity_compensation
    assert cfg.actions.arm_action.scale == cfg.actions.arm_action.max_delta == 0.05
    assert isinstance(cfg.actions.gripper_action, mdp.ResetBufferedGripperActionCfg)
    assert cfg.events.reset_from_state_buffer.func is mdp.StackResetStateTable
    assert cfg.curriculum.reset_sampling.func is mdp.StackResetTableCurriculum
    assert cfg.terminations.progress_context.func is mdp.StableOrderInvariantStackGoal
    assert cfg.rewards.success.func is mdp.stack_success_pulse
    assert cfg.observations.policy.object.func is mdp.role_conditioned_stack_obs
    assert cfg.observations.policy.cube_positions is None
    assert cfg.observations.policy.cube_orientations is None
    assert cfg.scene.cube_1.spawn.size == (0.04, 0.04, 0.04)
    assert cfg.scene.cube_1.spawn.physics_material.contact_stiffness == 1.0e4
    assert cfg.scene.table_contact_surface.spawn.physics_material.contact_stiffness == 1.0e4


def test_camera_actor_has_only_deployable_observations(stack_cfgs):
    cfg = stack_cfgs[FRANKA_CAMERA_TASK]
    runner = load_cfg_from_registry(FRANKA_CAMERA_TASK, "rsl_rl_cfg_entry_point")

    assert cfg.scene.base_camera.height == cfg.scene.base_camera.width == 128
    assert cfg.scene.base_camera.data_types == ["rgb"]
    assert cfg.num_rerenders_on_reset == 1
    assert cfg.events.reset_from_state_buffer.params["fixed_role_permutation"] == 0
    assert not hasattr(cfg.observations.policy, "object")
    assert hasattr(cfg.observations, "base_image")
    assert not hasattr(cfg.observations, "teacher")
    assert runner.obs_groups["actor"] == ["policy", "base_image"]
    assert runner.obs_groups["critic"] == ["policy", "privileged"]
    assert "privileged" not in runner.obs_groups["actor"]


def test_distillation_task_adds_privileged_labels_without_changing_the_student(stack_cfgs):
    cfg = stack_cfgs[FRANKA_DISTILLATION_TASK]
    runner = load_cfg_from_registry(FRANKA_DISTILLATION_TASK, "rsl_rl_cfg_entry_point")

    assert hasattr(cfg.observations, "teacher")
    assert hasattr(cfg.observations, "distillation_context")
    assert cfg.observations.distillation_context.recipe.func is mdp.stack_reset_recipe_one_hot
    assert cfg.events.reset_from_state_buffer.params["evaluation_recipe_ids"] == tuple(range(9))
    assert cfg.events.reset_from_state_buffer.params["evaluation_envs_per_recipe"] == 4
    assert cfg.curriculum.reset_sampling.params["evaluation_env_count"] == 36
    assert runner.obs_groups == {"student": ["policy", "base_image"], "teacher": ["teacher"]}
    assert isinstance(runner.student, StackVisualDistillationModelCfg)
    assert isinstance(runner.algorithm, StackDistillationAlgorithmCfg)
    assert not hasattr(runner.algorithm, "stepwise_student_control")
    assert not hasattr(runner.algorithm, "controller_warmup_updates")


def test_kuka_task_has_one_complete_23_dof_state_policy(stack_cfgs):
    cfg = stack_cfgs[KUKA_STATE_TASK]
    runner = load_cfg_from_registry(KUKA_STATE_TASK, "rsl_rl_cfg_entry_point")

    assert cfg.events.reset_from_state_buffer.func is mdp.KukaAllegroResetStateTable
    assert cfg.actions.arm_action.gravity_compensation
    assert cfg.actions.arm_action.scale == cfg.actions.arm_action.max_delta == 0.12
    assert isinstance(cfg.actions.gripper_action, mdp.ResetPreservingRelativeJointPositionActionCfg)
    assert tuple(cfg.actions.gripper_action.joint_names) == KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES
    assert len(cfg.actions.gripper_action.joint_names) == 16
    assert cfg.actions.gripper_action.scale == cfg.actions.gripper_action.max_delta == 0.10
    assert cfg.terminations.progress_context.func is mdp.StableFullHandOrderInvariantStackGoal
    assert cfg.curriculum.reset_sampling.params["global_sampling"] is True
    assert cfg.scene.cube_1.spawn.size == (KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,) * 3
    assert cfg.observations.policy.cube_x_axes.func is mdp.role_conditioned_cube_x_axes
    assert len(cfg.observations.policy.hand_joint_pos.params["asset_cfg"].joint_names) == 16
    assert len(cfg.observations.policy.hand_joint_vel.params["asset_cfg"].joint_names) == 16
    assert len(cfg.observations.policy.hand_tip_positions.params["body_cfg"].body_names) == 4
    assert not hasattr(cfg.observations.policy, "grasp_pair")
    assert runner.actor.distribution_cfg.arm_action_dim == 7


@pytest.mark.parametrize(
    "task_name",
    (FRANKA_STATE_TASK, FRANKA_CAMERA_TASK, FRANKA_DISTILLATION_TASK, KUKA_STATE_TASK),
)
def test_play_mode_uses_randomized_table_starts(task_name: str):
    cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=4)

    cfg.play_mode()

    assert cfg.events.reset_from_state_buffer.params["fixed_recipe"] == int(mdp.StackResetRecipe.TABLE)
    assert cfg.curriculum is None
    assert cfg.scene.num_envs == 4


def test_mixed_franka_distribution_matches_the_physical_action_space():
    distribution = StackGaussianDistribution(output_dim=8, init_std=0.45)
    output = torch.zeros((256, 8))
    distribution.update(output)

    samples = distribution.sample()

    assert samples.shape == (256, 8)
    assert set(samples[:, -1].unique().tolist()) <= {-1.0, 1.0}
    assert torch.equal(distribution.deterministic_output(output)[:, -1], torch.ones(256))
    assert distribution.log_prob(samples).shape == (256,)
    assert torch.isfinite(distribution.entropy).all()


def test_kuka_distribution_covers_all_arm_and_hand_actions():
    distribution = KukaAllegroGaussianDistribution(output_dim=23)
    output = torch.zeros((32, 23))
    distribution.update(output)

    assert distribution.sample().shape == (32, 23)
    assert torch.allclose(distribution.std[0, :7], torch.full((7,), 0.35), atol=1.0e-6)
    assert torch.allclose(distribution.std[0, 7:], torch.full((16,), 0.15), atol=1.0e-6)


def test_reset_runtime_state_has_one_typed_owner():
    env = SimpleNamespace(num_envs=3, device="cpu")

    state = create_stack_reset_runtime_state(env)

    assert get_stack_reset_runtime_state(env) is state
    assert set(vars(env)) == {"num_envs", "device", "stack_reset_state"}
    assert state.row_ids.shape == (3,)
    assert state.role_to_cube.shape == (3, 3)
    with pytest.raises(AttributeError):
        get_stack_reset_runtime_state(SimpleNamespace())


def _cube(positions: torch.Tensor):
    return SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=positions),
            root_vel_w=SimpleNamespace(torch=torch.zeros((positions.shape[0], 6))),
        )
    )


def test_stack_progress_is_independent_of_cube_identity_and_order():
    positions = (
        torch.tensor(((0.45, 0.00, 0.02), (0.45, 0.00, 0.10))),
        torch.tensor(((0.45, 0.00, 0.06), (0.45, 0.00, 0.02))),
        torch.tensor(((0.60, 0.10, 0.02), (0.45, 0.00, 0.06))),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={f"cube_{index + 1}": _cube(value) for index, value in enumerate(positions)},
    )

    progress = mdp.order_invariant_stack_progress(env)

    torch.testing.assert_close(progress, torch.tensor((1.0, 2.0)))
