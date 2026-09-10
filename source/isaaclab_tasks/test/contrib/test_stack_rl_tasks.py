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
from rsl_rl.algorithms import Distillation

from isaaclab.utils import modifiers
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlDistillationAlgorithmCfg, RslRlMLPModelCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.franka.agents.distillation import ClippedTeacherDistillation
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
            "FrankaCubeStackCameraRLEnvCfg",
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


def test_camera_default_batch_is_memory_safe():
    """The checked-in image rollout fits comfortably before CLI scaling."""
    cfg = parse_env_cfg(FRANKA_CAMERA_TASK, device="cuda:0")

    assert cfg.scene.num_envs == 256


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
    assert cfg.actions.arm_action.controller_owns_gravity_compensation
    assert cfg.actions.arm_action.scale == cfg.actions.arm_action.max_delta == 0.05
    assert isinstance(cfg.actions.gripper_action, mdp.ResetBufferedGripperActionCfg)
    assert cfg.events.reset_from_state_buffer.func is mdp.StackResetStateTable
    assert cfg.curriculum.reset_sampling.func is mdp.StackResetTableCurriculum
    assert cfg.terminations.progress_context.func is mdp.StableOrderInvariantStackGoal
    assert cfg.rewards.success.func is mdp.stack_success_pulse
    assert cfg.observations.policy.object.func is mdp.role_conditioned_stack_obs
    assert cfg.observations.policy.joint_target.func is mdp.joint_position_target
    assert not hasattr(cfg.observations.policy, "cube_positions")
    assert not hasattr(cfg.observations.policy, "cube_orientations")
    assert cfg.scene.cube_1.spawn.size == (0.04, 0.04, 0.04)
    assert cfg.scene.cube_1.spawn.physics_material.contact_stiffness == 1.0e4
    assert cfg.scene.table_contact_surface.spawn.physics_material.contact_stiffness == 1.0e4


def test_camera_actor_has_only_deployable_observations(stack_cfgs):
    cfg = stack_cfgs[FRANKA_CAMERA_TASK]
    runner = load_cfg_from_registry(FRANKA_CAMERA_TASK, "rsl_rl_cfg_entry_point")
    image = cfg.observations.base_image.rgb

    assert cfg.scene.base_camera.height == cfg.scene.base_camera.width == 128
    assert cfg.scene.base_camera.data_types == ["rgb"]
    assert image.func is mdp.image
    assert image.params["data_type"] == "rgb"
    assert image.params["normalize"] is False
    assert image.params["permute"] is True
    assert len(image.modifiers) == 1
    assert image.modifiers[0].func is modifiers.scale
    assert image.modifiers[0].params["multiplier"] == pytest.approx(1.0 / 255.0)
    assert isinstance(image.noise, mdp.EpisodeCameraNoiseCfg)
    assert image.clip == (0.0, 1.0)
    assert cfg.num_rerenders_on_reset == 1
    assert cfg.events.reset_from_state_buffer.params["fixed_role_permutation"] == 0
    assert cfg.observations.policy.joint_target.func is mdp.joint_position_target
    assert not hasattr(cfg.observations.policy, "object")
    assert not hasattr(cfg.observations.policy, "eef_position")
    assert not hasattr(cfg.observations.policy, "eef_axes")
    assert not hasattr(cfg.observations.policy, "eef_velocity")
    assert hasattr(cfg.observations, "base_image")
    assert not hasattr(cfg.observations, "teacher")
    assert runner.obs_groups["actor"] == ["policy", "base_image"]
    assert runner.obs_groups["critic"] == ["privileged"]
    assert "privileged" not in runner.obs_groups["actor"]


def test_deployment_inputs_precede_recurrent_action_state(stack_cfgs):
    """LEAPP sees a physical input before registering the previous-action state."""
    for task_name in (FRANKA_STATE_TASK, FRANKA_CAMERA_TASK):
        policy_terms = list(vars(stack_cfgs[task_name].observations.policy))
        assert policy_terms.index("joint_pos") < policy_terms.index("actions")


def test_distillation_task_adds_privileged_labels_without_changing_the_student(stack_cfgs):
    cfg = stack_cfgs[FRANKA_DISTILLATION_TASK]
    state_cfg = stack_cfgs[FRANKA_STATE_TASK]
    runner = load_cfg_from_registry(FRANKA_DISTILLATION_TASK, "rsl_rl_cfg_entry_point")
    camera_runner = load_cfg_from_registry(FRANKA_CAMERA_TASK, "rsl_rl_cfg_entry_point")
    state_runner = load_cfg_from_registry(FRANKA_STATE_TASK, "rsl_rl_cfg_entry_point")

    assert hasattr(cfg.observations, "privileged")
    assert not hasattr(cfg.observations, "teacher")
    assert not hasattr(cfg.observations, "distillation_context")
    assert runner.obs_groups == {"student": ["policy", "base_image"], "teacher": ["privileged"]}
    assert type(runner.student) is type(camera_runner.actor)
    assert runner.student.class_name == camera_runner.actor.class_name
    assert runner.student.hidden_dims == camera_runner.actor.hidden_dims
    assert runner.student.cnn_cfg == camera_runner.actor.cnn_cfg
    assert runner.teacher.class_name == state_runner.actor.class_name
    assert runner.teacher.hidden_dims == state_runner.actor.hidden_dims
    assert runner.teacher.to_dict() == state_runner.actor.to_dict()
    group_settings = {
        "enable_corruption",
        "concatenate_terms",
        "history_length",
        "flatten_history_dim",
        "concatenate_dim",
    }
    state_terms = [name for name in vars(state_cfg.observations.policy) if name not in group_settings]
    teacher_terms = [name for name in vars(cfg.observations.privileged) if name not in group_settings]
    assert teacher_terms == state_terms
    assert isinstance(runner.algorithm, RslRlDistillationAlgorithmCfg)
    assert runner.algorithm.class_name.endswith(":ClippedTeacherDistillation")
    assert runner.num_steps_per_env == 8
    assert runner.init_at_random_ep_len is False


def test_camera_noise_coefficients_are_stable_within_an_episode():
    """Episode photometric coefficients remain fixed until reset."""
    cfg = mdp.EpisodeCameraNoiseCfg(
        noise_cfg=UniformNoiseCfg(n_min=0.0, n_max=0.0),
        exposure_range=(0.5, 1.5),
        contrast_range=(0.8, 1.2),
        white_balance_range=(0.9, 1.1),
        brightness_range=(-0.1, 0.1),
    )
    noise = mdp.EpisodeCameraNoise(cfg, num_envs=4, device="cpu")
    image = torch.linspace(0.0, 1.0, 4 * 3 * 5 * 5).reshape(4, 3, 5, 5)

    first = noise(image)
    second = noise(image)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_camera_noise_reset_resamples_only_selected_environments():
    """A partial reset preserves every unselected environment's calibration."""
    with torch.random.fork_rng():
        torch.manual_seed(7)
        cfg = mdp.EpisodeCameraNoiseCfg(
            noise_cfg=UniformNoiseCfg(n_min=0.0, n_max=0.0),
            exposure_range=(0.5, 1.5),
            contrast_range=(1.0, 1.0),
            white_balance_range=(1.0, 1.0),
            brightness_range=(0.0, 0.0),
        )
        noise = mdp.EpisodeCameraNoise(cfg, num_envs=4, device="cpu")
        image = torch.ones((4, 3, 2, 2))
        before = noise(image)

        noise.reset(torch.tensor([1, 3]))
        after = noise(image)

    torch.testing.assert_close(after[[0, 2]], before[[0, 2]], rtol=0.0, atol=0.0)
    assert not torch.equal(after[[1, 3]], before[[1, 3]])


@pytest.mark.parametrize("shape", ((2, 4, 4, 3), (2, 3, 4)))
def test_camera_noise_rejects_non_nchw_rgb_input(shape: tuple[int, ...]):
    """Photometric randomization rejects non-NCHW RGB tensors."""
    noise = mdp.EpisodeCameraNoise(mdp.EpisodeCameraNoiseCfg(), num_envs=2, device="cpu")

    with pytest.raises(ValueError, match="expects NCHW RGB input"):
        noise(torch.zeros(shape))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("exposure_range", (0.0, 1.0)),
        ("contrast_range", (1.1, 0.9)),
        ("white_balance_range", (-0.1, 1.0)),
        ("brightness_range", (0.1, -0.1)),
    ),
)
def test_camera_noise_rejects_invalid_ranges(field: str, value: tuple[float, float]):
    """Photometric coefficient ranges must be ordered and physically valid."""
    cfg = mdp.EpisodeCameraNoiseCfg()
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=field):
        mdp.EpisodeCameraNoise(cfg, num_envs=2, device="cpu")


def test_distillation_labels_match_executed_action_clip(monkeypatch: pytest.MonkeyPatch):
    """Behavior cloning targets the teacher action after the environment clamp."""
    algorithm = object.__new__(ClippedTeacherDistillation)
    algorithm.transition = SimpleNamespace(privileged_actions=torch.tensor(((-2.0, 0.25, 3.0),)))
    student_actions = torch.tensor(((0.1, 0.2, 0.3),))
    monkeypatch.setattr(Distillation, "act", lambda self, obs: student_actions)

    returned_actions = algorithm.act({})

    assert returned_actions is student_actions
    torch.testing.assert_close(algorithm.transition.privileged_actions, torch.tensor(((-1.0, 0.25, 1.0),)))


def test_franka_policies_use_stock_rsl_rl_algorithms():
    """The Franka policies keep their task-specific networks in maintained RSL-RL workflows."""
    state_runner = load_cfg_from_registry(FRANKA_STATE_TASK, "rsl_rl_cfg_entry_point")
    camera_runner = load_cfg_from_registry(FRANKA_CAMERA_TASK, "rsl_rl_cfg_entry_point")

    assert state_runner.algorithm.class_name == camera_runner.algorithm.class_name == "PPO"
    assert state_runner.init_at_random_ep_len is False
    assert isinstance(camera_runner.actor, RslRlCNNModelCfg)
    assert isinstance(state_runner.actor.distribution_cfg, RslRlMLPModelCfg.GaussianDistributionCfg)
    assert isinstance(camera_runner.actor.distribution_cfg, RslRlMLPModelCfg.GaussianDistributionCfg)
    assert state_runner.actor.distribution_cfg.class_name == "GaussianDistribution"
    assert camera_runner.actor.distribution_cfg.class_name == "GaussianDistribution"
    assert state_runner.actor.distribution_cfg.std_type == camera_runner.actor.distribution_cfg.std_type == "log"
    assert (
        state_runner.actor.distribution_cfg.std_range == camera_runner.actor.distribution_cfg.std_range == (0.05, 0.3)
    )


def test_kuka_task_has_one_complete_23_dof_state_policy(stack_cfgs):
    cfg = stack_cfgs[KUKA_STATE_TASK]
    runner = load_cfg_from_registry(KUKA_STATE_TASK, "rsl_rl_cfg_entry_point")

    assert cfg.events.reset_from_state_buffer.func is mdp.KukaAllegroResetStateTable
    assert cfg.actions.arm_action.gravity_compensation
    assert not cfg.actions.arm_action.controller_owns_gravity_compensation
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
    if isinstance(cfg.actions.gripper_action, mdp.ResetBufferedGripperActionCfg):
        assert cfg.actions.gripper_action.force_close_steps == 0


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
