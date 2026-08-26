# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration tests for contributed heterogeneous manipulation training."""

import math
from importlib.metadata import version
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from isaaclab.managers import RewardTermCfg
from isaaclab.sim import UsdFileCfg

from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.contrib.multitask_manipulation.agents.models import TaskHeadedMLPModel, TaskHeadedValueModel
from isaaclab_tasks.contrib.multitask_manipulation.agents.ppo import TaskBalancedPPO
from isaaclab_tasks.contrib.multitask_manipulation.agents.rsl_rl_ppo_cfg import (
    MultitaskManipulationPPORunnerCfg,
    TaskBalancedPPOCfg,
    TaskHeadedGaussianDistributionCfg,
    TaskHeadedMLPModelCfg,
    TaskHeadedValueModelCfg,
)
from isaaclab_tasks.contrib.multitask_manipulation.mdp.actions import SelectedJointPositionAction
from isaaclab_tasks.contrib.multitask_manipulation.mdp.rewards import LiftGoalTracking, selected_joint_vel_l2
from isaaclab_tasks.contrib.multitask_manipulation.mdp.terminations import articulation_state_invalid, task_time_out
from isaaclab_tasks.contrib.multitask_manipulation.multitask_env_cfg import MultitaskManipulationEnvCfg
from isaaclab_tasks.contrib.multitask_manipulation.selection_utils import SceneEntitySelectionCfg


def test_registry_uses_task_specific_manager_environment() -> None:
    """The task should customize reset behavior without changing the core environment."""
    assert (
        gym.spec("Isaac-Multitask-Manipulation").entry_point
        == "isaaclab_tasks.contrib.multitask_manipulation.multitask_env:MultitaskManipulationEnv"
    )


def test_scene_composes_only_the_three_task_layouts() -> None:
    """The clone plan should contain only the intended task layouts."""
    cfg = MultitaskManipulationEnvCfg()

    combinations = [set(combination.assets) for combination in cfg.scene.clone_cfg.clone_combinations]

    assert combinations == [
        {"robot", "object"},
        {"robot_1", "cabinet"},
        {"robot_2"},
    ]
    assert getattr(cfg.scene, "table_1", None) is None
    assert cfg.scene.num_envs == 4096


def test_lift_layout_uses_openarm_cube_lift_assets() -> None:
    """The lift layout should reuse the maintained OpenArm cube-lift task assets."""
    cfg = MultitaskManipulationEnvCfg()

    assert isinstance(cfg.scene.robot.spawn, UsdFileCfg)
    assert cfg.scene.robot.spawn.usd_path.endswith("/Robots/OpenArm/openarm_unimanual/openarm_unimanual.usd")
    assert isinstance(cfg.scene.object.spawn, UsdFileCfg)
    assert cfg.scene.object.spawn.usd_path.endswith("/Props/Blocks/DexCube/dex_cube_instanceable.usd")
    assert cfg.scene.object.init_state.pos == [0.4, 0, 0.055]
    assert cfg.scene.table is None
    assert cfg.terminations.lift_object_dropped.params["minimum_height"] == -0.05


def test_lift_mdp_preserves_openarm_contract_with_task_reward_scale() -> None:
    """The lift branch should preserve OpenArm settings under one common reward scale."""
    cfg = MultitaskManipulationEnvCfg()

    assert cfg.actions.lift_arm_action.joint_names == ["openarm_joint.*"]
    assert cfg.actions.lift_arm_action.scale == 0.5
    assert not cfg.actions.lift_arm_action.relative
    assert cfg.actions.lift_gripper_action.open_command == 0.044
    assert cfg.actions.lift_gripper_action.close_command == 0.0
    assert cfg.observations.policy.lift_joint_pos.params["asset_cfg"].joint_names == [
        "openarm_joint.*",
        "openarm_finger_joint.*",
    ]
    assert cfg.rewards.lift_approach.params["robot_cfg"].body_names == "openarm_ee_tcp"
    assert cfg.commands.lift_pose.resampling_time_range == (5.0, 5.0)
    assert cfg.rewards.lift_approach.weight == pytest.approx(0.11)
    assert cfg.rewards.lift_height.weight == pytest.approx(1.5)
    assert cfg.rewards.lift_height.params["minimum_height"] == 0.04
    assert cfg.rewards.lift_goal.weight == pytest.approx(1.6)
    assert cfg.rewards.lift_goal_fine.weight == pytest.approx(0.5)
    assert cfg.rewards.lift_action_rate.weight == pytest.approx(-1.0e-5)
    assert cfg.rewards.lift_joint_vel.weight == pytest.approx(-1.0e-5)
    assert cfg.curriculum.lift_action_rate.params["weight"] == pytest.approx(-0.01)
    assert cfg.curriculum.lift_joint_vel.params["weight"] == pytest.approx(-0.01)


def test_mdp_uses_selection_aware_scene_entities() -> None:
    """Heterogeneous MDP terms should resolve entity members and physics-view rows together."""
    cfg = MultitaskManipulationEnvCfg()

    assert isinstance(cfg.observations.policy.lift_joint_pos.params["asset_cfg"], SceneEntitySelectionCfg)
    assert isinstance(cfg.rewards.cabinet_open.params["cabinet_cfg"], SceneEntitySelectionCfg)
    assert isinstance(cfg.terminations.reach_success.params["robot_cfg"], SceneEntitySelectionCfg)
    assert isinstance(cfg.commands.lift_pose.reference_cfg, SceneEntitySelectionCfg)
    assert isinstance(cfg.commands.lift_pose.tracked_cfg, SceneEntitySelectionCfg)
    assert cfg.rewards.cabinet_joint_vel.params["max_velocity"] == 50.0
    assert cfg.terminations.cabinet_state_invalid.params["max_joint_velocity"] == 50.0


def test_cabinet_action_clamps_targets_inside_soft_joint_limits() -> None:
    """The cabinet action should not command targets beyond the articulation's soft limits."""
    cfg = MultitaskManipulationEnvCfg()
    limits = torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]])
    applied = {}
    asset = SimpleNamespace(
        data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits)),
        set_joint_position_target_index=lambda target, joint_ids: applied.update(target=target, joint_ids=joint_ids),
    )
    action = SelectedJointPositionAction.__new__(SelectedJointPositionAction)
    action.cfg = SimpleNamespace(relative=False)
    action._asset = asset
    action._joint_ids = torch.tensor([0, 1])
    action._joint_limit_margin = cfg.actions.cabinet_arm_action.joint_limit_margin
    action._processed_actions = torch.tensor([[-3.0, 3.0]])

    action.apply_actions()

    assert action._joint_limit_margin == 0.02
    assert torch.allclose(applied["target"], torch.tensor([[-0.98, 1.98]]))
    assert torch.equal(applied["joint_ids"], action._joint_ids)


def test_cabinet_state_guard_terminates_outliers_and_bounds_velocity_penalty() -> None:
    """Non-finite or implausible cabinet states should reset with a finite reward penalty."""
    joint_pos = torch.tensor([[0.0, 0.0], [0.0, 0.0], [math.nan, 0.0]])
    joint_vel = torch.tensor([[1.0, 2.0], [60.0, -100.0], [math.nan, math.inf]])
    limits = torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]]).expand(3, -1, -1)
    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_pos),
            joint_vel=SimpleNamespace(torch=joint_vel),
            soft_joint_pos_limits=SimpleNamespace(torch=limits),
        )
    )
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SimpleNamespace(
        name="robot",
        joint_ids=slice(None),
        scatter_to_envs=lambda values: values,
    )

    invalid = articulation_state_invalid(
        env,
        asset_cfg,
        max_joint_velocity=50.0,
        joint_position_margin=0.1,
    )
    penalty = selected_joint_vel_l2(env, asset_cfg, max_velocity=50.0)

    assert torch.equal(invalid, torch.tensor([False, True, True]))
    assert torch.equal(penalty, torch.tensor([5.0, 5000.0, 2500.0]))


def test_play_mode_uses_short_horizons_and_task_specific_markers() -> None:
    """Playback should shorten task horizons and show position-appropriate markers."""
    cfg = MultitaskManipulationEnvCfg()
    cfg.play_mode()

    assert cfg.scene.table is None
    assert cfg.terminations.time_out.params["episode_lengths_s"] == (3.0, 4.0, 6.0)
    assert cfg.commands.lift_pose.debug_vis
    assert cfg.commands.reach_pose.debug_vis
    assert set(cfg.commands.lift_pose.goal_pose_visualizer_cfg.markers) == {"sphere"}
    assert set(cfg.commands.lift_pose.current_pose_visualizer_cfg.markers) == {"sphere"}
    assert "frame" in cfg.commands.reach_pose.goal_pose_visualizer_cfg.markers
    assert "frame" in cfg.commands.reach_pose.current_pose_visualizer_cfg.markers
    marker_paths = {
        cfg.commands.lift_pose.goal_pose_visualizer_cfg.prim_path,
        cfg.commands.lift_pose.current_pose_visualizer_cfg.prim_path,
        cfg.commands.reach_pose.goal_pose_visualizer_cfg.prim_path,
        cfg.commands.reach_pose.current_pose_visualizer_cfg.prim_path,
    }
    assert len(marker_paths) == 4


def test_task_requires_at_least_one_environment_per_layout() -> None:
    """Configuration validation should reject a batch that cannot instantiate every task view."""
    cfg = MultitaskManipulationEnvCfg()
    cfg.scene.num_envs = 2

    with pytest.raises(ValueError, match="at least three environments"):
        cfg.validate()


def test_task_timeouts_fire_after_each_complete_task_horizon() -> None:
    """Task-specific horizons should not terminate one step early."""
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=1.0 / 60.0,
        episode_length_buf=torch.tensor([359, 479, 719]),
    )
    task_asset_cfgs = tuple(
        SimpleNamespace(instance_ids=torch.tensor([0 if env_id == task_id else -1 for env_id in range(3)]))
        for task_id in range(3)
    )

    before_horizon = task_time_out(env, task_asset_cfgs, (6.0, 8.0, 12.0))
    env.episode_length_buf += 1
    at_horizon = task_time_out(env, task_asset_cfgs, (6.0, 8.0, 12.0))

    assert torch.equal(before_horizon, torch.zeros(3, dtype=torch.bool))
    assert torch.equal(at_horizon, torch.ones(3, dtype=torch.bool))


def test_lift_success_logs_sticky_episode_rate() -> None:
    """Lift success should remain set until reset and be averaged only over lift environments."""
    object_cfg = SimpleNamespace(instance_ids=torch.tensor([0, -1, 1, -1]))
    env = SimpleNamespace(num_envs=4, device="cpu", extras={})
    cfg = RewardTermCfg(func=LiftGoalTracking, weight=1.0, params={"object_cfg": object_cfg})
    term = LiftGoalTracking(cfg, env)
    term.succeeded[[0, 2]] = torch.tensor([True, False])

    term.reset(torch.arange(4))

    assert env.extras["log"]["Metrics/lift_success_rate"] == 0.5
    assert not torch.any(term.succeeded)


def test_rsl_rl_actor_uses_task_specific_gaussian_heads() -> None:
    """The default agent should route policy, value, and normalization by task identity."""
    cfg = MultitaskManipulationPPORunnerCfg()

    assert isinstance(cfg.actor, TaskHeadedMLPModelCfg)
    assert cfg.actor.task_action_dims == (8, 8, 6)
    assert cfg.actor.task_encoding_slice == (0, 3)
    assert isinstance(cfg.actor.distribution_cfg, TaskHeadedGaussianDistributionCfg)
    assert isinstance(cfg.critic, TaskHeadedValueModelCfg)
    assert cfg.critic.task_head_count == 3
    assert isinstance(cfg.algorithm, TaskBalancedPPOCfg)
    assert cfg.algorithm.task_names == ("lift", "cabinet", "reach")
    assert cfg.algorithm.task_encoding_obs_group == "policy"
    assert cfg.algorithm.task_encoding_slice == (0, 3)
    assert cfg.clip_actions is None
    assert not cfg.init_at_random_ep_len


def test_rsl_rl_constructs_task_headed_actor_with_fixed_action_output() -> None:
    """The production agent should resolve and complete a routed PPO rollout/update."""
    obs = _make_policy_observations([0, 1, 2], policy_dim=95)
    cfg = handle_deprecated_rsl_rl_cfg(MultitaskManipulationPPORunnerCfg(), version("rsl-rl-lib")).to_dict()
    cfg["num_steps_per_env"] = 4
    cfg["algorithm"]["num_learning_epochs"] = 1
    cfg["algorithm"]["num_mini_batches"] = 2
    cfg["multi_gpu"] = None
    env = type("StubEnv", (), {"num_actions": 22, "num_envs": 3})()

    algorithm = PPO.construct_algorithm(obs, env, cfg, "cpu")
    for step in range(cfg["num_steps_per_env"]):
        actions = algorithm.act(obs)
        reward_scale = torch.tensor([1.0, 10.0, 100.0])
        algorithm.process_env_step(obs, (step + 1) * reward_scale, torch.zeros(3, dtype=torch.long), {})
    algorithm.compute_returns(obs)

    task_ids = algorithm.storage.observations["policy"][..., :3].argmax(dim=-1)
    for task_id in range(3):
        task_advantages = algorithm.storage.advantages[task_ids == task_id]
        assert task_advantages.mean().item() == pytest.approx(0.0, abs=1.0e-6)
        assert task_advantages.std(unbiased=False).item() == pytest.approx(1.0)

    losses = algorithm.update()

    assert isinstance(algorithm, TaskBalancedPPO)
    assert isinstance(algorithm.actor, TaskHeadedMLPModel)
    assert isinstance(algorithm.critic, TaskHeadedValueModel)
    assert actions.shape == (3, 22)
    assert torch.equal(actions[0, 8:], torch.zeros(14))
    assert torch.equal(actions[1, :8], torch.zeros(8))
    assert torch.equal(actions[1, 16:], torch.zeros(6))
    assert torch.equal(actions[2, :16], torch.zeros(16))
    assert set(losses) >= {
        "lift_advantage_std",
        "lift_return_std",
        "cabinet_advantage_std",
        "cabinet_return_std",
        "reach_advantage_std",
        "reach_return_std",
    }
    assert all(math.isfinite(loss) for loss in losses.values())


def test_task_headed_critic_routes_scalar_values() -> None:
    """Each observation should select only its task-specific scalar value head."""
    obs = _make_policy_observations([0, 1, 2])
    critic = TaskHeadedValueModel(
        obs=obs,
        obs_groups={"critic": ["policy"]},
        obs_set="critic",
        output_dim=1,
        hidden_dims=[4],
        activation="elu",
        task_head_count=3,
        task_encoding_slice=(0, 3),
    )
    with torch.no_grad():
        for parameter in critic.backbone.parameters():
            parameter.zero_()
        for task_id, head in enumerate(critic.value_heads):
            head.weight.zero_()
            head.bias.fill_(task_id + 1.0)

    assert torch.equal(critic(obs), torch.tensor([[1.0], [2.0], [3.0]]))


def test_task_headed_actor_routes_deterministic_and_stochastic_actions() -> None:
    """Every sample should expose only the action block selected by its task identity."""
    obs = _make_policy_observations([0, 1, 2])
    actor = _make_task_headed_actor(obs)

    deterministic_actions = actor(obs)
    sampled_actions = actor(obs, stochastic_output=True)
    action_mask = _expected_action_mask([0, 1, 2])

    expected_actions = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
        ]
    )
    assert torch.equal(deterministic_actions, expected_actions)
    assert torch.equal(sampled_actions * (1.0 - action_mask), torch.zeros_like(sampled_actions))


def test_task_headed_distribution_ignores_inactive_actions() -> None:
    """Inactive global dimensions should not contribute to PPO probability terms."""
    task_ids = [0, 1, 2]
    obs = _make_policy_observations(task_ids)
    actor = _make_task_headed_actor(obs)
    actions = actor(obs, stochastic_output=True)
    action_mask = _expected_action_mask(task_ids)

    original_log_prob = actor.get_output_log_prob(actions)
    modified_actions = actions + 100.0 * (1.0 - action_mask)
    modified_log_prob = actor.get_output_log_prob(modified_actions)
    expected_entropy = torch.tensor([2.0, 3.0, 1.0]) * (0.5 * math.log(2.0 * math.pi * math.e))

    assert torch.allclose(original_log_prob, modified_log_prob)
    assert torch.allclose(actor.output_entropy, expected_entropy)

    old_params = tuple(param.detach().clone() for param in actor.output_distribution_params)
    new_params = tuple(param.detach().clone() for param in actor.output_distribution_params)
    new_params[0].add_(10.0 * (1.0 - action_mask))
    new_params[1].add_(1.0 - action_mask)
    assert torch.equal(actor.get_kl_divergence(old_params, new_params), torch.zeros(3))


def test_task_headed_actor_supports_single_task_evaluation_and_export() -> None:
    """A uniform task batch should route one head in eager and TorchScript inference."""
    obs = _make_policy_observations([1, 1])
    actor = _make_task_headed_actor(obs)

    expected_actions = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        ]
    )
    scripted_actor = torch.jit.script(actor.as_jit())

    assert torch.equal(actor(obs), expected_actions)
    assert torch.equal(scripted_actor(obs["policy"]), expected_actions)


def test_single_task_batch_updates_only_its_action_head() -> None:
    """A task-homogeneous PPO batch should not update inactive mean or exploration parameters."""
    active_obs = _make_policy_observations([0, 0])
    inactive_obs = _make_policy_observations([1, 1])
    actor = _make_task_headed_actor(active_obs)
    for parameter in actor.backbone.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)

    active_actions_before = actor(active_obs).detach().clone()
    inactive_actions_before = actor(inactive_obs).detach().clone()
    actor(active_obs, stochastic_output=True)
    std_before = actor.output_std.detach().clone()

    actor(active_obs, stochastic_output=True)
    target_actions = 3.0 * _expected_action_mask([0, 0])
    loss = -actor.get_output_log_prob(target_actions).mean()
    loss.backward()
    optimizer.step()

    active_actions_after = actor(active_obs).detach()
    inactive_actions_after = actor(inactive_obs).detach()
    actor(active_obs, stochastic_output=True)
    std_after = actor.output_std.detach()

    assert not torch.equal(active_actions_after, active_actions_before)
    assert torch.equal(inactive_actions_after, inactive_actions_before)
    assert not torch.equal(std_after[:2], std_before[:2])
    assert torch.equal(std_after[2:], std_before[2:])


def _make_policy_observations(task_ids: list[int], policy_dim: int = 5) -> TensorDict:
    """Create compact policy observations whose first three values encode the task."""
    policy = torch.zeros(len(task_ids), policy_dim)
    policy[torch.arange(len(task_ids)), torch.tensor(task_ids)] = 1.0
    return TensorDict({"policy": policy}, batch_size=[len(task_ids)])


def _make_task_headed_actor(obs: TensorDict) -> TaskHeadedMLPModel:
    """Create a small actor with easily inspected task-head outputs."""
    actor = TaskHeadedMLPModel(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=6,
        hidden_dims=[4],
        activation="elu",
        task_action_dims=(2, 3, 1),
        task_encoding_slice=(0, 3),
        distribution_cfg={
            "class_name": (
                "isaaclab_tasks.contrib.multitask_manipulation.agents.models:TaskHeadedGaussianDistribution"
            ),
            "init_std": 1.0,
        },
    )
    with torch.no_grad():
        for parameter in actor.backbone.parameters():
            parameter.zero_()
        for task_id, head in enumerate(actor.action_heads):
            head.weight.zero_()
            head.bias.fill_(task_id + 1.0)
    return actor


def _expected_action_mask(task_ids: list[int]) -> torch.Tensor:
    """Return the fixed 2D/3D/1D task-head mask for the compact test actor."""
    masks = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    return masks[torch.tensor(task_ids)]
