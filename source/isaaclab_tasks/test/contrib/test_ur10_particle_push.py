# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction and pure-tensor regressions for UR10 particle push."""

from __future__ import annotations

import math
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg
from rsl_rl.models import CNNModel, MLPModel
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.coupling import CouplerProxyCfg

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.ur10_particle_push.agents.rsl_rl_ppo_cfg import UR10ParticlePushPPORunnerCfg
from isaaclab_tasks.contrib.ur10_particle_push.mdp import (
    ClampedRelativeJointPositionActionCfg,
    PushCurriculum,
    build_bin_goal_mask,
    compute_capped_bin_goal_progress,
    compute_masked_particle_mean,
    compute_particle_metrics,
    compute_transport_progress,
    critic_observation,
    heightmap_observation,
    policy_observation,
    success,
    update_curriculum_levels,
    update_success_streak,
)
from isaaclab_tasks.contrib.ur10_particle_push.reset_randomization import (
    build_reset_paddle_targets,
    build_reset_pose_curriculum_levels,
    build_reset_pose_source_pile_indices,
    build_staged_particle_reset,
)
from isaaclab_tasks.contrib.ur10_particle_push.ur10_particle_push_env import UR10ParticlePushEnv
from isaaclab_tasks.contrib.ur10_particle_push.ur10_particle_push_env_cfg import (
    MPM_ENTRY,
    PADDLE_MASS,
    PILE_LATTICE_RESOLUTION,
    PUSH_ACTION_DIM,
    PUSH_CRITIC_OBSERVATION_DIM,
    PUSH_POLICY_OBSERVATION_DIM,
    RIGID_ENTRY,
    UR10ParticlePushEnvCfg,
    configure_sparse_mpm_capacities,
    get_mpm_solver_cfg,
)
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from isaaclab_assets.robots.universal_robots import UR10_CFG

_TASK_ID = "IsaacContrib-UR10-Particle-Push"
_TASK_DIR = Path(__file__).parents[2] / "isaaclab_tasks" / "contrib" / "ur10_particle_push"


def _success_dwell_steps(cfg: UR10ParticlePushEnvCfg) -> int:
    """Return the success dwell duration in environment steps."""
    return max(1, math.ceil(cfg.success_dwell_time_s / (cfg.sim.dt * cfg.decimation)))


def _particle_template(cfg: UR10ParticlePushEnvCfg, num_envs: int = 1) -> torch.Tensor:
    """Build the complete cell-centered particle lattice."""
    spawn = cfg.scene.media.spawn
    resolution = torch.tensor(PILE_LATTICE_RESOLUTION)
    extent = torch.tensor(spawn.upper) - torch.tensor(spawn.lower)
    spacing = extent / resolution
    axes = [
        torch.arange(int(count), dtype=torch.float32) * axis_spacing + lower + 0.5 * axis_spacing
        for count, axis_spacing, lower in zip(resolution, spacing, spawn.lower, strict=True)
    ]
    template = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, 3)
    template += torch.tensor(cfg.pile_nominal_center)
    return template.expand(num_envs, -1, -1).clone()


def _build_push_models(
    cfg: UR10ParticlePushPPORunnerCfg,
    observations: TensorDict,
    action_space: int,
) -> tuple[CNNModel, CNNModel]:
    """Construct actor and critic models from the registered runner config."""
    models = []
    for observation_set, output_dim in (("actor", action_space), ("critic", 1)):
        model_cfg = getattr(cfg, observation_set).to_dict()
        class_name = model_cfg.pop("class_name")
        model_type = {"CNNModel": CNNModel, "MLPModel": MLPModel}[class_name]
        models.append(
            model_type(
                observations,
                cfg.obs_groups,
                observation_set,
                output_dim,
                **model_cfg,
            )
        )
    return models[0], models[1]


def test_registers_one_task_with_current_play_mode():
    matching_ids = sorted(spec.id for spec in gym.registry.values() if "UR10-Particle-Push" in spec.id)

    assert matching_ids == [_TASK_ID]
    cfg = UR10ParticlePushEnvCfg()
    assert isinstance(cfg, ManagerBasedRLEnvCfg)
    cfg.play_mode()
    assert cfg.scene.num_envs == 4
    assert cfg.heightmap_depth_noise_std == 0.0
    assert cfg.heightmap_xy_noise_std == 0.0
    assert cfg.heightmap_dropout_probability == 0.0
    assert cfg.reset_cycle
    final_level = len(cfg.curriculum_pile_center_x) - 1
    assert cfg.curriculum_level_override is None
    assert cfg.reset_curriculum_level_cycle == (final_level - 1, final_level)
    assert len(cfg.sim.visualizer_cfgs) == 1
    assert isinstance(cfg.sim.visualizer_cfgs[0], NewtonVisualizerCfg)
    assert cfg.sim.visualizer_cfgs[0].show_particles
    assert get_mpm_solver_cfg(cfg).max_active_cell_count == cfg.mpm_active_cell_count_per_world * 4


def test_registered_task_resolves_through_hydra_serialization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["pytest", "hydra.job.chdir=False"])

    cfg, agent_cfg = resolve_task_config(_TASK_ID, None)

    assert agent_cfg is None
    assert cfg.scene.media.cloning_contexts == ("isaaclab_newton.cloner:NewtonReplicateContext",)
    assert all(type(context) is str for context in cfg.scene.media.cloning_contexts)


def test_uses_official_assets_and_declarative_coupled_solver():
    cfg = UR10ParticlePushEnvCfg()

    assert cfg.scene.robot.spawn.usd_path == UR10_CFG.spawn.usd_path
    assert cfg.scene.robot.spawn.func == UR10_CFG.spawn.func
    assert cfg.scene.table.spawn.usd_path == (
        f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    )
    assert isinstance(cfg.scene.media, MPMObjectCfg)
    assert not (_TASK_DIR / "assets").exists()
    assert cfg.scene.paddle.spawn.visible
    assert cfg.scene.paddle.spawn.rigid_props.rigid_body_enabled
    assert cfg.scene.paddle.spawn.mass_props.mass == pytest.approx(PADDLE_MASS)
    assert cfg.scene.paddle_visual.spawn.collision_props is None

    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, CouplerProxyCfg)
    assert cfg.sim.physics.use_cuda_graph
    assert cfg.sim.use_newton_actuators
    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}
    assert set(entries) == {RIGID_ENTRY, MPM_ENTRY}
    assert isinstance(entries[RIGID_ENTRY].solver_cfg, MJWarpSolverCfg)
    assert isinstance(entries[MPM_ENTRY].solver_cfg, MPMSolverCfg)
    assert entries[RIGID_ENTRY].include_static_shapes
    assert entries[RIGID_ENTRY].shape_label_patterns == []
    assert entries[MPM_ENTRY].all_particles
    assert entries[MPM_ENTRY].in_place

    mpm = entries[MPM_ENTRY].solver_cfg
    assert mpm.grid_type == "sparse"
    assert mpm.separate_worlds
    assert mpm.voxel_size == pytest.approx(0.025)
    assert mpm.collider_basis == "pic27"
    assert mpm.transfer_scheme == "apic"
    assert not mpm.project_outside_colliders
    proxy = cfg.sim.physics.solver_cfg.proxies[0]
    assert proxy.bodies == [r"/World/envs/env_.*/Robot/ee_link/Paddle"]
    assert proxy.mass_scale == pytest.approx(cfg.proxy_mass_scale)
    assert "Rigid" not in " ".join(entries[MPM_ENTRY].bodies)


def test_uses_manager_based_terms_with_checkpoint_compatible_spaces():
    cfg = UR10ParticlePushEnvCfg()

    assert issubclass(UR10ParticlePushEnv, ManagerBasedRLEnv)
    assert isinstance(cfg.actions.arm_action, ClampedRelativeJointPositionActionCfg)
    assert cfg.actions.arm_action.joint_names == list(cfg.scene.robot.init_state.joint_pos)
    assert list(cfg.actions.arm_action.scale.values()) == pytest.approx((0.035, 0.035, 0.035, 0.052, 0.052, 0.052))
    assert cfg.actions.arm_action.joint_limit_margin == pytest.approx(0.04)
    assert cfg.observations.policy.state.func is policy_observation
    assert cfg.observations.heightmap.image.func is heightmap_observation
    assert cfg.observations.critic.state.func is critic_observation
    assert cfg.terminations.success.func is success
    assert cfg.terminations.time_out.time_out
    assert cfg.events.reset_scene.mode == "reset"
    assert cfg.curriculum.task_levels.func is PushCurriculum


@pytest.mark.parametrize(
    ("num_envs", "expected_lower", "expected_upper"),
    (
        (2, 128, 32),
        (256, 16384, 256),
    ),
)
def test_sparse_capacities_scale_with_environment_count(
    num_envs: int,
    expected_lower: int,
    expected_upper: int,
):
    cfg = UR10ParticlePushEnvCfg()
    cfg.scene.num_envs = num_envs

    configure_sparse_mpm_capacities(cfg)

    mpm = get_mpm_solver_cfg(cfg)
    assert mpm.max_active_cell_count == num_envs * cfg.mpm_active_cell_count_per_world
    assert mpm.max_leaf_node_count == num_envs * cfg.mpm_leaf_node_count_per_world
    assert mpm.max_lower_node_count == expected_lower
    assert mpm.max_upper_node_count == expected_upper


def test_reset_pose_bank_is_deterministic_and_locally_randomized():
    cfg = UR10ParticlePushEnvCfg()

    position_a, quaternion_a = build_reset_paddle_targets(cfg, device="cpu")
    position_b, quaternion_b = build_reset_paddle_targets(cfg, device="cpu")
    levels = build_reset_pose_curriculum_levels(cfg, device="cpu")
    source_pile_index = build_reset_pose_source_pile_indices(cfg, device="cpu")

    torch.testing.assert_close(position_a, position_b)
    torch.testing.assert_close(quaternion_a, quaternion_b)
    assert position_a.shape == (cfg.reset_pose_count, 3)
    assert quaternion_a.shape == (cfg.reset_pose_count, 4)
    torch.testing.assert_close(torch.linalg.vector_norm(quaternion_a, dim=1), torch.ones(cfg.reset_pose_count))
    torch.testing.assert_close(position_a[:, 2], torch.full((cfg.reset_pose_count,), cfg.paddle_reset_center[2]))
    assert torch.unique(levels).tolist() == list(range(len(cfg.curriculum_pile_center_x)))
    pile_counts = torch.tensor(cfg.curriculum_source_pile_count)[levels]
    assert torch.all(source_pile_index >= 0)
    assert torch.all(source_pile_index < pile_counts)
    assert torch.unique(position_a, dim=0).shape[0] > cfg.reset_pose_count // 2
    assert torch.unique(quaternion_a, dim=0).shape[0] > cfg.reset_pose_count // 2
    final_rows = levels == len(cfg.curriculum_pile_center_x) - 1
    final_positions = position_a[final_rows]
    assert final_positions[:, 0].max() - final_positions[:, 0].min() > 0.12
    assert final_positions[:, 1].max() - final_positions[:, 1].min() > 0.20


def test_staged_reset_reuses_complete_particle_population():
    cfg = UR10ParticlePushEnvCfg()
    template = _particle_template(cfg, num_envs=2)

    state = build_staged_particle_reset(
        template,
        torch.full((2,), cfg.pile_nominal_center[0]),
        torch.tensor((1, 2)),
        torch.tensor((0.0, 0.15)),
        torch.tensor((0.0, 0.2)),
        torch.tensor((0, 1)),
        torch.zeros((2, 2)),
        torch.zeros(2),
        torch.zeros_like(template),
        cfg,
    )

    assert state.position_e.shape == template.shape
    assert torch.isfinite(state.position_e).all()
    assert torch.all(state.position_e[..., 2] > cfg.scene.mpm_ground.init_state.pos[2])
    assert state.focused_source_mask[0].all()
    expected_second_source_count = template.shape[1] - math.floor(0.2 * template.shape[1])
    assert state.focused_source_mask[1].sum().item() == expected_second_source_count // 2


def test_particle_metrics_use_complete_population_and_masks_select_semantic_groups():
    cfg = UR10ParticlePushEnvCfg()
    start_x = cfg.pile_nominal_center[0]
    particles = torch.tensor(
        [
            [
                [start_x, 0.0, 0.03],
                [start_x, 0.1, 0.04],
                [1.20, 0.0, 0.00],
                [-2.0, 2.0, -2.0],
            ]
        ]
    )
    focused = torch.tensor([[True, True, False, False]])

    bin_fraction, spill_fraction = compute_particle_metrics(particles, cfg)
    transport = compute_transport_progress(particles, cfg)
    focused_centroid = compute_masked_particle_mean(particles, focused)

    torch.testing.assert_close(bin_fraction, torch.full((1,), 0.25))
    torch.testing.assert_close(spill_fraction, torch.full((1,), 0.25))
    torch.testing.assert_close(transport, torch.zeros(1))
    torch.testing.assert_close(focused_centroid, particles[:, :2].mean(dim=1))


def test_curriculum_promotes_after_successes_and_demotes_after_misses():
    level = torch.tensor((0, 3))
    success_streak = torch.zeros(2, dtype=torch.long)
    failure_streak = torch.zeros(2, dtype=torch.long)

    level, success_streak, failure_streak, promoted, demoted = update_curriculum_levels(
        level,
        success_streak,
        failure_streak,
        torch.tensor((True, True)),
        max_level=3,
        successes_to_promote=2,
        failures_to_demote=3,
    )
    assert level.tolist() == [0, 3]
    assert success_streak.tolist() == [1, 1]
    assert not torch.any(promoted | demoted)

    level, success_streak, failure_streak, promoted, demoted = update_curriculum_levels(
        level,
        success_streak,
        failure_streak,
        torch.tensor((True, False)),
        max_level=3,
        successes_to_promote=2,
        failures_to_demote=3,
    )
    assert level.tolist() == [1, 3]
    assert promoted.tolist() == [True, False]
    assert failure_streak.tolist() == [0, 1]

    for _ in range(2):
        level, success_streak, failure_streak, promoted, demoted = update_curriculum_levels(
            level,
            success_streak,
            failure_streak,
            torch.tensor((False, False)),
            max_level=3,
            successes_to_promote=2,
            failures_to_demote=3,
        )
    assert level.tolist() == [1, 2]
    assert demoted.tolist() == [False, True]


def test_progress_terms_are_bounded_at_start_and_goal():
    cfg = UR10ParticlePushEnvCfg()
    start_x = cfg.pile_nominal_center[0]
    mouth_x = cfg.bin_inner_x_bounds[0]
    particles = torch.tensor(
        (
            ((start_x, 0.0, 0.01), (start_x, 0.0, 0.02)),
            ((0.5 * (start_x + mouth_x), 0.0, 0.01), (start_x, 0.0, 0.02)),
            ((mouth_x, 0.0, 0.01), (mouth_x + 0.1, 0.0, -0.1)),
        )
    )

    transport = compute_transport_progress(particles, cfg)
    goal = compute_capped_bin_goal_progress(
        torch.tensor((0.0, 0.01, 0.02, 0.08)),
        success_fraction=torch.tensor((0.02, 0.02, 0.02, 0.07)),
    )

    torch.testing.assert_close(transport, torch.tensor((0.0, 0.25, 1.0)))
    torch.testing.assert_close(goal, torch.tensor((0.0, 0.5, 1.0, 1.0)))
    assert torch.all((transport >= 0.0) & (transport <= 1.0))
    assert torch.all((goal >= 0.0) & (goal <= 1.0))


def test_success_boundary_requires_sustained_delivery_low_spill_and_below_rim():
    cfg = UR10ParticlePushEnvCfg()
    dwell_steps = _success_dwell_steps(cfg)

    streak, success = update_success_streak(
        torch.full((3,), dwell_steps - 1, dtype=torch.long),
        bin_fraction=torch.tensor((0.799, 0.800, 0.900)),
        spill_fraction=torch.tensor((0.0, 0.0, 0.021)),
        success_fraction=cfg.success_fraction,
        max_spill_fraction=cfg.success_max_spill_fraction,
        dwell_steps=dwell_steps,
    )

    assert streak.tolist() == [0, dwell_steps, 0]
    assert success.tolist() == [False, True, False]
    particles = torch.tensor([[[1.20, 0.0, 0.08], [1.20, 0.0, 0.11]]])
    delivered, spill = compute_particle_metrics(particles, cfg)
    assert delivered.item() == pytest.approx(0.5)
    assert spill.item() == pytest.approx(0.0)


def test_actor_observation_and_grouped_cnn_config_are_deployable():
    cfg = UR10ParticlePushEnvCfg()
    runner = UR10ParticlePushPPORunnerCfg()

    assert cfg.heightmap_shape == (50, 86)
    assert cfg.heightmap_history_steps == 4
    assert PUSH_POLICY_OBSERVATION_DIM == 31
    assert PUSH_CRITIC_OBSERVATION_DIM == 11
    assert build_bin_goal_mask(cfg).shape == cfg.heightmap_shape
    assert runner.obs_groups == {
        "actor": ["policy", "heightmap"],
        "critic": ["policy", "heightmap", "critic"],
    }
    assert runner.num_steps_per_env == 72
    assert not runner.init_at_random_ep_len
    assert runner.actor.class_name == "CNNModel"
    assert runner.critic.class_name == "CNNModel"
    assert runner.actor.cnn_cfg.output_channels == [8, 16, 32, 32]
    assert runner.actor.cnn_cfg.kernel_size == [5, 3, 3, 3]
    assert runner.actor.cnn_cfg.stride == [2, 2, 2, 2]
    assert runner.critic.cnn_cfg == runner.actor.cnn_cfg
    assert runner.actor.obs_normalization is False
    assert runner.critic.obs_normalization is False
    assert not runner.algorithm.share_cnn_encoders


def test_registered_agent_strictly_loads_and_trains_grouped_cnn_schema():
    env_cfg = UR10ParticlePushEnvCfg()
    observations = TensorDict(
        {
            "policy": torch.randn((2, PUSH_POLICY_OBSERVATION_DIM)),
            "heightmap": torch.rand((2, 3, *env_cfg.heightmap_shape)),
            "critic": torch.randn((2, PUSH_CRITIC_OBSERVATION_DIM)),
        },
        batch_size=[2],
    )
    installed_version = metadata.version("rsl-rl-lib")
    training_cfg = handle_deprecated_rsl_rl_cfg(UR10ParticlePushPPORunnerCfg(), installed_version)
    play_cfg = load_cfg_from_registry(_TASK_ID, "rsl_rl_cfg_entry_point")
    play_cfg = handle_deprecated_rsl_rl_cfg(play_cfg, installed_version)
    training_models = _build_push_models(training_cfg, observations, PUSH_ACTION_DIM)
    play_models = _build_push_models(play_cfg, observations, PUSH_ACTION_DIM)

    for training_model, play_model in zip(training_models, play_models, strict=True):
        checkpoint_state = training_model.state_dict()
        assert not any(key.startswith("obs_normalizer.") for key in checkpoint_state)
        assert any(key.startswith("cnns.heightmap.") for key in checkpoint_state)
        play_model.load_state_dict(checkpoint_state, strict=True)

    actor, critic = training_models
    parameter_count = sum(parameter.numel() for model in training_models for parameter in model.parameters())
    assert parameter_count < 450_000
    actions = actor(observations)
    assert actions.shape == (2, PUSH_ACTION_DIM)
    actions.square().mean().backward()
    actor_cnn_gradients = [
        parameter.grad
        for name, parameter in actor.named_parameters()
        if name.startswith("cnns.heightmap.") and parameter.grad is not None
    ]
    assert actor_cnn_gradients
    assert any(torch.isfinite(gradient).all() and gradient.abs().sum() > 0.0 for gradient in actor_cnn_gradients)
    assert critic(observations).shape == (2, 1)
    stochastic_actions = actor(observations, stochastic_output=True)
    assert stochastic_actions.shape == (2, PUSH_ACTION_DIM)
    assert actor.distribution is not None
    assert torch.all(actor.distribution.std > 0.15)
    assert torch.all(actor.distribution.std < 0.65)
