# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction tests for the registered Franka pour task (no simulator)."""

import subprocess
import sys

import gymnasium as gym
import pytest
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

import isaaclab_tasks.contrib.franka_pour.config.franka  # noqa: F401
import isaaclab_tasks.contrib.franka_pour.mdp as mdp
from isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg import (
    FrankaPourResetDatasetPPORunnerCfg,
)
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import (
    FRANKA_POUR_ARM_DRIVE_DAMPING,
    FRANKA_POUR_ARM_DRIVE_STIFFNESS,
    FRANKA_POUR_CUPS_USD_PATH,
    FRANKA_POUR_RESET_DATASET_CONTENT_SHA256,
    FRANKA_POUR_ROBOT_ASSET_ID,
    FRANKA_POUR_ROBOT_PHYSICS_OVERRIDE,
    FRANKA_POUR_ROBOT_PHYSICS_PAYLOAD_SHA256,
    FRANKA_POUR_ROBOT_USD_PATH,
    MPM_ENTRY,
    RESET_DATASET_PROFILE_FULL,
    RESET_DATASET_SUCCESS_SEMANTICS,
    RIGID_ENTRY,
    SPILL_FLOOR_LABEL_PATTERN,
    FrankaPourResetDatasetEnvCfg,
    _reset_dataset_task_contract,
    _resolve_pour_solver_tree,
    spawn_franka_with_arm_collisions,
)

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

_TASK_ID = "IsaacContrib-Franka-Pour"


def _term_names(group, term_type: type) -> tuple[str, ...]:
    """Return manager-term names in their configured order."""
    return tuple(name for name, value in vars(group).items() if isinstance(value, term_type))


def test_registration_is_import_light_and_points_to_the_only_pour_environment():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import gymnasium as gym; "
                "import isaaclab_tasks.contrib.franka_pour.pour_env_cfg; "
                "import isaaclab_tasks.contrib.franka_pour.config.franka; "
                "assert 'newton' not in sys.modules; "
                "assert 'pxr' not in sys.modules; "
                "assert gym.spec('IsaacContrib-Franka-Pour').entry_point == "
                "'isaaclab_tasks.contrib.franka_pour.pour_env:FrankaPourEnv'"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    spec = gym.spec(_TASK_ID)
    assert spec.entry_point == "isaaclab_tasks.contrib.franka_pour.pour_env:FrankaPourEnv"
    assert spec.disable_env_checker is True
    assert spec.kwargs == {
        "env_cfg_entry_point": "isaaclab_tasks.contrib.franka_pour.pour_env_cfg:FrankaPourResetDatasetEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg:FrankaPourResetDatasetPPORunnerCfg"
        ),
    }


def test_finalize_builds_independent_assets_and_propagates_overrides():
    cfg = FrankaPourResetDatasetEnvCfg()
    cfg.scene.num_envs = 8
    cfg.cup_mass = 0.08
    cfg.cup_grasp_box_friction = 1.5
    cfg.target_cup_friction = 0.6
    cfg.cup_reset_pos = (0.45, 0.02, 0.0)
    cfg.target_cup_reset_pos = (0.52, -0.20, 0.0)

    resolved = cfg.finalize()
    second = cfg.finalize()

    assert cfg.scene.source_cup is None
    assert cfg.scene.target_cup is None
    assert cfg.scene.media is None
    assert _resolve_pour_solver_tree(cfg).media_solver.max_active_cell_count == -1
    assert resolved is not cfg
    assert resolved.scene is not cfg.scene
    assert isinstance(resolved.scene.source_cup, RigidObjectCfg)
    assert isinstance(resolved.scene.target_cup, RigidObjectCfg)
    assert isinstance(resolved.scene.media, MPMObjectCfg)
    assert resolved.scene.source_cup is not second.scene.source_cup
    assert resolved.scene.media is not second.scene.media

    source = resolved.scene.source_cup
    target = resolved.scene.target_cup
    media = resolved.scene.media
    assert isinstance(source.spawn, UsdFileCfg)
    assert source.spawn.usd_path == FRANKA_POUR_CUPS_USD_PATH
    assert source.spawn.variants == {"cupType": "source"}
    assert source.spawn.mass_props.mass == pytest.approx(0.08)
    assert source.spawn.physics_material.static_friction == pytest.approx(1.5)
    assert source.init_state.pos == (0.45, 0.02, 0.0)
    assert isinstance(target.spawn, UsdFileCfg)
    assert target.spawn.usd_path == FRANKA_POUR_CUPS_USD_PATH
    assert target.spawn.variants == {"cupType": "target"}
    assert target.spawn.physics_material.dynamic_friction == pytest.approx(0.6)
    assert target.init_state.pos == (0.52, -0.20, 0.0)
    assert media.init_state.pos == cfg.cup_reset_pos
    assert isinstance(media.spawn, MPMGridCfg)
    assert media.spawn.material.density == pytest.approx(cfg.media_material.density)
    assert media.spawn.mass is None
    assert media.spawn.radius is None
    assert _reset_dataset_task_contract(resolved)["particle_count"] == 245


def test_robot_override_is_task_local_and_preserves_asset_provenance():
    cfg = FrankaPourResetDatasetEnvCfg()
    robot = cfg.scene.robot

    assert cfg.robot_asset_identity == FRANKA_POUR_ROBOT_ASSET_ID
    assert cfg.robot_physics_payload_sha256 == FRANKA_POUR_ROBOT_PHYSICS_PAYLOAD_SHA256
    assert cfg.robot_physics_override == FRANKA_POUR_ROBOT_PHYSICS_OVERRIDE
    assert robot is not FRANKA_PANDA_CFG
    assert robot.spawn is not FRANKA_PANDA_CFG.spawn
    assert robot.spawn.usd_path == FRANKA_POUR_ROBOT_USD_PATH
    assert robot.spawn.func is spawn_franka_with_arm_collisions
    assert robot.spawn.articulation_props.enabled_self_collisions is True
    assert FRANKA_PANDA_CFG.spawn.usd_path != FRANKA_POUR_ROBOT_USD_PATH

    shoulder = robot.actuators["panda_shoulder"]
    forearm = robot.actuators["panda_forearm"]
    assert shoulder.stiffness == {
        key: FRANKA_POUR_ARM_DRIVE_STIFFNESS[key] for key in ("panda_joint[1-2]", "panda_joint[3-4]")
    }
    assert shoulder.damping == {
        key: FRANKA_POUR_ARM_DRIVE_DAMPING[key] for key in ("panda_joint[1-2]", "panda_joint[3-4]")
    }
    assert forearm.stiffness == {"panda_joint[5-7]": FRANKA_POUR_ARM_DRIVE_STIFFNESS["panda_joint[5-7]"]}
    assert forearm.damping == {"panda_joint[5-7]": FRANKA_POUR_ARM_DRIVE_DAMPING["panda_joint[5-7]"]}
    assert all(actuator.effort_limit_sim is None for actuator in robot.actuators.values())
    assert all(actuator.velocity_limit_sim is None for actuator in robot.actuators.values())
    assert robot.spawn.joint_drive_props == [MujocoJointCfg(actuatorgravcomp=True)]

    joint_positions = cfg.finalize().scene.robot.init_state.joint_pos
    assert tuple(joint_positions[f"panda_joint{index}"] for index in range(1, 8)) == cfg.arm_home
    assert joint_positions["panda_finger_joint.*"] == pytest.approx(cfg.gripper_open_pos)
    assert FRANKA_PANDA_CFG.actuators["panda_shoulder"].stiffness != shoulder.stiffness


def test_coupled_solver_routes_rigid_bodies_media_and_proxies_once():
    cfg = FrankaPourResetDatasetEnvCfg()
    solver = _resolve_pour_solver_tree(cfg)

    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(solver.coupled, CouplerProxyCfg)
    assert [entry.name for entry in solver.coupled.entries] == [RIGID_ENTRY, MPM_ENTRY]
    assert isinstance(solver.arm_entry, CouplerEntryCfg)
    assert isinstance(solver.arm_entry.solver_cfg, MJWarpSolverCfg)
    assert solver.arm_entry.bodies == [
        r"/World/envs/env_.*/Robot",
        r"/World/envs/env_.*/Table",
        r"/World/envs/env_.*/SourceCup",
        r"/World/envs/env_.*/TargetCup",
    ]
    assert solver.arm_entry.include_static_shapes is True
    assert solver.arm_entry.substeps == 3

    assert isinstance(solver.media_entry, CouplerEntryCfg)
    assert isinstance(solver.media_solver, MPMSolverCfg)
    assert solver.media_entry.bodies == [SPILL_FLOOR_LABEL_PATTERN]
    assert solver.media_entry.all_particles is True
    assert solver.media_entry.include_static_shapes is False
    assert solver.media_entry.include_child_joints is False
    assert solver.media_entry.in_place is True
    assert solver.media_solver.grid_type == "sparse"
    assert solver.media_solver.grid_padding == 0
    assert solver.media_solver.separate_worlds is True
    assert solver.media_solver.transfer_scheme == "apic"
    assert solver.media_solver.strain_basis == "P0"
    assert solver.media_solver.velocity_basis == "Q1"
    assert solver.media_solver.solver == "jacobi"

    assert len(solver.coupled.proxies) == 1
    assert isinstance(solver.proxy, CouplerProxyMappingCfg)
    assert (solver.proxy.source, solver.proxy.destination) == (RIGID_ENTRY, MPM_ENTRY)
    assert solver.proxy.bodies == [r"/World/envs/env_.*/SourceCup", r"/World/envs/env_.*/TargetCup"]
    assert solver.proxy.mode == "lagged"
    assert solver.proxy.mass_scale == pytest.approx(100.0)
    assert solver.proxy.collision_pipeline is None

    resolved = cfg.finalize()
    resolved_solver = _resolve_pour_solver_tree(resolved)
    assert resolved_solver.media_solver.max_active_cell_count == 2 * 512
    assert resolved_solver.media_solver.max_lower_node_count == 64
    assert resolved_solver.media_solver.max_upper_node_count == 32


def test_finalize_propagates_solver_and_capacity_overrides():
    cfg = FrankaPourResetDatasetEnvCfg()
    cfg.scene.num_envs = 8
    cfg.voxel_size = 0.012
    cfg.mpm_iterations = 17
    cfg.mpm_tolerance = 2.0e-4
    cfg.mpm_collider_basis = "S2"
    cfg.mpm_collider_velocity_mode = "backward"
    cfg.rigid_entry_substeps = 4
    cfg.mpm_entry_substeps = 2
    cfg.physics_substeps = 3
    cfg.proxy_iterations = 2
    cfg.proxy_mass_scale = 75.0
    cfg.use_cuda_graph = False
    cfg.mpm_cell_cap_override = 8192

    resolved = cfg.finalize()
    solver = _resolve_pour_solver_tree(resolved)

    assert solver.media_solver.voxel_size == pytest.approx(0.012)
    assert solver.media_solver.max_iterations == 17
    assert solver.media_solver.tolerance == pytest.approx(2.0e-4)
    assert solver.media_solver.collider_basis == "S2"
    assert solver.media_solver.collider_velocity_mode == "backward"
    assert solver.arm_entry.substeps == 4
    assert solver.media_entry.substeps == 2
    assert resolved.sim.physics.num_substeps == 3
    assert solver.coupled.iterations == 2
    assert solver.proxy.mass_scale == pytest.approx(75.0)
    assert resolved.sim.physics.use_cuda_graph is False
    assert solver.media_solver.max_active_cell_count == 8192
    assert solver.media_solver.max_lower_node_count == 8 * 32
    assert solver.media_solver.max_upper_node_count == 32


def test_actions_keep_the_checkpoint_compatible_abi():
    cfg = FrankaPourResetDatasetEnvCfg()
    arm = cfg.actions.arm_action
    gripper = cfg.actions.gripper_action

    assert tuple(vars(cfg.actions)) == ("arm_action", "gripper_action")
    assert arm.class_type == "isaaclab_tasks.contrib.franka_pour.mdp.actions:EMARelativeJointPositionAction"
    assert arm.asset_name == "robot"
    assert arm.joint_names == [f"panda_joint{index}" for index in range(1, 8)]
    assert arm.preserve_order is True
    assert arm.scale == pytest.approx(0.03)
    assert arm.offset == pytest.approx(0.0)
    assert arm.use_zero_offset is True
    assert arm.alpha == pytest.approx(0.20)

    assert gripper.class_type == ("isaaclab_tasks.contrib.franka_pour.mdp.actions:CurriculumGripperPositionAction")
    assert gripper.asset_name == "robot"
    assert gripper.joint_names == ["panda_finger.*"]
    assert gripper.scale == pytest.approx(0.016)
    assert gripper.alpha == pytest.approx(1.0 - (1.0 - 0.2) ** (1.0 / 3.0))
    assert gripper.binary_threshold == pytest.approx(0.0)
    assert gripper.close_position == pytest.approx(0.021)
    assert gripper.default_position == pytest.approx(0.024)
    assert gripper.neutral_position == pytest.approx(0.04)
    assert gripper.contact_min_deflection == pytest.approx(0.001)


def test_observations_keep_the_checkpoint_compatible_abi():
    observations = FrankaPourResetDatasetEnvCfg().observations

    assert _term_names(observations.policy, ObservationTermCfg) == (
        "arm_q",
        "arm_qd",
        "time_remaining",
        "pour_target_fraction",
        "tcp_pose",
        "cup_pose",
        "target_pose",
        "tcp_to_grasp_position_c",
        "grasp_to_tcp_quat",
        "target_position_c",
        "finger_position",
        "finger_velocity",
        "gripper_target",
        "gripper_contact",
        "last_action",
    )
    assert observations.policy.history_length == 13
    assert observations.policy.concatenate_terms is True
    assert observations.policy.enable_corruption is False
    assert observations.policy.arm_q.func is mdp.joint_pos_rel
    assert observations.policy.gripper_contact.scale == pytest.approx(250.0)

    assert _term_names(observations.media, ObservationTermCfg) == (
        "cup_velocity",
        "particle_fractions",
        "particle_source_state",
        "particle_transfer",
    )
    assert observations.media.history_length == 0
    assert observations.media.concatenate_terms is True
    assert observations.media.enable_corruption is False
    assert observations.media.cup_velocity.func is mdp.normalized_cup_velocity_obs
    assert observations.media.cup_velocity.params == {"max_surface_speed": 0.5, "surface_radius": 0.13}
    assert observations.media.cup_velocity.clip == (-2.0, 2.0)

    assert _term_names(observations.privileged, ObservationTermCfg) == (
        "success_dwell",
        "lost_grasp_dwell",
        "held_delivery_history",
    )
    assert observations.privileged.concatenate_terms is True
    assert observations.privileged.enable_corruption is False


def test_rewards_terminations_events_and_curriculum_are_minimal():
    cfg = FrankaPourResetDatasetEnvCfg()

    assert _term_names(cfg.rewards, RewardTermCfg) == (
        "success",
        "action_magnitude",
        "action_rate",
        "failure",
    )
    assert cfg.rewards.success.func is mdp.pour_success_bonus
    assert cfg.rewards.success.weight == pytest.approx(5.0)
    assert cfg.rewards.action_magnitude.func is mdp.action_l2
    assert cfg.rewards.action_magnitude.weight == pytest.approx(-1.0e-4)
    assert cfg.rewards.action_rate.func is mdp.action_rate_l2
    assert cfg.rewards.action_rate.weight == pytest.approx(-1.0e-4)
    assert cfg.rewards.failure.func is mdp.terminal_failure
    assert cfg.rewards.failure.params == {"include_time_out": False}

    assert _term_names(cfg.terminations, TerminationTermCfg) == (
        "failure",
        "extreme_rigid_state",
        "source_receiver_overlap",
        "lost_grasp",
        "spill",
        "particle_out_of_bounds",
        "success",
        "learning_progress_context",
        "time_out",
    )
    assert cfg.terminations.success.func is mdp.immediate_pour_success
    assert cfg.terminations.success.params == {}
    assert cfg.terminations.lost_grasp.params["terminate"] is False
    assert cfg.terminations.spill.params == {"terminate": True}
    assert cfg.terminations.time_out.func is mdp.unsuccessful_time_out
    assert cfg.terminations.time_out.time_out is True
    assert cfg.terminations.learning_progress_context.func is mdp.PourResetLearningProgress
    assert cfg.terminations.learning_progress_context.params["minimum_progress"] == pytest.approx(0.08)
    assert cfg.terminations.learning_progress_context.params["minimum_episode_steps"] == 3
    assert cfg.events.reset_scene.func is mdp.reset_pour_scene
    assert cfg.events.reset_scene.mode == "reset"
    assert tuple(vars(cfg.curriculum)) == ("reset_dataset",)
    assert cfg.curriculum.reset_dataset.func is mdp.PourResetDatasetCurriculum


def test_play_mode_freezes_single_world_playback_without_changing_policy_abi():
    cfg = FrankaPourResetDatasetEnvCfg()
    action_contract = _reset_dataset_task_contract(cfg)["action_abi"]
    observation_names = tuple(
        _term_names(group, ObservationTermCfg)
        for group in (cfg.observations.policy, cfg.observations.media, cfg.observations.privileged)
    )

    cfg.play_mode()
    resolved = cfg.finalize()
    solver = _resolve_pour_solver_tree(resolved).media_solver

    assert cfg.scene.num_envs == 1
    assert cfg.curriculum_freeze is True
    assert cfg.sim.default_visualizer_cfg.eye == (0.9, 0.65, 0.5)
    assert cfg.sim.default_visualizer_cfg.lookat == (0.5, 0.0, 0.1)
    assert cfg.sim.default_visualizer_cfg.origin_type == "env"
    assert cfg.sim.default_visualizer_cfg.origin_env_index == 0
    assert _reset_dataset_task_contract(cfg)["action_abi"] == action_contract
    assert (
        tuple(
            _term_names(group, ObservationTermCfg)
            for group in (cfg.observations.policy, cfg.observations.media, cfg.observations.privileged)
        )
        == observation_names
    )
    assert solver.max_active_cell_count == 512
    assert solver.max_lower_node_count == 32
    assert solver.max_upper_node_count == 32


def test_reset_dataset_defaults_runner_and_task_contract_match_training():
    cfg = FrankaPourResetDatasetEnvCfg()
    sampler = cfg.reset_dataset_sampler
    runner = FrankaPourResetDatasetPPORunnerCfg()
    contract = _reset_dataset_task_contract(cfg)

    assert cfg.reset_dataset_path == "datasets/franka_pour/reset_dataset.pt"
    assert cfg.reset_dataset_content_sha256 == FRANKA_POUR_RESET_DATASET_CONTENT_SHA256
    assert cfg.reset_dataset_top_grasp_count is None
    assert cfg.reset_dataset_sampling_mode == "adaptive"
    assert cfg.reset_dataset_expected_sampling_profile == RESET_DATASET_PROFILE_FULL
    assert cfg.reset_dataset_expected_grasp_side_ids == (0, 1, 2, 3)
    assert cfg.curriculum_freeze is False
    assert sampler.monitored_history_len == 50
    assert sampler.target_success_rate == pytest.approx(0.50)
    assert sampler.kappa == pytest.approx(1.0)
    assert sampler.epsilon == pytest.approx(1.0e-4)
    assert sampler.uniform_fraction_initial == pytest.approx(0.50)
    assert sampler.uniform_fraction == pytest.approx(0.25)

    assert runner.init_at_random_ep_len is False
    assert runner.num_steps_per_env == 96
    assert runner.clip_actions == pytest.approx(1.0)
    assert runner.obs_groups == {
        "actor": ["policy", "media"],
        "critic": ["policy", "media", "privileged"],
    }
    assert contract["robot_asset"] == FRANKA_POUR_ROBOT_ASSET_ID
    assert contract["robot_physics_payload_sha256"] == FRANKA_POUR_ROBOT_PHYSICS_PAYLOAD_SHA256
    assert contract["robot_physics_override"] == FRANKA_POUR_ROBOT_PHYSICS_OVERRIDE
    assert contract["use_newton_actuators"] is True
    assert contract["action_abi"]["term_order"] == ("arm_action", "gripper_action")
    assert contract["success_semantics"] == RESET_DATASET_SUCCESS_SEMANTICS
    assert contract["simulation_dt"] == pytest.approx(1.0 / 120.0)
    assert contract["policy_decimation"] == 4
    assert contract["physics_substeps"] == 2
    assert contract["rigid_entry_substeps"] == 3
    assert contract["mpm_entry_substeps"] == 1
    assert contract["mpm_iterations"] == 24
    assert contract["proxy_mass_scale"] == pytest.approx(100.0)
    assert contract["particle_count"] == 245
    assert contract["particle_spacing"] == pytest.approx(0.005)
    assert contract["particle_radius"] == pytest.approx(0.0025)
    assert contract["particle_mass"] == pytest.approx(0.0001875)
    assert contract["pour_target_fraction"] == pytest.approx(0.70)
    assert contract["maximum_spill_fraction"] == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reset_dataset_path", "", "reset_dataset_path"),
        ("reset_dataset_path", None, "reset_dataset_path"),
        ("reset_dataset_content_sha256", "A" * 64, "reset_dataset_content_sha256"),
        ("reset_dataset_content_sha256", "abc", "reset_dataset_content_sha256"),
        ("reset_dataset_sampling_mode", "weighted", "reset_dataset_sampling_mode"),
        ("reset_dataset_expected_sampling_profile", "small", "reset_dataset_expected_sampling_profile"),
        ("reset_dataset_expected_grasp_side_ids", (0, 0), "reset_dataset_expected_grasp_side_ids"),
        ("reset_dataset_expected_grasp_side_ids", (4,), "reset_dataset_expected_grasp_side_ids"),
        ("reset_dataset_top_grasp_count", 0, "reset_dataset_top_grasp_count"),
        ("reset_dataset_top_grasp_count", 8, "curriculum_freeze"),
    ],
)
def test_reset_dataset_rejects_invalid_artifact_selection(field, value, message):
    with pytest.raises(ValueError, match=message):
        FrankaPourResetDatasetEnvCfg(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("mpm_iterations", 0, ValueError),
        ("mpm_cell_capacity_alignment", 0, ValueError),
        ("mpm_cell_cap_override", 0, ValueError),
        ("mpm_collider_basis", "Q1", ValueError),
        ("mpm_collider_velocity_mode", "central", ValueError),
        ("physics_substeps", True, ValueError),
        ("use_cuda_graph", 1, TypeError),
    ],
)
def test_finalize_rejects_invalid_solver_overrides(field, value, error_type):
    cfg = FrankaPourResetDatasetEnvCfg()
    setattr(cfg, field, value)

    with pytest.raises(error_type, match=field):
        cfg.finalize()
