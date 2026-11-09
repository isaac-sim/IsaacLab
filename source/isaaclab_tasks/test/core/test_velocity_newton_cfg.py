# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

import pytest
from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab.managers import RewardTermCfg

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.config.g1.rough_29dof_env_cfg import G1RoughTeacherEnvCfg
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


@pytest.mark.parametrize("env_type", [G1RoughEnvCfg, G1RoughTeacherEnvCfg])
def test_g1_rough_newton_has_sufficient_constraint_capacity(env_type):
    env_cfg = env_type()

    solver_cfg = env_cfg.sim.physics.newton_mjwarp.solver_cfg

    assert isinstance(solver_cfg, MJWarpSolverCfg)
    assert solver_cfg.njmax == 300


@pytest.mark.parametrize(
    "task,env_type,asset,obs_groups,iterations,save_interval",
    [
        ("Isaac-Velocity-Rough-G1", G1RoughEnvCfg, "g1_minimal.usd", ["policy"], 5000, 50),
        (
            "Isaac-Velocity-Rough-G1-29dof",
            G1RoughTeacherEnvCfg,
            "g1_29dof.usd",
            ["proprio", "height_scan"],
            6000,
            250,
        ),
    ],
)
def test_g1_rough_task_registrations(task, env_type, asset, obs_groups, iterations, save_interval):
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")

    assert type(env_cfg) is env_type
    assert env_cfg.scene.robot.spawn.usd_path.endswith(f"/{asset}")
    assert agent_cfg.obs_groups == {"actor": obs_groups, "critic": obs_groups}
    for group in obs_groups:
        assert getattr(env_cfg.observations, group) is not None
    assert agent_cfg.max_iterations == iterations
    assert agent_cfg.save_interval == save_interval
    assert (agent_cfg.algorithm.symmetry_cfg is not None) == (env_type is G1RoughTeacherEnvCfg)


@pytest.mark.parametrize("env_type", [G1RoughEnvCfg, G1RoughTeacherEnvCfg])
def test_g1_rough_configuration_settings(env_type):
    """Preserve the minimal and deployment configurations across shared reward refactors."""
    cfg = env_type()
    teacher = env_type is G1RoughTeacherEnvCfg
    weights = {
        "track_lin_vel_xy_exp": 2.0 if teacher else 1.0,
        "track_ang_vel_z_exp": 2.0,
        "lin_vel_z_l2": 0.0,
        "ang_vel_xy_l2": -0.05,
        "dof_torques_l2": -1.5e-7,
        "dof_acc_l2": -1.25e-7,
        "action_rate_l2": -0.005,
        "feet_air_time": 1.0 if teacher else 0.25,
        "flat_orientation_l2": -1.0,
        "dof_pos_limits": -1.0,
        "termination_penalty": -200.0,
        "feet_slide": -0.5 if teacher else -0.1,
        "joint_deviation_hip": -1.0 if teacher else -0.1,
        "joint_deviation_arms": -0.1,
        "joint_deviation_fingers": -0.05,
        "joint_deviation_torso": -1.0 if teacher else -0.1,
    }
    if teacher:
        weights.update(pelvis_height=-10.0, torso_ang_vel=-0.1, root_downward_speed=-1.0, feet_support_edge=-5.0)
    assert {name: term.weight for name, term in vars(cfg.rewards).items() if isinstance(term, RewardTermCfg)} == weights
    assert cfg.rewards.undesired_contacts is None
    assert cfg.rewards.dof_acc_l2.params["asset_cfg"].joint_names == [".*_hip_.*", ".*_knee_joint"]
    assert cfg.rewards.dof_torques_l2.params["asset_cfg"].joint_names == [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
    assert cfg.rewards.joint_deviation_hip.func is (mdp.joint_deviation_l2 if teacher else mdp.joint_deviation_l1)
    assert cfg.rewards.joint_deviation_torso.func is (mdp.joint_deviation_l2 if teacher else mdp.joint_deviation_l1)
    torso = cfg.rewards.joint_deviation_torso.params["asset_cfg"].joint_names
    assert torso == (["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] if teacher else "torso_joint")
    assert cfg.actions.joint_pos.scale == (None if teacher else 0.5)
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
        assert cfg.scene.terrain.terrain_generator.sub_terrains[name].step_height_range == (
            0.05,
            0.35 if teacher else 0.23,
        )
    assert cfg.scene.height_scanner.prim_path == "{ENV_REGEX_NS}/Robot/torso_link"
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.0, 1.0)
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (0.0, 0.0)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (-1.0, 1.0)
    assert cfg.events.base_com is None
    assert cfg.events.add_base_mass.params["asset_cfg"].body_names == "torso_link"
    assert cfg.events.base_external_force_torque.params["asset_cfg"].body_names == "torso_link"
    assert cfg.events.reset_robot_joints.params["position_range"] == (1.0, 1.0)
    assert cfg.terminations.base_contact.params["sensor_cfg"].body_names == "torso_link"
    assert (getattr(cfg.events, "asset_parameters", None) is not None) == teacher
    if teacher:
        assert cfg.rewards.feet_support_edge.func is mdp.feet_support_edge_distance
        assert cfg.events.asset_parameters.mode == "prestartup"


@pytest.mark.parametrize("env_type", [G1RoughEnvCfg, G1RoughTeacherEnvCfg])
def test_g1_rough_configuration_instances_are_independent(env_type):
    cfg = env_type()
    others = [G1RoughEnvCfg(), G1RoughTeacherEnvCfg()]
    snapshots = [deepcopy(other.to_dict()) for other in others]

    cfg.rewards.feet_slide.weight = 0.0
    cfg.rewards.dof_acc_l2.params["asset_cfg"].joint_names.append("test_joint")
    cfg.events.add_base_mass.params["asset_cfg"].body_names = "test_body"
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)

    assert [other.to_dict() for other in others] == snapshots
    assert env_type().to_dict() == snapshots[int(env_type is G1RoughTeacherEnvCfg)]
