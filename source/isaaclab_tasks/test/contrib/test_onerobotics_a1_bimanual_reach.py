# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused registration, contract, kinematics, and runtime tests for A1 bimanual Reach."""

import sys

# Import pinocchio before AppLauncher so Isaac Lab's dependency wins over Isaac Sim's bundled copy.
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, device="cpu")
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.reach.config.onerobotics_a1.bimanual import reach_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

_TASK_ID = "IsaacContrib-Reach-OneRobotics-A1-Bimanual"
_UNIMANUAL_TASK_ID = "IsaacContrib-Reach-OneRobotics-A1"
_RIGHT_JOINT_PATTERNS = [f".*joint_r{index}.*" for index in range(1, 8)]
_LEFT_JOINT_PATTERNS = [f".*joint_l{index}.*" for index in range(1, 8)]
_JOINT_PATTERNS = _RIGHT_JOINT_PATTERNS + _LEFT_JOINT_PATTERNS
_JOINT_LIMITS = [
    [-1.04, 3.14],
    [-3.14, 0.26],
    [-2.75, 2.75],
    [-1.91, 1.91],
    [-2.75, 2.75],
    [-1.57, 1.57],
    [-2.75, 2.75],
    [-3.14, 1.04],
    [-0.26, 3.14],
    [-2.75, 2.75],
    [-1.91, 1.91],
    [-2.75, 2.75],
    [-1.57, 1.57],
    [-2.75, 2.75],
]
_JOINT_EFFORT_LIMITS = ([26.859] * 3 + [5.975] * 4) * 2
_JOINT_VELOCITY_LIMITS = ([2.6179938779914944] * 3 + [12.566370614359172] * 4) * 2
_JOINT_STIFFNESS = ([150.0] * 4 + [40.0] * 3) * 2
_JOINT_DAMPING = ([4.0] * 4 + [1.0] * 3) * 2
_JOINT_ARMATURE = ([0.050927514873] * 3 + [0.002193] * 4) * 2
_NONZERO_JOINT_POSITION = [
    0.3,
    -0.9,
    0.4,
    0.7,
    -0.5,
    0.8,
    0.6,
    -0.3,
    0.9,
    -0.4,
    -0.7,
    0.5,
    -0.8,
    -0.6,
]


def _assert_finite(value) -> None:
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, tuple | list):
        for item in value:
            _assert_finite(item)


def _assert_fk_matches_body(env, command_name: str, joint_ids: list[int], body_id: int) -> None:
    """Compare the command's mount-and-axis-aware FK with the spawned body pose."""
    robot = env.unwrapped.scene["robot"]
    command_term = env.unwrapped.command_manager.get_term(command_name)
    fk_position, fk_orientation = command_term._forward_kinematics(robot.data.joint_pos.torch[:, joint_ids])
    body_position, body_orientation = subtract_frame_transforms(
        robot.data.root_link_pose_w.torch[:, :3],
        robot.data.root_link_pose_w.torch[:, 3:7],
        robot.data.body_link_pose_w.torch[:, body_id, :3],
        robot.data.body_link_pose_w.torch[:, body_id, 3:7],
    )
    torch.testing.assert_close(fk_position, body_position, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(
        quat_error_magnitude(fk_orientation, body_orientation),
        torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device),
        atol=1.0e-5,
        rtol=0.0,
    )


def test_task_registration_and_configuration():
    """The task exposes only RSL-RL and preserves the 14-D right-then-left contract."""
    spec = gym.spec(_TASK_ID)
    assert spec.kwargs["default_agent"] == "rsl_rl"
    assert "rsl_rl_cfg_entry_point" in spec.kwargs
    assert "rl_games_cfg_entry_point" not in spec.kwargs
    assert "skrl_cfg_entry_point" not in spec.kwargs

    env_cfg = load_cfg_from_registry(_TASK_ID, "env_cfg_entry_point")
    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert list(env_cfg.actions.to_dict()) == ["right_arm_action", "left_arm_action"]
    assert env_cfg.actions.right_arm_action.joint_names == _RIGHT_JOINT_PATTERNS
    assert env_cfg.actions.left_arm_action.joint_names == _LEFT_JOINT_PATTERNS
    assert env_cfg.actions.right_arm_action.preserve_order
    assert env_cfg.actions.left_arm_action.preserve_order
    assert env_cfg.actions.right_arm_action.scale == env_cfg.actions.left_arm_action.scale == 0.5
    assert env_cfg.actions.right_arm_action.use_default_offset
    assert env_cfg.actions.left_arm_action.use_default_offset
    assert env_cfg.commands.right_ee_pose.body_name == ".*Link_r7.*"
    assert env_cfg.commands.left_ee_pose.body_name == ".*Link_l7.*"
    assert env_cfg.commands.right_ee_pose.joint_range_scale == 0.8
    assert env_cfg.commands.left_ee_pose.joint_range_scale == 0.8
    assert env_cfg.commands.right_ee_pose.fixed_transform is not None
    assert env_cfg.commands.left_ee_pose.fixed_transform is not None
    assert all(len(entry) == 4 for entry in env_cfg.commands.right_ee_pose.chain)
    assert all(len(entry) == 4 for entry in env_cfg.commands.left_ee_pose.chain)
    assert env_cfg.scene.robot.init_state.joint_pos == {"joint_[rl][1-7]": 0.0}
    assert env_cfg.sim.dt == 1.0 / 200.0
    assert env_cfg.decimation == 4
    assert env_cfg.sim.render_interval == 4

    assert env_cfg.rewards.right_end_effector_position_tracking.weight == -0.2
    assert env_cfg.rewards.left_end_effector_position_tracking.weight == -0.2
    assert env_cfg.rewards.right_end_effector_orientation_tracking.weight == -0.1
    assert env_cfg.rewards.left_end_effector_orientation_tracking.weight == -0.1
    assert env_cfg.rewards.success.weight == 10.0
    assert env_cfg.terminations.success.params["command_names"] == ("right_ee_pose", "left_ee_pose")
    assert env_cfg.terminations.success.func is reach_env_cfg.AllPoseCommandsSuccess

    agent_cfg = load_cfg_from_registry(_TASK_ID, "rsl_rl_cfg_entry_point")
    unimanual_agent_cfg = load_cfg_from_registry(_UNIMANUAL_TASK_ID, "rsl_rl_cfg_entry_point")
    assert agent_cfg.experiment_name == "onerobotics_a1_bimanual_reach"
    agent_dict = agent_cfg.to_dict()
    unimanual_agent_dict = unimanual_agent_cfg.to_dict()
    agent_dict.pop("experiment_name")
    unimanual_agent_dict.pop("experiment_name")
    assert agent_dict == unimanual_agent_dict


def test_bimanual_success_requires_both_commands(monkeypatch):
    """The single success termination is the logical AND of both standard pose checks."""
    side_success = {
        "right_ee_pose": torch.tensor([True, True, False, False]),
        "left_ee_pose": torch.tensor([True, False, True, False]),
    }
    monkeypatch.setattr(
        reach_env_cfg.mdp,
        "pose_command_success",
        lambda _env, command_name: side_success[command_name],
    )
    success = reach_env_cfg.all_pose_commands_success(object(), ("right_ee_pose", "left_ee_pose"))
    assert torch.equal(success, torch.tensor([True, False, False, False]))


def test_environment_contract_fk_reset_and_steps_are_finite():
    """The one-articulation task validates its 56-D observation and both FK chains at runtime."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(_TASK_ID, device="cpu", num_envs=2)
        env = gym.make(_TASK_ID, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        observations, _ = env.reset()
        _assert_finite(observations)
        assert observations["policy"].shape == (env.unwrapped.num_envs, 56)
        assert env.unwrapped.observation_manager.group_obs_dim["policy"] == (56,)
        assert env.unwrapped.action_manager.active_terms == ["right_arm_action", "left_arm_action"]
        assert env.unwrapped.action_manager.action_term_dim == [7, 7]
        assert env.unwrapped.action_manager.total_action_dim == 14
        assert env.unwrapped.command_manager.active_terms == ["right_ee_pose", "left_ee_pose"]
        assert env.unwrapped.physics_dt == 1.0 / 200.0
        assert env.unwrapped.step_dt == 1.0 / 50.0

        robot = env.unwrapped.scene["robot"]
        joint_ids, joint_names = robot.find_joints(_JOINT_PATTERNS, preserve_order=True)
        right_joint_ids, _ = robot.find_joints(_RIGHT_JOINT_PATTERNS, preserve_order=True)
        left_joint_ids, _ = robot.find_joints(_LEFT_JOINT_PATTERNS, preserve_order=True)
        right_body_ids, right_body_names = robot.find_bodies(".*Link_r7.*")
        left_body_ids, left_body_names = robot.find_bodies(".*Link_l7.*")
        assert len(joint_ids) == 14
        assert joint_ids == right_joint_ids + left_joint_ids
        assert [f"joint_r{index}" in name for index, name in enumerate(joint_names[:7], start=1)] == [True] * 7
        assert [f"joint_l{index}" in name for index, name in enumerate(joint_names[7:], start=1)] == [True] * 7
        assert len(right_body_ids) == 1 and "Link_r7" in right_body_names[0]
        assert len(left_body_ids) == 1 and "Link_l7" in left_body_names[0]
        assert len(robot.body_names) == 17
        assert "Link_r0" in robot.body_names and "Link_l0" in robot.body_names
        assert set(robot.actuators) == {"arm_4340", "arm_4310"}
        torch.testing.assert_close(
            robot.data.default_joint_pos.torch[0, joint_ids],
            torch.zeros(14, device=env.unwrapped.device),
        )
        torch.testing.assert_close(
            robot.data.joint_pos.torch[:, joint_ids],
            torch.zeros((env.unwrapped.num_envs, 14), device=env.unwrapped.device),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            robot.data.joint_pos_limits.torch[0, joint_ids],
            torch.tensor(_JOINT_LIMITS, device=env.unwrapped.device),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            robot.data.joint_effort_limits.torch[0, joint_ids],
            torch.tensor(_JOINT_EFFORT_LIMITS, device=env.unwrapped.device),
        )
        torch.testing.assert_close(
            robot.data.joint_velocity_limits.torch[0, joint_ids],
            torch.tensor(_JOINT_VELOCITY_LIMITS, device=env.unwrapped.device),
        )
        torch.testing.assert_close(
            robot.data.soft_joint_vel_limits.torch[0, joint_ids],
            torch.tensor(_JOINT_VELOCITY_LIMITS, device=env.unwrapped.device),
        )
        torch.testing.assert_close(
            robot.data.joint_stiffness.torch[0, joint_ids],
            torch.tensor(_JOINT_STIFFNESS, device=env.unwrapped.device),
        )
        torch.testing.assert_close(
            robot.data.joint_damping.torch[0, joint_ids],
            torch.tensor(_JOINT_DAMPING, device=env.unwrapped.device),
        )
        torch.testing.assert_close(
            robot.data.joint_armature.torch[0, joint_ids],
            torch.tensor(_JOINT_ARMATURE, device=env.unwrapped.device),
        )

        right_command = env.unwrapped.command_manager.get_term("right_ee_pose")
        left_command = env.unwrapped.command_manager.get_term("left_ee_pose")
        assert right_command._chain_joint_ids == right_joint_ids
        assert left_command._chain_joint_ids == left_joint_ids
        assert not right_command._use_legacy_z_axis_fk
        assert not left_command._use_legacy_z_axis_fk
        assert torch.linalg.vector_norm(right_command._fixed_transform[:3, 3]) > 0.0
        assert torch.linalg.vector_norm(left_command._fixed_transform[:3, 3]) > 0.0
        torch.testing.assert_close(
            torch.linalg.vector_norm(right_command._joint_axes, dim=-1),
            torch.ones(7, dtype=torch.float64, device=env.unwrapped.device),
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(left_command._joint_axes, dim=-1),
            torch.ones(7, dtype=torch.float64, device=env.unwrapped.device),
        )
        _assert_fk_matches_body(env, "right_ee_pose", right_joint_ids, right_body_ids[0])
        _assert_fk_matches_body(env, "left_ee_pose", left_joint_ids, left_body_ids[0])

        # Standard pose commands write the same metric key. Verify the later
        # termination reset replaces their per-arm value with logical-AND episode success.
        success_term = env.unwrapped.termination_manager.get_term_cfg("success").func
        assert isinstance(success_term, reach_env_cfg.AllPoseCommandsSuccess)
        right_command._succeeded[:] = False
        left_command._succeeded[:] = True
        success_term._succeeded[:] = torch.tensor([True, False], device=env.unwrapped.device)
        observations, reset_extras = env.reset()
        _assert_finite(observations)
        assert reset_extras["log"]["Metrics/success_rate"] == 0.5
        assert not success_term._succeeded.any()

        nonzero_joint_pos = torch.tensor(_NONZERO_JOINT_POSITION, device=env.unwrapped.device).repeat(
            env.unwrapped.num_envs, 1
        )
        robot.write_joint_position_to_sim_index(position=nonzero_joint_pos, joint_ids=joint_ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(nonzero_joint_pos), joint_ids=joint_ids)
        env.unwrapped.sim.forward()
        robot.update(0.0)
        _assert_fk_matches_body(env, "right_ee_pose", right_joint_ids, right_body_ids[0])
        _assert_fk_matches_body(env, "left_ee_pose", left_joint_ids, left_body_ids[0])

        observations, _ = env.reset()
        _assert_finite(observations)
        zero_actions = torch.zeros((env.unwrapped.num_envs, 14), device=env.unwrapped.device)
        zero_transition = env.step(zero_actions)
        _assert_finite(zero_transition[:-1])
        assert torch.equal(zero_transition[2], env.unwrapped.termination_manager.get_term("success"))
        assert not zero_transition[3].any()

        random_actions = 2.0 * torch.rand_like(zero_actions) - 1.0
        random_transition = env.step(random_actions)
        _assert_finite(random_transition[:-1])
        assert torch.equal(random_transition[2], env.unwrapped.termination_manager.get_term("success"))
        assert not random_transition[3].any()
        _assert_finite(robot.data.joint_pos.torch)
        _assert_finite(robot.actuators.applied_effort.torch)
        assert torch.all(
            torch.abs(robot.actuators.applied_effort.torch[:, joint_ids])
            <= torch.tensor(_JOINT_EFFORT_LIMITS, device=env.unwrapped.device) + 1.0e-5
        )
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()
