# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused registration, configuration, and runtime tests for OneRobotics A1 Reach."""

import sys

# Import pinocchio before AppLauncher so Isaac Lab's dependency wins over Isaac Sim's bundled copy.
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_physx.physics import PhysxCfg

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply, quat_error_magnitude, subtract_frame_transforms

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

from isaaclab_assets import ONEROBOTICS_A1_UNIMANUAL_CFG

_TASK_ID = "IsaacContrib-Reach-OneRobotics-A1"
_JOINT_PATTERNS = [f".*joint{index}.*" for index in range(1, 8)]
_HOME_POSITION = [0.0, -0.6, 0.0, 1.0, 0.0, 0.5, 0.0]
_JOINT_LIMITS = [
    [-1.04, 3.14],
    [-3.14, 0.26],
    [-2.76, 2.76],
    [-1.92, 1.92],
    [-2.23, 2.23],
    [-1.57, 1.57],
    [-2.76, 2.76],
]
_JOINT_EFFORT_LIMITS = [26.859] * 3 + [5.975] * 4
_JOINT_VELOCITY_LIMITS = [2.6179938779914944] * 3 + [12.566370614359172] * 4
_JOINT_STIFFNESS = [150.0] * 4 + [40.0] * 3
_JOINT_DAMPING = [4.0] * 4 + [1.0] * 3
_JOINT_ARMATURE = [0.050927514873] * 3 + [0.002193] * 4
_TABLE_POSITION = (0.5, 0.0, -0.5)
_TABLE_ROTATION = (0.0, 0.0, 0.0, 1.0)
_TABLE_SIZE = (0.9, 1.3, 1.0)
_TABLE_TOP_Z = 0.0
_TABLE_X_BOUNDS = (0.05, 0.95)
_TABLE_Y_BOUNDS = (-0.65, 0.65)
_A1_BASE_LOCAL_X_BOUNDS = (-0.0421904102, 0.0420310199)
_A1_BASE_LOCAL_MIN_Z = -0.01575
_A1_ROOT_POSITION = (0.1, 0.0, 0.01575)
_A1_ROOT_ROTATION = (0.0, 0.0, 1.0, 0.0)
_VIEWER_EYE = (1.35, -1.25, 0.85)
_VIEWER_LOOKAT = (0.42, 0.0, 0.22)
_TF32_REGRESSION_JOINT_POSITION = [
    0.8695318698883057,
    -1.2529727220535278,
    1.9887187480926514,
    -0.49152350425720215,
    -1.273512363433838,
    1.0959815979003906,
    -0.5835989713668823,
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


def test_task_registration_and_configuration():
    """The task exposes one RSL-RL entry point with the exact A1 control contract."""
    spec = gym.spec(_TASK_ID)
    assert spec.kwargs["default_agent"] == "rsl_rl"
    assert "rsl_rl_cfg_entry_point" in spec.kwargs
    assert "rl_games_cfg_entry_point" not in spec.kwargs
    assert "skrl_cfg_entry_point" not in spec.kwargs

    env_cfg = load_cfg_from_registry(_TASK_ID, "env_cfg_entry_point")
    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert env_cfg.actions.arm_action.joint_names == _JOINT_PATTERNS
    assert env_cfg.actions.arm_action.preserve_order
    assert env_cfg.actions.arm_action.scale == 0.5
    assert env_cfg.actions.arm_action.use_default_offset
    assert env_cfg.commands.ee_pose.body_name == ".*Link7.*"
    assert env_cfg.commands.ee_pose.joint_range_scale == 0.8
    assert len(env_cfg.commands.ee_pose.chain) == 7
    assert env_cfg.scene.table.prim_path == "{ENV_REGEX_NS}/Table"
    assert env_cfg.scene.table.init_state.pos == _TABLE_POSITION
    assert env_cfg.scene.table.init_state.rot == _TABLE_ROTATION
    assert isinstance(env_cfg.scene.table.spawn, sim_utils.CuboidCfg)
    assert env_cfg.scene.table.spawn.size == _TABLE_SIZE
    assert env_cfg.scene.table.spawn.collision_props is not None
    assert env_cfg.scene.table.spawn.rigid_props is None
    assert env_cfg.scene.robot.init_state.pos == _A1_ROOT_POSITION
    assert env_cfg.scene.robot.init_state.rot == _A1_ROOT_ROTATION
    assert env_cfg.scene.robot.init_state is not ONEROBOTICS_A1_UNIMANUAL_CFG.init_state
    assert ONEROBOTICS_A1_UNIMANUAL_CFG.init_state.pos == (0.0, 0.0, 0.0)
    assert ONEROBOTICS_A1_UNIMANUAL_CFG.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert env_cfg.scene.robot.spawn.fix_base
    assert env_cfg.sim.default_visualizer_cfg.eye == _VIEWER_EYE
    assert env_cfg.sim.default_visualizer_cfg.lookat == _VIEWER_LOOKAT
    table_top_z = _TABLE_POSITION[2] + 0.5 * _TABLE_SIZE[2]
    base_bottom_z = _A1_ROOT_POSITION[2] + _A1_BASE_LOCAL_MIN_Z
    base_x_bounds = (
        _A1_ROOT_POSITION[0] - _A1_BASE_LOCAL_X_BOUNDS[1],
        _A1_ROOT_POSITION[0] - _A1_BASE_LOCAL_X_BOUNDS[0],
    )
    assert table_top_z == _TABLE_TOP_Z
    assert abs(base_bottom_z - table_top_z) < 1.0e-9
    assert _TABLE_X_BOUNDS[0] <= base_x_bounds[0] <= base_x_bounds[1] <= _TABLE_X_BOUNDS[1]
    assert env_cfg.sim.dt == 1.0 / 200.0
    assert env_cfg.decimation == 4
    assert env_cfg.sim.render_interval == 4
    assert env_cfg.rewards.end_effector_position_tracking.weight == -0.2
    assert env_cfg.rewards.end_effector_orientation_tracking.weight == -0.1

    agent_cfg = load_cfg_from_registry(_TASK_ID, "rsl_rl_cfg_entry_point")
    assert agent_cfg.actor.hidden_dims == [128, 128]
    assert agent_cfg.critic.hidden_dims == [128, 128]
    assert agent_cfg.algorithm.learning_rate == 1.0e-3
    assert agent_cfg.num_steps_per_env == 24
    assert agent_cfg.max_iterations == 3000
    assert agent_cfg.obs_groups == {"actor": ["policy"], "critic": ["policy"]}


def test_environment_reset_zero_and_random_steps_are_finite():
    """The A1 spawns and completes reset, zero-action, and random-action steps."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(_TASK_ID, device="cuda:0", num_envs=2)
        env = gym.make(_TASK_ID, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        observations, _ = env.reset()
        _assert_finite(observations)

        robot = env.unwrapped.scene["robot"]
        assert "table" in env.unwrapped.scene.keys()
        joint_ids, joint_names = robot.find_joints(_JOINT_PATTERNS, preserve_order=True)
        body_ids, body_names = robot.find_bodies(".*Link7.*")
        assert len(joint_ids) == 7
        assert [f"joint{index}" in name for index, name in enumerate(joint_names, start=1)] == [True] * 7
        assert len(body_ids) == 1 and "Link7" in body_names[0]
        assert set(robot.actuators) == {"arm_4340", "arm_4310"}
        assert env.unwrapped.action_manager.total_action_dim == 7
        assert env.unwrapped.physics_dt == 1.0 / 200.0
        assert env.unwrapped.step_dt == 1.0 / 50.0
        expected_root_position = torch.tensor(_A1_ROOT_POSITION, device=env.unwrapped.device).repeat(
            env.unwrapped.num_envs, 1
        )
        expected_root_rotation = torch.tensor(_A1_ROOT_ROTATION, device=env.unwrapped.device).repeat(
            env.unwrapped.num_envs, 1
        )
        torch.testing.assert_close(
            robot.data.root_link_pose_w.torch[:, :3] - env.unwrapped.scene.env_origins,
            expected_root_position,
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            quat_error_magnitude(robot.data.root_link_pose_w.torch[:, 3:7], expected_root_rotation),
            torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device),
            atol=1.0e-6,
            rtol=0.0,
        )

        # The inherited core Reach table is a replicated static collider with an exact z=0 top.
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        for env_index in range(env.unwrapped.num_envs):
            table_prim = env.unwrapped.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Table")
            assert table_prim.IsValid()
            table_prims = list(Usd.PrimRange(table_prim))
            collision_prims = [prim for prim in table_prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
            assert collision_prims
            assert all(not prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in table_prims)
            collision_ranges = [bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange() for prim in collision_prims]
            table_top_z = max(collision_range.GetMax()[2] for collision_range in collision_ranges)
            assert abs(table_top_z - env.unwrapped.scene.env_origins[env_index, 2].item() - _TABLE_TOP_Z) < 1.0e-6

            # The spawned source mesh independently confirms a flush mount fully over the tabletop.
            base_prim = env.unwrapped.sim.stage.GetPrimAtPath(
                f"/World/envs/env_{env_index}/Robot/Geometry/base_link/base_link"
            )
            assert base_prim.IsValid()
            base_range = bbox_cache.ComputeWorldBound(base_prim).ComputeAlignedRange()
            assert abs(base_range.GetMin()[2] - table_top_z) < 1.0e-6
            assert min(collision_range.GetMin()[0] for collision_range in collision_ranges) <= base_range.GetMin()[0]
            assert max(collision_range.GetMax()[0] for collision_range in collision_ranges) >= base_range.GetMax()[0]
            assert min(collision_range.GetMin()[1] for collision_range in collision_ranges) <= base_range.GetMin()[1]
            assert max(collision_range.GetMax()[1] for collision_range in collision_ranges) >= base_range.GetMax()[1]
        torch.testing.assert_close(
            robot.data.default_joint_pos.torch[0, joint_ids],
            torch.tensor(_HOME_POSITION, device=env.unwrapped.device),
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

        # Cross-check the hard-coded kinematic chain against the spawned Link7 pose.
        command_term = env.unwrapped.command_manager.get_term("ee_pose")
        forward_kinematics = getattr(command_term, "_forward_kinematics")
        fk_position, fk_orientation = forward_kinematics(robot.data.joint_pos.torch[:, joint_ids])
        link7_position, link7_orientation = subtract_frame_transforms(
            robot.data.root_link_pose_w.torch[:, :3],
            robot.data.root_link_pose_w.torch[:, 3:7],
            robot.data.body_link_pose_w.torch[:, body_ids[0], :3],
            robot.data.body_link_pose_w.torch[:, body_ids[0], 3:7],
        )
        torch.testing.assert_close(fk_position, link7_position, atol=1.0e-5, rtol=1.0e-5)
        torch.testing.assert_close(
            quat_error_magnitude(fk_orientation, link7_orientation),
            torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device),
            atol=1.0e-5,
            rtol=0.0,
        )

        # The unified RSL-RL entry point enables TF32. Exercise a training-sized
        # command batch under that setting to prevent invalid rotation quaternions.
        joint_limits = robot.data.joint_pos_limits.torch[0, joint_ids]
        generator = torch.Generator(device=env.unwrapped.device).manual_seed(42)
        sampled_joint_pos = joint_limits[:, 0] + (joint_limits[:, 1] - joint_limits[:, 0]) * torch.rand(
            (4096, 7), device=env.unwrapped.device, generator=generator
        )
        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            # This seed-42 training sample made the former float32 chain return NaN.
            regression_fk_position, regression_fk_orientation = forward_kinematics(
                torch.tensor([_TF32_REGRESSION_JOINT_POSITION], device=env.unwrapped.device)
            )
            sampled_fk_position, sampled_fk_orientation = forward_kinematics(sampled_joint_pos)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        _assert_finite((regression_fk_position, regression_fk_orientation))
        torch.testing.assert_close(
            torch.linalg.vector_norm(regression_fk_orientation, dim=-1),
            torch.ones(1, device=env.unwrapped.device),
            atol=1.0e-5,
            rtol=0.0,
        )
        _assert_finite((sampled_fk_position, sampled_fk_orientation))
        torch.testing.assert_close(
            torch.linalg.vector_norm(sampled_fk_orientation, dim=-1),
            torch.ones(4096, device=env.unwrapped.device),
            atol=1.0e-5,
            rtol=0.0,
        )

        # Measure the actual FK target distribution against the table without changing it.
        sample_center = 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])
        table_sampled_joint_pos = sample_center + command_term.cfg.joint_range_scale * (
            sampled_joint_pos - sample_center
        )
        table_sampled_fk_position, _ = forward_kinematics(table_sampled_joint_pos)
        target_position_w = (
            quat_apply(robot.data.root_link_pose_w.torch[0, 3:7].expand(4096, -1), table_sampled_fk_position)
            + robot.data.root_link_pose_w.torch[0, :3]
        )
        target_position_e = target_position_w - env.unwrapped.scene.env_origins[0]
        target_above_table = target_position_e[:, 2] >= _TABLE_TOP_Z
        target_inside_table_xy = (
            (target_position_e[:, 0] >= _TABLE_X_BOUNDS[0])
            & (target_position_e[:, 0] <= _TABLE_X_BOUNDS[1])
            & (target_position_e[:, 1] >= _TABLE_Y_BOUNDS[0])
            & (target_position_e[:, 1] <= _TABLE_Y_BOUNDS[1])
        )
        target_inside_table = target_inside_table_xy & ~target_above_table
        assert target_above_table.float().mean() >= 0.98
        assert target_inside_table.float().mean() <= 0.02

        # Exercise every joint at a non-zero angle before comparing the full chain again.
        non_default_joint_pos = torch.tensor([0.3, -0.9, 0.4, 0.7, -0.5, 0.8, 0.6], device=env.unwrapped.device).repeat(
            env.unwrapped.num_envs, 1
        )
        robot.write_joint_position_to_sim_index(position=non_default_joint_pos, joint_ids=joint_ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(non_default_joint_pos), joint_ids=joint_ids)
        env.unwrapped.sim.forward()
        robot.update(0.0)

        fk_position, fk_orientation = forward_kinematics(non_default_joint_pos)
        link7_position, link7_orientation = subtract_frame_transforms(
            robot.data.root_link_pose_w.torch[:, :3],
            robot.data.root_link_pose_w.torch[:, 3:7],
            robot.data.body_link_pose_w.torch[:, body_ids[0], :3],
            robot.data.body_link_pose_w.torch[:, body_ids[0], 3:7],
        )
        torch.testing.assert_close(fk_position, link7_position, atol=1.0e-5, rtol=1.0e-5)
        torch.testing.assert_close(
            quat_error_magnitude(fk_orientation, link7_orientation),
            torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device),
            atol=1.0e-5,
            rtol=0.0,
        )

        observations, _ = env.reset()
        _assert_finite(observations)

        zero_actions = torch.zeros((env.unwrapped.num_envs, 7), device=env.unwrapped.device)
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
        _assert_finite(robot.data.joint_vel.torch)
        _assert_finite(robot.actuators.applied_effort.torch)
        assert torch.all(
            torch.abs(robot.actuators.applied_effort.torch[:, joint_ids])
            <= torch.tensor(_JOINT_EFFORT_LIMITS, device=env.unwrapped.device) + 1.0e-5
        )
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()
