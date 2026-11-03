# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral checks for source recovery resets, command gating and actual-effort protection."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names

from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import MICRODUCK_JOINT_NAMES
from isaaclab_tasks.contrib.microduck.velstand import mdp

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def view(tensor):
    return SimpleNamespace(torch=tensor)


class Robot:
    def __init__(self, n):
        self.joint_names = MICRODUCK_JOINT_NAMES
        self.num_joints = 14
        quat = torch.zeros(n, 4)
        quat[:, 3] = 1
        self.data = SimpleNamespace(
            root_link_pos_w=view(torch.tensor([0.0, 0.0, 0.125]).repeat(n, 1)),
            root_link_quat_w=view(quat),
            default_joint_pos=view(torch.zeros(n, 14)),
            joint_pos=view(torch.zeros(n, 14)),
            joint_vel=view(torch.zeros(n, 14)),
            joint_pos_limits=view(torch.tensor([-1.0, 1.0]).repeat(n, 14, 1)),
        )
        self.actuators = SimpleNamespace(applied_effort=view(torch.zeros(n, 14)))
        self.pose_written = torch.zeros(n, dtype=torch.bool)

    def find_joints(self, names, preserve_order=False):
        return resolve_matching_names(names, self.joint_names, preserve_order=preserve_order)

    def write_root_link_pose_to_sim_index(self, root_pose, env_ids):
        self.data.root_link_pos_w.torch[env_ids] = root_pose[:, :3]
        self.data.root_link_quat_w.torch[env_ids] = root_pose[:, 3:]
        self.pose_written[env_ids] = True

    def write_root_com_velocity_to_sim_index(self, root_velocity, env_ids):
        assert torch.count_nonzero(root_velocity) == 0

    def write_joint_position_to_sim_index(self, position, env_ids):
        self.data.joint_pos.torch[env_ids] = position

    def write_joint_velocity_to_sim_index(self, velocity, env_ids):
        self.data.joint_vel.torch[env_ids] = velocity


class Scene(dict):
    pass


def environment(n):
    robot = Robot(n)
    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(n, 3)
    scene.env_origins[:, 2] = torch.linspace(0.1, 0.3, n)
    robot.data.root_link_pos_w.torch[:, 2] += scene.env_origins[:, 2]
    return SimpleNamespace(
        num_envs=n, device="cpu", step_dt=0.02, scene=scene, episode_length_buf=torch.full((n,), 2, dtype=torch.long)
    )


def test_prone_mixture_uses_terrain_height_and_or_randomization_clearance():
    torch.manual_seed(42)
    env = environment(20000)
    ac = SceneEntityCfg("robot", joint_ids=list(range(14)))
    mdp.reset_velstand_state(env, None, ac, prone_prob=1.0, sit_prob=0.0)
    robot = env.scene["robot"]
    z = robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    random_joints = robot.data.joint_pos.torch.abs().amax(1) > 0
    # Independent half draw OR always-random SO(3) quarter: .5 + .25 - .125 = .625.
    assert random_joints.float().mean().item() == pytest.approx(0.625, abs=0.015)
    assert z[~random_joints].min() >= 0.05 - 1e-6
    assert z[~random_joints].max() <= 0.09 + 1e-6
    assert z[random_joints].min() >= 0.11 - 1e-6
    assert z.max() <= 0.19 + 1e-6
    assert robot.data.joint_pos.torch.abs().max() <= 0.9
    torch.testing.assert_close(robot.data.root_link_quat_w.torch.norm(dim=1), torch.ones(env.num_envs))
    # Axis-aligned quarter-turn starts account for 3/4; uniform SO(3) supplies the rest.
    q = robot.data.root_link_quat_w.torch
    cos_tilt = 1 - 2 * (q[:, 0].square() + q[:, 1].square())
    assert (cos_tilt.abs() > 1e-5).float().mean().item() == pytest.approx(0.25, abs=0.015)


def test_reset_slices_are_exclusive_and_unselected_standing_is_untouched():
    torch.manual_seed(43)
    env = environment(12000)
    ac = SceneEntityCfg("robot", joint_ids=list(range(14)))
    original = env.scene["robot"].data.root_link_pos_w.torch.clone()
    mdp.reset_velstand_state(env, slice(None), ac, prone_prob=0.45, crouch_prob=0.2, sit_prob=0.05)
    robot = env.scene["robot"]
    assert robot.pose_written.float().mean().item() == pytest.approx(0.7, abs=0.015)
    torch.testing.assert_close(robot.data.root_link_pos_w.torch[~robot.pose_written], original[~robot.pose_written])
    height = robot.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    assert height.min() > 0.05 - 1e-6


def test_body_commands_gate_again_on_twist_changes_without_losing_sampled_value():
    env = environment(4)
    twist = SimpleNamespace(is_standing_env=torch.tensor([True, False, True, False]))
    env.command_manager = SimpleNamespace(get_term=lambda name: twist)
    cfg = mdp.StandingGatedPoseCommandCfg(
        ranges=((0.1, 0.1),) * 6, axis_zero_prob=(0.0,) * 6, zero_command_prob=0.0, resampling_time_range=(2.0, 5.0)
    )
    term = mdp.StandingGatedPoseCommand(cfg, env)
    term.reset()
    torch.testing.assert_close(term.command[:, 0], torch.tensor([0.1, 0.0, 0.1, 0.0]))
    twist.is_standing_env.logical_not_()
    term.compute(0.02)
    torch.testing.assert_close(term.command[:, 0], torch.tensor([0.0, 0.1, 0.0, 0.1]))


def test_protection_uses_native_actual_effort_and_first_frame_acceleration_is_free():
    env = environment(2)
    robot = env.scene["robot"]
    ac = SceneEntityCfg("robot", joint_ids=[0, 1])
    robot.actuators.applied_effort.torch[:, :2] = torch.tensor([[0.5, 0.5], [0.1, 0.8]])
    robot.data.joint_vel.torch[:, :2] = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    torch.testing.assert_close(mdp.servo_stall_penalty(env, ac), torch.tensor([1.0, 1.0]))
    cfg = RewardTermCfg(func=mdp.servo_acc_spike_penalty, weight=-0.001, params={"asset_cfg": ac})
    term = mdp.servo_acc_spike_penalty(cfg, env)
    assert torch.count_nonzero(term(env, ac)) == 0
    robot.data.joint_vel.torch[0, 0] = 10.0
    env.episode_length_buf[1] = 1
    robot.data.joint_vel.torch[1, 0] = 100.0
    torch.testing.assert_close(term(env, ac), torch.tensor([200.0, 0.0]))
    env.action_manager = SimpleNamespace(action=torch.ones(2, 14), prev_action=torch.zeros(2, 14))
    robot.data.root_link_quat_w.torch[1] = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    torch.testing.assert_close(mdp.action_rate_l2_fallen_scaled(env), torch.tensor([14.0, 1.4]))


def test_invalid_physics_before_reset_cannot_poison_new_rewards_or_persistent_state():
    env = environment(2)
    robot = env.scene["robot"]
    robot.data.GRAVITY_VEC_W = view(torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1))
    robot.data.root_link_ang_vel_b = view(torch.zeros(2, 3))
    env.action_manager = SimpleNamespace(action=torch.zeros(2, 14), prev_action=torch.zeros(2, 14))
    commands = {"base_velocity": torch.zeros(2, 3), "body_pose": torch.zeros(2, 6)}
    env.command_manager = SimpleNamespace(
        get_command=lambda name: commands[name],
        get_term=lambda name: SimpleNamespace(is_standing_env=torch.ones(2, dtype=torch.bool)),
    )
    ac = SceneEntityCfg("robot", joint_ids=list(range(14)))
    posture = {name: {".*": 0.2} for name in ("std_standing", "std_walking", "std_running", "std_body")}
    posture_cfg = RewardTermCfg(func=mdp.variable_posture_body_relaxed, weight=1.0, params={"asset_cfg": ac, **posture})
    pose = mdp.variable_posture_body_relaxed(posture_cfg, env)
    torque = mdp.joint_torque_rate_l2_fallen_scaled(
        RewardTermCfg(func=mdp.joint_torque_rate_l2_fallen_scaled, weight=-0.002, params={"asset_cfg": ac}), env
    )
    acceleration = mdp.servo_acc_spike_penalty(
        RewardTermCfg(func=mdp.servo_acc_spike_penalty, weight=-0.001, params={"asset_cfg": ac}), env
    )
    torque(env, ac)
    acceleration(env, ac)
    robot.data.root_link_quat_w.torch[0, 0] = float("nan")
    robot.data.root_link_ang_vel_b.torch[0, 2] = float("nan")
    robot.data.joint_pos.torch[0, 0] = float("nan")
    robot.actuators.applied_effort.torch[0, 0] = float("nan")
    robot.data.joint_vel.torch[0, 0] = float("inf")
    env.action_manager.action[0, 0] = float("nan")
    values = [
        mdp.upright_body_cmd_relative(env, 0.5, "body_pose"),
        mdp.track_yaw_rate_fine(env, 0.35, "base_velocity"),
        pose(env, "base_velocity", walking_threshold=0.1, running_threshold=0.5, asset_cfg=ac, **posture),
        mdp.body_pose_tracking_standing(env, "body_pose", 0.125, 0.01, 0.1),
        mdp.action_rate_l2_fallen_scaled(env),
        torque(env, ac),
        acceleration(env, ac),
    ]
    assert all(torch.isfinite(value).all() for value in values)
    for value in values[:4]:
        assert value[0] == 0
        assert value[1] == pytest.approx(1.0)
    # A later repaired/reset sample must resume meaningful finite costs, rather than
    # retaining a nonfinite previous torque/velocity and silently losing that term.
    robot.actuators.applied_effort.torch.fill_(0.2)
    robot.data.joint_vel.torch.fill_(10.0)
    robot.data.root_link_quat_w.torch[0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(torque(env, ac), torch.full((2,), 0.56))
    torch.testing.assert_close(acceleration(env, ac), torch.full((2,), 2800.0))


def test_reward_guard_keeps_good_terms_and_history_and_preserves_backlash_ema_reset():
    from isaaclab.envs import mdp as native_mdp
    from isaaclab.managers import RewardManager

    import isaaclab_tasks.contrib.microduck.mdp as shared

    env = environment(2)
    env.sim = SimpleNamespace(is_playing=lambda: True)
    env.max_episode_length_s = 20.0
    robot = env.scene["robot"]
    robot.joint_names = list(robot.joint_names) + [f"passive_{name}_backlash" for name in MICRODUCK_JOINT_NAMES]
    robot.num_joints = 28
    for name in ("joint_pos", "default_joint_pos"):
        setattr(robot.data, name, view(torch.zeros(2, 28)))
    env.action_manager = SimpleNamespace(action=torch.ones(2, 14), prev_action=torch.zeros(2, 14))
    env.command_manager = SimpleNamespace(get_command=lambda name: torch.zeros(2, 4))
    head = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES[5:9], preserve_order=True)
    play = SceneEntityCfg(
        "robot", joint_names=[f"passive_{name}_backlash" for name in MICRODUCK_JOINT_NAMES[5:9]], preserve_order=True
    )
    manager = RewardManager(
        {
            "constant": RewardTermCfg(func=lambda env: torch.full((env.num_envs,), 2.0), weight=1.0),
            "stock": RewardTermCfg(func=native_mdp.action_rate_l2, weight=1.0),
            "head": RewardTermCfg(
                func=shared.head_pose_bias_penalty_gated,
                weight=1.0,
                params={"command_name": "head_pose", "tau_s": 1.0, "asset_cfg": head, "backlash_cfg": play},
            ),
        },
        env,
    )
    env.reward_manager = manager
    mdp.install_finite_reward_guard(env, None)
    torch.testing.assert_close(manager.compute(0.02), torch.full((2,), 0.32))
    env.action_manager.action[0, 0] = float("nan")
    robot.data.joint_pos.torch[0, 5] = float("nan")
    # The valid constant must survive the stock and EMA terms' invalid first row.
    torch.testing.assert_close(manager.compute(0.02), torch.tensor([0.04, 0.32]))
    history = manager.reset([0])
    assert history["Episode_Reward/constant"] == pytest.approx(0.08 / 20.0)
    assert history["Episode_Reward/stock"] == pytest.approx(0.28 / 20.0)
    assert history["Episode_Reward/head"] == 0
    robot.data.joint_pos.torch[:, 5:9] = 0.1
    # Source output-side measurements remain intact beneath the guard.
    robot.data.joint_pos.torch[:, 19:23] = -0.1
    values = manager.compute(0.02)
    assert torch.isfinite(values).all()
    assert dict(manager.get_active_iterable_terms(0))["head"][0] == 0
    robot.data.joint_pos.torch[:, 19:23] = 0.0
    assert torch.isfinite(manager.compute(0.02)).all()
    assert dict(manager.get_active_iterable_terms(0))["head"][0] < 0
