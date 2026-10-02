# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment contract and reset/step checks for MicroDuck walking."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
ROUGH_TASK = "IsaacContrib-Velocity-Rough-MicroDuck"
BACKLASH_TASK = "IsaacContrib-Velocity-Flat-Backlash-MicroDuck"
ROUGH_BACKLASH_TASK = "IsaacContrib-Velocity-Rough-Backlash-MicroDuck"
TASKS = [TASK, ROUGH_TASK, BACKLASH_TASK, ROUGH_BACKLASH_TASK]
JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
POLICY_TERMS = [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "velocity_commands",
    "head_pose_commands",
    "body_pose_commands",
]


@pytest.mark.parametrize("task", TASKS)
def test_microduck_policy_contract(task):
    """Keep the original walking policy's joint order and 50 Hz interface."""
    cfg = parse_env_cfg(task, device="cpu", num_envs=2)
    assert cfg.sim.dt * cfg.decimation == pytest.approx(0.02)
    assert cfg.actions.joint_pos.joint_names == JOINT_NAMES
    assert cfg.actions.joint_pos.preserve_order
    assert cfg.actions.joint_pos.scale == 1.0
    terms = [name for name in vars(cfg.observations.policy) if name in POLICY_TERMS]
    assert terms == POLICY_TERMS
    assert cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names == JOINT_NAMES
    assert cfg.scene.terrain.terrain_type == ("generator" if "Rough" in task else "plane")
    assert cfg.sim.use_newton_actuators


def test_microduck_foot_height_on_steps():
    """Use the closest terrain sample per foot, ignoring misses and world elevation."""
    from isaaclab_tasks.contrib.microduck.mdp.observations import foot_height_safe

    foot_positions = torch.tensor([[[0.0, 0.0, 0.05], [0.0, 0.0, 0.07]]])
    left_hits = torch.tensor([[[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]]])
    right_hits = torch.tensor([[[0.0, 0.0, 0.03], [float("inf"), float("inf"), float("inf")]]])
    left_normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    scene = {
        "robot": SimpleNamespace(data=SimpleNamespace(body_link_pos_w=SimpleNamespace(torch=foot_positions))),
        "left": SimpleNamespace(
            data=SimpleNamespace(
                ray_hits_w=SimpleNamespace(torch=left_hits), ray_normals_w=SimpleNamespace(torch=left_normals)
            ),
            cfg=SimpleNamespace(max_distance=1.0),
        ),
        "right": SimpleNamespace(
            data=SimpleNamespace(
                ray_hits_w=SimpleNamespace(torch=right_hits), ray_normals_w=SimpleNamespace(torch=left_normals.clone())
            ),
            cfg=SimpleNamespace(max_distance=1.0),
        ),
    }

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    env = SimpleNamespace(scene=Scene(scene))
    params = {"asset_cfg": SceneEntityCfg("robot", body_ids=[0, 1]), "height_sensor_names": ("left", "right")}
    expected = torch.tensor([[0.03, 0.04]])
    torch.testing.assert_close(foot_height_safe(env, **params), expected)
    foot_positions[..., 2] += 4.0
    left_hits[..., 2] += 4.0
    right_hits[..., 2] += 4.0
    env.scene.env_origins[:, 2] += 4.0
    torch.testing.assert_close(foot_height_safe(env, **params), expected)
    right_hits.fill_(float("inf"))
    torch.testing.assert_close(foot_height_safe(env, **params), torch.tensor([[0.03, 0.07]]))
    # A ray starting inside a step hits its underside; that foot has no clearance.
    left_normals[0, 0, 2] = -1.0
    torch.testing.assert_close(foot_height_safe(env, **params), torch.tensor([[0.0, 0.07]]))


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("task", TASKS)
def test_microduck_reset_and_step(device, task):
    """Resolve the policy interface on the USD and exercise reset-time BAM writes."""
    cfg = parse_env_cfg(task, device=device, num_envs=8)
    if "Rough" in task:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 4
        cfg.scene.terrain.terrain_generator.seed = 42
        for sub_terrain in cfg.scene.terrain.terrain_generator.sub_terrains.values():
            sub_terrain.proportion = 0.25
    sim_utils.create_new_stage()
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (8, 61)
        assert obs["critic"].shape == (8, 76)
        assert env.observation_manager.active_terms["policy"] == POLICY_TERMS
        robot = env.scene["robot"]
        ids = env.action_manager.get_term("joint_pos")._joint_ids
        assert [robot.joint_names[i] for i in ids] == JOINT_NAMES
        before = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale").clone()
        env.reset()
        after = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale")
        assert not torch.equal(before, after)
        torch.testing.assert_close(after, after[:, :1].expand_as(after))
        assert ((after >= 0.9) & (after <= 1.1)).all()
        if "Rough" in task:
            from isaaclab_newton.physics import NewtonManager

            from isaaclab_tasks.contrib.microduck.mdp.observations import foot_height_safe

            solver = NewtonManager._solver
            terrain_shape = NewtonManager.backend.model.shape_label.index("/World/ground/terrain/mesh")
            terrain_geom = np.flatnonzero(solver.mjc_geom_to_newton_shape.numpy()[0] == terrain_shape).item()
            for name, expected in (("geom_solref", [0.04, 1.0]), ("geom_solimp", [0.85, 0.95, 0.001, 0.5, 2.0])):
                actual = getattr(solver.mjw_model, name).numpy()[:, terrain_geom]
                np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=1e-6)
            assert env.scene.env_origins[:, 2].max() > 0.05
            for name in ("left_foot_height", "right_foot_height"):
                hits = env.scene[name].data.ray_hits_w.torch
                assert hits.shape == (8, 2, 3)
                assert torch.isfinite(hits).all()
                torch.testing.assert_close(
                    hits[..., 2], env.scene.env_origins[:, 2, None].expand(-1, 2), atol=2e-3, rtol=0
                )
            heights = foot_height_safe(env, **env.observation_manager.cfg.critic.foot_height.params)
            assert ((heights > 0.01) & (heights < 0.04)).all()
        if "Backlash" in task:
            assert robot.num_joints == 28
            passive_ids, _ = robot.find_joints("passive_.*_backlash")
            torch.testing.assert_close(
                robot.data.joint_pos.torch[:, passive_ids], torch.zeros(8, 14, device=env.device)
            )
        if task == BACKLASH_TASK:
            # Distinct play values catch pairing/order mistakes in the policy's encoder view.
            from isaaclab_tasks.contrib.microduck.mdp.events import encoder_bias

            play_ids = [robot.joint_names.index(f"passive_{name}_backlash") for name in JOINT_NAMES]
            play = torch.linspace(-0.014, 0.014, 8 * 14, device=env.device).reshape(8, 14)
            q = robot.data.default_joint_pos.torch.clone()
            qd = torch.zeros_like(q)
            q[:, play_ids] = play
            qd[:, play_ids] = play * 10.0
            robot.write_joint_state_to_sim_index(position=q, velocity=qd)
            for group in (env.observation_manager.cfg.policy, env.observation_manager.cfg.critic):
                term = group.joint_pos
                expected = play + (encoder_bias(env)[:, ids] if term.params["biased"] else 0.0)
                torch.testing.assert_close(term.func(env, **term.params), expected)
            term = env.observation_manager.cfg.policy.joint_vel
            torch.testing.assert_close(term.func(env, **term.params), play * 10.0)
            term = env.observation_manager.cfg.critic.joint_vel
            torch.testing.assert_close(term.func(env, **term.params), play * 10.0)
            env.command_manager.get_command("head_pose").zero_()
            head = env.reward_manager.get_term_cfg("head_pose_tracking")
            expected = torch.exp(-((play[:, 5:9] / head.params["std"]) ** 2)).mean(dim=-1)
            torch.testing.assert_close(head.func(env, **head.params), expected)
            head_bias = env.reward_manager.get_term_cfg("head_pose_bias")
            head_bias.func.reset()
            expected = -play[:, 5:9].abs().mean(dim=-1) * env.step_dt / head_bias.params["tau_s"]
            torch.testing.assert_close(head_bias.func(env, **head_bias.params), expected)
            # Riding the play stops must not incur a servo soft-limit penalty.
            q[:, play_ids] = robot.data.soft_joint_pos_limits.torch[:, play_ids, 1] + 0.001
            robot.write_joint_state_to_sim_index(position=q, velocity=qd)
            limits = env.reward_manager.get_term_cfg("dof_pos_limits")
            torch.testing.assert_close(limits.func(env, **limits.params), torch.zeros(8, device=env.device))
            env.reset()
        with torch.inference_mode():
            for _ in range(64):
                obs, reward, terminated, truncated, _ = env.step(torch.randn(8, 14, device=env.device) * 0.1)
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
        assert terminated.shape == truncated.shape == (8,)
        if "Backlash" in task:
            torch.testing.assert_close(
                robot.actuators.applied_effort.torch[:, passive_ids], torch.zeros(8, 14, device=env.device)
            )
    finally:
        env.close()
        SimulationContext.clear_instance()
