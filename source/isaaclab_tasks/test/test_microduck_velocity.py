# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck walking: deployed policy interface, rough-terrain foot sensing, and the backlash preset.

Stepping every registered contributed task is covered by ``test/contrib/test_contrib_environments_kitless.py``.
"""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.flat_env_cfg import MICRODUCK_JOINT_NAMES
from isaaclab_tasks.contrib.microduck.mdp.observations import foot_height_safe
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
ROUGH_TASK = "IsaacContrib-Velocity-Rough-MicroDuck"

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
"""Actor input order of the deployed ONNX policy."""


def _make_env(task: str, device: str, num_envs: int, overrides: tuple[str, ...] = (), cfg_hook=None):
    cfg = parse_env_cfg(task, device=device, num_envs=num_envs, overrides=overrides)
    if cfg_hook is not None:
        cfg_hook(cfg)
    sim_utils.create_new_stage()
    return gym.make(task, cfg=cfg).unwrapped


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_deploy_interface_and_friction_reset(device):
    """The deployed policy's 61/76 inputs and joint order hold, and each reset redraws BAM friction."""
    env = _make_env(TASK, device, num_envs=2)
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, 61)
        assert obs["critic"].shape == (2, 76)
        assert env.observation_manager.active_terms["policy"] == POLICY_TERMS
        assert env.action_manager.get_term("joint_pos").IO_descriptor.joint_names == MICRODUCK_JOINT_NAMES

        robot = env.scene["robot"]
        before = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale").clone()
        env.reset()
        after = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale")
        assert not (before == after).all()
        assert (after == after[:, :1]).all(), "one friction scale per environment"
        assert ((after >= 0.9) & (after <= 1.1)).all()
    finally:
        env.close()
        SimulationContext.clear_instance()


def test_microduck_foot_height_on_steps():
    """Use the closest terrain sample per foot, ignoring misses, step undersides, and world elevation.

    Missed rays and rays starting inside a step cannot be placed reliably in a live scene.
    """
    foot_positions = torch.tensor([[[0.0, 0.0, 0.05], [0.0, 0.0, 0.07]]])
    left_hits = torch.tensor([[[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]]])
    right_hits = torch.tensor([[[0.0, 0.0, 0.03], [float("inf"), float("inf"), float("inf")]]])
    left_normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])

    def ray_sensor(hits, normals):
        data = SimpleNamespace(ray_hits_w=SimpleNamespace(torch=hits), ray_normals_w=SimpleNamespace(torch=normals))
        return SimpleNamespace(data=data, cfg=SimpleNamespace(max_distance=1.0))

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene(
        robot=SimpleNamespace(data=SimpleNamespace(body_link_pos_w=SimpleNamespace(torch=foot_positions))),
        left=ray_sensor(left_hits, left_normals),
        right=ray_sensor(right_hits, left_normals.clone()),
    )
    env = SimpleNamespace(scene=scene)
    params = {"asset_cfg": SceneEntityCfg("robot", body_ids=[0, 1]), "height_sensor_names": ("left", "right")}
    expected = torch.tensor([[0.03, 0.04]])
    torch.testing.assert_close(foot_height_safe(env, **params), expected)
    for tensor in (foot_positions, left_hits, right_hits, scene.env_origins):
        tensor[..., 2] += 4.0
    torch.testing.assert_close(foot_height_safe(env, **params), expected)
    right_hits.fill_(float("inf"))
    torch.testing.assert_close(foot_height_safe(env, **params), torch.tensor([[0.03, 0.07]]))
    left_normals[0, 0, 2] = -1.0
    torch.testing.assert_close(foot_height_safe(env, **params), torch.tensor([[0.0, 0.07]]))


def _small_terrain(cfg):
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows, generator.num_cols, generator.seed = 2, 4, 42
    for sub_terrain in generator.sub_terrains.values():
        sub_terrain.proportion = 0.25


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_rough_terrain_contact_and_foot_rays(device):
    """The terrain carries upstream's contact response, and foot rays see the ground under each foot."""
    env = _make_env(ROUGH_TASK, device, num_envs=8, cfg_hook=_small_terrain)
    try:
        env.reset()
        mesh = sim_utils.get_current_stage().GetPrimAtPath("/World/ground/terrain/mesh")
        assert tuple(mesh.GetAttribute("mjc:solref").Get()) == pytest.approx((0.04, 1.0))
        assert tuple(mesh.GetAttribute("mjc:solimp").Get()) == pytest.approx((0.85, 0.95, 0.001, 0.5, 2.0))
        assert env.scene.env_origins[:, 2].max() > 0.05, "some environments must start on raised tiles"
        for name in ("left_foot_height", "right_foot_height"):
            hits = env.scene[name].data.ray_hits_w.torch
            assert torch.isfinite(hits).all()
            torch.testing.assert_close(hits[..., 2], env.scene.env_origins[:, 2, None].expand(-1, 2), atol=2e-3, rtol=0)
        heights = foot_height_safe(env, **env.observation_manager.cfg.critic.foot_height.params)
        assert ((heights > 0.01) & (heights < 0.04)).all()
    finally:
        env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_backlash_preset_reads_output_side_encoders(device):
    """With ``presets=backlash``, encoders and head rewards see servo plus play, and play joints stay undriven.

    Distinct play values per joint and environment catch pairing or ordering mistakes. The kitless contributed-task
    test only steps default presets, so this test also steps the backlash robot.
    """
    env = _make_env(TASK, device, num_envs=8, overrides=("presets=backlash",))
    try:
        env.reset()
        robot = env.scene["robot"]
        assert robot.num_joints == 28
        play_ids = [robot.joint_names.index(f"passive_{name}_backlash") for name in MICRODUCK_JOINT_NAMES]
        torch.testing.assert_close(robot.data.joint_pos.torch[:, play_ids], torch.zeros(8, 14, device=env.device))

        play = torch.linspace(-0.014, 0.014, 8 * 14, device=env.device).reshape(8, 14)
        q = robot.data.default_joint_pos.torch.clone()
        qd = torch.zeros_like(q)
        q[:, play_ids] = play
        qd[:, play_ids] = play * 10.0
        robot.write_joint_state_to_sim_index(position=q, velocity=qd)
        bias = env.action_manager.get_term("joint_pos").encoder_bias
        for group in (env.observation_manager.cfg.policy, env.observation_manager.cfg.critic):
            term = group.joint_pos
            expected = play + (bias if term.params["biased"] else 0.0)
            torch.testing.assert_close(term.func(env, **term.params), expected)
            term = group.joint_vel
            torch.testing.assert_close(term.func(env, **term.params), play * 10.0)
        env.command_manager.get_command("head_pose").zero_()
        head = env.reward_manager.get_term_cfg("head_pose_tracking")
        expected = torch.exp(-((play[:, 5:9] / head.params["std"]) ** 2)).mean(dim=-1)
        torch.testing.assert_close(head.func(env, **head.params), expected)
        head_bias = env.reward_manager.get_term_cfg("head_pose_bias")
        head_bias.func.reset()
        expected = play[:, 5:9].abs().mean(dim=-1) * env.step_dt / head_bias.params["tau_s"]
        torch.testing.assert_close(head_bias.func(env, **head_bias.params), expected)
        # Riding the play stops must not incur a servo soft-limit penalty.
        q[:, play_ids] = robot.data.soft_joint_pos_limits.torch[:, play_ids, 1] + 0.001
        robot.write_joint_state_to_sim_index(position=q, velocity=qd)
        limits = env.reward_manager.get_term_cfg("dof_pos_limits")
        torch.testing.assert_close(limits.func(env, **limits.params), torch.zeros(8, device=env.device))

        env.reset()
        with torch.inference_mode():
            for _ in range(20):
                obs, reward, *_ = env.step(torch.randn(8, 14, device=env.device) * 0.1)
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
        effort = robot.actuators.applied_effort.torch[:, play_ids]
        torch.testing.assert_close(effort, torch.zeros_like(effort))
    finally:
        env.close()
        SimulationContext.clear_instance()
