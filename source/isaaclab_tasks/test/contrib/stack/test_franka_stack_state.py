# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the published state-policy input and full-stack success contracts."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import ObservationTermCfg, TerminationManager
from isaaclab.utils.string import resolve_matching_names

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.mdp.runtime_state import create_stack_reset_runtime_state
from isaaclab_tasks.utils import parse_env_cfg

pytestmark = pytest.mark.kitless


@pytest.fixture(scope="module")
def stack_cfg():
    """Load the registered state task without starting a renderer."""
    cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Franka-RL", device="cpu", num_envs=3)
    cfg.validate_config()
    return cfg


def test_state_policy_observation_order_matches_published_checkpoints(stack_cfg):
    policy = stack_cfg.observations.policy
    assert [name for name, term in vars(policy).items() if isinstance(term, ObservationTermCfg)] == [
        "actions",
        "joint_pos",
        "joint_vel",
        "object",
        "gripper_pos",
        "eef_velocity",
        "eef_axes",
    ]


def test_explicit_newton_selector_keeps_the_task_physics(stack_cfg):
    explicit = parse_env_cfg(
        "IsaacContrib-Stack-Cube-Franka-RL", device="cpu", num_envs=3, overrides=["physics=newton_mjwarp"]
    )
    assert explicit.sim.physics == stack_cfg.sim.physics


def test_play_mode_selects_randomized_table_starts(stack_cfg):
    cfg = stack_cfg.copy()
    cfg.play_mode()

    assert cfg.events.reset_from_state_buffer.params["fixed_recipe"] == int(mdp.StackResetRecipe.TABLE)
    assert cfg.curriculum is None
    assert cfg.actions.gripper_action.force_close_steps == 0


def test_success_requires_an_order_invariant_released_stack_held_at_rest(stack_cfg):
    class Scene(dict):
        env_origins = torch.zeros(3, 3)

    # The physical stack is asset 2 -> asset 3 -> asset 1 in each environment.
    scene = Scene()
    for cube_index, height in enumerate((0.10, 0.02, 0.06), start=1):
        positions = torch.zeros(3, 3)
        positions[:, 2] = height
        pose = torch.zeros(3, 7)
        pose[:, :3] = positions
        pose[:, 6] = 1.0
        velocities = torch.zeros(3, 6)
        if cube_index == 1:
            velocities[2, 2] = 0.2
        scene[f"cube_{cube_index}"] = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=positions),
                root_pose_w=SimpleNamespace(torch=pose),
                root_vel_w=SimpleNamespace(torch=velocities),
            )
        )
    joint_names = [f"panda_joint{number}" for number in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
    joint_positions = torch.zeros(3, 9)
    joint_positions[:, 7:] = torch.tensor([[0.04, 0.04], [0.02, 0.02], [0.04, 0.04]])
    scene["robot"] = SimpleNamespace(
        num_joints=9,
        joint_names=joint_names,
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_positions),
            joint_vel=SimpleNamespace(torch=torch.zeros_like(joint_positions)),
            body_pos_w=SimpleNamespace(torch=torch.zeros(3, 1, 3)),
            body_quat_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(3, -1, -1)),
        ),
        find_joints=lambda names, preserve_order=False: resolve_matching_names(names, joint_names, preserve_order),
        find_bodies=lambda name: ([0], [name]),
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        cfg=stack_cfg,
        scene=scene,
        sim=SimpleNamespace(is_playing=lambda: True),
        max_episode_length=100,
        episode_length_buf=torch.full((3,), 3),
        step_dt=0.02,
        extras={},
    )
    create_stack_reset_runtime_state(env)
    env.termination_manager = TerminationManager(stack_cfg.terminations, env)
    reward = stack_cfg.rewards.success

    for _ in range(4):
        assert not env.termination_manager.compute().any()
        assert not reward.func(env, **reward.params).any()
        env.episode_length_buf += 1

    assert env.termination_manager.compute().tolist() == [True, False, False]
    assert env.termination_manager.get_term("success").tolist() == [True, False, False]
    torch.testing.assert_close(
        reward.func(env, **reward.params) * env.step_dt * reward.weight, torch.tensor([2.0, 0.0, 0.0])
    )
    env.termination_manager.reset(env_ids=torch.tensor([0]))
    assert env.extras["successes"].tolist() == [True, False, False]
    assert not reward.func(env, **reward.params).any()
