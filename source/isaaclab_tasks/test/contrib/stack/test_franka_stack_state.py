# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the published state-policy input and full-stack success contracts."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import ObservationTermCfg

from isaaclab_tasks.contrib.stack import mdp
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
    # The physical stack is asset 2 -> asset 3 -> asset 1 in each environment.
    scene = {}
    for cube_index, height in enumerate((0.10, 0.02, 0.06), start=1):
        positions = torch.zeros(3, 3)
        positions[:, 2] = height
        velocities = torch.zeros(3, 6)
        if cube_index == 1:
            velocities[2, 2] = 0.2
        scene[f"cube_{cube_index}"] = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=positions), root_vel_w=SimpleNamespace(torch=velocities)
            )
        )
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.tensor([[0.04, 0.04], [0.02, 0.02], [0.04, 0.04]]))),
        find_joints=lambda names: ([0, 1], names),
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        cfg=stack_cfg,
        scene=scene,
        episode_length_buf=torch.full((3,), 3),
        step_dt=0.02,
        extras={},
    )
    progress_cfg = stack_cfg.terminations.progress_context
    context = progress_cfg.func(progress_cfg, env)
    env.termination_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=context))
    success = stack_cfg.terminations.success
    reward = stack_cfg.rewards.success

    for _ in range(4):
        context(env, **progress_cfg.params)
        assert not success.func(env, **success.params).any()
        env.episode_length_buf += 1

    context(env, **progress_cfg.params)
    assert success.func(env, **success.params).tolist() == [True, False, False]
    torch.testing.assert_close(reward.func(env, **reward.params) * env.step_dt, torch.tensor([1.0, 0.0, 0.0]))

    context(env, **progress_cfg.params)
    assert not reward.func(env, **reward.params).any()
