# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backlash presets preserve encoder, latency and independent standard configurations."""

import re
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import clone

from isaaclab_tasks.contrib.microduck import backlash
from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg
from isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

pytestmark = pytest.mark.unit


def test_backlash_encoders_preserve_pair_order_defaults_and_servo_bias():
    def view(values):
        return SimpleNamespace(torch=torch.tensor([values], dtype=torch.float32))

    data = SimpleNamespace(
        joint_pos=view([0.01, 0.7, 0.04, -0.2]),
        default_joint_pos=view([0.0, 0.5, 0.0, -0.1]),
        joint_vel=view([0.4, 2.0, 0.6, -1.0]),
        default_joint_vel=view([0.0, 0.1, 0.0, -0.2]),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data, num_joints=4)},
        _microduck_encoder_bias={"robot": torch.tensor([[0.0, 0.03, 0.0, -0.02]])},
    )
    servos = SceneEntityCfg("robot", joint_ids=[3, 1])
    play = SceneEntityCfg("robot", joint_ids=[0, 2])
    torch.testing.assert_close(
        backlash.joint_pos_rel_backlash_biased(env, servos, play, biased=False),
        torch.tensor([[-0.09, 0.24]]),
    )
    torch.testing.assert_close(
        backlash.joint_pos_rel_backlash_biased(env, servos, play, biased=True),
        torch.tensor([[-0.11, 0.27]]),
    )
    torch.testing.assert_close(
        backlash.joint_vel_rel_backlash(env, servos, play),
        torch.tensor([[-0.4, 2.5]]),
    )


def test_backlash_selection_preserves_default_initial_pose_and_delayed_sensor_contract():
    cfg = MicroDuckVelocityFlatEnvCfg()
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.245)
    standard_path = cfg.scene.robot.spawn.usd_path
    variant = clone(cfg.scene.robot)
    variant.spawn.usd_path = "/unloaded-test-asset/microduck_walk_backlash.usd"
    variant.actuators["servos"].backlash_joint_template = "passive_{}_backlash"
    backlash.apply_backlash_preset(cfg, variant)
    standard = resolve_presets(deepcopy(cfg))
    selected = resolve_presets(deepcopy(cfg), selected=("backlash",))
    assert standard.scene.robot.spawn.usd_path == standard_path
    assert selected.scene.robot.spawn.usd_path == variant.spawn.usd_path
    assert selected.scene.robot.init_state.pos == (0.0, 0.0, 0.245)
    assert selected.scene.robot.init_state.joint_pos["passive_.*_backlash"] == 0.0
    assert "backlash_cfg" not in standard.observations.policy.joint_pos.params
    velocity = selected.observations.policy.joint_vel
    original_velocity = standard.observations.policy.joint_vel
    assert velocity.func is original_velocity.func
    assert velocity.params["min_lag"] == original_velocity.params["min_lag"] == 1
    assert velocity.params["max_lag"] == original_velocity.params["max_lag"] == 1
    assert velocity.noise.n_min == original_velocity.noise.n_min == -0.25
    assert velocity.params["term_func"] is backlash.joint_vel_rel_backlash
    assert velocity.params["term_params"]["backlash_cfg"].joint_names == [
        f"passive_{name}_backlash" for name in selected.actions.joint_pos.joint_names
    ]
    assert selected.sim.physics.solver_cfg.njmax >= standard.sim.physics.solver_cfg.njmax + 14
    assert MicroDuckVelocityFlatEnvCfg().scene.robot.spawn.usd_path == standard_path


def test_backlash_agent_experiments_are_distinct_without_changing_training_budget():
    agent = MicroDuckPPORunnerCfg()
    name, budget = agent.experiment_name, agent.max_iterations
    backlash.backlash_agent_preset(agent)
    standard = resolve_presets(deepcopy(agent))
    selected = resolve_presets(deepcopy(agent), selected=("backlash",))
    assert standard.experiment_name == name
    assert selected.experiment_name == name + "_backlash"
    assert selected.max_iterations == standard.max_iterations == budget


@pytest.mark.parametrize("existing_pattern", ["passive_.*", "passive_left_.*"])
def test_backlash_initialization_does_not_duplicate_existing_passive_joint_patterns(existing_pattern):
    cfg = MicroDuckVelocityFlatEnvCfg()
    cfg.scene.robot.init_state.joint_pos[existing_pattern] = 0.0
    variant = clone(cfg.scene.robot)
    variant.actuators["servos"].backlash_joint_template = "passive_{}_backlash"
    backlash.apply_backlash_preset(cfg, variant)
    cfg = resolve_presets(cfg, selected=("backlash",))
    defaults = cfg.scene.robot.init_state.joint_pos
    for name in cfg.actions.joint_pos.joint_names:
        joint = f"passive_{name}_backlash"
        matches = [pattern for pattern in defaults if re.fullmatch(pattern, joint)]
        assert len(matches) == 1, f"Ambiguous initial state for {joint}: {matches}"
        assert defaults[matches[0]] == 0.0


@pytest.mark.parametrize("kind", ["tracking", "bias"])
def test_head_rewards_measure_output_compensation_without_observation_bias(kind):
    from isaaclab_tasks.contrib.microduck.pollen_mdp import rewards

    positions = torch.tensor([[-0.2, 0.01, 0.4, -0.01], [-0.2, 0.01, 0.4, -0.01]])
    commands = torch.tensor([[0.41, -0.21], [0.4, -0.2]])
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=SimpleNamespace(torch=positions),
                    default_joint_pos=SimpleNamespace(torch=torch.zeros_like(positions)),
                )
            )
        },
        command_manager=SimpleNamespace(get_command=lambda name: commands),
        _microduck_encoder_bias={"robot": torch.full_like(positions, 999.0)},
    )
    params = {
        "command_name": "head_pose",
        "asset_cfg": SceneEntityCfg("robot", joint_ids=[2, 0]),
        "backlash_cfg": SceneEntityCfg("robot", joint_ids=[1, 3]),
    }
    if kind == "tracking":
        result = rewards.head_pose_tracking(env, std=0.5, **params)
        expected = torch.tensor([1.0, torch.exp(torch.tensor(-((0.01 / 0.5) ** 2)))])
    else:
        params["tau_s"] = 1.0
        cfg = RewardTermCfg(func=rewards.head_pose_bias_penalty, weight=1.0, params=params)
        term = rewards.head_pose_bias_penalty(cfg, env)
        result = term(env, **params)
        expected = torch.tensor([0.0, -0.0002])
    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-5)
